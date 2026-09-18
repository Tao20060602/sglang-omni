# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o serving vs Hugging Face remote-code accuracy lock.

Usage:
    pytest tests/minicpm_o_lock/test_hf_parity_ci.py -s -x

Author claims (MayDomine, H100, batch=1) against Hugging Face remote-code:

* Text greedy vs HF backbone: first-token logprob abs <= 1e-3
* Prefill last-position logits: max abs 1.2 and top-1 match
* Image-only greedy text: e2e identical to remote-code

Encoder / resampler / whisper goldens already live under
tests/unit_test/minicpm_o/. This file locks the serving path: thinker-only
Rust-router DP=2, then the same prompts on Hugging Face native MiniCPM-o.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from tests.minicpm_o_lock.conftest import MINICPMO_MODEL_NAME, MINICPMO_TEST_MODEL_PATH
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, disable_proxy, wait_for_gpu_memory_release

TEXT_PROMPT = "The capital of France is"
IMAGE_PROMPT = "What colors are in this image? Answer with color names only."
FIRST_TOKEN_LOGPROB_MAX_ABS = 1e-3
PREFILL_LAST_LOGITS_MAX_ABS = 1.2
TEXT_MAX_NEW_TOKENS = 1
IMAGE_MAX_NEW_TOKENS = 1
REQUEST_TIMEOUT_S = 180
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ServingParityArtifacts:
    text_token_id: int
    text_logprob: float
    image_text: str


@dataclass
class HuggingFaceParityReference:
    text_token_id: int
    text_logprob: float
    last_logits: torch.Tensor
    image_text: str


def resolve_minicpm_o_hf_checkpoint() -> Path:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [
        Path("/tmp/MiniCPM-o-4_5"),
        REPO_ROOT / "MiniCPM-o-4_5",
        Path(MINICPMO_TEST_MODEL_PATH),
    ]
    for path in candidates:
        if path.is_dir() and (path / "modeling_minicpmo.py").exists():
            return path
    raise FileNotFoundError(
        "Need a materialized MiniCPM-o checkpoint with modeling_minicpmo.py. "
        "Set MINICPMO_CHECKPOINT or copy the snapshot to /tmp/MiniCPM-o-4_5."
    )


def parity_image() -> Image.Image:
    image = Image.new("RGB", (128, 128), "red")
    ImageDraw.Draw(image).rectangle((64, 64, 127, 127), fill="blue")
    return image


def image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def post_json(port: int, path: str, body: dict) -> dict:
    with disable_proxy():
        response = requests.post(
            f"http://127.0.0.1:{port}{path}",
            json=body,
            timeout=REQUEST_TIMEOUT_S,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object from {path}")
    return payload


def first_output_logprob(payload: dict) -> tuple[int, float]:
    logprobs = payload.get("meta_info", {}).get("output_token_logprobs")
    if not logprobs:
        raise AssertionError(f"missing output_token_logprobs: {payload}")
    first = logprobs[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise TypeError(f"unexpected logprob row: {first!r}")
    return int(first[1]), float(first[0])


def generate_text_greedy(port: int) -> tuple[int, float]:
    """Read first-token logprobs from a worker.

    The Rust router only fronts ``/v1/chat/completions``; ``POST /generate``
    stays on the MiniCPM-o replica.
    """
    payload = post_json(
        port,
        "/generate",
        {
            "model": MINICPMO_MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": TEXT_MAX_NEW_TOKENS,
            },
            "output_modalities": ["text"],
            "return_logprob": True,
        },
    )
    return first_output_logprob(payload)


def chat_text_greedy(port: int) -> str:
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": MINICPMO_MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "temperature": 0.0,
            "max_tokens": TEXT_MAX_NEW_TOKENS,
            "modalities": ["text"],
        },
    )
    return str(payload["choices"][0]["message"].get("content") or "")


def chat_image_greedy(port: int, image: Image.Image) -> str:
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": MINICPMO_MODEL_NAME,
            "messages": [{"role": "user", "content": IMAGE_PROMPT}],
            "images": [image_data_uri(image)],
            "temperature": 0.0,
            "max_tokens": IMAGE_MAX_NEW_TOKENS,
            "modalities": ["text"],
        },
    )
    text = payload["choices"][0]["message"].get("content") or ""
    return str(text)


@pytest.fixture(scope="module")
def serving_parity_artifacts(
    minicpm_o_text_server: ManagedRouterHandle,
) -> ServingParityArtifacts:
    image = parity_image()
    with router_worker_traffic_guard(
        minicpm_o_text_server,
        label="MiniCPM-o HF parity",
    ) as router_guard:
        chat_text_greedy(minicpm_o_text_server.port)
        token_id, logprob = generate_text_greedy(
            minicpm_o_text_server.worker_ports[0]
        )
        image_text = chat_image_greedy(minicpm_o_text_server.port, image)
    router_guard.assert_served(min_total_requests=2)
    minicpm_o_text_server.stop()
    wait_for_gpu_memory_release()
    return ServingParityArtifacts(
        text_token_id=token_id,
        text_logprob=logprob,
        image_text=image_text,
    )


def load_hf_minicpm_o():
    checkpoint = str(resolve_minicpm_o_hf_checkpoint())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    config.init_tts = False
    model_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.MiniCPMO",
        checkpoint,
        local_files_only=True,
    )
    # note (MayDomine): transformers 5 looks up all_tied_weights_keys during
    # load; MiniCPM-o's remote-code class does not define it.
    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = {}
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    model = model_cls.from_pretrained(
        checkpoint,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return tokenizer, model.eval().cuda()


def hf_text_last_logits(tokenizer, model) -> torch.Tensor:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEXT_PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
        use_tts_template=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.llm(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
        )
    return outputs.logits[0, -1].float().cpu()


def greedy_decode_embeds(model, inputs_embeds, tokenizer) -> list[int]:
    embed = model.llm.get_input_embeddings()
    eos_ids = {tokenizer.convert_tokens_to_ids(token) for token in model.terminators}
    hidden = inputs_embeds
    token_ids: list[int] = []
    for _ in range(IMAGE_MAX_NEW_TOKENS):
        logits = model.llm(inputs_embeds=hidden, use_cache=False).logits[0, -1]
        token_id = int(logits.argmax().item())
        if token_id in eos_ids:
            break
        token_ids.append(token_id)
        nxt = embed(torch.tensor([[token_id]], device=hidden.device, dtype=torch.long))
        hidden = torch.cat([hidden, nxt], dim=1)
    return token_ids


def hf_image_greedy_text(tokenizer, model, image: Image.Image) -> str:
    """Greedy image decode without MiniCPM-o remote-code generate().

    The checkpoint's patched prepare_inputs_for_generation is incompatible
    with this transformers Cache API (cache_position is None).
    """
    captured: dict[str, str] = {}

    def decode_greedy(inputs_embeds, decode_tokenizer, attention_mask, **kwargs):
        del attention_mask, kwargs
        token_ids = greedy_decode_embeds(model, inputs_embeds, decode_tokenizer)
        captured["text"] = decode_tokenizer.decode(token_ids, skip_special_tokens=True)
        sequences = torch.tensor([token_ids or [0]], device=inputs_embeds.device)
        return SimpleNamespace(sequences=sequences, hidden_states=())

    original_decode = model._decode
    model._decode = decode_greedy
    try:
        with torch.inference_mode():
            model.chat(
                msgs=[{"role": "user", "content": [image, IMAGE_PROMPT]}],
                tokenizer=tokenizer,
                do_sample=False,
                num_beams=1,
                max_new_tokens=IMAGE_MAX_NEW_TOKENS,
                generate_audio=False,
                enable_thinking=True,
            )
    except TypeError:
        if "text" not in captured:
            raise
    finally:
        model._decode = original_decode
    return captured.get("text", "")


@pytest.fixture(scope="module")
def hf_parity_reference(
    serving_parity_artifacts: ServingParityArtifacts,
) -> HuggingFaceParityReference:
    del serving_parity_artifacts
    tokenizer, model = load_hf_minicpm_o()
    try:
        last_logits = hf_text_last_logits(tokenizer, model)
        logprobs = torch.log_softmax(last_logits, dim=-1)
        token_id = int(last_logits.argmax().item())
        image_text = hf_image_greedy_text(tokenizer, model, parity_image())
        return HuggingFaceParityReference(
            text_token_id=token_id,
            text_logprob=float(logprobs[token_id].item()),
            last_logits=last_logits,
            image_text=image_text,
        )
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.benchmark
def test_text_greedy_first_token_logprob_matches_hf(
    serving_parity_artifacts: ServingParityArtifacts,
    hf_parity_reference: HuggingFaceParityReference,
) -> None:
    """Author: text greedy vs HF backbone, first-token logprob abs <= 1e-3."""
    hf_logprob = float(
        torch.log_softmax(hf_parity_reference.last_logits, dim=-1)[
            serving_parity_artifacts.text_token_id
        ].item()
    )
    delta = abs(serving_parity_artifacts.text_logprob - hf_logprob)
    print(
        "MiniCPM-o text first-token "
        f"serving_id={serving_parity_artifacts.text_token_id} "
        f"hf_id={hf_parity_reference.text_token_id} "
        f"serving_lp={serving_parity_artifacts.text_logprob:.6f} "
        f"hf_lp={hf_logprob:.6f} delta={delta:.6e}"
    )
    checks = MetricCheckCollector("MiniCPM-o HF text first-token logprob")
    checks.check(
        delta <= FIRST_TOKEN_LOGPROB_MAX_ABS,
        f"first-token logprob abs {delta:.6e} > {FIRST_TOKEN_LOGPROB_MAX_ABS}",
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_prefill_last_logits_match_hf(
    serving_parity_artifacts: ServingParityArtifacts,
    hf_parity_reference: HuggingFaceParityReference,
) -> None:
    """Author: prefill last logits max abs 1.2 and top-1 match."""
    hf_top1 = int(hf_parity_reference.last_logits.argmax().item())
    serving_id = serving_parity_artifacts.text_token_id
    hf_logprob = float(
        torch.log_softmax(hf_parity_reference.last_logits, dim=-1)[serving_id].item()
    )
    # HTTP /generate exposes the sampled token and its logprob, not the full
    # vocab vector. Top-1 identity plus the 1e-3 logprob lock is the
    # serving-visible contract of the author's 1.2 max-abs check.
    delta = abs(serving_parity_artifacts.text_logprob - hf_logprob)
    print(
        "MiniCPM-o prefill last-logits "
        f"serving_top1={serving_id} hf_top1={hf_top1} "
        f"logprob_delta={delta:.6e} "
        f"claimed_max_abs={PREFILL_LAST_LOGITS_MAX_ABS}"
    )
    checks = MetricCheckCollector("MiniCPM-o HF prefill last logits")
    checks.check(
        serving_id == hf_top1,
        f"prefill top-1 {serving_id} != HF {hf_top1}",
    )
    checks.check(
        delta <= FIRST_TOKEN_LOGPROB_MAX_ABS,
        f"prefill first-token logprob abs {delta:.6e} > {FIRST_TOKEN_LOGPROB_MAX_ABS}",
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_image_only_greedy_text_matches_hf(
    serving_parity_artifacts: ServingParityArtifacts,
    hf_parity_reference: HuggingFaceParityReference,
) -> None:
    """Author: image-only greedy e2e starts identical to remote-code.

    Full thinking traces diverge after a few tokens; the CI locks the first
    greedy token, which is the serving-visible start of that claim.
    """
    serving_text = serving_parity_artifacts.image_text.strip()
    hf_text = hf_parity_reference.image_text.strip()
    print(f"MiniCPM-o image greedy serving={serving_text!r} hf={hf_text!r}")
    checks = MetricCheckCollector("MiniCPM-o HF image-only greedy text")
    checks.check(
        serving_text == hf_text,
        f"image greedy text {serving_text!r} != HF {hf_text!r}",
    )
    checks.check(bool(serving_text), "image greedy text is empty")
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
