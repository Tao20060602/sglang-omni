# SPDX-License-Identifier: Apache-2.0
"""Thinker request-builder contracts MiniCPM-o refactoring must preserve."""

from types import SimpleNamespace

import torch
import xxhash

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import (
    _apply_mm_pad_values,
    apply_thinker_result,
    build_encoder_request,
    build_sglang_thinker_request,
    resolve_sampling_seed,
)


def test_resolve_sampling_seed_reads_seed_then_sampling_seed() -> None:
    assert resolve_sampling_seed({"seed": 7, "sampling_seed": 9}) == 7
    assert resolve_sampling_seed({"sampling_seed": 9}) == 9
    assert resolve_sampling_seed({}) is None


def test_empty_encoder_inputs_skip_the_forward() -> None:
    request = build_encoder_request(MiniCPMOPipelineState(), stage_name="image_encoder")
    assert request.model_inputs == {}
    assert request.skip_result == {}


def test_encoder_request_strips_routing_metadata() -> None:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            "image_encoder": {
                "pixel_values": [1],
                "cache_key": "img",
                "_active": True,
            }
        }
    )
    request = build_encoder_request(state, stage_name="image_encoder")
    assert request.cache_key == "img"
    assert request.skip_result is None
    assert request.model_inputs == {"pixel_values": [1]}


def test_mm_pad_values_are_derived_from_cache_key_hash() -> None:
    vocab_size = 100
    cache_key = "img-key"
    input_ids = torch.tensor([1, 0, 0, 2], dtype=torch.long)
    model_inputs: dict = {}
    padded, positions = _apply_mm_pad_values(
        input_ids,
        mm_inputs={
            "image": {
                "bounds": torch.tensor([[1, 3]]),
                "cache_key": cache_key,
            }
        },
        model_inputs=model_inputs,
        vocab_size=vocab_size,
    )
    expected = vocab_size + xxhash.xxh3_64(cache_key.encode()).intdigest() % (1 << 62)
    assert padded.tolist() == [1, expected, expected, 2]
    assert positions["image"].tolist() == [1, 2]
    assert positions["audio"].numel() == 0
    assert model_inputs["pad_values"]["image"] == expected
    assert input_ids.tolist() == [1, 0, 0, 2]


def test_thinker_request_injects_pad_values_and_sampling_defaults() -> None:
    tokenizer = SimpleNamespace(
        additional_stop_token_ids=set(),
        eos_token_id=2,
    )
    state = MiniCPMOPipelineState(
        prompt={
            "input_ids": torch.tensor([5, 0, 0, 6], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
        },
        mm_inputs={
            "audio": {
                "bounds": torch.tensor([[1, 3]]),
                "cache_key": "aud",
            }
        },
        thinker_inputs={"model_inputs": {"audio_embeds": torch.zeros(2, 3)}},
    )
    req = build_sglang_thinker_request(
        state,
        params={"seed": 42, "max_new_tokens": 16, "temperature": 0.0},
        tokenizer=tokenizer,
        vocab_size=32,
        request_id="thinker-1",
    )
    expected = 32 + xxhash.xxh3_64(b"aud").intdigest() % (1 << 62)
    assert req.req.rid == "thinker-1"
    assert req.req.origin_input_ids[1:3] == [expected, expected]
    assert req.req.sampling_params.sampling_seed == 42
    assert req.req.sampling_params.max_new_tokens == 16
    assert req.model_inputs["pad_values"]["audio"] == expected
    assert req.req._omni_mm_positions["audio"].tolist() == [1, 2]


def test_apply_thinker_result_copies_output_ids_and_hidden_states() -> None:
    state = MiniCPMOPipelineState()
    result = SimpleNamespace(
        output_ids=[11, 12],
        extra_model_outputs={"hidden_states_seq": [torch.zeros(2)]},
        finish_reason="stop",
        weight_version="v1",
        output_token_logprobs=None,
    )
    thinker_out = apply_thinker_result(state, stage_name="thinker", result=result)
    assert thinker_out["output_ids"] == [11, 12]
    assert thinker_out["is_final"] is True
    assert thinker_out["finish_reason"] == "stop"
    assert state.engine_outputs["thinker"] is thinker_out
