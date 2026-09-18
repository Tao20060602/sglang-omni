# SPDX-License-Identifier: Apache-2.0
"""Routing and merge contracts MiniCPM-o refactoring must preserve."""

import pytest
import torch

from sglang_omni.models.minicpm_o.config import (
    MiniCPMOPipelineConfig,
    MiniCPMOSpeechPipelineConfig,
)
from sglang_omni.models.minicpm_o.merge import merge_for_thinker
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    AUDIO_STAGE,
    CODE2WAV_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
    project_encoder_to_thinker,
    project_preprocessing_to_audio_encoder,
    project_preprocessing_to_image_encoder,
    project_preprocessing_to_thinker,
    project_thinker_to_decode,
    project_thinker_to_talker,
    resolve_preprocessing_next_stages,
    resolve_terminal_stages,
    resolve_thinker_next_stages,
    resolve_thinker_wait_sources,
    should_generate_audio_output,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


def payload_with_state(
    state: MiniCPMOPipelineState,
    *,
    modalities: list[str] | None = None,
    params: dict | None = None,
) -> StagePayload:
    metadata = {} if modalities is None else {"output_modalities": modalities}
    return StagePayload(
        request_id="req",
        request=OmniRequest(
            inputs=None, metadata=metadata, params=params or {}
        ),
        data=state.to_dict(),
    )


def test_missing_modalities_defaults_to_audio_output() -> None:
    payload = payload_with_state(MiniCPMOPipelineState())
    assert should_generate_audio_output(payload) is True
    assert resolve_thinker_next_stages("req", payload) == [DECODE_STAGE, TALKER_STAGE]
    assert resolve_terminal_stages(payload.request) == [DECODE_STAGE, CODE2WAV_STAGE]


def test_text_modalities_skip_talker_and_code2wav() -> None:
    payload = payload_with_state(MiniCPMOPipelineState(), modalities=["text"])
    assert should_generate_audio_output(payload) is False
    assert resolve_thinker_next_stages("req", payload) == [DECODE_STAGE]
    assert resolve_terminal_stages(payload.request) == [DECODE_STAGE]


def test_audio_in_modalities_keeps_speech_path() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(), modalities=["text", "audio"]
    )
    assert resolve_thinker_next_stages("req", payload) == [DECODE_STAGE, TALKER_STAGE]


def test_text_only_encoder_inputs_skip_encoder_stages() -> None:
    payload = payload_with_state(MiniCPMOPipelineState())
    assert resolve_preprocessing_next_stages("req", payload) == [THINKER_STAGE]
    assert resolve_thinker_wait_sources("req", "preprocessing", payload) == [
        "preprocessing"
    ]


def test_image_and_audio_features_fan_into_thinker() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(
            encoder_inputs={
                IMAGE_STAGE: {"pixel_values": [torch.zeros(1)], "cache_key": "img"},
                AUDIO_STAGE: {
                    "audio_features": torch.zeros(1, 4),
                    "cache_key": "aud",
                },
            }
        )
    )
    assert resolve_preprocessing_next_stages("req", payload) == [
        IMAGE_STAGE,
        AUDIO_STAGE,
        THINKER_STAGE,
    ]
    assert resolve_thinker_wait_sources("req", "preprocessing", payload) == [
        "preprocessing",
        IMAGE_STAGE,
        AUDIO_STAGE,
    ]


def test_active_flag_overrides_missing_encoder_tensors() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(
            encoder_inputs={IMAGE_STAGE: {"_active": True, "cache_key": "img"}}
        )
    )
    assert IMAGE_STAGE in resolve_preprocessing_next_stages("req", payload)


def test_inactive_flag_skips_encoder_even_with_tensors() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(
            encoder_inputs={
                IMAGE_STAGE: {"pixel_values": [torch.zeros(1)], "_active": False}
            }
        )
    )
    assert resolve_preprocessing_next_stages("req", payload) == [THINKER_STAGE]


def test_preprocessing_projections_keep_only_the_target_stage() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(
            prompt={"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            encoder_inputs={
                IMAGE_STAGE: {"pixel_values": [1], "cache_key": "img"},
                AUDIO_STAGE: {"audio_features": [1], "cache_key": "aud"},
            },
        )
    )
    image = MiniCPMOPipelineState.from_dict(
        project_preprocessing_to_image_encoder(payload).data
    )
    audio = MiniCPMOPipelineState.from_dict(
        project_preprocessing_to_audio_encoder(payload).data
    )
    thinker = MiniCPMOPipelineState.from_dict(
        project_preprocessing_to_thinker(payload).data
    )
    assert list(image.encoder_inputs) == [IMAGE_STAGE]
    assert list(audio.encoder_inputs) == [AUDIO_STAGE]
    assert thinker.prompt["prompt_text"] == "hi"
    assert thinker.encoder_inputs[IMAGE_STAGE] == {
        "cache_key": "img",
        "_active": True,
    }


def test_encoder_projection_requires_exactly_one_output() -> None:
    payload = payload_with_state(
        MiniCPMOPipelineState(
            encoder_outs={
                IMAGE_STAGE: {"embeds": torch.zeros(1, 2)},
                AUDIO_STAGE: {"embeds": torch.zeros(1, 2)},
            }
        )
    )
    with pytest.raises(ValueError, match="exactly one encoder output"):
        project_encoder_to_thinker(payload)


def test_merge_for_thinker_joins_encoder_embeddings_and_drops_branches() -> None:
    request = OmniRequest(inputs=None, metadata={"output_modalities": ["text"]})
    preprocessing = StagePayload(
        request_id="req",
        request=request,
        data=MiniCPMOPipelineState(
            prompt={"input_ids": torch.tensor([7]), "prompt_text": "hi"}
        ).to_dict(),
    )
    image = StagePayload(
        request_id="req",
        request=request,
        data=MiniCPMOPipelineState(
            encoder_outs={IMAGE_STAGE: {"image_embeds": torch.ones(2, 3)}}
        ).to_dict(),
    )
    audio = StagePayload(
        request_id="req",
        request=request,
        data=MiniCPMOPipelineState(
            encoder_outs={AUDIO_STAGE: {"audio_embeds": torch.zeros(4, 3)}}
        ).to_dict(),
    )
    merged = MiniCPMOPipelineState.from_dict(
        merge_for_thinker(
            {"preprocessing": preprocessing, IMAGE_STAGE: image, AUDIO_STAGE: audio}
        ).data
    )
    assert merged.prompt["prompt_text"] == "hi"
    assert merged.encoder_inputs == {}
    assert merged.encoder_outs == {}
    torch.testing.assert_close(
        merged.thinker_inputs["model_inputs"]["image_embeds"], torch.ones(2, 3)
    )
    torch.testing.assert_close(
        merged.thinker_inputs["model_inputs"]["audio_embeds"], torch.zeros(4, 3)
    )


def test_thinker_to_talker_keeps_hidden_states_and_decode_strips_them() -> None:
    hidden = [torch.arange(4.0)]
    payload = payload_with_state(
        MiniCPMOPipelineState(
            prompt={"input_ids": torch.tensor([1])},
            thinker_inputs={"model_inputs": {"image_embeds": torch.ones(1)}},
            thinker_out={
                "output_ids": [9, 8],
                "extra_model_outputs": {
                    "hidden_states_seq": hidden,
                    "other": "drop-me",
                },
            },
        )
    )
    talker = MiniCPMOPipelineState.from_dict(project_thinker_to_talker(payload).data)
    decode = MiniCPMOPipelineState.from_dict(project_thinker_to_decode(payload).data)
    assert talker.thinker_out["output_ids"] == [9, 8]
    assert talker.thinker_out["extra_model_outputs"] == {
        "hidden_states_seq": hidden
    }
    assert decode.thinker_inputs == {}
    assert "extra_model_outputs" not in decode.thinker_out


def test_speech_variant_is_the_default_entry_class() -> None:
    from sglang_omni.models.minicpm_o.config import EntryClass, Variants

    assert EntryClass is MiniCPMOSpeechPipelineConfig
    assert Variants["text"] is MiniCPMOPipelineConfig
    assert Variants["speech"] is MiniCPMOSpeechPipelineConfig


def test_speech_pipeline_passes_speech_enabled_to_thinker_and_preprocessor() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="model")
    assert config.stage_factory_kwargs("thinker") == {"speech_enabled": True}
    assert config.stage_factory_kwargs("preprocessing") == {"speech_enabled": True}
    assert config.stage_factory_kwargs("talker") == {}
