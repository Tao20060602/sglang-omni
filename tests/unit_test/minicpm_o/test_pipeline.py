# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o pipeline topology, registration, and routing contracts."""

import importlib
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.config import (
    MiniCPMOPipelineConfig,
    MiniCPMOSpeechPipelineConfig,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.utils.imports import import_string


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _stage_index(config, name: str) -> int:
    return next(
        index for index, stage in enumerate(config.stages) if stage.name == name
    )


def test_text_pipeline_constructs_thinker_before_encoders() -> None:
    config = MiniCPMOPipelineConfig(model_path="model")

    assert _stage_index(config, "thinker") < _stage_index(config, "image_encoder")
    assert _stage_index(config, "thinker") < _stage_index(config, "audio_encoder")
    assert _stage(config, "thinker").process == "pipeline"
    assert _stage(config, "image_encoder").process == "pipeline"
    assert _stage(config, "audio_encoder").process == "pipeline"


def test_speech_pipeline_preserves_tp_and_process_boundaries() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="model")

    assert _stage_index(config, "thinker") < _stage_index(config, "image_encoder")
    assert _stage_index(config, "thinker") < _stage_index(config, "audio_encoder")
    assert _stage(config, "thinker").process == "pipeline"
    assert _stage(config, "talker").process == "talker"
    assert _stage(config, "code2wav").process == "code2wav"


def test_gpu_placed_factories_declare_gpu_id() -> None:
    """GPU-placed factories accept the placement layer's gpu_id argument."""
    import inspect

    from sglang_omni.models.minicpm_o import stages

    factories = [
        stages.create_image_encoder_executor,
        stages.create_audio_encoder_executor,
        stages.create_code2wav_executor,
    ]
    for factory in factories:
        params = inspect.signature(factory).parameters
        assert "gpu_id" in params, f"{factory.__qualname__} is missing gpu_id"


def test_hf_config_registration_is_explicit(monkeypatch) -> None:
    from sglang_omni.models.minicpm_o import hf_config

    register = Mock()
    monkeypatch.setattr(hf_config.AutoConfig, "register", register)

    importlib.reload(hf_config)
    register.assert_not_called()
    hf_config.register_minicpm_o_hf_config()
    register.assert_called_once_with(
        "minicpmo", hf_config.MiniCPMOConfig, exist_ok=True
    )


@pytest.mark.parametrize(
    "config_type", [MiniCPMOPipelineConfig, MiniCPMOSpeechPipelineConfig]
)
def test_configured_routes_preserve_text_prompt(config_type) -> None:
    config = config_type(model_path="model")
    input_ids = torch.tensor([1, 2, 3])
    payload = StagePayload(
        request_id="text",
        request=OmniRequest(inputs=None, metadata={"output_modalities": ["text"]}),
        data=MiniCPMOPipelineState(
            prompt={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "prompt_text": "hello",
            }
        ).to_dict(),
    )
    preprocessing = _stage(config, "preprocessing")
    thinker = _stage(config, "thinker")

    assert import_string(preprocessing.route_fn)(payload.request_id, payload) == [
        "thinker"
    ]
    projected = import_string(preprocessing.project_payload["thinker"])(payload)
    assert import_string(thinker.wait_for_fn)(
        payload.request_id, "preprocessing", projected
    ) == ["preprocessing"]
    merged = import_string(thinker.merge_fn)({"preprocessing": projected})

    torch.testing.assert_close(merged.data["prompt"]["input_ids"], input_ids)
    assert merged.data["thinker_inputs"] == {"model_inputs": {}}
    if thinker.route_fn:
        assert import_string(thinker.route_fn)(payload.request_id, merged) == ["decode"]
    if config.terminal_stages_fn:
        assert import_string(config.terminal_stages_fn)(payload.request) == ["decode"]
