# SPDX-License-Identifier: Apache-2.0
"""Preprocessor contracts MiniCPM-o refactoring must preserve."""

import io
import wave

import numpy as np
import pytest

from sglang_omni.models.minicpm_o.components.preprocessor import (
    ASR_PROMPT_EN,
    ASR_PROMPT_ZH,
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    MiniCPMOPreprocessor,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


def bare_preprocessor(*, speech_enabled: bool) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.speech_enabled = speech_enabled
    return preprocessor


def pcm16_wav_bytes(*, num_samples: int = 1600, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_samples)
    return buffer.getvalue()


def test_openai_image_url_parts_are_dropped_from_text() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "describe this"},
            ],
        }
    ]
    normalized = MiniCPMOPreprocessor._normalize_message_contents(messages)
    assert normalized == [{"role": "user", "content": "describe this"}]


def test_media_placeholders_are_prepended_to_the_last_user_turn() -> None:
    preprocessor = bare_preprocessor(speech_enabled=True)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look"},
    ]
    rewritten = preprocessor._messages_with_media_placeholders(
        messages, num_images=2, num_audios=1
    )
    assert rewritten[0] == {"role": "system", "content": "sys"}
    assert rewritten[1]["content"] == "\n".join(
        [IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER, AUDIO_PLACEHOLDER, "look"]
    )


def test_tts_template_requires_speech_pipeline_and_audio_output() -> None:
    speech = bare_preprocessor(speech_enabled=True)
    text = bare_preprocessor(speech_enabled=False)
    audio_payload = StagePayload(
        request_id="r",
        request=OmniRequest(
            inputs=None, metadata={"output_modalities": ["text", "audio"]}
        ),
        data=None,
    )
    text_payload = StagePayload(
        request_id="r",
        request=OmniRequest(inputs=None, metadata={"output_modalities": ["text"]}),
        data=None,
    )
    assert speech._use_tts_template(audio_payload) is True
    assert speech._use_tts_template(text_payload) is False
    assert text._use_tts_template(audio_payload) is False


@pytest.mark.parametrize(
    "language, prompt",
    [("en", ASR_PROMPT_EN), ("zh", ASR_PROMPT_ZH), ("zh-CN", ASR_PROMPT_ZH), ("", ASR_PROMPT_EN)],
)
def test_transcription_prompt_follows_request_language(language: str, prompt: str) -> None:
    preprocessor = bare_preprocessor(speech_enabled=True)
    payload = StagePayload(
        request_id="asr",
        request=OmniRequest(inputs=None, params={"language": language}),
        data=None,
    )
    messages, audios = preprocessor._speech_to_text_inputs(
        payload, {"audio_bytes": pcm16_wav_bytes()}
    )
    assert messages == [{"role": "user", "content": prompt}]
    assert len(audios) == 1
    assert isinstance(audios[0], np.ndarray)
    assert audios[0].ndim == 1
