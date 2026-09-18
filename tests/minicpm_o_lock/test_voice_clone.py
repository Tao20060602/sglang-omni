# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o SeedTTS client puts the speaker reference in audio.ref_audio."""

from __future__ import annotations

from pathlib import Path

from benchmarks.dataset.seedtts import SampleInput
from tests.minicpm_o_lock.voice_clone import VoiceCloneMiniCPMO, wav_path_to_data_uri


def test_wav_path_to_data_uri_encodes_file_bytes(tmp_path: Path) -> None:
    wav_path = tmp_path / "ref.wav"
    wav_path.write_bytes(b"RIFF-ref")
    assert wav_path_to_data_uri(str(wav_path)) == "data:audio/wav;base64,UklGRi1yZWY="


def test_minicpm_o_reference_uses_audio_ref_audio_data_uri(tmp_path: Path) -> None:
    wav_path = tmp_path / "ref.wav"
    wav_path.write_bytes(b"RIFF-ref")
    payload = {"audio": {"format": "wav"}}
    sample = SampleInput(
        sample_id="s0",
        ref_text="hello",
        ref_audio=str(wav_path),
        target_text="world",
    )
    VoiceCloneMiniCPMO().attach_reference(payload, sample)
    assert payload["audio"]["ref_audio"] == wav_path_to_data_uri(str(wav_path))
    assert "audios" not in payload


def test_minicpm_o_prompt_is_read_aloud_not_listen_above() -> None:
    sample = SampleInput(
        sample_id="s0",
        ref_text="ignored",
        ref_audio="/tmp/ref.wav",
        target_text="Please say this.",
    )
    prompt = VoiceCloneMiniCPMO().build_prompt_text(sample, "en", voice_clone=True)
    assert "Listen to the audio above" not in prompt
    assert "Please say this." in prompt
