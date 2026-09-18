# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o SeedTTS client. Lives here so benchmarks/tasks/tts.py stays clean."""

from __future__ import annotations

import base64
from pathlib import Path

import aiohttp

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.tts import VoiceCloneOmni


def wav_path_to_data_uri(path: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class VoiceCloneMiniCPMO(VoiceCloneOmni):
    """Speaker ref is ``audio.ref_audio``; prompt is read-aloud, seed=42."""

    REQUEST_SEED = 42

    def build_prompt_text(
        self,
        sample: SampleInput,
        lang: str,
        voice_clone: bool = True,
    ) -> str:
        del voice_clone
        if lang == "en":
            return (
                f"Please read the following text out loud in English: "
                f"{sample.target_text}"
            )
        return f"请用中文朗读以下文本: {sample.target_text}"

    def attach_reference(self, payload: dict, sample: SampleInput) -> None:
        audio = payload.setdefault("audio", {"format": "wav"})
        audio["ref_audio"] = wav_path_to_data_uri(sample.ref_audio)
        payload.pop("audios", None)

    async def generate_speech(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        lang: str,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        voice_clone: bool = False,
        stream: bool = False,
        system_prompt: str | None = None,
        chunk_times_out: list[float] | None = None,
        text_first_time_holder: list[float] | None = None,
    ) -> tuple[bytes, float, dict]:
        original_post = session.post

        def post_minicpm_o_payload(url, **kwargs):
            payload = kwargs.get("json")
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.pop("audios", None)
                payload["seed"] = self.REQUEST_SEED
                if voice_clone:
                    self.attach_reference(payload, sample)
                kwargs["json"] = payload
            return original_post(url, **kwargs)

        # Parent builds the Qwen payload; swap the POST body for MiniCPM-o.
        session.post = post_minicpm_o_payload
        try:
            return await super().generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker,
                max_tokens,
                temperature,
                False,
                stream,
                system_prompt,
                chunk_times_out,
                text_first_time_holder,
            )
        finally:
            session.post = original_post
