# SPDX-License-Identifier: Apache-2.0
"""Run shared SeedTTS eval with the MiniCPM-o client, without a task= hook."""

from __future__ import annotations

import asyncio

from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from tests.minicpm_o_lock.voice_clone import VoiceCloneMiniCPMO


def run_minicpm_o_seedtts_benchmark(config: OmniSeedttsBenchmarkConfig) -> dict:
    import benchmarks.eval.benchmark_omni_seedtts as seedtts

    original = seedtts.VoiceCloneOmni
    seedtts.VoiceCloneOmni = VoiceCloneMiniCPMO
    try:
        return asyncio.run(run_omni_seedtts_benchmark(config))
    finally:
        seedtts.VoiceCloneOmni = original
