# SPDX-License-Identifier: Apache-2.0
"""SeedTTS accuracy CI for MiniCPM-o speech (Text -> Audio, voice clone).

Usage:
    pytest tests/minicpm_o_lock/test_tts_ci.py -s -x

Quality claim (BruceLoveDecimal, RTX 5090, Seed-TTS Arrow, 25 EN + 25 ZH,
with reference, Whisper large-v3): EN WER 2.93%, ZH CER 8.81%. Serving
claim (MayDomine, H100): every speech request returned finite, non-silent
24 kHz audio. CI transcribes with Qwen3-ASR behind the Rust router
(DP=2) and keeps the standard 1.25x WER slack.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    evaluate_generated_audio,
)
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from tests.minicpm_o_lock.conftest import MINICPMO_MODEL_NAME
from tests.minicpm_o_lock.seedtts_runner import run_minicpm_o_seedtts_benchmark
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    apply_wer_slack,
    assert_summary_metrics,
    assert_wer_partitioned,
    wait_for_gpu_memory_release,
)

MAX_SAMPLES = 25
CONCURRENCY = 2
CLAIMED_EN_WER = 0.0293
CLAIMED_ZH_CER = 0.0881
EN_WER_MAX = apply_wer_slack(CLAIMED_EN_WER)
ZH_CER_MAX = apply_wer_slack(CLAIMED_ZH_CER)
MAX_N_ABOVE_50 = 2
DATASET_REPO = DATASETS["seedtts"]
SEEDTTS_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts",
    repo_id=DATASET_REPO,
)


def run_minicpm_o_seedtts(
    port: int,
    output_dir: str,
    *,
    lang: str,
) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model=MINICPMO_MODEL_NAME,
        port=port,
        meta=DATASET_REPO,
        output_dir=output_dir,
        lang=lang,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        max_new_tokens=256,
        temperature=0.0,
        warmup=0,
        voice_clone=True,
    )
    results = run_minicpm_o_seedtts_benchmark(config)
    if "summary" not in results or "per_request" not in results:
        raise AssertionError(f"Incomplete SeedTTS results: {list(results)}")
    return results


def transcribe_minicpm_o_seedtts(
    output_dir: str,
    *,
    asr_router_port: int,
    lang: str,
) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model=MINICPMO_MODEL_NAME,
        meta=DATASET_REPO,
        output_dir=output_dir,
        lang=lang,
        port=asr_router_port,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )
    evaluate_generated_audio(config)
    results_path = Path(output_dir) / "wer_results.json"
    if not results_path.exists():
        raise AssertionError(f"WER results file not found: {results_path}")
    wer_results = json.loads(results_path.read_text())
    if "summary" not in wer_results or "per_sample" not in wer_results:
        raise AssertionError(f"Incomplete WER results: {list(wer_results)}")
    return wer_results


@dataclass
class MiniCPMOTTSArtifacts:
    en_dir: str
    zh_dir: str
    en_summary: dict
    zh_summary: dict


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    download_dataset(DATASET_REPO, quiet=True)
    return DATASET_REPO


@pytest.fixture(scope="module")
def tts_artifacts(
    minicpm_o_speech_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> MiniCPMOTTSArtifacts:
    del dataset_repo
    en_dir = str(tmp_path_factory.mktemp("minicpm_o_tts_en"))
    zh_dir = str(tmp_path_factory.mktemp("minicpm_o_tts_zh"))
    with router_worker_traffic_guard(
        minicpm_o_speech_server,
        label="MiniCPM-o SeedTTS",
    ) as router_guard:
        en_results = run_minicpm_o_seedtts(
            minicpm_o_speech_server.port, en_dir, lang="en"
        )
        zh_results = run_minicpm_o_seedtts(
            minicpm_o_speech_server.port, zh_dir, lang="zh"
        )
    router_guard.assert_served(min_total_requests=MAX_SAMPLES * 2)
    return MiniCPMOTTSArtifacts(
        en_dir=en_dir,
        zh_dir=zh_dir,
        en_summary=en_results["summary"],
        zh_summary=zh_results["summary"],
    )


@pytest.fixture(scope="module")
def wer_audio_dirs(
    minicpm_o_speech_server: ManagedRouterHandle,
    tts_artifacts: MiniCPMOTTSArtifacts,
) -> MiniCPMOTTSArtifacts:
    minicpm_o_speech_server.stop()
    wait_for_gpu_memory_release()
    return tts_artifacts


@pytest.mark.benchmark
def test_minicpm_o_seedtts_generation_succeeds(
    tts_artifacts: MiniCPMOTTSArtifacts,
) -> None:
    """All 25+25 speech requests must return finite, non-silent audio."""
    for lang, summary in (
        ("en", tts_artifacts.en_summary),
        ("zh", tts_artifacts.zh_summary),
    ):
        print_speed_summary(
            summary,
            MINICPMO_MODEL_NAME,
            CONCURRENCY,
            title=f"MiniCPM-o SeedTTS {lang.upper()} Speed",
            dataset=SEEDTTS_DATASET_LABEL,
        )
    checks = MetricCheckCollector("MiniCPM-o SeedTTS generation")
    assert_summary_metrics(
        tts_artifacts.en_summary, check_tokens=False, collector=checks
    )
    assert_summary_metrics(
        tts_artifacts.zh_summary, check_tokens=False, collector=checks
    )
    checks.check(
        tts_artifacts.en_summary.get("completed_requests") == MAX_SAMPLES,
        f"EN completed {tts_artifacts.en_summary.get('completed_requests')} "
        f"!= {MAX_SAMPLES}",
    )
    checks.check(
        tts_artifacts.zh_summary.get("completed_requests") == MAX_SAMPLES,
        f"ZH completed {tts_artifacts.zh_summary.get('completed_requests')} "
        f"!= {MAX_SAMPLES}",
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_minicpm_o_seedtts_en_wer(
    wer_audio_dirs: MiniCPMOTTSArtifacts,
    qwen3_asr_wer_router: ManagedRouterHandle,
) -> None:
    results = transcribe_minicpm_o_seedtts(
        wer_audio_dirs.en_dir,
        asr_router_port=qwen3_asr_wer_router.port,
        lang="en",
    )
    print_wer_summary(
        results["summary"], MINICPMO_MODEL_NAME, dataset=SEEDTTS_DATASET_LABEL
    )
    checks = MetricCheckCollector("MiniCPM-o SeedTTS EN WER")
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=EN_WER_MAX,
        max_n_above_50=MAX_N_ABOVE_50,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_minicpm_o_seedtts_zh_cer(
    wer_audio_dirs: MiniCPMOTTSArtifacts,
    qwen3_asr_wer_router: ManagedRouterHandle,
) -> None:
    results = transcribe_minicpm_o_seedtts(
        wer_audio_dirs.zh_dir,
        asr_router_port=qwen3_asr_wer_router.port,
        lang="zh",
    )
    print_wer_summary(
        results["summary"], MINICPMO_MODEL_NAME, dataset=SEEDTTS_DATASET_LABEL
    )
    checks = MetricCheckCollector("MiniCPM-o SeedTTS ZH CER")
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=ZH_CER_MAX,
        max_n_above_50=MAX_N_ABOVE_50,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
