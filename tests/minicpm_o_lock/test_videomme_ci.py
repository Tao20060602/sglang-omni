# SPDX-License-Identifier: Apache-2.0
"""Video-MME accuracy CI for MiniCPM-o thinker-only (Video -> Text).

Usage:
    pytest tests/minicpm_o_lock/test_videomme_ci.py -s -x

Author claim (MayDomine, MiniCPM-o-4.5, H100, videomme-ci-50, 16 frames at
1 FPS, thinker-only): 30/50 correct. The gate keeps four samples of slack
for ordinary runtime drift.

Thinker-only video accuracy. Speech quality is gated in
test_tts_ci.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.minicpm_o_lock.conftest import MINICPMO_MODEL_NAME
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector

MAX_SAMPLES = 50
CLAIMED_CORRECT = 30
SLACK_SAMPLES = 4
MIN_CORRECT = CLAIMED_CORRECT - SLACK_SAMPLES
MINICPMO_VIDEOMME_MIN_ACCURACY = MIN_CORRECT / MAX_SAMPLES
CONCURRENCY = 2
VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 16
VIDEO_MAX_PIXELS = 401408


@pytest.mark.benchmark
def test_minicpm_o_videomme_thinker_only_accuracy(
    minicpm_o_text_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run videomme-ci-50 on thinker-only Rust-router DP=2 and check 30/50."""
    config = VideoEvalConfig(
        model=MINICPMO_MODEL_NAME,
        port=minicpm_o_text_server.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=VIDEO_FPS,
        video_max_frames=VIDEO_MAX_FRAMES,
        video_max_pixels=VIDEO_MAX_PIXELS,
        disable_tqdm=False,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        minicpm_o_text_server,
        label="MiniCPM-o Video-MME",
    ) as router_guard:
        results = asyncio.run(
            run_video_eval(
                config,
                task_label="MiniCPM-o Video-MME",
                output_filename="videomme_results.json",
                audio_output_dir_default="results/minicpm_o_videomme_audio",
            )
        )

    summary = results["summary"]
    dataset_label = format_benchmark_dataset_label(
        dataset="videomme-ci-50",
        repo_id=config.repo_id,
    )
    print_videomme_accuracy_summary(
        summary,
        config.model,
        dataset=dataset_label,
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="MiniCPM-o Video-MME Speed",
        dataset=dataset_label,
    )

    correct = summary.get("correct")
    total = summary.get("total_samples")
    checks = MetricCheckCollector("MiniCPM-o Video-MME accuracy")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=MAX_SAMPLES,
    )
    checks.check(
        total == MAX_SAMPLES,
        f"Expected {MAX_SAMPLES} samples, got {total}",
    )
    checks.check(
        summary.get("failed", 0) == 0,
        f"Expected 0 failed samples, got {summary.get('failed')}",
    )
    checks.check(
        isinstance(correct, int) and correct >= MIN_CORRECT,
        f"Video-MME correct {correct!r}/{total} < {MIN_CORRECT}/{MAX_SAMPLES} "
        f"(author claim {CLAIMED_CORRECT}/{MAX_SAMPLES})",
    )
    accuracy = summary.get("accuracy")
    checks.check(
        accuracy is not None and accuracy >= MINICPMO_VIDEOMME_MIN_ACCURACY,
        f"Video-MME accuracy {accuracy!r} < "
        f"threshold {MINICPMO_VIDEOMME_MIN_ACCURACY} "
        f"(author claim {CLAIMED_CORRECT / MAX_SAMPLES:.2f})",
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
