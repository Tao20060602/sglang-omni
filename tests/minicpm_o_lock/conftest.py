# SPDX-License-Identifier: Apache-2.0
"""Isolated fixtures for the MiniCPM-o accuracy lock.

Delete ``tests/minicpm_o_lock`` after the refactor. These fixtures must not
live in ``tests/test_model/conftest.py``.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest_plugins = ["tests.utils"]

if TYPE_CHECKING:
    from tests.test_model.omni_router_utils import ManagedRouterHandle

MINICPMO_MODEL_PATH = "openbmb/MiniCPM-o-4_5"
MINICPMO_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_MINICPMO_MODEL", MINICPMO_MODEL_PATH
)
MINICPMO_MODEL_NAME = "minicpm-o"
SGL_OMNI_SHIM_DIR = Path("/tmp/sgl-omni-ci-bin")
DEFAULT_VISIBLE_GPUS = "0,1"
MINICPMO_NUM_WORKERS = 2
MINICPMO_GPUS_PER_WORKER = 1


def model_cache_present(model_path: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    if Path(model_path).exists():
        return True
    try:
        snapshot_download(model_path, local_files_only=True)
    except Exception:
        return False
    return True


def ensure_sgl_omni_on_path() -> None:
    if shutil.which("sgl-omni") is not None:
        return
    SGL_OMNI_SHIM_DIR.mkdir(parents=True, exist_ok=True)
    shim = SGL_OMNI_SHIM_DIR / "sgl-omni"
    if not shim.exists():
        shim.write_text('#!/bin/sh\nexec python -m sglang_omni.cli "$@"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = f"{SGL_OMNI_SHIM_DIR}{os.pathsep}{os.environ.get('PATH', '')}"


def ensure_two_visible_gpus() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if len(devices) < 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_VISIBLE_GPUS


def worker_extra_args(*, text_only: bool) -> str:
    media_root = os.environ.get("HF_HOME") or str(
        Path.home() / ".cache" / "huggingface"
    )
    args = f"--allowed-local-media-path {media_root}"
    if text_only:
        return f"--text-only {args}"
    return args


def start_minicpm_o_router(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    text_only: bool,
) -> Iterator[ManagedRouterHandle]:
    if not model_cache_present(MINICPMO_TEST_MODEL_PATH):
        pytest.skip(
            f"{MINICPMO_TEST_MODEL_PATH} is not in the local HF cache; "
            "set SGLANG_OMNI_TEST_MINICPMO_MODEL to a local path."
        )
    from tests.test_model.omni_router_utils import (
        CiRouterTopology,
        launch_managed_router,
    )

    ensure_sgl_omni_on_path()
    ensure_two_visible_gpus()
    topology = (
        CiRouterTopology.OMNI_TEXT if text_only else CiRouterTopology.OMNI_AUDIO
    )
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=MINICPMO_TEST_MODEL_PATH,
        model_name=MINICPMO_MODEL_NAME,
        worker_extra_args=worker_extra_args(text_only=text_only),
        router_topology=topology,
        num_workers=MINICPMO_NUM_WORKERS,
        num_gpus_per_worker=MINICPMO_GPUS_PER_WORKER,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def minicpm_o_text_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    yield from start_minicpm_o_router(tmp_path_factory, text_only=True)


@pytest.fixture(scope="module")
def minicpm_o_speech_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    yield from start_minicpm_o_router(tmp_path_factory, text_only=False)
