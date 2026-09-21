# SPDX-License-Identifier: Apache-2.0
"""Exact-(B, T) CUDA graphs over the dots.tts slot-pool AudioVAE step."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)
_HIT_RATE_LOG_EVERY = 1024

_StepInputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
_StepOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_StepForward = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    _StepOutputs,
]


class _CapturedStepGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    packed: torch.Tensor
    hidden_h: torch.Tensor
    hidden_c: torch.Tensor
    window: torch.Tensor
    valid: torch.Tensor
    outputs: _StepOutputs


class DotsVocoderGraphRunner:
    """Replays one captured graph per (batch, frames) key; misses return None.

    Inputs are copied into static buffers, outputs are cloned out, so callers
    keep the eager step's value semantics and a replay never aliases the
    previous step's result.
    """

    def __init__(
        self,
        *,
        forward: _StepForward,
        new_inputs: Callable[[int, int], _StepInputs],
        device: torch.device,
        warmup_iters: int = 2,
    ) -> None:
        self._forward = forward
        self._new_inputs = new_inputs
        self._device = device
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[tuple[int, int], _CapturedStepGraph] = {}
        self._replays = 0
        self._misses = 0

    @property
    def captured_keys(self) -> list[tuple[int, int]]:
        return sorted(self._graphs)

    @property
    def replays(self) -> int:
        return self._replays

    @property
    def misses(self) -> int:
        return self._misses

    @torch.no_grad()
    def capture(self, keys: Iterable[tuple[int, int]]) -> None:
        if self._device.type != "cuda":
            raise RuntimeError("dots.tts vocoder CUDA graphs require a CUDA device")
        # note (lennox): graphs share one mempool, so capture the largest key
        # first; growing the pool after a smaller capture invalidates its addresses.
        ordered = sorted({(int(b), int(t)) for b, t in keys}, reverse=True)
        with torch.cuda.device(self._device):
            current_stream = torch.cuda.current_stream(self._device)
            capture_stream = torch.cuda.Stream(device=self._device)
            capture_stream.wait_stream(current_stream)
            graph_pool = torch.cuda.graph_pool_handle()
            for key in ordered:
                if key in self._graphs:
                    continue
                inputs = self._new_inputs(*key)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.stream(capture_stream):
                        for _ in range(self._warmup_iters):
                            self._forward(*inputs)
                    capture_stream.synchronize()
                    with torch.cuda.graph(
                        graph,
                        pool=graph_pool,
                        stream=capture_stream,
                        capture_error_mode="thread_local",
                    ):
                        outputs = self._forward(*inputs)
                except Exception as exc:
                    graph.reset()
                    logger.warning(
                        "dots.tts vocoder CUDA graph capture failed for "
                        "(batch, frames)=%s: %s; that shape stays eager",
                        key,
                        exc,
                    )
                    continue
                self._graphs[key] = _CapturedStepGraph(graph, *inputs, outputs)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)
        logger.info(
            "dots.tts streaming vocoder CUDA graphs captured: %s", self.captured_keys
        )

    @torch.no_grad()
    def run(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
    ) -> _StepOutputs | None:
        key = (int(packed.shape[0]), int(packed.shape[2]))
        captured = self._graphs.get(key)
        if captured is None or packed.device != self._device:
            self._misses += 1
            self._maybe_log_hit_rate()
            return None
        captured.packed.copy_(packed)
        captured.hidden_h.copy_(hidden_h)
        captured.hidden_c.copy_(hidden_c)
        captured.window.copy_(window)
        captured.valid.copy_(valid)
        captured.graph.replay()
        self._replays += 1
        self._maybe_log_hit_rate()
        return tuple(output.clone() for output in captured.outputs)

    def _maybe_log_hit_rate(self) -> None:
        calls = self._replays + self._misses
        if calls != 1 and calls % _HIT_RATE_LOG_EVERY:
            return
        logger.info(
            "dots.tts streaming vocoder CUDA graph replays=%d misses=%d (%.1f%% hit)",
            self._replays,
            self._misses,
            100.0 * self._replays / calls,
        )


__all__ = ["DotsVocoderGraphRunner"]
