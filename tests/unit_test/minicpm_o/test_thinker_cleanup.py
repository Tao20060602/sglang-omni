import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


def _runner():
    runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
    runner.pending_hidden = {}
    return runner


@pytest.mark.parametrize("speech_enabled", [False, True])
def test_bootstrap_wires_abort_cleanup(monkeypatch, speech_enabled):
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.models.minicpm_o import bootstrap as minicpm_bootstrap
    from sglang_omni.models.minicpm_o import request_builders, thinker_model_runner
    from sglang_omni.scheduling import bootstrap
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    runner = _runner()
    runner.pending_hidden = {"aborted": [torch.ones(4)], "other": [torch.zeros(4)]}
    monkeypatch.setattr(
        thinker_model_runner, "MiniCPMOThinkerModelRunner", lambda *args: runner
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            object(),
            None,
            None,
            None,
            SimpleNamespace(model_path="model", vocab_size=100),
        ),
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        request_builders,
        "make_thinker_scheduler_adapters",
        lambda **kwargs: (None, None),
    )

    scheduler_signature = inspect.signature(OmniScheduler.__init__)

    def init_scheduler(scheduler, **kwargs):
        scheduler_signature.bind(scheduler, **kwargs)
        scheduler._abort_callback = kwargs.get("abort_callback")

    monkeypatch.setattr(OmniScheduler, "__init__", init_scheduler)
    scheduler = minicpm_bootstrap.create_thinker_scheduler(
        SimpleNamespace(disable_cuda_graph=True), speech_enabled=speech_enabled
    )
    assert scheduler._abort_callback == runner.reset_request
    scheduler._run_abort_callback("aborted")
    scheduler._run_abort_callback("aborted")
    scheduler._run_abort_callback("unknown")
    assert set(runner.pending_hidden) == {"other"}


def test_finish_flushes_cloned_hidden_states_and_abort_drops_them():
    runner = _runner()
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(
                request_id="finished",
                data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=0)),
            )
        ]
    )
    outputs = {"finished": SimpleNamespace(extra={"hidden_states": hidden})}
    runner.post_process_outputs(None, scheduler_output, outputs)
    expected = hidden[-1].clone()
    hidden.zero_()
    assert outputs["finished"].extra == {}

    request_data = SimpleNamespace(extra_model_outputs={})
    runner.on_request_finished("finished", request_data)
    runner.reset_request("finished")
    assert not runner.pending_hidden
    sequence = request_data.extra_model_outputs["hidden_states_seq"]
    assert len(sequence) == 1
    torch.testing.assert_close(sequence[0], expected)
    assert sequence[0].device.type == "cpu"

    runner.pending_hidden["aborted"] = [torch.ones(4)]
    runner.reset_request("aborted")
    aborted_data = SimpleNamespace(extra_model_outputs={})
    runner.on_request_finished("aborted", aborted_data)
    assert aborted_data.extra_model_outputs == {}
    assert not runner.pending_hidden


def test_middle_prefill_chunk_does_not_capture_hidden_or_advance_state():
    runner = _runner()
    middle_req = SimpleNamespace(
        request_id="middle",
        data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=1)),
    )
    final_req = SimpleNamespace(
        request_id="final",
        data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=0)),
    )
    scheduler_output = SimpleNamespace(requests=[middle_req, final_req])
    outputs = {
        "middle": SimpleNamespace(extra={"hidden_states": torch.ones(1, 4)}),
        "final": SimpleNamespace(extra={"hidden_states": torch.zeros(1, 4)}),
    }

    runner.post_process_outputs(None, scheduler_output, outputs)

    assert set(runner.pending_hidden) == {"final"}
    assert outputs["middle"].extra == {}
    assert outputs["final"].extra == {}
    assert runner.finalize_skip_rids(scheduler_output) == {"middle"}


@pytest.mark.parametrize("missing_output", [False, True])
def test_requests_without_hidden_output_do_not_accumulate(missing_output):
    runner = _runner()
    scheduler_output = SimpleNamespace(requests=[SimpleNamespace(request_id="text")])
    outputs = {} if missing_output else {"text": SimpleNamespace(extra=None)}

    runner.post_process_outputs(None, scheduler_output, outputs)

    assert not runner.pending_hidden


def test_missing_chunk_state_is_not_treated_as_final():
    runner = _runner()
    request = SimpleNamespace(request_id="broken", data=SimpleNamespace(req=object()))
    scheduler_output = SimpleNamespace(requests=[request])
    outputs = {"broken": SimpleNamespace(extra={"hidden_states": torch.ones(1, 4)})}

    with pytest.raises(AttributeError, match="inflight_middle_chunks"):
        runner.post_process_outputs(None, scheduler_output, outputs)
    with pytest.raises(AttributeError, match="inflight_middle_chunks"):
        runner.finalize_skip_rids(scheduler_output)
    assert not runner.pending_hidden


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("middle_chunks,token_id", [(1, 7), (0, None), (0, 7)])
def test_stream_output_respects_request_and_chunk_state(
    stream, middle_chunks, token_id
):
    from sglang_omni.models.minicpm_o.request_builders import (
        build_thinker_stream_output,
    )
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
    from sglang_omni.scheduling.types import RequestOutput

    data = SGLangARRequestData(
        req=SimpleNamespace(inflight_middle_chunks=middle_chunks),
        stage_payload=StagePayload(
            request_id="stream",
            request=OmniRequest(inputs=None, params={"stream": stream}),
            data={},
        ),
    )
    output = RequestOutput(request_id="stream", data=token_id)

    messages = build_thinker_stream_output("stream", data, output)

    if not stream or middle_chunks or token_id is None:
        assert messages == []
    else:
        assert len(messages) == 1
        assert messages[0].target == "decode"
        assert messages[0].data.tolist() == [7]
        assert messages[0].metadata == {"token_id": 7}


def test_single_request_prefill_preserves_all_hidden_rows():
    class ForwardMode:
        def is_extend(self):
            return True

    req = SimpleNamespace(extend_range=SimpleNamespace(length=3))
    scheduler_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="req")],
        batch_data=SimpleNamespace(
            reqs=[req],
            forward_mode=ForwardMode(),
        ),
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    selected = SGLangOutputProcessor._slice_per_request_tensor(
        hidden,
        request_index=0,
        scheduler_output=scheduler_output,
    )

    torch.testing.assert_close(selected, hidden)
