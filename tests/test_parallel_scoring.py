from contextlib import contextmanager, nullcontext
from threading import Barrier, Event, Thread, get_ident, local
from types import SimpleNamespace

import pytest
import torch

from qavs.evidence_gap import parallel_scoring
# The scheduler only needs wrapper identity and a model/device pair.
ModelInternvl = type("ModelInternvl", (), {"__module__": "qavs.models.modeling_internvl"})
ModelGlobalLocal = type("ModelGlobalLocal", (), {"__module__": "qavs.models.modeling_llava"})


class DeviceModel(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.parameter = SimpleNamespace(device=torch.device(device))
        self.eval()

    def parameters(self):
        yield self.parameter


def wrappers():
    generator = ModelInternvl.__new__(ModelInternvl)
    verifier = ModelGlobalLocal.__new__(ModelGlobalLocal)
    for wrapper in (generator, verifier):
        wrapper.model = DeviceModel()
        wrapper.device = "cuda:0"
    return generator, verifier


@pytest.fixture
def cuda(monkeypatch):
    state = local()
    log, streams = [], []
    producer = SimpleNamespace(name="producer")

    class Stream:
        def __init__(self, device):
            self.device = torch.device(device)
            self.name = f"role-{len(streams)}"
            streams.append(self)

        def wait_stream(self, prior):
            log.append(("wait", self.name, prior.name))

        def synchronize(self):
            log.append(("sync", self.name))

    @contextmanager
    def use_stream(stream):
        previous = getattr(state, "stream", producer)
        state.stream = stream
        try:
            yield
        finally:
            state.stream = previous

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: getattr(state, "stream", producer))
    return SimpleNamespace(log=log, streams=streams, state=state, producer=producer)


@pytest.mark.parametrize("reason", ["unknown", "cpu", "different_devices", "identical", "training", "child_training", "no_cuda"])
def test_unsupported_models_yield_none_without_allocating_resources(cuda, monkeypatch, reason):
    generator, verifier = wrappers()
    if reason == "unknown":
        generator = SimpleNamespace(model=generator.model, device=generator.device)
    elif reason == "cpu":
        for wrapper in (generator, verifier):
            wrapper.model.parameter.device = torch.device("cpu")
            wrapper.device = "cpu"
    elif reason == "different_devices":
        verifier.model.parameter.device = torch.device("cuda:1")
        verifier.device = "cuda:1"
    elif reason == "identical":
        verifier.model = generator.model
    elif reason == "training":
        generator.model.train()
    elif reason == "child_training":
        verifier.model.child = torch.nn.Identity().train()
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair is None
    assert cuda.streams == []


def test_reuses_two_worker_streams_waiting_for_each_callers_producer(cuda):
    owner = get_ident()
    with parallel_scoring.parallel_model_scoring(*wrappers()) as pair:
        assert callable(pair)
        for producer_name in ("producer", "next-producer"):
            cuda.producer.name = producer_name
            ready = Barrier(2)
            def operation(role):
                ready.wait(timeout=2)
                stream = torch.cuda.current_stream()
                cuda.log.append(("call", stream.name, role))
                return role, stream.name, get_ident()
            result = pair(lambda: operation("generator"), lambda: operation("verifier"))
            assert [item[0] for item in result] == ["generator", "verifier"]
            assert [item[1] for item in result] == ["role-0", "role-1"]
            assert len({item[2] for item in result}) == 2
            assert all(item[2] != owner for item in result)
        assert len(cuda.streams) == 2
    for role in ("role-0", "role-1"):
        events = [item for item in cuda.log if item[1] == role]
        assert [event[0] for event in events] == ["wait", "call", "sync"] * 2
        assert [event[2] for event in events if event[0] == "wait"] == ["producer", "next-producer"]
    with pytest.raises(RuntimeError, match="shutdown"):
        pair(lambda: None, lambda: None)


def test_successive_samples_reuse_streams_with_fresh_executors(cuda):
    generator, verifier = wrappers()
    for producer_name in ("first-sample", "second-sample"):
        cuda.producer.name = producer_name
        with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
            ready = Barrier(2)

            def operation():
                ready.wait(timeout=2)
                stream = torch.cuda.current_stream()
                cuda.log.append(("call", stream.name))
                return stream.name

            assert pair(operation, operation) == ("role-0", "role-1")
        with pytest.raises(RuntimeError, match="shutdown"):
            pair(operation, operation)
    assert len(cuda.streams) == 2
    for role in ("role-0", "role-1"):
        events = [event for event in cuda.log if event[1] == role]
        assert [event[0] for event in events] == ["wait", "call", "sync"] * 2
        assert [event[2] for event in events if event[0] == "wait"] == [
            "first-sample", "second-sample",
        ]


def test_cached_streams_recheck_training_and_parameter_devices(cuda):
    generator, verifier = wrappers()
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair(lambda: 1, lambda: 2) == (1, 2)
    generator.model.train()
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair is None
    generator.model.eval()
    generator.model.parameter.device = torch.device("cuda:1")
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair is None
    assert len(cuda.streams) == 2
    verifier.model.parameter.device = torch.device("cuda:1")
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair(lambda: torch.cuda.current_stream().device,
                    lambda: torch.cuda.current_stream().device) == (
            torch.device("cuda:1"), torch.device("cuda:1"),
        )
    assert len(cuda.streams) == 4


def modes():
    return (
        torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
        torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
        torch.is_autocast_cache_enabled(),
    )


def test_workers_copy_each_calls_grad_inference_and_autocast_modes(cuda):
    with parallel_scoring.parallel_model_scoring(*wrappers()) as pair:
        assert pair(modes, modes) == (modes(), modes())
        with torch.no_grad():
            assert pair(modes, modes) == (modes(), modes())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.autocast("cpu"):
            assert pair(modes, modes) == (modes(), modes())
        assert pair(modes, modes) == (modes(), modes())


def test_exception_drains_other_role_and_keeps_executor_usable(cuda):
    ready = Barrier(2)
    waiting, release, drained, returned = Event(), Event(), Event(), Event()
    failures = []

    def fail():
        ready.wait(timeout=2)
        raise ValueError("generator failed")

    def finish():
        ready.wait(timeout=2)
        waiting.set()
        assert release.wait(timeout=2)
        drained.set()
        return "verifier finished"

    generator, verifier = wrappers()
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        def caller():
            try:
                pair(fail, finish)
            except BaseException as error:
                failures.append((error, drained.is_set()))
            finally:
                returned.set()
        thread = Thread(target=caller)
        thread.start()
        assert waiting.wait(timeout=2)
        assert not returned.wait(timeout=0.05)
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(failures) == 1 and isinstance(failures[0][0], ValueError)
        assert str(failures[0][0]) == "generator failed"
        assert failures[0][1]
        assert sum(event[0] == "sync" for event in cuda.log) == 2
        assert pair(lambda: 1, lambda: 2) == (1, 2)
    with parallel_scoring.parallel_model_scoring(generator, verifier) as pair:
        assert pair(lambda: 3, lambda: 4) == (3, 4)
    assert len(cuda.streams) == 2
    assert sum(event[0] == "sync" for event in cuda.log) == 6


def test_verifier_exception_propagates_after_generator_finishes(cuda):
    def fail():
        raise RuntimeError("verifier failed")
    with parallel_scoring.parallel_model_scoring(*wrappers()) as pair:
        with pytest.raises(RuntimeError, match="verifier failed"):
            pair(lambda: "done", fail)
    assert sum(event[0] == "sync" for event in cuda.log) == 2
