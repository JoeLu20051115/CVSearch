"""Scoped execution of independent generator and verifier scoring."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager

import torch


def _device(generator, verifier):
    if ((type(generator).__module__, type(generator).__name__),
        (type(verifier).__module__, type(verifier).__name__)) != (
        ("qavs.models.modeling_internvl", "ModelInternvl"),
        ("qavs.models.modeling_llava", "ModelGlobalLocal"),
    ):
        return None
    models = (getattr(generator, "model", None), getattr(verifier, "model", None))
    if (models[0] is models[1] or not all(isinstance(model, torch.nn.Module) for model in models)
            or any(module.training for model in models for module in model.modules())
            or not torch.cuda.is_available()):
        return None
    devices = {parameter.device for model in models for parameter in model.parameters()}
    if len(devices) != 1:
        return None
    device = devices.pop()
    return device if device.type == "cuda" else None


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


@contextmanager
def parallel_model_scoring(generator, verifier):
    """Yield a blocking pair(fn, fn), or None for unsupported model pairs.

    Callbacks retain their complete prompts and return independent results.
    One sample owns the executor; callbacks must not submit nested pairs.
    Sequential samples reuse each wrapper's stream so allocator blocks can be reused.
    """
    device = _device(generator, verifier)
    if device is None:
        yield None
        return
    streams = []
    for wrapper in (generator, verifier):
        stream = getattr(wrapper, "_parallel_scoring_stream", None)
        if stream is None or stream.device != device:
            stream = torch.cuda.Stream(device=device)
            wrapper._parallel_scoring_stream = stream
        streams.append(stream)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-scoring") as executor:
        def pair(generator_fn, verifier_fn):
            producer = torch.cuda.current_stream(device)
            grad = torch.is_grad_enabled()
            inference = torch.is_inference_mode_enabled()
            autocast = [(kind, torch.is_autocast_enabled(kind), torch.get_autocast_dtype(kind))
                        for kind in ("cpu", "cuda")]
            cache_enabled = torch.is_autocast_cache_enabled()

            def run(function, stream):
                with ExitStack() as stack:
                    stack.enter_context(torch.cuda.device(device))
                    stack.enter_context(torch.cuda.stream(stream))
                    stack.enter_context(torch.inference_mode(inference))
                    stack.enter_context(torch.set_grad_enabled(grad))
                    for kind, enabled, dtype in autocast:
                        stack.enter_context(torch.autocast(kind, enabled=enabled, dtype=dtype,
                                                           cache_enabled=cache_enabled))
                    stream.wait_stream(producer)
                    try:
                        return function()
                    finally:
                        stream.synchronize()

            futures = []
            try:
                for function, stream in zip((generator_fn, verifier_fn), streams):
                    futures.append(executor.submit(run, function, stream))
                results = tuple(future.result() for future in futures)
                _record_stream(results, producer)
                return results
            finally:
                # Propagate the first role's exception only after both roles stop.
                wait(futures)

        yield pair
