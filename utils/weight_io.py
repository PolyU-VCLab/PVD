def load_weights(path):
    from pathlib import Path
    import torch

    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model", state)
    if (
        not isinstance(state, dict)
        or not state
        or (
            not all(
                (
                    isinstance(k, str) and isinstance(v, torch.Tensor)
                    for (k, v) in state.items()
                )
            )
        )
    ):
        raise ValueError("Expected a nonempty tensor state dictionary")
    return {k.removeprefix("module."): v for (k, v) in state.items()}


def load_checkpoint(path, *args, **kwargs):
    """Load tensor exports or training checkpoints using a bounded type allowlist."""
    import argparse
    import importlib
    from pathlib import Path
    import torch

    if Path(path).suffix == ".safetensors":
        return {"model": load_weights(path)}
    kwargs["weights_only"] = True
    allowed = [argparse.Namespace]
    known = {
        "deepspeed.runtime.fp16.loss_scaler.LossScaler",
        "deepspeed.runtime.fp16.loss_scaler.DynamicLossScaler",
        "deepspeed.runtime.zero.config.ZeroStageEnum",
        "deepspeed.utils.tensor_fragment.fragment_address",
    }
    while True:
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(path, *args, **kwargs)
        except Exception as exc:
            match = next(
                (
                    name
                    for name in known
                    if "Unsupported global: GLOBAL " + name in str(exc)
                ),
                None,
            )
            if match is None:
                raise
            module, name = match.rsplit(".", 1)
            allowed.append(getattr(importlib.import_module(module), name))
            known.remove(match)
