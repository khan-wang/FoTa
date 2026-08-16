# utils/amp.py
from __future__ import annotations
import contextlib
import torch

def _resolve_autocast(device_type: str = "cuda", **kwargs):
    """Return a real context manager. When disabled -> nullcontext()."""
    enabled = kwargs.pop("enabled", True)
    if not enabled:
        return contextlib.nullcontext()

    try:
        return torch.autocast(device_type=device_type, **kwargs)
    except Exception:
        try:
            from torch.cuda.amp import autocast as cuda_autocast

            return cuda_autocast(**kwargs)
        except Exception:

            return contextlib.nullcontext()

class _AutocastShim:
    def __init__(self, device_type: str = "cuda", **kwargs):
        self.device_type = device_type
        self.kwargs = kwargs
        self._ctx = None

    # context manager
    def __enter__(self):
        self._ctx = _resolve_autocast(self.device_type, **self.kwargs)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)

    # decorator
    def __call__(self, fn):
        def wrapped(*a, **k):
            with _resolve_autocast(self.device_type, **self.kwargs):
                return fn(*a, **k)
        wrapped.__name__ = getattr(fn, "__name__", "wrapped_autocast")
        return wrapped

def autocast(*args, **kwargs):

    if args and isinstance(args[0], str):

        kwargs["device_type"] = args[0]
        args = args[1:]
    return _AutocastShim(**kwargs)


try:
    from torch.cuda.amp import GradScaler  # noqa: F401
except Exception:  # pragma: no cover
    class GradScaler:  # type: ignore
        def __init__(self, *a, **k): print("WARNING: GradScaler unavailable, AMP disabled.")
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass
        def state_dict(self): return {}
        def load_state_dict(self, s): pass
