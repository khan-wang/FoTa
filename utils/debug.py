"""Numerical diagnostics used by FoTa-Net training and inference."""

import torch

try:
    import torch._dynamo as _dynamo  # type: ignore[attr-defined]

    def _is_compiling() -> bool:
        try:
            return bool(_dynamo.is_compiling())
        except Exception:
            return False
except Exception:  # pragma: no cover - dynamo may be unavailable
    def _is_compiling() -> bool:
        return False


def _cfg_get(obj: object, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def sanity_state(cfg: object) -> dict:
    s = _cfg_get(cfg, 'SANITY', None)
    return {
        'NAN_GUARD': bool(_cfg_get(s, 'NAN_GUARD', False)),
        'NAN_GUARD_UNTIL': int(_cfg_get(s, 'NAN_GUARD_UNTIL', 0) or 0),
        'NAN_GUARD_EVERY': int(_cfg_get(s, 'NAN_GUARD_EVERY', 50) or 50),
        'SANITIZE_ALWAYS': bool(_cfg_get(s, 'SANITIZE_ALWAYS', False)),
    }


def format_sanity_state(cfg: object) -> str:
    st = sanity_state(cfg)
    return (
        "[Sanity] "
        f"NAN_GUARD={st['NAN_GUARD']} "
        f"until={st['NAN_GUARD_UNTIL']} "
        f"every={st['NAN_GUARD_EVERY']} "
        f"SANITIZE_ALWAYS={st['SANITIZE_ALWAYS']}"
    )


def assert_finite_maybe(name: str, t: torch.Tensor, cfg: object, gs: int):
    """Run a conditional NaN/Inf probe during the configured training window."""
    if _is_compiling():
        return

    st = sanity_state(cfg)

    if not st['NAN_GUARD']:
        return


    if gs >= st['NAN_GUARD_UNTIL']:
        return


    if (gs % st['NAN_GUARD_EVERY']) != 0:
        return



    is_all_finite = torch.all(torch.isfinite(t))
    if not is_all_finite.item():

        bad_ratio = float((~torch.isfinite(t)).float().mean().item())
        print(f"\n\033[91m[NaNGuard] At step {gs}, {name}: non-finite ratio={bad_ratio*100:.6f}%\033[0m")


        raise RuntimeError(f"[NaNGuard] Training stopped. Tensor '{name}' at step {gs} contains non-finite values.")

def sanitize_maybe(t: torch.Tensor, cfg: object) -> torch.Tensor:
    """Replace non-finite tensor values when sanitization is enabled."""
    if sanity_state(cfg)['SANITIZE_ALWAYS']:
        return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
    return t
