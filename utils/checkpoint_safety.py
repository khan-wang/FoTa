from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch
import torch.nn as nn


def _summarize_keys(keys: Iterable[str], limit: int = 20) -> str:
    keys = list(keys)
    if not keys:
        return "[]"
    shown = ", ".join(keys[:limit])
    suffix = "" if len(keys) <= limit else f", ... (+{len(keys) - limit} more)"
    return f"[{shown}{suffix}]"


def load_state_dict_report(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    label: str,
    *,
    strict: bool = False,
    critical_prefixes: Optional[Iterable[str]] = None,
    fail_on_critical: bool = False,
):
    """Load a state dict and always report missing/unexpected keys.

    The default remains compatible with legacy checkpoints (`strict=False`), but
    critical prefix mismatches can be escalated to fail-fast by config.
    """
    result = module.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    if missing or unexpected:
        print(
            f"[ckpt:{label}] strict={strict} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[ckpt:{label}] missing keys: {_summarize_keys(missing)}")
        if unexpected:
            print(f"[ckpt:{label}] unexpected keys: {_summarize_keys(unexpected)}")
    else:
        print(f"[ckpt:{label}] loaded with no missing/unexpected keys.")

    critical_prefixes = tuple(critical_prefixes or ())
    if critical_prefixes:
        critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
        critical_unexpected = [k for k in unexpected if k.startswith(critical_prefixes)]
        if critical_missing or critical_unexpected:
            msg = (
                f"[ckpt:{label}] CRITICAL module mismatch under prefixes "
                f"{critical_prefixes}: missing={len(critical_missing)} "
                f"unexpected={len(critical_unexpected)}"
            )
            if fail_on_critical:
                raise RuntimeError(msg)
            print(f"\033[93m{msg}\033[0m")

    return result


def filter_matching_tensors(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
):
    keep = {}
    mismatched = []
    for key, value in source.items():
        target_value = target.get(key)
        if (
            target_value is not None
            and hasattr(value, "shape")
            and hasattr(target_value, "shape")
            and value.shape == target_value.shape
        ):
            keep[key] = value
        else:
            mismatched.append(key)
    return keep, mismatched
