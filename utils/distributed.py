from __future__ import annotations

import builtins
import atexit
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class DistributedContext:
    launched: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(timeout_minutes: int = 30) -> DistributedContext:
    launched = all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if launched and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        atexit.register(cleanup_distributed)

    return DistributedContext(
        launched=launched,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend if launched else None,
        device=device,
    )


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if distributed_ready():
        dist.barrier()


def cleanup_distributed() -> None:
    if distributed_ready():
        dist.destroy_process_group()


def configure_process_output(is_main: bool) -> None:
    if is_main:
        return
    original_print = builtins.print

    def rank_print(*args, **kwargs):
        force = bool(kwargs.pop("force", False))
        if force:
            original_print(*args, **kwargs)

    builtins.print = rank_print


def broadcast_object(value: Any, src: int = 0, device: torch.device | None = None) -> Any:
    if not distributed_ready():
        return value
    payload = [value]
    try:
        dist.broadcast_object_list(payload, src=src, device=device)
    except TypeError:
        dist.broadcast_object_list(payload, src=src)
    return payload[0]


def broadcast_bool(value: bool, device: torch.device, src: int = 0) -> bool:
    if not distributed_ready():
        return bool(value)
    tensor = torch.tensor([int(bool(value))], dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=src)
    return bool(tensor.item())


def any_true(value: bool, device: torch.device) -> bool:
    if not distributed_ready():
        return bool(value)
    tensor = torch.tensor([int(bool(value))], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def mean_scalar(value: float, device: torch.device) -> float:
    if not distributed_ready():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def mean_scalars(values: list[float] | tuple[float, ...], device: torch.device) -> tuple[float, ...]:
    if not distributed_ready():
        return tuple(float(value) for value in values)
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tuple(float(value) for value in tensor.tolist())


def max_scalar(value: float, device: torch.device) -> float:
    if not distributed_ready():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def unwrap_model(module: nn.Module) -> nn.Module:
    current = module
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        break
    return current
