import os
from pathlib import Path
from typing import Any, Dict


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get(cfg, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _warn_or_fail(message: str, strict: bool = False) -> None:
    if strict:
        raise SystemExit(message)
    print(f"\033[93m{message}\033[0m")


def _phase_payload(phase: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in phase.items()
        if k not in {"name", "start", "step", "dur", "duration"}
    }


def _ensure_path_exists(path_value: Any, label: str) -> None:
    if not path_value:
        raise SystemExit(f"[ConfigLint] {label} is required but empty.")
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(f"[ConfigLint] {label} path does not exist: {path}")


def validate_config(cfg) -> None:
    """Performs strict sanity checks after YAML merging."""

    resume = _as_bool(_get(cfg, "RESUME", False))
    finetune_mode = _as_bool(_get(cfg, "FINETUNE_MODE", False))
    schedule_absolute = _as_bool(_get(cfg, "SCHEDULE_ABSOLUTE", True))
    reset_opt = _as_bool(_get(cfg, "RESET_OPT_AND_SCHED", False))
    override_lr_on_resume = _as_bool(_get(cfg, "OVERRIDE_LR_ON_RESUME", False))
    resume_ckpt = str(_get(cfg, "RESUME_CHECKPOINT", "") or "")
    strict_lint = _as_bool(_get(cfg, "STRICT_CONFIG_LINT", False))

    mode_flags: Dict[str, bool] = {
        "hot_resume_latest": resume and resume_ckpt.endswith("latest.pth") and not finetune_mode and schedule_absolute,
        "finetune_from_best": resume and resume_ckpt.endswith("best_generator_ema.pth") and finetune_mode and reset_opt,
    }

    if resume:
        if finetune_mode and not reset_opt:
            raise SystemExit("[ConfigLint] FINETUNE_MODE requires RESET_OPT_AND_SCHED=true.")
        if resume_ckpt.endswith("latest.pth") and not schedule_absolute:
            raise SystemExit("[ConfigLint] Resuming from latest.pth requires SCHEDULE_ABSOLUTE=true.")
        if sum(1 for ok in mode_flags.values() if ok) == 0:
            raise SystemExit("[ConfigLint] Unsupported resume workflow. Only hot_resume_latest or finetune_from_best allowed.")
        resume_path = Path(resume_ckpt) if resume_ckpt else None
        if not resume_path or not resume_path.exists():
            raise SystemExit(f"[ConfigLint] RESUME=true but RESUME_CHECKPOINT missing: {resume_ckpt}")
    elif finetune_mode:
        raise SystemExit("[ConfigLint] FINETUNE_MODE cannot be true when RESUME=false.")

    fsm_cfg = getattr(cfg, "FSM", {}) or {}
    if (fsm_cfg.get("enabled") or fsm_cfg.get("wake_d", {}).get("override") or fsm_cfg.get("soften_d", {}).get("override")) and not override_lr_on_resume:
        raise SystemExit("[ConfigLint] FSM/WakeD requires OVERRIDE_LR_ON_RESUME=true.")

    min_expected_lr_d = float(_get(cfg, "MIN_EXPECTED_LR_D", 1e-5))
    if min_expected_lr_d <= 0:
        raise SystemExit("[ConfigLint] MIN_EXPECTED_LR_D must be > 0.")

    mask_bank = getattr(cfg, "MASK_BANK", None)
    if mask_bank:
        mask_root = mask_bank.get("IRR_ROOT")
        if mask_root:
            _ensure_path_exists(mask_root, "MASK_BANK.IRR_ROOT")
    if _get(cfg, "MASK_BANK", None) is None and _get(cfg, "MASK_BANK_PATH", None):
        _ensure_path_exists(cfg.MASK_BANK_PATH, "MASK_BANK_PATH")

    if getattr(cfg, "MASK_BANK", None) is None:
        raise SystemExit("[ConfigLint] MASK_BANK configuration is required.")

    for attr in ("VAL_IMG_FLIST", "MASK_FLIST_EVAL"):
        value = getattr(cfg, attr, None)
        if isinstance(value, (str, os.PathLike)):
            _ensure_path_exists(value, attr)

    eval_cfg = getattr(cfg, "EVAL", None)
    if eval_cfg:
        for attr in ("VAL_IMG_FLIST", "MASK_FLIST_EVAL"):
            nested_val = None
            if isinstance(eval_cfg, dict):
                nested_val = eval_cfg.get(attr)
            else:
                nested_val = getattr(eval_cfg, attr, None)
            if isinstance(nested_val, (str, os.PathLike)):
                _ensure_path_exists(nested_val, f"EVAL.{attr}")

    phases = getattr(cfg, "PHASES", None) or []
    phases_with_lr = any(
        isinstance(phase, dict) and any(
            key in {"lr_g", "lr_d", "lrD", "lr_g_target", "lr_d_target"}
            for key in phase.keys()
        )
        for phase in phases
    )
    cosine_candidates = [
        str(_get(cfg, "LR_SCHEDULER", "") or ""),
        str(_get(cfg, "LR_POLICY", "") or ""),
    ]
    cosine_flags = [
        _as_bool(_get(cfg, "USE_COSINE_LR", False)),
        _as_bool(_get(cfg, "USE_COSINE_LR_G", False)),
        _as_bool(_get(cfg, "USE_COSINE_LR_D", False)),
    ]
    has_cosine = any(flag for flag in cosine_flags)
    if not has_cosine:
        has_cosine = any('cosine' in cand.lower() for cand in cosine_candidates if cand)
    if not has_cosine:
        for name in dir(cfg):
            if not name or name.startswith('_'):
                continue
            if 'cosine' in name.lower():
                value = getattr(cfg, name)
                if isinstance(value, bool) and value:
                    has_cosine = True
                    break
                if isinstance(value, str) and 'cosine' in value.lower():
                    has_cosine = True
                    break
    if has_cosine and phases_with_lr:
        raise SystemExit(
            "[ConfigLint] Cosine LR scheduling cannot be combined with PHASES lr targets."
        )

    if getattr(cfg, "torch_compile", False):
        try:
            import torch  # noqa: F401
            if not hasattr(torch, "compile"):
                raise SystemExit("[ConfigLint] torch.compile requested but not supported by this PyTorch build.")
        except ImportError as exc:
            raise SystemExit(f"[ConfigLint] torch not installed while torch.compile requested: {exc}")

    if phases:
        from uformer_pytorch.scheduler_manager import SchedulerManager

        known_phase_keys = set(SchedulerManager._PHASE_KEY_MAP.keys())
        known_phase_keys.update(SchedulerManager._PHASE_KEY_MAP.values())
        known_phase_keys.update({
            "lr_g_target", "lr_d_target", "reset_d_optimizer",
        })
        KEY_KEYS_TO_CHECK = {
            "lr_g_target", "lr_d_target", "adversarial_weight", "r1_every",
            "gan_cap", "gate_temp", "spectral_dropout_rate", "fm_weight",
            "perc_in_weight", "perc_out_weight", "l1_hole_weight"
        }

        for i, phase in enumerate(phases):
            payload = _phase_payload(phase) if isinstance(phase, dict) else {}
            if not payload:
                continue

            unknown_keys = sorted(k for k in payload.keys() if k not in known_phase_keys)
            if unknown_keys:
                _warn_or_fail(
                    f"[ConfigLint] Phase {i} ('{phase.get('name', 'N/A')}') has unknown keys: {unknown_keys}",
                    strict_lint,
                )

            mapped_keys = {SchedulerManager._PHASE_KEY_MAP.get(k, k) for k in payload.keys()}
            missing_keys = KEY_KEYS_TO_CHECK - mapped_keys

            if missing_keys:
                print(
                    f"  \033[93m[WARN] Phase {i} ('{phase.get('name', 'N/A') if isinstance(phase, dict) else 'N/A'}') "
                    f"is missing critical keys: {sorted(list(missing_keys))}. Values will fall back to defaults.\033[0m"
                )

        d_floor = float(_get(cfg, "D_LR_FLOOR", 0.0) or 0.0)
        phase_lr_d = []
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            payload = _phase_payload(phase)
            lr_val = payload.get("lr_d_target", payload.get("lr_d", payload.get("lrD")))
            if lr_val is None:
                continue
            if isinstance(lr_val, dict):
                vals = [lr_val.get("start"), lr_val.get("end")]
            else:
                vals = [lr_val]
            for val in vals:
                if val is None:
                    continue
                try:
                    phase_lr_d.append((phase.get("name", "N/A"), float(val)))
                except (TypeError, ValueError):
                    pass
        below_floor = [(name, lr) for name, lr in phase_lr_d if 0.0 < lr < d_floor]
        if below_floor:
            sample = ", ".join(f"{name}:{lr:.2e}" for name, lr in below_floor[:6])
            _warn_or_fail(
                f"[ConfigLint] D_LR_FLOOR={d_floor:.2e} is above phase lr_d_target values ({sample}). "
                "Those phase LR targets will be clamped.",
                strict_lint,
            )

    config_path = str(_get(cfg, "_CONFIG_PATH", "") or "")
    config_name = Path(config_path).name
    if config_name:
        if "A1_noFNO" in config_name and list(_get(cfg, "FNO_STAGES", [])):
            _warn_or_fail("[ConfigLint] A1_noFNO config name conflicts with non-empty FNO_STAGES.", strict_lint)
        if "A3_noGatedFusion" in config_name and str(_get(cfg, "FUSION_MODE", "gated")).lower() == "gated":
            _warn_or_fail("[ConfigLint] A3_noGatedFusion config name conflicts with FUSION_MODE=gated.", strict_lint)
        if "B1_noCMA" in config_name and _as_bool(_get(cfg, "USE_CMT", False)):
            _warn_or_fail("[ConfigLint] B1_noCMA config name conflicts with USE_CMT=true.", strict_lint)
        if "B2_noESD" in config_name and _get(cfg, "SPECTRAL_DROPOUT_FORCE", None) not in (0, 0.0):
            _warn_or_fail("[ConfigLint] B2_noESD config should force SPECTRAL_DROPOUT_FORCE=0.0.", strict_lint)
        if "B3_noGatePrior" in config_name and _get(cfg, "GATE_PRIOR_FORCE", None) not in (0, 0.0):
            _warn_or_fail("[ConfigLint] B3_noGatePrior config should force GATE_PRIOR_FORCE=0.0.", strict_lint)

    if _as_bool(_get(cfg, "REQUIRE_HRF_LOSS_WEIGHTS", False)):
        _ensure_path_exists(cfg.HRF_LOSS_WEIGHTS_PATH, "HRF_LOSS_WEIGHTS_PATH")
