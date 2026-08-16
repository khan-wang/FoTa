from pathlib import Path
from typing import Dict


def _unwrap_module(m):
    for a in ('_orig_mod', 'module'):
        if hasattr(m, a):
            return getattr(m, a)
    return m


def _first_conv_in_channels(m):
    m = _unwrap_module(m)
    if hasattr(m, 'in_channels') and isinstance(getattr(m, 'in_channels'), int):
        try:
            return int(getattr(m, 'in_channels'))
        except Exception:
            pass
    for c in m.children():
        c = _unwrap_module(c)
        if hasattr(c, 'in_channels') and isinstance(getattr(c, 'in_channels'), int):
            try:
                return int(getattr(c, 'in_channels'))
            except Exception:
                pass
        v = _first_conv_in_channels(c)
        if v is not None:
            return v
    return None


def _bool(cfg, name: str, default: bool = False) -> bool:
    return bool(getattr(cfg, name, default))


def _get(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def preflight_check(cfg, sched_mgr, optim_g, optim_d, global_step: int,
                    discriminator=None) -> Dict[str, float]:
    """Fail-fast validation after checkpoint loading and scheduler setup."""

    sched_mgr.step(int(global_step))
    lr_g = float(optim_g.param_groups[0]['lr']) if optim_g.param_groups else 0.0
    lr_d = float(optim_d.param_groups[0]['lr']) if optim_d.param_groups else 0.0
    gan_cap = float(sched_mgr.current_gan_cap())
    current_adv_weight = float(sched_mgr.value('adversarial_weight', getattr(cfg, 'ADVERSARIAL_WEIGHT', 0.0)))
    adv_delay_steps = float(getattr(cfg, 'ADV_DELAY_STEPS', 0))
    adv_scale = 0.0 if global_step < adv_delay_steps else 1.0
    eff_adv = current_adv_weight * gan_cap * adv_scale
    upd_d = int(sched_mgr.update_d_every())
    r1_every = int(sched_mgr.r1_every())
    r1_gamma = float(sched_mgr.r1_gamma())

    resume = _bool(cfg, 'RESUME', False)
    finetune_mode = _bool(cfg, 'FINETUNE_MODE', False)
    schedule_absolute = _bool(cfg, 'SCHEDULE_ABSOLUTE', True)

    if finetune_mode and not resume:
        raise SystemExit('[Preflight] FINETUNE_MODE requires RESUME=true.')


    if finetune_mode and not _bool(cfg, 'RESET_OPT_AND_SCHED', False):
        raise SystemExit('[Preflight] FINETUNE_MODE requires RESET_OPT_AND_SCHED=true.')


    if finetune_mode and not _bool(cfg, 'OVERRIDE_LR_ON_RESUME', False):
        print('\033[93m[Preflight] WARNING: OVERRIDE_LR_ON_RESUME is usually true for finetune.\033[0m')

    resume_ckpt = str(getattr(cfg, 'RESUME_CHECKPOINT', '') or '')
    if resume and resume_ckpt.endswith('latest.pth') and not schedule_absolute:
        raise SystemExit('[Preflight] Resuming from latest.pth requires SCHEDULE_ABSOLUTE=true.')

    fsm_cfg = getattr(cfg, 'FSM', {}) or {}
    wake_cfg = _get(fsm_cfg, 'wake_d', {}) or {}
    if (_get(fsm_cfg, 'enabled', False) or _get(wake_cfg, 'override', None)) and not _bool(cfg, 'OVERRIDE_LR_ON_RESUME', False):
        raise SystemExit('[Preflight] FSM requires OVERRIDE_LR_ON_RESUME=true to ensure lrD overrides apply.')

    min_expected_lr_d = float(getattr(cfg, 'MIN_EXPECTED_LR_D', 1e-5))
    if lr_d < min_expected_lr_d:
        raise SystemExit(f'[Preflight] lrD={lr_d:.3e} below MIN_EXPECTED_LR_D={min_expected_lr_d:.3e}.')

    if eff_adv < 0.02:
        is_in_warmup_or_delay = (global_step < adv_delay_steps) or (current_adv_weight == 0.0)

        if is_in_warmup_or_delay:

            print(f"\033[93m[Preflight] WARNING: Effective adversarial weight is {eff_adv:.3f} (GS={global_step}). "
                  f"This is OK during ADV_DELAY_STEPS or if PHASES sets adv_weight=0.\033[0m")
        else:

            raise SystemExit(f'[Preflight] Effective adversarial weight too small: {eff_adv:.3f}. '
                             f'Check ADV_WEIGHT and GAN_CAP in config.')

    d_floor = float(getattr(cfg, 'D_LR_FLOOR', 0.0) or 0.0)
    lr_d_target = float(sched_mgr.value('lr_d_target', getattr(cfg, 'LEARNING_RATE_D', lr_d)))
    if d_floor > 0 and lr_d_target < d_floor:
        print(
            f"\033[93m[Preflight] WARNING: D_LR_FLOOR={d_floor:.3e} is above current "
            f"phase lr_d_target={lr_d_target:.3e}; D lr will be clamped.\033[0m"
        )

    expected_disc_nc = int(getattr(cfg, 'IMG_CHANNEL', 3)) + (1 if getattr(cfg, 'USE_MASK_IN_D', False) else 0)
    configured_disc_nc = int(getattr(cfg, 'DISC_INPUT_NC', expected_disc_nc))
    if configured_disc_nc != expected_disc_nc:
        prefix = "When USE_MASK_IN_D=True" if getattr(cfg, 'USE_MASK_IN_D', False) else "When USE_MASK_IN_D=False"
        raise SystemExit(f"[Preflight] {prefix}, DISC_INPUT_NC must be {expected_disc_nc}.")

    if discriminator is not None:
        disc_mod = _unwrap_module(discriminator)
        disc_in_channels = _first_conv_in_channels(disc_mod)

        if disc_in_channels is None:
            print("\033[93m[Preflight] WARNING: Could not determine discriminator in_channels.\033[0m")
        elif int(disc_in_channels) != expected_disc_nc:
            raise SystemExit(
                f"[Preflight] netD.in_channels={disc_in_channels} but expected {expected_disc_nc} (mask_in_d={getattr(cfg, 'USE_MASK_IN_D', False)})."
            )

    summary = {
        'lrG': lr_g,
        'lrD': lr_d,
        'ganCap': gan_cap,
        'effAdv': eff_adv,
        'advW': current_adv_weight,
        'updD': upd_d,
        'r1Every': r1_every,
        'r1Gamma': r1_gamma,
        'ABS': schedule_absolute,
        'FINETUNE': finetune_mode,
        'RESUME': resume,
    }

    def _fmt_val(val):
        return f"{val:.2e}" if isinstance(val, float) and abs(val) < 1000 else str(val)

    panel = ' '.join(f"{k}={_fmt_val(v)}" for k, v in summary.items())
    print(f"[Preflight] {panel}")
    resume_ckpt_path = getattr(cfg, 'RESUME_CHECKPOINT', None)
    if resume and resume_ckpt_path:
        try:
            resolved = Path(resume_ckpt_path).resolve()
        except Exception:
            resolved = resume_ckpt_path
        print(f"[Preflight] Resume checkpoint => {resolved}")
    return summary
