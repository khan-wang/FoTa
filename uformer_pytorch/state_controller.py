from collections import deque
from typing import Any, Dict, Optional

from utils.ckpt_ops import override_lr_and_reset


class GANStateController:
    """Finite state machine governing discriminator-related overrides."""

    STATES = ("Normal", "WakeD", "SoftenD", "Cooldown")

    def __init__(self, cfg):
        self.cfg = cfg
        fsm_cfg = getattr(cfg, "FSM", {}) or {}
        self.enabled = bool(fsm_cfg.get("enabled", False))
        self.state = "Normal"
        self.state_until: int = -1
        self._phase_context: Dict[str, Any] = {}
        self._history = deque(maxlen=16)
        self._psnr_history = deque(maxlen=16)
        self._wake_cfg = fsm_cfg.get("wake_d", {})
        self._soft_cfg = fsm_cfg.get("soften_d", {})
        self._cool_cfg = fsm_cfg.get("cooldown", {})
        self._consecutive_wake = 0
        self._consecutive_soften = 0
        self._optimizer_d = None
        self._lr_reset_pending = False
        self._entry_stats: Dict[str, Dict[str, int]] = {s: {"count": 0, "steps": 0} for s in self.STATES if s != "Normal"}
        self._last_state_step: Optional[int] = None

    # ------------------------------------------------------------------ public
    def attach_optimizer(self, optimizer_d) -> None:
        self._optimizer_d = optimizer_d

    def on_eval(self, gs: int, dgap: float, psnr_ema: float) -> None:
        if not self.enabled:
            return
        self._history.append((gs, dgap))
        self._psnr_history.append((gs, psnr_ema))
        self._maybe_trigger_wake(gs, dgap)
        self._maybe_trigger_soften(gs, dgap)
        self._maybe_trigger_cooldown(gs)

    def overrides(self, gs: int) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if self.state != "Normal" and gs >= self.state_until >= 0:
            self._close_state(gs)
        if self.state == "Normal":
            return {}
        override = dict(self._phase_context)
        override["state"] = self.state
        return override

    def consume_lr_reset(self) -> bool:
        flag = self._lr_reset_pending
        self._lr_reset_pending = False
        return flag

    def statistics(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self._entry_stats.items()}

    # ----------------------------------------------------------------- helpers
    def _set_state(self, new_state: str, gs: int, window: int, overrides: Dict[str, Any]) -> None:
        if self.state == new_state:
            self.state_until = max(self.state_until, gs + window)
            self._phase_context.update(overrides)
            return
        if self.state != "Normal":
            self._close_state(gs)
        self.state = new_state
        self.state_until = gs + window
        self._phase_context = dict(overrides)
        self._lr_reset_pending = "lr_d" in overrides or "lrD" in overrides or "lr_d_target" in overrides
        self._entry_stats.setdefault(new_state, {"count": 0, "steps": 0})
        self._entry_stats[new_state]["count"] += 1
        self._last_state_step = gs
        print(self._fmt_log("enter", gs, window, overrides))
        if self._lr_reset_pending and self._optimizer_d is not None:
            target_lr = overrides.get("lr_d") or overrides.get("lrD") or overrides.get("lr_d_target")
            if target_lr is not None:
                override_lr_and_reset(self._optimizer_d, float(target_lr))
                self._lr_reset_pending = False

    def _close_state(self, gs: int) -> None:
        if self.state == "Normal":
            return
        if self._last_state_step is not None:
            steps = max(0, gs - self._last_state_step)
            stats = self._entry_stats.setdefault(self.state, {"count": 0, "steps": 0})
            stats["steps"] += steps
        print(self._fmt_log("exit", gs, 0, {}))
        self.state = "Normal"
        self.state_until = -1
        self._phase_context = {}
        self._lr_reset_pending = False
        self._last_state_step = None

    def _maybe_trigger_wake(self, gs: int, dgap: float) -> None:
        thr = float(self._wake_cfg.get("trigger_dgap_lt", 0.05))
        required = int(self._wake_cfg.get("trigger_consecutive", 2))
        if dgap < thr:
            self._consecutive_wake += 1
        else:
            self._consecutive_wake = 0
        if self._consecutive_wake >= required:
            window = int(self._wake_cfg.get("window_steps", 3000))
            overrides = self._wake_cfg.get("override", {})
            self._set_state("WakeD", gs, window, self._normalise_override(overrides))
            self._consecutive_wake = 0

    def _maybe_trigger_soften(self, gs: int, dgap: float) -> None:
        thr = float(self._soft_cfg.get("trigger_dgap_gt", 0.65))
        required = int(self._soft_cfg.get("trigger_consecutive", 2))
        if dgap > thr:
            self._consecutive_soften += 1
        else:
            self._consecutive_soften = 0
        if self._consecutive_soften >= required:
            window = int(self._soft_cfg.get("window_steps", 3000))
            overrides = self._soft_cfg.get("override", {})
            self._set_state("SoftenD", gs, window, self._normalise_override(overrides))
            self._consecutive_soften = 0

    def _maybe_trigger_cooldown(self, gs: int) -> None:
        if not self._cool_cfg or len(self._psnr_history) < 3:
            return
        recent_vals = [v for _, v in list(self._psnr_history)[-3:]]
        mean_drop = ((recent_vals[0] + recent_vals[1] + recent_vals[2]) / 3.0) - recent_vals[-1]
        threshold = float(self._cool_cfg.get("psnr_drop_ema_db", 0.3))
        if mean_drop >= threshold and self.state != "Cooldown":
            window = int(self._cool_cfg.get("window_steps", 500))
            overrides = self._cool_cfg.get("override", {})
            self._set_state("Cooldown", gs, window, self._normalise_override(overrides))

    def _fmt_log(self, action: str, gs: int, window: int, overrides: Dict[str, Any]) -> str:
        base = f"[FSM] {action:5s} {self.state:7s} @GS={gs}"
        if action == "enter":
            base += f" for W={window}"
            extra = []
            for key in ("lr_d", "lrD", "lr_d_target", "r1_every", "gan_cap", "update_d_every", "inst_noise"):
                if key in overrides:
                    extra.append(f"{key}={overrides[key]}")
            if extra:
                base += " | " + " ".join(extra)
        return base

    @staticmethod
    def _normalise_override(override: Dict[str, Any]) -> Dict[str, Any]:
        normalised = {}
        for key, value in (override or {}).items():
            lk = key.lower()
            if lk in {"lr_d", "lrd"}:
                normalised["lr_d"] = value
            elif lk in {"gan_cap", "gancap"}:
                normalised["gan_cap"] = value
            elif lk in {"r1_gamma", "r1gamma"}:
                normalised["r1_gamma"] = value
            elif lk in {"r1_every", "r1every"}:
                normalised["r1_every"] = int(value)
            elif lk in {"update_d_every", "updatedevery"}:
                normalised["update_d_every"] = int(value)
            elif lk == "inst_noise":
                normalised["inst_noise"] = value
            else:
                normalised[key] = value
        return normalised
