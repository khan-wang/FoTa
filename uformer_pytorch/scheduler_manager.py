from typing import Any, Dict, List, Optional, Tuple


def _parse_step_value(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        mult = 1
        if value.endswith('k'):
            mult = 1000
            value = value[:-1]
        elif value.endswith('m'):
            mult = 1_000_000
            value = value[:-1]
        try:
            return int(float(value) * mult)
        except ValueError as exc:
            raise ValueError(f"Invalid step specification: {value}") from exc
    raise ValueError(f"Unsupported step specification type: {type(value)}")


def _interp_schedule(step: int, schedule: List[Dict[str, Any]], default: float) -> float:
    if not schedule:
        return float(default)
    for segment in schedule:
        start = _parse_step_value(segment.get("start_step", 0))
        end = _parse_step_value(segment.get("end_step", start))
        if start <= step < end:
            span = max(1, end - start)
            progress = (step - start) / span
            start_val = float(segment.get("start_val", default))
            end_val = float(segment.get("end_val", start_val))
            return start_val + (end_val - start_val) * progress
    last_segment = schedule[-1]
    return float(last_segment.get("end_val", default))


def _schedule_pairs(step: int, schedule: List[Any], default: int) -> int:
    if not schedule:
        return int(default)
    value = int(default)
    for entry in schedule:
        if isinstance(entry, dict):
            start = _parse_step_value(entry.get("step", entry.get("start", 0)))
            freq = entry.get("value", entry.get("every", entry.get("freq", entry.get("frequency", None))))
        else:
            try:
                start, freq = entry
            except (TypeError, ValueError):
                continue
        start = _parse_step_value(start)
        if step >= start:
            value = int(freq)
        else:
            break
    return int(value)


class SchedulerManager:
    """Centralised accessor for all scheduled training values."""

    _SCHEDULE_SPECS: Tuple[Tuple[str, str, str], ...] = (
        ("TV_INHOLE_WEIGHT_SCHEDULE", "tv_inhole_weight", "TV_INHOLE_WEIGHT"),
        ("ROI_RAMP_SCHEDULE", "roi_scale", "ROI_SCALE_BASE"),
        ("ROI_ADV_WEIGHT_SCHEDULE", "roi_adv_weight", "ROI_ADV_WEIGHT"),
        ("GATE_TEMP_SCHEDULE", "gate_temp", "GATE_TEMP"),
        ("SPECTRAL_DROPOUT_SCHEDULE", "spectral_dropout_rate", "SPECTRAL_DROPOUT_RATE"),
        ("L1_HOLE_WEIGHT_SCHEDULE", "l1_hole_weight", "L1_HOLE_WEIGHT"),
        ("GAN_CAP_SCHEDULE", "gan_cap", "GAN_CAP"),
        ("SEAM_WEIGHT_SCHEDULE", "seam_weight", "SEAM_LOSS_WEIGHT"),
        ("PERC_WEIGHT_SCHEDULE", "perc_weight", "PERC_WEIGHT"),
        ("FM_WEIGHT_SCHEDULE", "fm_weight", "FEATURE_MATCH_WEIGHT"),
        ("LPIPS_WEIGHT_SCHEDULE", "lpips_weight", "LPIPS_WEIGHT"),
        ("EDGE_WEIGHT_SCHEDULE", "edge_weight", "EDGE_WEIGHT"),
        ("SPECTRAL_L1_WEIGHT_SCHEDULE", "spectral_l1_weight", "SPECTRAL_L1_WEIGHT"),
    )


    _PHASE_KEY_MAP: Dict[str, str] = {

        "update_d_every": "update_d_every",
        "r1_every": "r1_every",
        "r1_gamma": "r1_gamma",
        "lr_g": "lr_g_target",
        "lrG": "lr_g_target",
        "lr_d": "lr_d_target",
        "lrD": "lr_d_target",


        "gan_cap": "gan_cap",
        "ganCap": "gan_cap",
        "adversarial_weight": "adversarial_weight",
        "roi_scale": "roi_scale",
        "roi_adv_weight": "roi_adv_weight",
        "inst_noise": "inst_noise",


        "l1_hole": "l1_hole_weight",
        "l1_hole_weight": "l1_hole_weight",
        "tv_inhole": "tv_inhole_weight",
        "tv_inhole_weight": "tv_inhole_weight",
        "seam_weight": "seam_weight",
        "edge_weight": "edge_weight",


        "perc_weight": "perc_weight",
        "perc_in_w": "perc_in_weight",
        "perc_out_w": "perc_out_weight",
        "style_in_w": "style_in_weight",
        "style_out_w": "style_out_weight",
        "fm_weight": "fm_weight",
        "lpips_weight": "lpips_weight",


        "spec_l1_weight": "spectral_l1_weight",
        "spectral_l1_weight": "spectral_l1_weight",
        "spectral_dropout_rate": "spectral_dropout_rate",


        "gate_temp": "gate_temp",
        "gate_prior_weight": "gate_prior_weight",


        "roi": "roi",
        "freq": "freq",
        "ring": "ring",
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self._current_step: Optional[int] = None
        self._current: Dict[str, Any] = {}
        self._phase_name: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._phases = self._parse_phases(getattr(cfg, "PHASES", None))
        self._defaults = {

            "update_d_every": int(getattr(cfg, "UPDATE_D_EVERY", 1)),
            "r1_every": int(getattr(cfg, "R1_EVERY_STEPS", 16)),
            "r1_gamma": float(getattr(cfg, "R1_GAMMA", 6.0)),
            "lr_g_target": float(getattr(cfg, "LEARNING_RATE_G", 2e-4)),
            "lr_d_target": float(getattr(cfg, "LEARNING_RATE_D", 1e-4)),


            "gan_cap": float(getattr(cfg, "GAN_CAP", 0.3)),
            "adversarial_weight": float(getattr(cfg, "ADVERSARIAL_WEIGHT", 0.0)),
            "roi_scale": float(getattr(cfg, "ROI_SCALE_BASE", 0.0)),
            "roi_adv_weight": float(getattr(cfg, "ROI_ADV_WEIGHT", 0.0)),
            "inst_noise": None,


            "l1_hole_weight": float(getattr(cfg, "L1_HOLE_WEIGHT", 0.0)),
            "tv_inhole_weight": float(getattr(cfg, "TV_INHOLE_WEIGHT", 0.0)),
            "seam_weight": float(getattr(cfg, "SEAM_LOSS_WEIGHT", 0.0)),
            "edge_weight": float(getattr(cfg, "EDGE_WEIGHT", 0.0)),


            "perc_weight": float(getattr(cfg, "PERC_WEIGHT", 0.0)),
            "perc_in_weight": float(getattr(cfg, "PERC_WEIGHT", 0.0)),
            "perc_out_weight": float(getattr(cfg, "PERC_WEIGHT", 0.0)),
            "style_in_weight": float(getattr(cfg, "STYLE_WEIGHT", 0.0)),
            "style_out_weight": float(getattr(cfg, "STYLE_WEIGHT", 0.0)),
            "fm_weight": float(getattr(cfg, "FEATURE_MATCH_WEIGHT", 0.0)),
            "lpips_weight": float(getattr(cfg, "LPIPS_WEIGHT", 0.0)),


            "spectral_l1_weight": float(getattr(cfg, "SPECTRAL_L1_WEIGHT", 0.0)),
            "spectral_dropout_rate": float(getattr(cfg, "SPECTRAL_DROPOUT_RATE", 0.0)),


            "gate_temp": float(getattr(cfg, "GATE_TEMP", 1.0)),
            "gate_prior_weight": float(getattr(cfg, "GATE_PRIOR_WEIGHT", 0.0)),


            "roi": None,
            "freq": None,
            "ring": None,
        }

    def _parse_phases(self, phases_raw: Optional[Any]) -> List[Dict[str, Any]]:
        if not phases_raw:
            return []
        parsed: List[Dict[str, Any]] = []
        for entry in phases_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", f"phase_{len(parsed)}")
            start = _parse_step_value(entry.get("start", entry.get("step", 0)))
            dur = entry.get("dur")
            if dur is None:
                raise ValueError(f"Phase '{name}' missing 'dur'.")
            duration = _parse_step_value(dur)
            end = start + duration
            payload = {k: v for k, v in entry.items() if k not in {"name", "start", "step", "dur", "duration"}}
            parsed.append({"name": name, "start": start, "end": end, "payload": payload})
        parsed.sort(key=lambda x: x["start"])
        return parsed

    def _phase_values(self, step: int) -> Tuple[Dict[str, Any], Optional[str]]:
        if not self._phases:
            return {}, None
        current = None
        for phase in self._phases:
            if phase["start"] <= step < phase["end"]:
                current = phase
                break
        if current is None:
            if step >= self._phases[-1]["end"]:
                current = self._phases[-1]
            else:
                current = self._phases[0]
        payload = {}
        span = max(1, current["end"] - current["start"])
        progress = min(1.0, max(0.0, (step - current["start"]) / span))
        for key, raw_value in current["payload"].items():
            norm_key = self._PHASE_KEY_MAP.get(key, key)
            if isinstance(raw_value, dict) and {"start", "end"}.issubset(raw_value.keys()):
                start_val = raw_value.get("start")
                end_val = raw_value.get("end")
                if start_val is None or end_val is None:
                    continue
                val = float(start_val) + (float(end_val) - float(start_val)) * progress
            else:
                val = raw_value
            payload[norm_key] = val
        return payload, current["name"]

    def step(self, global_step: int) -> None:
        schedule_values = {k: v for k, v in self._defaults.items()}
        for attr, name, default_name in self._SCHEDULE_SPECS:
            schedule = getattr(self.cfg, attr, None)
            default = getattr(self.cfg, default_name, schedule_values.get(name, 0.0))
            if schedule is not None:
                schedule_values[name] = _interp_schedule(global_step, schedule, default)
        schedule_values["update_d_every"] = _schedule_pairs(
            global_step,
            getattr(self.cfg, "UPDATE_D_EVERY_SCHEDULE", []) or [],
            schedule_values.get("update_d_every", 1),
        )
        schedule_values["r1_every"] = _schedule_pairs(
            global_step,
            getattr(self.cfg, "R1_EVERY_STEPS_SCHEDULE", []) or [],
            schedule_values.get("r1_every", self._defaults["r1_every"]),
        )
        schedule_values["r1_gamma"] = float(
            getattr(self.cfg, "R1_GAMMA", schedule_values.get("r1_gamma", 6.0))
        )
        schedule_values["lr_d_target"] = float(
            getattr(self.cfg, "LEARNING_RATE_D", schedule_values.get("lr_d_target", 1e-4))
        )

        phase_values, phase_name = self._phase_values(global_step)


        if phase_values:
            schedule_values.update(phase_values)

            for phase in self._phases:
                if phase.get("name") == phase_name:
                    schedule_values["_phase_start_step"] = phase.get("start", -1)

                    if "reset_d_optimizer" in phase.get("payload", {}):
                        schedule_values["reset_d_optimizer"] = phase["payload"]["reset_d_optimizer"]
                    break


        self._current = schedule_values
        ctx_snapshot = dict(schedule_values)
        if phase_name is not None:
            ctx_snapshot.setdefault("ctx_phase_name", phase_name)
        self._context = ctx_snapshot
        self._current_step = global_step
        self._phase_name = phase_name

    def _require_step(self) -> None:
        if self._current_step is None:
            raise RuntimeError("SchedulerManager.step() must be called before accessing values.")

    def value(self, name: str, default: Any = None) -> Any:
        self._require_step()
        return self._current.get(name, default)

    def current_gan_cap(self) -> float:
        return float(self.value("gan_cap", self._defaults.get("gan_cap", 0.3)))

    def update_d_every(self) -> int:
        return int(self.value("update_d_every", self._defaults.get("update_d_every", 1)))

    def r1_every(self) -> int:
        return int(self.value("r1_every", self._defaults.get("r1_every", 16)))

    def r1_gamma(self) -> float:
        return float(self.value("r1_gamma", self._defaults.get("r1_gamma", 6.0)))

    def lr_d_target(self) -> float:
        return float(self.value("lr_d_target", self._defaults.get("lr_d_target", 1e-4)))

    def current_phase(self) -> Optional[str]:
        return self._phase_name

    def context(self) -> Dict[str, Any]:
        self._require_step()
        return dict(self._context)
