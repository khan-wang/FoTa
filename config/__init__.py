import yaml
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from config_lint import validate_config


def _deep_merge(a: dict, b: dict) -> dict:
    """
    Recursively merges dict `b` into `a`. Modifies `a` in place.
    """
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def _expand_env_vars(value):
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


DEFAULTS = {

    'D_ROI_SIZE': 128,
    'EVAL_MICROBATCH': 16,
    'VIZ_UPSCALE': 1,
    'VIZ_PER_ROW': 4,
    'VIZ_SAVE_INDIVIDUAL': False,
    'D_LR_FLOOR': 0.0,
    'SEAM_KERNEL_SIZE': 33,


    'FINETUNE_LR_MULT': 1.0,
    'FREEZE_ENCODER_STEPS': 0,
    'RECOVERY_STEPS': 0,
    'INIT_FROM_G_EMA': None,
    'SCHEDULE_ABSOLUTE': True,
    'FINETUNE_MODE': False,
    'RESET_OPT_AND_SCHED': False,
    'OVERRIDE_LR_ON_RESUME': False,
    'MIN_EXPECTED_LR_D': 5.0e-5,
    'START_GLOBAL_STEP': 0,


    'USE_MASK_IN_D': False,
    'DISC_INPUT_NC': 3,


    'ALLOW_SOFT_STOP_WITH_ROI_ZERO': False,
}
def load_config(run_mode: str = 'public', config_path: str = None) -> SimpleNamespace:
    """
    Loads, merges, and resolves configurations from YAML files.

    Priority:
    1. Command-line specified config_path (if provided)
    2. Environment-specific config (based on run_mode)
    3. default.yaml

    Args:
        run_mode (str): Name of an optional file under ``config/env``.
                        ``auto`` maps to the portable ``public`` configuration.
        config_path (str, optional): Direct path to a YAML config file.

    Returns:
        SimpleNamespace: A nested namespace object containing the final configuration.
    """
    if run_mode == 'auto':
        run_mode = 'public'

    # Base path for config files is the directory of this __init__.py file
    base_path = Path(__file__).parent.resolve()

    # --- Load configurations ---
    # 1. Start from hard-coded defaults and layer config/default.yaml on top
    config_dict = deepcopy(DEFAULTS)
    default_config_path = base_path / 'default.yaml'
    if default_config_path.exists():
        with open(default_config_path, 'r', encoding='utf-8') as f:
            default_yaml = yaml.safe_load(f) or {}
            _deep_merge(config_dict, default_yaml)
    else:
        legacy_default = base_path / 'env/default.yaml'
        if legacy_default.exists():
            with open(legacy_default, 'r', encoding='utf-8') as f:
                default_yaml = yaml.safe_load(f) or {}
                _deep_merge(config_dict, default_yaml)
        else:
            print("WARNING: No default configuration file found. Starting from in-code defaults.")

    # 2. Load and merge environment-specific config
    env_config_file = base_path / f'env/{run_mode}.yaml'
    if env_config_file.exists():
        with open(env_config_file, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f) or {}
            _deep_merge(config_dict, env_config)
    else:
        fallback_env = base_path / 'env/default.yaml'
        if fallback_env.exists():
            print(f"WARNING: Environment config file not found at {env_config_file}. Falling back to env/default.yaml")
            with open(fallback_env, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f) or {}
                _deep_merge(config_dict, env_config)
        else:
            print(f"WARNING: Environment config file not found at {env_config_file}. Using defaults.")

    # 3. Load and merge user-specified config files (highest priority).
    # A comma-separated list is applied from left to right.
    if config_path:
        config_paths = [
            Path(item.strip())
            for item in str(config_path).split(",")
            if item.strip()
        ]
        loaded_paths = []
        for user_config_path in config_paths:
            if user_config_path.exists():
                with open(user_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    _deep_merge(config_dict, user_config)
                loaded_paths.append(str(user_config_path))
                print(f"INFO: Loaded custom config from {user_config_path}")
            else:
                print(f"WARNING: Custom config file not found at {user_config_path}. Ignoring.")
        config_dict['_CONFIG_PATH'] = ",".join(loaded_paths)

    config_dict = _expand_env_vars(config_dict)

    # Derive backward-compatible defaults.
    config_dict.setdefault('EVAL_MICROBATCH', config_dict.get('BATCH_SIZE', DEFAULTS['EVAL_MICROBATCH']))
    config_dict.setdefault('SEAM_KERNEL_SIZE', config_dict.get('BORDER_KERNEL', DEFAULTS['SEAM_KERNEL_SIZE']))
    # Convert the legacy ROI ramp fields when needed.
    start_step = config_dict.get('ROI_START_STEP')
    ramp_steps = config_dict.get('ROI_RAMP_STEPS')
    if start_step is not None and ramp_steps is not None and not config_dict.get("ROI_RAMP_SCHEDULE"):
        print(
            f"INFO: Converting legacy ROI_START_STEP({start_step})/ROI_RAMP_STEPS({ramp_steps}) to ROI_RAMP_SCHEDULE.")
        config_dict["ROI_RAMP_SCHEDULE"] = [
            {"start_step": start_step, "end_step": start_step + ramp_steps, "start_val": 0.0, "end_val": 1.0}
        ]

    # Keep discriminator input channels consistent with mask conditioning.
    use_mask_in_d = bool(config_dict.get('USE_MASK_IN_D', False))
    img_channels = int(config_dict.get('IMG_CHANNEL', 3))
    expected_disc_nc = img_channels + (1 if use_mask_in_d else 0)
    configured_disc_nc = int(config_dict.get('DISC_INPUT_NC', expected_disc_nc))
    if configured_disc_nc != expected_disc_nc:
        requirement = expected_disc_nc
        prefix = "When USE_MASK_IN_D=True" if use_mask_in_d else "When USE_MASK_IN_D=False"
        raise AssertionError(f"{prefix}, DISC_INPUT_NC must be {requirement}.")
    config_dict['DISC_INPUT_NC'] = expected_disc_nc

    # Promote nested EVAL_* keys for backward compatibility.
    _e = config_dict.get('EVAL', {}) or {}
    for k, v in _e.items():
        if k.startswith('EVAL_') and k not in config_dict:
            config_dict[k] = v
    # Convert dictionary to SimpleNamespace for attribute-style access
    config = SimpleNamespace(**config_dict)

    # --- Resolve path strings to Path objects ---
    validate_config(config)

    # This makes path handling OS-agnostic
    for key, value in vars(config).items():
        if isinstance(value, str) and (
                'DIR' in key or 'ROOT' in key or 'PATH' in key or 'FLIST' in key or 'CHECKPOINT' in key):
            setattr(config, key, Path(value))

    # Calculate DISC_INPUT_NC based on USE_MASK_IN_D
    config.DISC_INPUT_NC = config.IMG_CHANNEL + 1 if config.USE_MASK_IN_D else config.IMG_CHANNEL

    if not config.USE_MASK_IN_D:
        expected_nc_runtime = config.IMG_CHANNEL
        assert int(getattr(config, "DISC_INPUT_NC", expected_nc_runtime)) == expected_nc_runtime, (
            f"When USE_MASK_IN_D=False, DISC_INPUT_NC must be {expected_nc_runtime}.")
    else:
        expected_nc_runtime = config.IMG_CHANNEL + 1
        assert int(getattr(config, "DISC_INPUT_NC", expected_nc_runtime)) == expected_nc_runtime, (
            f"When USE_MASK_IN_D=True, DISC_INPUT_NC must be {expected_nc_runtime}.")

    return config


if __name__ == '__main__':
    public_config = load_config(run_mode='public')
    print(f"Run mode: {public_config.RUN_MODE}")
    print(f"Data root: {public_config.RAW_DATA_ROOT}")
