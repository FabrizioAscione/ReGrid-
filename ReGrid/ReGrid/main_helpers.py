import argparse
import json
import os


def build_output_path(base_dir, *parts):
    return os.path.join(base_dir, *parts)


def fiber_family_spec(spec, *, name):
    if not isinstance(spec, (list, tuple)) or len(spec) != 5:
        raise ValueError(
            f"{name} must be a list/tuple like [n_web_y, n_web_z, n_flange_y, n_flange_z, unused_rotation_deg]."
        )
    return {
        "n_web_y": int(spec[0]),
        "n_web_z": int(spec[1]),
        "n_flange_y": int(spec[2]),
        "n_flange_z": int(spec[3]),
        # Rotation is intentionally disabled across elastic and nonlinear models.
        "rotation_deg": 0.0,
    }


def build_fiber_section_opts(n_circ, n_rad, fiber_L, fiber_2L, fiber_4L):
    return {
        "n_circ": int(n_circ),
        "n_rad": int(n_rad),
        "L": fiber_family_spec(fiber_L, name="fiber_L"),
        "2L": fiber_family_spec(fiber_2L, name="fiber_2L"),
        "4L": fiber_family_spec(fiber_4L, name="fiber_4L"),
        "use_GJ": True,
    }


def parse_batch_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def load_batch_overrides(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Batch config must contain a JSON object.")
    return data


def load_batch_runtime(argv=None):
    cli_args = parse_batch_args(argv)
    batch_overrides = load_batch_overrides(cli_args.batch_config)
    batch_mode = bool(cli_args.batch_config)
    return cli_args, batch_overrides, batch_mode
