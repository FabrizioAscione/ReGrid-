import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT_DIR / "ga_batch_plan_night.json"


def _safe_label(raw_label, index):
    label = str(raw_label or f"run_{index:02d}").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return safe or f"run_{index:02d}"


def _load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if not isinstance(plan, dict):
        raise ValueError("Batch plan must be a JSON object.")
    runs = plan.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Batch plan must contain a non-empty 'runs' list.")
    return plan


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_summary(run_dir):
    summary_path = run_dir / "reuse_genetic_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"summary_error": str(exc)}


def _status_is_success(status_path):
    if not status_path.exists():
        return False
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        return int(status.get("returncode", -1)) == 0
    except Exception:
        return False


def _run_one(index, total, run, defaults, output_root, args):
    label = _safe_label(run.get("label"), index)
    run_dir = output_root / f"{index:02d}_{label}"
    config_path = run_dir / "batch_config.json"
    log_path = run_dir / "run_stdout.log"
    status_path = run_dir / "run_status.json"

    if args.resume and _status_is_success(status_path):
        print(f"[{index}/{total}] SKIP {label} | already completed")
        return {
            "label": label,
            "run_dir": str(run_dir),
            "returncode": 0,
            "skipped": True,
            "summary": _read_summary(run_dir),
        }

    config = dict(defaults)
    overrides = dict(run)
    overrides.pop("label", None)
    config.update(overrides)

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)

    cmd = [
        sys.executable,
        "unitled.py",
        "--batch-config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    started = time.time()
    print(f"[{index}/{total}] START {label}")
    print(f"[{index}/{total}] OUT   {run_dir}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            print(text)
            log.write(text + "\n")
            log.flush()
        returncode = proc.wait()

    elapsed_s = time.time() - started
    summary = _read_summary(run_dir)
    status = {
        "label": label,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "returncode": int(returncode),
        "elapsed_s": elapsed_s,
        "summary": summary,
    }
    _write_json(status_path, status)

    result = "OK" if returncode == 0 else f"FAIL code={returncode}"
    print(f"[{index}/{total}] {result} {label} | elapsed={elapsed_s / 60.0:.1f} min")
    return status


def main():
    parser = argparse.ArgumentParser(description="Run unitled.py GA experiments sequentially.")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="JSON plan path.")
    parser.add_argument("--resume", action="store_true", help="Skip runs already completed successfully.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the run plan without launching analyses.")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = _load_plan(plan_path)
    defaults = plan.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a JSON object.")

    output_root = Path(plan.get("output_root", "outputs/night_ga_batch"))
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    runs = plan["runs"]
    batch_started = time.time()
    batch_results = []

    print(f"# GA BATCH START | plan={plan_path}")
    print(f"# OUTPUT ROOT    | {output_root}")

    if args.dry_run:
        print("# DRY RUN        | no analyses will be launched")
        for index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                raise ValueError(f"Run {index} must be a JSON object.")
            label = _safe_label(run.get("label"), index)
            config = dict(defaults)
            overrides = dict(run)
            overrides.pop("label", None)
            config.update(overrides)
            run_dir = output_root / f"{index:02d}_{label}"
            objectives = [
                config.get("genetic_obj_1"),
                config.get("genetic_obj_2"),
                config.get("genetic_obj_3"),
            ]
            objectives = [str(obj) for obj in objectives if obj is not None and str(obj).strip()]
            print(
                f"[{index:02d}/{len(runs):02d}] {label} | "
                f"obj={objectives} | ref={config.get('genetic_reference_mode')} | "
                f"gen={config.get('genetic_generations')} | "
                f"range_x={config.get('genetic_move_range_x')} | "
                f"range_y={config.get('genetic_move_range_y')} | "
                f"out={run_dir}"
            )
        print("# DRY RUN OK")
        return

    try:
        for index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                raise ValueError(f"Run {index} must be a JSON object.")
            batch_results.append(_run_one(index, len(runs), run, defaults, output_root, args))
    except KeyboardInterrupt:
        print("# GA BATCH INTERRUPTED")
        batch_results.append({"interrupted": True})

    batch_status = {
        "plan_path": str(plan_path),
        "output_root": str(output_root),
        "elapsed_s": time.time() - batch_started,
        "results": batch_results,
    }
    _write_json(output_root / "batch_status.json", batch_status)
    print(f"# GA BATCH END | elapsed={batch_status['elapsed_s'] / 3600.0:.2f} h")
    print(f"# STATUS       | {output_root / 'batch_status.json'}")


if __name__ == "__main__":
    main()
