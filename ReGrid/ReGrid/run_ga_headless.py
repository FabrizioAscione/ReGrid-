"""Run/continue the genetic optimization without opening the Tkinter GUI.

Uso tipico da terminale, nella cartella del progetto:

    python run_ga_headless.py --extra-generations 5 --workers 8

Lo script crea un batch-config temporaneo e lancia unitled.py in batch mode.
Alla fine puoi aprire la GUI solo per guardare/selezionare i risultati.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _build_config(args: argparse.Namespace, project_dir: Path) -> Path:
    base = _load_json(Path(args.base_config).resolve()) if args.base_config else {}
    output_dir = Path(args.output_dir).as_posix()

    overrides = dict(base)
    overrides.update(
        {
            "reuse_genetic_opt": True,
            "genetic_resume_from_metrics": True,
            "genetic_headless_exit_after_ga": True,
            "genetic_select_pareto_for_nl": False,
            "genetic_generations": int(args.extra_generations),
            "genetic_n_workers": int(args.workers),
            "genetic_terminal_progress_every": int(args.progress_every),
            "genetic_candidate_stall_timeout_s": float(args.stall_timeout),
            "genetic_suppress_evaluation_output": True,
            "genetic_verbose": int(args.verbose),
            "output_dir": output_dir,
        }
    )
    if args.population is not None:
        overrides["genetic_population"] = int(args.population)
    if args.seed is not None:
        overrides["genetic_seed"] = int(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir = project_dir / "outputs" / "headless_configs"
    config_path = config_dir / f"ga_headless_continue_{stamp}.json"
    _write_json(config_path, overrides)
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue GA without GUI")
    parser.add_argument("--extra-generations", type=int, default=5, help="Generazioni aggiuntive per blocco")
    parser.add_argument("--workers", type=int, default=8, help="Numero worker paralleli")
    parser.add_argument("--blocks", type=int, default=1, help="Quanti blocchi consecutivi lanciare")
    parser.add_argument("--progress-every", type=int, default=100, help="Stampa progresso ogni N candidati")
    parser.add_argument("--stall-timeout", type=float, default=1200.0, help="Secondi senza candidati completati prima di skip/gestione stallo")
    parser.add_argument("--population", type=int, default=None, help="Override popolazione GA")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Cartella output")
    parser.add_argument("--base-config", type=str, default=None, help="JSON con override di base da riusare")
    parser.add_argument("--verbose", type=int, default=1, help="0=silent, 1=progress")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    unitled_path = project_dir / "unitled.py"
    if not unitled_path.exists():
        raise FileNotFoundError(f"unitled.py not found in {project_dir}")

    for block_idx in range(1, int(args.blocks) + 1):
        config_path = _build_config(args, project_dir)
        print(
            f"# HEADLESS GA | block={block_idx}/{args.blocks} | "
            f"extra_generations={args.extra_generations} | workers={args.workers} | "
            f"config={config_path}",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(unitled_path),
            "--batch-config",
            str(config_path),
            "--output-dir",
            str(args.output_dir),
        ]
        completed = subprocess.run(cmd, cwd=str(project_dir))
        if completed.returncode != 0:
            print(f"# HEADLESS GA | block={block_idx} failed | returncode={completed.returncode}", flush=True)
            return int(completed.returncode)
        print(f"# HEADLESS GA | block={block_idx}/{args.blocks} completed", flush=True)

    print("# HEADLESS GA | all requested blocks completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
