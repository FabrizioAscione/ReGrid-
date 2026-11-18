import argparse
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

from pareto_selection_gui import load_saved_pareto_front
from saved_pareto_runtime import SavedParetoRuntime


def parse_args():
    parser = argparse.ArgumentParser(description="Continue saved GA Pareto runs without opening the GUI.")
    parser.add_argument("output_dirs", nargs="+", help="Saved run output folders to continue.")
    parser.add_argument("--generations", type=int, default=10, help="Extra generations to run for each folder.")
    parser.add_argument("--unitled-path", default="unitled.py", help="Path to unitled.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be continued without launching.")
    return parser.parse_args()


def main():
    args = parse_args()
    generations = max(1, int(args.generations))

    for index, raw_dir in enumerate(args.output_dirs, start=1):
        output_dir = Path(raw_dir)
        print(f"# CONTINUE {index}/{len(args.output_dirs)} | output_dir={output_dir} | generations={generations}")
        pareto_front, all_evaluations, objective_names = load_saved_pareto_front(output_dir=str(output_dir))
        print(
            f"# LOADED | pareto={len(pareto_front)} | all={len(all_evaluations)} | "
            f"objectives={','.join(objective_names)}"
        )
        if args.dry_run:
            continue
        runtime = SavedParetoRuntime(
            output_dir=str(output_dir),
            objective_names=objective_names,
            unitled_path=args.unitled_path,
        )
        message = runtime.continue_generations(generations)
        print(f"# DONE | {message}")

    if args.dry_run:
        print("# DRY RUN OK")


if __name__ == "__main__":
    main()
