import argparse
import os
import subprocess
import sys

from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reopen the saved Pareto GUI from a previous optimization output folder."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Folder containing saved Pareto files. By default the global Pareto is used when available.",
    )
    parser.add_argument(
        "--pareto-json",
        type=str,
        default=None,
        help="Optional explicit path to a Pareto JSON. Overrides the default global-Pareto preference.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Optional explicit path to the all-candidate metrics CSV. Default: output-dir\\genetic_search\\genetic_candidate_metrics.csv",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Saved Pareto results",
        help="Window title",
    )
    parser.add_argument(
        "--unitled-path",
        type=str,
        default="unitled.py",
        help="Path to the unitled.py file used to rebuild candidates before launching nonlinear runs",
    )
    parser.add_argument(
        "--run-fixed-base",
        action="store_true",
        help="Run the fixed-base nonlinear analysis directly, without opening the GUI.",
    )
    parser.add_argument(
        "--run-all-nonlinear",
        choices=("reused", "new"),
        default=None,
        help="Run all saved Pareto nonlinear analyses headlessly, without opening the GUI.",
    )
    parser.add_argument(
        "--export-material-csvs",
        action="store_true",
        help="Export reused/new/scrap material CSVs for all saved Pareto solutions without running nonlinear analyses.",
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help="Regenerate saved Pareto and parallel-coordinate plots from the current saved results.",
    )
    parser.add_argument(
        "--material-output-dir",
        type=str,
        default=None,
        help="Optional folder for --export-material-csvs. Default: output-dir\\pareto_material_takeoffs.",
    )
    parser.add_argument(
        "--continue-generations",
        type=int,
        default=None,
        help="Run extra generations headlessly from the saved Pareto state, then exit.",
    )
    return parser.parse_args()


def _run_headless_continue(args, extra_generations: int) -> str:
    os.environ.setdefault("MPLBACKEND", "Agg")

    from saved_pareto_loader import load_saved_pareto_front
    from saved_pareto_runtime import SavedParetoRuntime

    _, _, objective_names = load_saved_pareto_front(
        output_dir=args.output_dir,
        pareto_json_path=args.pareto_json,
        metrics_csv_path=args.metrics_csv,
    )
    runtime = SavedParetoRuntime(
        output_dir=args.output_dir,
        objective_names=objective_names,
        unitled_path=args.unitled_path,
    )
    return runtime.continue_generations(int(extra_generations))


def _spawn_headless_continue(args, extra_generations: int) -> None:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--output-dir",
        str(args.output_dir),
        "--unitled-path",
        str(args.unitled_path),
        "--continue-generations",
        str(int(extra_generations)),
    ]
    if args.pareto_json:
        cmd.extend(["--pareto-json", str(args.pareto_json)])
    if args.metrics_csv:
        cmd.extend(["--metrics-csv", str(args.metrics_csv)])
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)


def _run_headless_all_nonlinear(args, solution_kind: str) -> str:
    os.environ["MPLBACKEND"] = "Agg"

    from saved_pareto_loader import load_saved_pareto_front
    from saved_pareto_runtime import SavedParetoRuntime

    pareto_front, _, objective_names = load_saved_pareto_front(
        output_dir=args.output_dir,
        pareto_json_path=args.pareto_json,
        metrics_csv_path=args.metrics_csv,
    )
    runtime = SavedParetoRuntime(
        output_dir=args.output_dir,
        objective_names=objective_names,
        unitled_path=args.unitled_path,
    )
    kind = str(solution_kind or "reused").strip().lower()
    if kind == "new":
        return runtime.on_run_reference_all(pareto_front)
    return runtime.on_run_all(pareto_front)


def _spawn_headless_all_nonlinear(args, solution_kind: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--output-dir",
        str(args.output_dir),
        "--unitled-path",
        str(args.unitled_path),
        "--run-all-nonlinear",
        str(solution_kind),
    ]
    if args.pareto_json:
        cmd.extend(["--pareto-json", str(args.pareto_json)])
    if args.metrics_csv:
        cmd.extend(["--metrics-csv", str(args.metrics_csv)])
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(cmd, env=env, creationflags=creationflags)


def _continue_from_gui(args, extra_generations: int) -> None:
    try:
        _spawn_headless_continue(args, int(extra_generations))
    except subprocess.CalledProcessError as exc:
        print(
            "# SAVED PARETO | continue_generations=failed | "
            f"returncode={exc.returncode} | reopening_previous_gui=1",
            flush=True,
        )
    except Exception as exc:
        print(
            "# SAVED PARETO | continue_generations=failed | "
            f"error={type(exc).__name__}: {exc} | reopening_previous_gui=1",
            flush=True,
        )
    else:
        print("# SAVED PARETO | continue_generations=completed | reopening_gui=1", flush=True)


def main():
    args = parse_args()
    if args.continue_generations is not None:
        message = _run_headless_continue(args, int(args.continue_generations))
        print(f"# SAVED PARETO | continue_generations=completed | message={message}")
        return
    if args.run_all_nonlinear is not None:
        message = _run_headless_all_nonlinear(args, args.run_all_nonlinear)
        print(
            "# SAVED PARETO | nonlinear_run_all=completed | "
            f"solution_kind={args.run_all_nonlinear} | message={message}"
        )
        return

    from pareto_selection_gui import (
        load_saved_pareto_front,
        save_candidate_metrics_parallel_coords_plot,
        save_pareto_fixed_base_parallel_coords_plot,
        save_pareto_metrics_parallel_coords_plot,
        select_pareto_solution_gui,
    )
    from reuse_genetic_opt import plot_pareto_front
    from saved_pareto_runtime import SavedParetoRuntime

    runtime = None
    while True:
        pareto_front, all_evaluations, objective_names = load_saved_pareto_front(
            output_dir=args.output_dir,
            pareto_json_path=args.pareto_json,
            metrics_csv_path=args.metrics_csv,
        )
        if runtime is None:
            runtime = SavedParetoRuntime(
                output_dir=args.output_dir,
                objective_names=objective_names,
                unitled_path=args.unitled_path,
            )
        if args.run_fixed_base:
            message = runtime.on_run_fixed_base()
            print(f"# SAVED PARETO | fixed_base_run=completed | message={message}")
            return
        if args.export_material_csvs:
            message = runtime.export_material_takeoffs(
                pareto_front,
                solution_kind="reused",
                output_dir=args.material_output_dir,
            )
            print(f"# SAVED PARETO | material_csv_export=completed | message={message}")
            if not args.regenerate_plots:
                return
        if args.regenerate_plots:
            pareto_plot = os.path.join(args.output_dir, "reuse_genetic_pareto_front.png")
            parallel_plot = os.path.join(args.output_dir, "reuse_genetic_parallel_coords.png")
            pareto_metrics_plot = os.path.join(args.output_dir, "reuse_genetic_parallel_coords_pareto_metrics.png")
            fixed_parallel_plot = os.path.join(args.output_dir, "reuse_genetic_parallel_coords_pareto_fixed_base.png")
            pareto_written = plot_pareto_front(
                pareto_plot,
                pareto_front,
                objective_names,
                all_evaluations=all_evaluations,
            )
            parallel_written = save_candidate_metrics_parallel_coords_plot(
                all_evaluations=all_evaluations,
                save_path=parallel_plot,
            )
            pareto_metrics_written = save_pareto_metrics_parallel_coords_plot(
                pareto_front=pareto_front,
                save_path=pareto_metrics_plot,
            )
            fixed_parallel_written = save_pareto_fixed_base_parallel_coords_plot(
                output_dir=args.output_dir,
                pareto_front=pareto_front,
                save_path=fixed_parallel_plot,
            )
            print(
                "# SAVED PARETO | plots_regenerated=1 | "
                f"pareto_plot={pareto_plot if pareto_written else 'n/a'} | "
                f"parallel={parallel_written or 'n/a'} | "
                f"pareto_metrics={pareto_metrics_written or 'n/a'} | "
                f"fixed_parallel={fixed_parallel_written or 'n/a'}"
            )
            return
        result = select_pareto_solution_gui(
            pareto_front,
            all_evaluations,
            objective_names,
            title=args.title,
            on_run_selected=runtime.on_run_selected,
            on_run_all=lambda _evaluations: (
                f"Started headless reused nonlinear runs in background "
                f"(pid={_spawn_headless_all_nonlinear(args, 'reused').pid}). "
                f"Outputs will appear under {os.path.join(args.output_dir, 'pareto_nonlinear_runs')}."
            ),
            on_run_reference_selected=runtime.on_run_reference_selected,
            on_run_reference_all=lambda _evaluations: (
                f"Started headless new-reference nonlinear runs in background "
                f"(pid={_spawn_headless_all_nonlinear(args, 'new').pid}). "
                f"Outputs will appear under {os.path.join(args.output_dir, 'pareto_nonlinear_runs')}."
            ),
            on_run_fixed_base=runtime.on_run_fixed_base,
            save_pareto_only_path=os.path.join(args.output_dir, "reuse_genetic_pareto_only.png"),
            save_parallel_coords_path=os.path.join(args.output_dir, "reuse_genetic_parallel_coords.png"),
        )
        if result is not None and result.continue_requested:
            _continue_from_gui(args, int(result.continue_generations or 1))
            runtime = None
            continue
        if result is not None and result.run_all_executed:
            run_kind = str(getattr(result, "run_all_solution_kind", "reused") or "reused").strip().lower()
            if run_kind == "new":
                message = runtime.on_run_reference_all(pareto_front)
            else:
                message = runtime.on_run_all(pareto_front)
            print(f"# SAVED PARETO | nonlinear_run_all=completed | message={message}")
            continue
        if result is not None and result.evaluation is not None and result.index is not None:
            print(
                f"# SAVED PARETO | selected=1 | pareto_rank={int(result.index) + 1} | "
                f"candidate_id={result.evaluation.meta.get('candidate_id', 'n/a')}"
            )
        return


if __name__ == "__main__":
    main()
