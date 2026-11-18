import argparse
import csv
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pareto_selection_gui import _candidate_from_metrics_row
from reuse_genetic_opt import _extract_pareto_front, export_pareto_front_csv, export_pareto_front_json, plot_pareto_front


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover a Pareto front from a partial genetic_candidate_metrics.csv file."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Run output folder. Default: outputs",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Metrics CSV path. Default: output-dir\\genetic_search\\genetic_candidate_metrics.csv",
    )
    parser.add_argument(
        "--objectives",
        default="pi_mass,pi_ec,n_cuts",
        help="Comma-separated objective columns used by the GA. Default: pi_mass,pi_ec,n_cuts",
    )
    parser.add_argument(
        "--prefix",
        default="reuse_genetic_pareto_front_recovered",
        help="Output filename prefix inside output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    metrics_csv = args.metrics_csv or os.path.join(output_dir, "genetic_search", "genetic_candidate_metrics.csv")
    objective_names = tuple(part.strip() for part in str(args.objectives).split(",") if part.strip())
    if len(objective_names) not in {2, 3}:
        raise ValueError("Use exactly 2 or 3 objectives, for example: pi_mass,pi_ec,n_cuts")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(metrics_csv)

    with open(metrics_csv, "r", encoding="utf-8", newline="") as fobj:
        rows = list(csv.DictReader(fobj))
    evaluations = [_candidate_from_metrics_row(row, objective_names) for row in rows]
    evaluations = [evaluation for evaluation in evaluations if getattr(evaluation, "x", None)]
    pareto_front = _extract_pareto_front(evaluations, objective_names)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{args.prefix}.json")
    csv_path = os.path.join(output_dir, f"{args.prefix}.csv")
    plot_path = os.path.join(output_dir, f"{args.prefix}.png")
    summary_path = os.path.join(output_dir, f"{args.prefix}_summary.json")

    export_pareto_front_json(json_path, pareto_front, objective_names)
    export_pareto_front_csv(csv_path, pareto_front, objective_names)
    plot_written = False
    plot_error = None
    try:
        plot_written = bool(plot_pareto_front(plot_path, pareto_front, objective_names, all_evaluations=evaluations))
    except Exception as exc:
        plot_error = str(exc)
    with open(summary_path, "w", encoding="utf-8") as fobj:
        json.dump(
            {
                "metrics_csv": metrics_csv,
                "objective_names": list(objective_names),
                "n_evaluations": len(evaluations),
                "pareto_front_size": len(pareto_front),
                "json_path": json_path,
                "csv_path": csv_path,
                "plot_path": plot_path if plot_written else None,
                "plot_error": plot_error,
            },
            fobj,
            indent=2,
        )
    print(
        f"# RECOVERED | evaluations={len(evaluations)} | pareto={len(pareto_front)} | "
        f"json={json_path} | csv={csv_path}"
    )


if __name__ == "__main__":
    main()
