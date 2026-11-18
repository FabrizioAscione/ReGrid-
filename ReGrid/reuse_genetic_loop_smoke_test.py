import argparse
import json
import os

from OS_Loads import sf_load_as_nodal_xy_projection
from OS_constraints import apply_costr
from OS_elastic_elements import el_ele
from OS_nodes import define_nodes
from OS_run import run_elastic_static_analysis
from OS_start import init_model
from export_dxf import export_dxf
from genetic_grid_helpers import build_quarter_mirror_move_spec, describe_move_spec
from geometry import define_geometry, write_nodes_to_txt
from import_section import get_section_properties, load_section_catalog
from materials import define_material
from reuse_genetic_evaluator import (
    ReuseGeneticConstraints,
    ReuseGeneticEvaluationContext,
    ReuseGeneticObjectives,
)
from reuse_genetic_opt import (
    ReuseGeneticAlgorithmSettings,
    export_pareto_front_csv,
    export_pareto_front_json,
    plot_pareto_front,
    run_reuse_genetic_optimization,
)


def _build_path(output_dir, prefix, suffix):
    return os.path.join(output_dir, f"{prefix}_{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Small smoke test for the reuse genetic loop.")
    parser.add_argument("--geom", type=int, default=1)
    parser.add_argument("--B", type=float, default=30.0)
    parser.add_argument("--L", type=float, default=30.0)
    parser.add_argument("--f", type=float, default=None)
    parser.add_argument("--nx", type=int, default=9)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--csv-path", type=str, default="sections_1.csv")
    parser.add_argument("--reuse-csv-path", type=str, default="Database_reuse_1.csv")
    parser.add_argument("--units-in", type=str, default="cm")
    parser.add_argument("--units-out", type=str, default="m")
    parser.add_argument("--objective", dest="objectives", action="append")
    parser.add_argument("--mecc-opt", type=int, default=1, choices=(1, 2))
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", type=str, default="auto", choices=("auto", "builtin", "pymoo"))
    parser.add_argument("--multi-objective", action="store_true", help="Use multi-objective mode with all listed objectives.")
    parser.add_argument("--buckling-mode", type=int, default=1)
    parser.add_argument("--x-range", type=str, default="-0.30,+0.30")
    parser.add_argument("--y-range", type=str, default="-0.30,+0.30")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="reuse_genetic_smoke")
    args = parser.parse_args()
    objective_names = tuple(args.objectives or ["mass"])

    os.makedirs(args.output_dir, exist_ok=True)
    f = float(args.f) if args.f is not None else float(args.B) / 8.0

    G = define_geometry(args.geom, args.B, args.L, f, args.nx, args.ny, args.grid)
    move_spec = build_quarter_mirror_move_spec(G, x_move_range=args.x_range, y_move_range=args.y_range)
    print(f"# REUSE GENETIC SMOKE | {describe_move_spec(move_spec)}")

    catalog = load_section_catalog(args.csv_path, sep=";", encoding="utf-8", name_col_index=0)
    candidate_sections = get_section_properties(
        catalog,
        units_in=args.units_in,
        units_out=args.units_out,
        print_props=False,
        print_all=False,
    )

    material_el = define_material(("elastic", 355000, 210000000, 80000000), name="M1")
    q_load = (0.0, 0.0, 0.6, "symmetrical", 0.0, 0.0, 1.2, "asymmetrical", 1, 4)

    ctx = ReuseGeneticEvaluationContext(
        base_G=G,
        move_spec=move_spec,
        grid=args.grid,
        material=material_el,
        catalog=catalog,
        units_in=args.units_in,
        units_out=args.units_out,
        c1=(1, 1, 1, 0, 0, 0),
        c2=(1, 1, 1, 0, 0, 0),
        c3=(1, 0, 1, 0, 0, 0),
        c4=(1, 0, 1, 0, 0, 0),
        c5=(1, 1, 1, 0, 0, 0),
        q_load=q_load,
        ndm=3,
        ndf=6,
        init_model=init_model,
        define_nodes=define_nodes,
        apply_costr=apply_costr,
        el_ele=el_ele,
        apply_loads=sf_load_as_nodal_xy_projection,
        run_static=run_elastic_static_analysis,
        candidate_sections=candidate_sections,
        mecc_opt=args.mecc_opt,
        initial_section_name="Ø 139,7 s 14,2",
        start_from_first_section=False,
        start_section_row=40,
        opt_max_iter=20,
        opt_dcr_limit=0.90,
        opt_dlim=0.15,
        opt_check_disp=True,
        opt_check_buckling=True,
        opt_buckling_lambda_min=1.0,
        opt_buckling_retries=4,
        opt_buckling_group_limit=None,
        gamma_M0=1.05,
        gamma_M1=1.05,
        gamma_M2=1.25,
        beta=1.0,
        alpha_LT=0.76,
        compression_sign="negative",
        subdivide=1,
        area_mode="faces",
        gamma_steel=78.5,
        ecf_new=0.0,
        ecf_reused=0.0,
        ecf_work=0.0,
        ecf_scrap=0.0,
        cost_new=0.0,
        cost_reused=0.0,
        cost_work=0.0,
        cost_scrap=0.0,
        reuse_enabled=True,
        reuse_csv_path=args.reuse_csv_path,
        reuse_length_units_in="m",
        reuse_max_cut_length=10.0,
        reuse_min_remainder_length=0.5,
        reuse_group_sizes=(1, 2, 4),
        constraints=ReuseGeneticConstraints(
            dcr_limit=1.0,
            disp_limit=0.15,
            buckling_lambda_min=1.0,
            buckling_mode_index=args.buckling_mode,
        ),
        objectives=ReuseGeneticObjectives(
            names=objective_names,
            multi_objective=bool(args.multi_objective or len(objective_names) > 1),
        ),
    )

    settings = ReuseGeneticAlgorithmSettings(
        population_size=args.population,
        n_generations=args.generations,
        seed=args.seed,
        multi_objective=bool(args.multi_objective or len(objective_names) > 1),
        engine=args.engine,
    )

    result = run_reuse_genetic_optimization(ctx, settings)
    best = result.best
    if best is None:
        raise RuntimeError("No best solution was produced by the genetic smoke test.")

    history_path = _build_path(args.output_dir, args.prefix, "history.json")
    with open(history_path, "w", encoding="utf-8") as fobj:
        json.dump(result.history, fobj, indent=2)

    if best.moved_G is not None and best.moved_M is not None:
        write_nodes_to_txt(best.moved_G, _build_path(args.output_dir, args.prefix, "best_nodes.txt"))
        export_dxf(best.moved_G, best.moved_M, path=_build_path(args.output_dir, args.prefix, "best_mesh.dxf"), print_summary=False)

    summary = {
        "method": result.method,
        "objective_names": list(result.objective_names),
        "n_evaluations": result.n_evaluations,
        "cache_hits": result.cache_hits,
        "feasible": best.feasible,
        "mass": best.objectives["mass"],
        "cost": best.objectives.get("cost", None),
        "reuse_ratio": best.objectives["reuse_ratio"],
        "lambda_ref": best.lambda_ref,
        "constraints": best.constraints,
        "meta": best.meta,
        "vector": list(best.x),
        "pareto_front_size": len(result.pareto_front),
        "pareto_front_global_size": len(result.pareto_front_global),
    }
    summary_path = _build_path(args.output_dir, args.prefix, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fobj:
        json.dump(summary, fobj, indent=2)
    pareto_json = _build_path(args.output_dir, args.prefix, "pareto_front.json")
    pareto_csv = _build_path(args.output_dir, args.prefix, "pareto_front.csv")
    pareto_global_json = _build_path(args.output_dir, args.prefix, "pareto_front_global.json")
    pareto_global_csv = _build_path(args.output_dir, args.prefix, "pareto_front_global.csv")
    pareto_png = _build_path(args.output_dir, args.prefix, "pareto_front.png")
    export_pareto_front_json(pareto_json, result.pareto_front, result.objective_names)
    export_pareto_front_csv(pareto_csv, result.pareto_front, result.objective_names)
    export_pareto_front_json(pareto_global_json, result.pareto_front_global, result.objective_names)
    export_pareto_front_csv(pareto_global_csv, result.pareto_front_global, result.objective_names)
    pareto_plot_written = plot_pareto_front(
        pareto_png,
        result.pareto_front,
        result.objective_names,
        all_evaluations=result.all_evaluations,
    )

    print(f"# REUSE GENETIC SMOKE | method={result.method} | evaluations={result.n_evaluations} | cache_hits={result.cache_hits}")
    print(
        f"# REUSE GENETIC SMOKE | feasible={best.feasible} | "
        f"mass={best.objectives['mass']:.6f} | "
        f"cost={float(best.objectives.get('cost', float('inf'))):.6f} | "
        f"reuse_ratio={best.objectives['reuse_ratio']:.6f}"
    )
    print(f"# REUSE GENETIC SMOKE | lambda_ref={best.lambda_ref} | constraints={best.constraints}")
    print(f"# REUSE GENETIC SMOKE | wrote {history_path}")
    print(f"# REUSE GENETIC SMOKE | wrote {summary_path}")
    print(f"# REUSE GENETIC SMOKE | wrote {pareto_json}")
    print(f"# REUSE GENETIC SMOKE | wrote {pareto_csv}")
    print(f"# REUSE GENETIC SMOKE | wrote {pareto_global_json}")
    print(f"# REUSE GENETIC SMOKE | wrote {pareto_global_csv}")
    if pareto_plot_written:
        print(f"# REUSE GENETIC SMOKE | wrote {pareto_png}")


if __name__ == "__main__":
    main()
