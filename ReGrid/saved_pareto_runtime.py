from __future__ import annotations

import json
import os
import shutil
import textwrap
from dataclasses import replace
from typing import Optional, Sequence

from pareto_selection_gui import (
    save_candidate_metrics_parallel_coords_plot,
    save_pareto_fixed_base_parallel_coords_plot,
    save_pareto_metrics_parallel_coords_plot,
)
from saved_pareto_loader import load_saved_pareto_front
from reuse_genetic_evaluator import (
    ReuseGeneticConstraints,
    ReuseGeneticDecisionControls,
    ReuseGeneticEvaluationContext,
    ReuseGeneticObjectives,
    compute_fixed_reference_design,
    evaluate_candidate_design,
)
from reuse_genetic_opt import (
    ReuseGeneticAlgorithmSettings,
    build_resume_settings,
    export_candidate_metrics_csv,
    export_pareto_front_csv,
    export_pareto_front_json,
    plot_pareto_front,
    run_reuse_genetic_optimization,
)


def _slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return textwrap.dedent(text[start:end])


def _load_unitled_namespace(unitled_path: str) -> dict:
    # Accept UTF-8 files saved with BOM by stripping it during decoding.
    with open(unitled_path, "r", encoding="utf-8-sig") as fobj:
        source = fobj.read()

    main_marker = "# =================================================================MAINCODE"
    prefix_code = source[: source.index(main_marker)]

    geometry_code = _slice_between(
        source,
        'G = define_geometry(',
        "#=====================================================================MESH",
    )
    mesh_code = _slice_between(
        source,
        "M = define_mesh(G, grid)",
        "#=============================================================EXPORT GEOMETRICAL MODEL",
    )
    catalog_material_code = _slice_between(
        source,
        'catalog = load_section_catalog(',
        "def _candidate_run_folder_name",
    )
    helper_code = _slice_between(
        source,
        "def _candidate_run_folder_name",
        "genetic_result = None",
    )

    namespace = {"__name__": "_saved_pareto_runtime"}
    exec(compile(prefix_code, unitled_path, "exec"), namespace)
    exec(compile(geometry_code, unitled_path, "exec"), namespace)
    exec(compile(mesh_code, unitled_path, "exec"), namespace)
    exec(compile(catalog_material_code, unitled_path, "exec"), namespace)
    exec(compile(helper_code, unitled_path, "exec"), namespace)
    return namespace


def _build_genetic_context(namespace: dict, objective_names: Sequence[str], *, output_dir: Optional[str] = None):
    objective_names = tuple(dict.fromkeys(str(name).strip() for name in objective_names if str(name).strip()))
    genetic_multi_obj = len(objective_names) > 1
    genetic_decision_controls = namespace["ReuseGeneticDecisionControls"](
        ga_dcr_limit=namespace["_build_ga_variable_control"](
            namespace["genetic_control_ga_dcr_limit"],
            namespace["genetic_control_ga_dcr_limit_range"],
            (namespace["genetic_dcr_limit"], namespace["genetic_dcr_limit"]),
        ),
        ga_disp_limit=namespace["_build_ga_variable_control"](
            namespace["genetic_control_ga_disp_limit"],
            namespace["genetic_control_ga_disp_limit_range"],
            (namespace["genetic_disp_limit"], namespace["genetic_disp_limit"]),
        ),
        ga_buckling_lambda_min=namespace["_build_ga_variable_control"](
            namespace["genetic_control_ga_buckling_limit"],
            namespace["genetic_control_ga_buckling_limit_range"],
            (namespace["genetic_buckling_lambda_min"], namespace["genetic_buckling_lambda_min"]),
        ),
        ga_merchant_rankine_min=namespace["_build_ga_variable_control"](
            namespace["genetic_control_ga_mr_limit"],
            namespace["genetic_control_ga_mr_limit_range"],
            (namespace["genetic_merchant_rankine_min"], namespace["genetic_merchant_rankine_min"]),
        ),
        opt_dcr_limit=namespace["_build_ga_variable_control"](
            namespace["genetic_control_opt_dcr_limit"],
            namespace["genetic_control_opt_dcr_limit_range"],
            (namespace["opt_dcr_limit"], namespace["opt_dcr_limit"]),
        ),
        opt_dlim=namespace["_build_ga_variable_control"](
            namespace["genetic_control_opt_dlim"],
            namespace["genetic_control_opt_dlim_range"],
            (namespace["opt_dlim"], namespace["opt_dlim"]),
        ),
        opt_buckling_lambda_min=namespace["_build_ga_variable_control"](
            namespace["genetic_control_opt_buckling_limit"],
            namespace["genetic_control_opt_buckling_limit_range"],
            (namespace["opt_buckling_lambda_min"], namespace["opt_buckling_lambda_min"]),
        ),
        opt_merchant_rankine_min=namespace["_build_ga_variable_control"](
            namespace["genetic_control_opt_mr_limit"],
            namespace["genetic_control_opt_mr_limit_range"],
            (namespace["opt_merchant_rankine_min"], namespace["opt_merchant_rankine_min"]),
        ),
    )

    comparison_reference_mode = str(namespace["genetic_reference_mode"]).strip().lower()
    if comparison_reference_mode not in {"candidate_new", "fixed_base"}:
        raise ValueError("genetic_reference_mode must be 'candidate_new' or 'fixed_base'.")

    ctx = namespace["ReuseGeneticEvaluationContext"](
        base_G=namespace["G"],
        move_spec=namespace["genetic_grid_spec"],
        grid=namespace["grid"],
        material=namespace["material_el"],
        catalog=namespace["catalog"],
        units_in=namespace["units_in"],
        units_out=namespace["units_out"],
        c1=namespace["c1"],
        c2=namespace["c2"],
        c3=namespace["c3"],
        c4=namespace["c4"],
        c5=namespace["c5"],
        q_load=namespace["load_el"],
        ndm=namespace["ndm"],
        ndf=namespace["ndf"],
        init_model=namespace["init_model"],
        define_nodes=namespace["define_nodes"],
        apply_costr=namespace["apply_costr"],
        el_ele=namespace["el_ele"],
        apply_loads=namespace["sf_load_as_nodal_xy_projection"],
        run_static=namespace["run_elastic_static_analysis"],
        section_rotation_opts=namespace["_build_fiber_section_opts"](
            namespace["fiber_n_circ"],
            namespace["fiber_n_rad"],
            namespace["fiber_L"],
            namespace["fiber_2L"],
            namespace["fiber_4L"],
        ),
        base_M=namespace["M"],
        reuse_base_mesh_topology=True,
        diagonal_topology_enabled=bool(namespace.get("genetic_diagonal_topology_enabled", False)),
        diagonal_topology_threshold=float(namespace.get("genetic_diagonal_topology_threshold", 0.5)),
        diagonal_topology_pattern=str(namespace.get("genetic_diagonal_topology_pattern", "alternating")),
        diagonal_topology_groups=tuple((namespace.get("genetic_diagonal_spec") or {}).get("groups", [])),
        diagonal_topology_candidate_members=tuple((namespace.get("genetic_diagonal_spec") or {}).get("candidate_members", [])),
        candidate_sections=namespace["candidate_sections"],
        mech_solver=namespace["genetic_mech_solver"],
        mecc_opt=namespace["genetic_opt_mech"],
        initial_section_name=namespace["s_1"],
        start_from_first_section=False,
        start_section_row=namespace["genetic_opt_start_section_row"],
        opt_max_iter=namespace["genetic_opt_max_iter"],
        opt_dcr_limit=namespace["genetic_opt_dcr_limit"],
        opt_dlim=namespace["genetic_opt_dlim"],
        opt_check_disp=namespace["genetic_opt_check_disp"],
        opt_check_buckling=namespace["genetic_opt_check_buckling"],
        opt_buckling_lambda_min=namespace["genetic_opt_buckling_lambda_min"],
        opt_buckling_mode_index=namespace["genetic_opt_buckling_mode"],
        opt_check_merchant_rankine=namespace["genetic_opt_check_merchant_rankine"],
        opt_merchant_rankine_min=namespace["genetic_opt_merchant_rankine_min"],
        opt_merchant_rankine_mode_index=namespace["genetic_opt_merchant_rankine_mode"],
        opt_buckling_retries=namespace["genetic_opt_buckling_retries"],
        opt_buckling_group_limit=namespace["genetic_opt_buckling_group_limit"],
        gamma_M0=namespace["gamma_M0"],
        gamma_M1=namespace["gamma_M1"],
        gamma_M2=namespace["gamma_M2"],
        beta=namespace["beta"],
        alpha_LT=namespace["alpha_LT"],
        compression_sign=namespace["compression_sign"],
        subdivide=1,
        first_new_node_tag=2000,
        update_G=False,
        area_mode="faces",
        gamma_steel=namespace["gamma_steel"],
        ecf_new=namespace["ecf_new"],
        ecf_reused=namespace["ecf_reused"],
        ecf_work=namespace["ecf_work"],
        ecf_scrap=namespace["ecf_scrap"],
        cost_new=namespace["cost_new"],
        cost_reused=namespace["cost_reused"],
        cost_work=namespace["cost_work"],
        cost_scrap=namespace["cost_scrap"],
        reuse_enabled=bool(namespace["genetic_matching_reuse"]),
        reuse_csv_path=(namespace["reuse_csv_path"] if namespace["genetic_matching_reuse"] else None),
        reuse_length_units_in=namespace["reuse_length_units_in"],
        reuse_max_cut_length=namespace["reuse_max_cut_length"],
        reuse_min_remainder_length=namespace["reuse_min_remainder_length"],
        reuse_group_sizes=namespace["reuse_group_sizes"],
        reuse_member_mask_enabled=bool(namespace["genetic_matching_reuse"] and namespace["genetic_reuse_member_mask_enabled"]),
        reuse_member_mask_threshold=float(namespace["genetic_reuse_member_mask_threshold"]),
        reuse_member_mask_mode=str(namespace["genetic_reuse_member_mask_mode"]),
        reuse_member_mask_groups=tuple((namespace.get("genetic_reuse_mask_spec") or {}).get("groups", [])),
        output_dir=output_dir or namespace["genetic_output_dir"],
        save_candidate_artifacts=True,
        reference_mode=comparison_reference_mode,
        constraints=namespace["ReuseGeneticConstraints"](
            dcr_limit=namespace["genetic_dcr_limit"],
            disp_limit=namespace["genetic_disp_limit"],
            buckling_lambda_min=namespace["genetic_buckling_lambda_min"],
            buckling_mode_index=namespace["genetic_buckling_mode"],
            merchant_rankine_enabled=bool(namespace["genetic_use_merchant_rankine_constraint"]),
            merchant_rankine_min=float(namespace["genetic_merchant_rankine_min"]),
            merchant_rankine_mode_index=namespace["genetic_merchant_rankine_mode"],
        ),
        objectives=namespace["ReuseGeneticObjectives"](
            names=tuple(objective_names),
            multi_objective=bool(genetic_multi_obj),
        ),
        decision_controls=genetic_decision_controls,
    )

    if comparison_reference_mode == "fixed_base":
        fixed_reference_ctx = replace(
            ctx,
            mecc_opt=namespace["mecc_opt"],
            start_section_row=namespace["section_num"],
            opt_max_iter=namespace["opt_max_iter"],
            opt_dcr_limit=namespace["opt_dcr_limit"],
            opt_dlim=namespace["opt_dlim"],
            opt_check_disp=namespace["check_disp"],
            opt_check_buckling=namespace["opt_check_buckling"],
            opt_buckling_lambda_min=namespace["opt_buckling_lambda_min"],
            opt_buckling_mode_index=namespace["opt_buckling_mode"],
            opt_check_merchant_rankine=namespace["opt_check_merchant_rankine"],
            opt_merchant_rankine_min=namespace["opt_merchant_rankine_min"],
            opt_merchant_rankine_mode_index=namespace["opt_merchant_rankine_mode"],
            opt_buckling_retries=namespace["opt_buckling_retries"],
            opt_buckling_group_limit=namespace["opt_buckling_group_limit"],
        )
        ctx.fixed_reference = compute_fixed_reference_design(fixed_reference_ctx)

    return ctx


def _compute_fixed_reference(namespace: dict, ctx: ReuseGeneticEvaluationContext):
    fixed_reference_ctx = replace(
        ctx,
        mecc_opt=namespace["mecc_opt"],
        start_section_row=namespace["section_num"],
        opt_max_iter=namespace["opt_max_iter"],
        opt_dcr_limit=namespace["opt_dcr_limit"],
        opt_dlim=namespace["opt_dlim"],
        opt_check_disp=namespace["check_disp"],
        opt_check_buckling=namespace["opt_check_buckling"],
        opt_buckling_lambda_min=namespace["opt_buckling_lambda_min"],
        opt_buckling_mode_index=namespace["opt_buckling_mode"],
        opt_check_merchant_rankine=namespace["opt_check_merchant_rankine"],
        opt_merchant_rankine_min=namespace["opt_merchant_rankine_min"],
        opt_merchant_rankine_mode_index=namespace["opt_merchant_rankine_mode"],
        opt_buckling_retries=namespace["opt_buckling_retries"],
        opt_buckling_group_limit=namespace["opt_buckling_group_limit"],
    )
    return compute_fixed_reference_design(fixed_reference_ctx)


class SavedParetoRuntime:
    def __init__(self, *, output_dir: str, objective_names: Sequence[str], unitled_path: str = "main.py"):
        self.output_dir = os.path.abspath(output_dir)
        self.unitled_path = os.path.abspath(unitled_path)
        self.objective_names = tuple(str(name) for name in objective_names)
        self.namespace = _load_unitled_namespace(self.unitled_path)
        self.ctx = _build_genetic_context(
            self.namespace,
            self.objective_names,
            output_dir=os.path.join(self.output_dir, "genetic_search"),
        )
        self.pareto_nl_runs_dir = self.namespace["_build_output_path"](self.output_dir, "pareto_nonlinear_runs")
        self.run_state = {"next_id": 1}
        self.history_json_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_history.json")
        self.summary_json_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_summary.json")
        self.pareto_json_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_pareto_front.json")
        self.pareto_csv_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_pareto_front.csv")
        self.pareto_global_json_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_pareto_front_global.json")
        self.pareto_global_csv_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_pareto_front_global.csv")
        self.pareto_metrics_csv_path = self.namespace["_build_output_path"](self.output_dir, "pareto_candidate_metrics.csv")
        self.all_metrics_csv_path = self.namespace["_build_output_path"](
            str(self.ctx.output_dir),
            "genetic_candidate_metrics.csv",
        )
        self.pareto_plot_path = self.namespace["_build_output_path"](self.output_dir, "reuse_genetic_pareto_front.png")
        self.parallel_coords_plot = self.namespace["_build_output_path"](
            self.output_dir,
            "reuse_genetic_parallel_coords.png",
        )
        self.pareto_metrics_parallel_coords_plot = self.namespace["_build_output_path"](
            self.output_dir,
            "reuse_genetic_parallel_coords_pareto_metrics.png",
        )
        self.pareto_fixed_parallel_coords_plot = self.namespace["_build_output_path"](
            self.output_dir,
            "reuse_genetic_parallel_coords_pareto_fixed_base.png",
        )
        self.material_takeoff_dir = self.namespace["_build_output_path"](self.output_dir, "pareto_material_takeoffs")

    def _base_settings(self) -> ReuseGeneticAlgorithmSettings:
        return ReuseGeneticAlgorithmSettings(
            population_size=self.namespace["genetic_population"],
            n_generations=self.namespace["genetic_generations"],
            crossover_prob=self.namespace["genetic_crossover_prob"],
            mutation_prob=self.namespace["genetic_mutation_prob"],
            mutation_scale=self.namespace["genetic_mutation_scale"],
            seed=self.namespace["genetic_seed"],
            multi_objective=(len(self.objective_names) > 1),
            engine=self.namespace["genetic_engine"],
            verbose=self.namespace["genetic_verbose"],
            n_offsprings=self.namespace["genetic_n_offsprings"],
            sampling=self.namespace["genetic_sampling"],
            eliminate_duplicates=bool(self.namespace["genetic_eliminate_duplicates"]),
            crossover_eta=self.namespace["genetic_crossover_eta"],
            mutation_eta=self.namespace["genetic_mutation_eta"],
            n_workers=self.namespace["genetic_n_workers"],
            progress_heartbeat_s=self.namespace.get("genetic_progress_heartbeat_s", 30.0),
            candidate_stall_timeout_s=self.namespace.get("genetic_candidate_stall_timeout_s", None),
        )

    def continue_generations(self, extra_generations: int) -> str:
        extra_generations = max(1, int(extra_generations))
        pareto_front, all_evaluations, objective_names = load_saved_pareto_front(output_dir=self.output_dir)
        if tuple(str(name) for name in objective_names) != self.objective_names:
            self.objective_names = tuple(str(name) for name in objective_names)
        previous_result = type("ResumeResult", (), {})()
        previous_result.all_evaluations = list(all_evaluations)
        previous_result.history = []
        if os.path.exists(self.history_json_path):
            try:
                with open(self.history_json_path, "r", encoding="utf-8") as fobj:
                    previous_result.history = list(json.load(fobj) or [])
            except Exception:
                previous_result.history = []
        resume_settings = build_resume_settings(previous_result, self._base_settings(), extra_generations)
        result = run_reuse_genetic_optimization(self.ctx, resume_settings)

        combined_history = list(result.history or [])
        if previous_result.history:
            combined_history = list(previous_result.history)
            appended = list(result.history[len(previous_result.history):] or [])
            combined_history.extend(appended)
        with open(self.history_json_path, "w", encoding="utf-8") as fobj:
            json.dump(combined_history, fobj, indent=2)

        displayed_pareto_front = list(result.pareto_front_global or result.pareto_front)
        export_pareto_front_json(self.pareto_json_path, displayed_pareto_front, result.objective_names)
        export_pareto_front_csv(self.pareto_csv_path, displayed_pareto_front, result.objective_names)
        export_pareto_front_json(self.pareto_global_json_path, result.pareto_front_global, result.objective_names)
        export_pareto_front_csv(self.pareto_global_csv_path, result.pareto_front_global, result.objective_names)
        export_candidate_metrics_csv(self.pareto_metrics_csv_path, self.ctx, displayed_pareto_front)
        os.makedirs(os.path.dirname(self.all_metrics_csv_path), exist_ok=True)
        export_candidate_metrics_csv(self.all_metrics_csv_path, self.ctx, result.all_evaluations)
        material_takeoff_pareto_front = self._reconstruct_missing_geometry(displayed_pareto_front)
        takeoff_export = self.namespace["_export_pareto_material_takeoff_set"](
            material_takeoff_pareto_front,
            self.material_takeoff_dir,
            solution_kind="reused",
        )
        pareto_metrics_parallel_plot = save_pareto_metrics_parallel_coords_plot(
            pareto_front=displayed_pareto_front,
            save_path=self.pareto_metrics_parallel_coords_plot,
        )
        candidate_parallel_plot = save_candidate_metrics_parallel_coords_plot(
            all_evaluations=result.all_evaluations,
            save_path=self.parallel_coords_plot,
        )
        fixed_parallel_plot = save_pareto_fixed_base_parallel_coords_plot(
            output_dir=self.output_dir,
            pareto_front=displayed_pareto_front,
            save_path=self.pareto_fixed_parallel_coords_plot,
        )
        plot_pareto_front(
            self.pareto_plot_path,
            displayed_pareto_front,
            result.objective_names,
            all_evaluations=result.all_evaluations,
        )
        with open(self.summary_json_path, "w", encoding="utf-8") as fobj:
            json.dump(
                {
                    "method": result.method,
                    "objective_names": list(result.objective_names),
                    "n_evaluations": int(result.n_evaluations),
                    "cache_hits": int(result.cache_hits),
                    "pareto_front_size": len(displayed_pareto_front),
                    "pareto_front_final_population_size": len(result.pareto_front),
                    "pareto_front_global_size": len(result.pareto_front_global),
                    "selected_candidate_id": None if result.best is None else result.best.meta.get("candidate_id", None),
                    "continued_generations": extra_generations,
                    "candidate_metrics_csv": self.all_metrics_csv_path,
                    "material_takeoffs": takeoff_export,
                    "parallel_coords_plot": candidate_parallel_plot,
                    "pareto_metrics_parallel_coords_plot": pareto_metrics_parallel_plot,
                    "fixed_parallel_coords_plot": fixed_parallel_plot,
                },
                fobj,
                indent=2,
            )
        return (
            f"Completed {extra_generations} extra generations from the last saved population. "
            f"Pareto solutions now available in {self.output_dir}."
        )

    def _candidate_run_folder_name(self, evaluation, pareto_index: int, run_index: int, *, solution_kind: str = "reused") -> str:
        return self.namespace["_candidate_run_folder_name"](
            evaluation,
            pareto_index,
            run_index,
            solution_kind=solution_kind,
        )

    def reconstruct_evaluation(self, saved_evaluation):
        evaluation = evaluate_candidate_design(self.ctx, saved_evaluation.x)
        for key in ("candidate_id", "generation"):
            if key in saved_evaluation.meta and key not in evaluation.meta:
                evaluation.meta[key] = saved_evaluation.meta[key]
        if "candidate_id" in saved_evaluation.meta:
            evaluation.meta["candidate_id"] = saved_evaluation.meta["candidate_id"]
        return evaluation

    def _reconstruct_missing_geometry(self, evaluations):
        reconstructed = []
        for evaluation in evaluations or []:
            if evaluation.moved_G is not None and evaluation.moved_M is not None:
                reconstructed.append(evaluation)
            else:
                reconstructed.append(self.reconstruct_evaluation(evaluation))
        return reconstructed

    def _run_selected(self, saved_evaluation, pareto_index: int, *, solution_kind: str = "reused") -> str:
        evaluation = self.reconstruct_evaluation(saved_evaluation)
        run_index = int(self.run_state["next_id"])
        self.run_state["next_id"] = run_index + 1
        folder_name = self._candidate_run_folder_name(
            evaluation,
            pareto_index,
            run_index,
            solution_kind=solution_kind,
        )
        run_output_dir = self.namespace["_build_output_path"](self.pareto_nl_runs_dir, folder_name)
        message = self.namespace["_run_candidate_gmna_from_evaluation"](
            evaluation,
            pareto_index,
            run_output_dir,
            solution_kind=solution_kind,
        )
        print(
            f"# SAVED PARETO | nonlinear_run=single | solution_kind={solution_kind} | pareto_rank={int(pareto_index) + 1} | "
            f"candidate_id={evaluation.meta.get('candidate_id', 'n/a')} | out_dir={run_output_dir}"
        )
        return message

    def export_material_takeoffs(self, saved_evaluations, *, solution_kind: str = "reused", output_dir: Optional[str] = None) -> str:
        total = len(saved_evaluations)
        if total <= 0:
            return "No Pareto solutions available for material CSV export."
        kind = str(solution_kind or "reused").strip().lower()
        base_dir = output_dir or self.namespace["_build_output_path"](self.output_dir, "pareto_material_takeoffs")
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        completed = 0
        failed = 0
        failed_rows = []
        for idx, saved_evaluation in enumerate(saved_evaluations):
            candidate_id = saved_evaluation.meta.get("candidate_id", "n/a")
            try:
                evaluation = self.reconstruct_evaluation(saved_evaluation)
                candidate_id = evaluation.meta.get("candidate_id", candidate_id)
                folder_name = f"pareto_{idx + 1:03d}_candidate_{candidate_id}_{kind}"
                run_output_dir = self.namespace["_build_output_path"](base_dir, folder_name)
                self.namespace["_export_pareto_material_takeoff_csvs"](
                    evaluation,
                    run_output_dir,
                    pareto_index=idx,
                    solution_kind=kind,
                )
                completed += 1
            except Exception as exc:
                failed += 1
                failed_rows.append(f"solution_kind={kind}, pareto_rank={idx + 1}, candidate_id={candidate_id}, error={exc}")
                print(
                    f"# SAVED PARETO | material_csv_export_failed | "
                    f"solution_kind={kind} | pareto_rank={idx + 1} | candidate_id={candidate_id} | error={exc}"
                )
            print(
                f"# SAVED PARETO | material_csv_export_progress={idx + 1}/{total} | "
                f"solution_kind={kind} | candidate_id={candidate_id} | completed={completed} | failed={failed}"
            )

        if failed_rows:
            with open(
                self.namespace["_build_output_path"](base_dir, f"material_csv_export_failures_{kind}.txt"),
                "w",
                encoding="utf-8",
            ) as fobj:
                for row in failed_rows:
                    fobj.write(f"{row}\n")
        if failed:
            return f"Exported material CSVs for {completed}/{total} Pareto {kind} solutions in {base_dir}. Failed: {failed}."
        return f"Exported material CSVs for {total} Pareto {kind} solutions in {base_dir}."

    def on_run_selected(self, saved_evaluation, pareto_index: int) -> str:
        return self._run_selected(saved_evaluation, pareto_index, solution_kind="reused")

    def on_run_reference_selected(self, saved_evaluation, pareto_index: int) -> str:
        return self._run_selected(saved_evaluation, pareto_index, solution_kind="new")

    def _run_all(self, saved_evaluations, *, solution_kind: str = "reused") -> str:
        total = len(saved_evaluations)
        completed = 0
        failed = 0
        failed_runs = []
        for idx, saved_evaluation in enumerate(saved_evaluations):
            candidate_id = saved_evaluation.meta.get("candidate_id", "n/a")
            try:
                self._run_selected(saved_evaluation, idx, solution_kind=solution_kind)
                completed += 1
            except Exception as exc:
                failed += 1
                failed_runs.append(
                    f"solution_kind={solution_kind}, pareto_rank={idx + 1}, candidate_id={candidate_id}, error={exc}"
                )
                print(
                    f"# SAVED PARETO | nonlinear_run_all_failed | "
                    f"solution_kind={solution_kind} | pareto_rank={idx + 1} | candidate_id={candidate_id} | error={exc}"
                )
            print(
                f"# SAVED PARETO | nonlinear_run_all_progress={idx + 1}/{total} | "
                f"solution_kind={solution_kind} | candidate_id={candidate_id} | completed={completed} | failed={failed}"
            )

        if failed_runs:
            os.makedirs(self.pareto_nl_runs_dir, exist_ok=True)
            with open(
                self.namespace["_build_output_path"](self.pareto_nl_runs_dir, f"run_all_failures_{solution_kind}.txt"),
                "w",
                encoding="utf-8",
            ) as fobj:
                for row in failed_runs:
                    fobj.write(f"{row}\n")
        if failed:
            return (
                f"Completed {completed}/{total} {solution_kind} nonlinear runs in {self.pareto_nl_runs_dir}. "
                f"Failed runs: {failed}."
            )
        return f"Completed nonlinear runs for {total} Pareto {solution_kind} solutions in {self.pareto_nl_runs_dir}"

    def on_run_all(self, saved_evaluations) -> str:
        return self._run_all(saved_evaluations, solution_kind="reused")

    def on_run_reference_all(self, saved_evaluations) -> str:
        return self._run_all(saved_evaluations, solution_kind="new")

    def _fixed_reference(self):
        fixed_reference = getattr(self.ctx, "fixed_reference", None)
        if not isinstance(fixed_reference, dict):
            print("# SAVED PARETO | computing fixed-base reference design")
            fixed_reference = _compute_fixed_reference(self.namespace, self.ctx)
            self.ctx.fixed_reference = fixed_reference
        return fixed_reference

    def on_run_fixed_base(self) -> str:
        fixed_reference = self._fixed_reference()
        fixed_evaluation = self.namespace["_build_fixed_reference_evaluation"](fixed_reference)
        run_index = int(self.run_state["next_id"])
        self.run_state["next_id"] = run_index + 1
        folder_name = self._candidate_run_folder_name(
            fixed_evaluation,
            pareto_index=0,
            run_index=run_index,
            solution_kind="fixed_base",
        )
        run_output_dir = self.namespace["_build_output_path"](self.pareto_nl_runs_dir, folder_name)
        message = self.namespace["_run_candidate_gmna_from_evaluation"](
            fixed_evaluation,
            pareto_index=0,
            run_output_dir=run_output_dir,
            solution_kind="fixed_base",
        )
        print(f"# SAVED PARETO | nonlinear_run=fixed_base | out_dir={run_output_dir}")
        return message
