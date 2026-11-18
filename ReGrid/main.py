#---------------------------------------------------------------------------------------------------------------------------------------
#===============================================================FUNCTIONS===============================================================
#---------------------------------------------------------------------------------------------------------------------------------------
import csv
import json
import os
import multiprocessing
import shutil
from pathlib import Path
import yaml
from dataclasses import replace
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

from geometry import define_geometry,write_nodes_to_txt
from mesh import define_mesh, write_members_to_txt, write_barrel_rows_report
from plot import (
    plot_geometry,
    plot_boundary_sets,
    plot_mesh,
    plot_section_assignment_plan,
    plot_ec3_dcr_plan,
    plot_nodal_displacement_plan,
)
from pareto_selection_gui import (
    select_pareto_solution_gui,
    save_candidate_metrics_parallel_coords_plot,
    save_pareto_fixed_base_parallel_coords_plot,
    save_pareto_metrics_parallel_coords_plot,
)
from export_dxf import export_dxf
from py_writer import PyWriter
from materials import define_material
from import_section import get_section_properties, load_section_catalog
from section_assignment_gui import resolve_section_definition
from reuse_matching import match_reused_members, write_sections_used_catalog
from corrosion import calculate_and_plot_three_period_corrosion
from section_corrosion import corrode_section_catalog_csv
from corrosion_database_plots import plot_corroded_database_fig12_style
#----------------------------------------------------------Opensees code import----------------------------------------------------------
from OS_start import init_model
from OS_nodes import define_nodes
from OS_constraints import apply_costr
from OS_elastic_elements import el_ele
from OS_self_weigth import apply_selfweight
from OS_Loads import sf_load_as_nodal_xy_projection
from OS_run import run_elastic_static_analysis
from OS_plot import plot_model_3d, plot_deformed_mesh_mpl,plot_deformed_mesh_tension_compression_mpl
from OS_LB_fast import get_tangent_stiffness, linear_buckling_sparse, vectors_to_modes, plot_buckling_modes
from OS_opt_sections_ec3 import optimize_sections_ec3
from OS_hinge_radau import fb_H_ele
from OS_Lobatto import fb_L_ele
from OS_pushdown import run_gmna_force_control,run_gmna_disp_control,run_gmna_arclength_control
from OS_pushdown_plot import plot_gmna_results, save_gmna_history_txt
from OS_local_buckling_post_final import postprocess_local_instability
from OS_local_buckling_summary_plots import plot_gmna_vs_local_buckling_normalized
from OS_EC3_structuralcheck import run_ec3_check
from OS_results import extract_results, print_results_summary, write_force_report, write_member_force_report
from Merchant_Rankine import merchant_rankine_factors_from_results
from Merchant_Rankine_plot import plot_merchant_rankine_comparison
from Elastic_force_plots import plot_elastic_force_suite, write_elastic_force_report
from genetic_grid_helpers import build_quarter_mirror_move_spec, build_reuse_member_mask_spec, describe_move_spec
from reuse_genetic_evaluator import (
    CandidateEvaluation,
    ReuseGeneticConstraints,
    ReuseGeneticDecisionControls,
    ReuseGeneticEvaluationContext,
    ReuseGeneticObjectives,
    ReuseGeneticVariableControl,
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
    load_candidate_metrics_csv,
    ReuseGeneticOptimizationResult,
)
from main_helpers import build_output_path as _build_output_path
from main_helpers import build_fiber_section_opts as _build_fiber_section_opts
from main_helpers import load_batch_runtime
#----------------------------------------------------------PYTON-CODE-WRITERS-----------------------------------------------------------
W_el = PyWriter()                                                                       # .py code - elastic analysis
W_non_lin = PyWriter()                                                                  # .py code - non linear analysis
_cli_args, _batch_overrides, batch_mode = load_batch_runtime()                          # batch runtime for multiple analysis
_IS_MAIN_PROCESS = multiprocessing.current_process().name == "MainProcess"


def _parse_ga_range(raw_value, default_pair):
    if raw_value is None:
        values = default_pair
    elif isinstance(raw_value, str):
        parts = [p.strip() for p in raw_value.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid GA range '{raw_value}'. Expected 'min,max'.")
        values = (float(parts[0]), float(parts[1]))
    elif isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
        values = (float(raw_value[0]), float(raw_value[1]))
    else:
        raise ValueError(f"Invalid GA range value: {raw_value!r}")
    return (min(float(values[0]), float(values[1])), max(float(values[0]), float(values[1])))


def _build_ga_variable_control(enabled, raw_range, default_pair):
    bounds = _parse_ga_range(raw_range, default_pair)
    return ReuseGeneticVariableControl(
        enabled=bool(enabled),
        lower=float(bounds[0]),
        upper=float(bounds[1]),
    )
#---------------------------------------------------------------------------------------------------------------------------------------
#==============================================================USER INPUT===============================================================
#---------------------------------------------------------------------------------------------------------------------------------------
# All user-configurable parameters are loaded from config.yaml.
# Existing command-line/batch overrides keep precedence over the YAML values.

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"ReGrid configuration file not found: {_CONFIG_PATH}. "
        "Place config.yaml in the same directory as main.py."
    )

with _CONFIG_PATH.open("r", encoding="utf-8") as _config_file:
    CFG = yaml.safe_load(_config_file)

if not isinstance(CFG, dict):
    raise ValueError("config.yaml must contain a YAML mapping at the top level.")


def _cfg(*keys):
    """Return a required nested value from config.yaml."""
    value = CFG
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            path = ".".join(str(k) for k in keys)
            raise KeyError(f"Missing required configuration value: {path}")
        value = value[key]
    return value


def _cfg_batch(batch_name, *keys):
    """Batch/CLI override takes precedence over the corresponding YAML value."""
    return _batch_overrides.get(batch_name, _cfg(*keys))


#---------------------------------------------------------------GEOMETRY----------------------------------------------------------------
geom = _cfg_batch("geom", "geometry", "geom")
B = _cfg_batch("B", "geometry", "B")
L = _cfg_batch("L", "geometry", "L")
f = _cfg_batch("f", "geometry", "f")
nx = _cfg_batch("nx", "geometry", "nx")
ny = _cfg_batch("ny", "geometry", "ny")
grid = _cfg_batch("grid", "geometry", "grid")

#-----------------------------------------------------------SECTION AND MATERIAL---------------------------------------------------------
output_dir = _cli_args.output_dir or _batch_overrides.get("output_dir", _cfg("files", "output_dir"))
os.makedirs(output_dir, exist_ok=True)
dxf_export = _build_output_path(output_dir, "geom_mesh.dxf")
csv_path = _cfg("files", "section_catalogue")
units_in = _cfg("units", "section_catalogue_input")
units_out = _cfg("units", "model_output")

#-------------------------------------------------------------CORROSION----------------------------------------------------------------
corrosion_enabled = _cfg_batch("corrosion_enabled", "corrosion", "enabled")
corrosion_period_end_years = tuple(
    _cfg_batch("corrosion_period_end_years", "corrosion", "period_end_years")
)
corrosion_categories = tuple(
    _cfg_batch("corrosion_categories", "corrosion", "categories")
)
corrosion_curve_path = _build_output_path(output_dir, "corrosion_curve.png")
corrosion_curve_data_path = _build_output_path(output_dir, "corrosion_curve_data.csv")
corrosion_apply_to_database = _cfg_batch(
    "corrosion_apply_to_database", "corrosion", "apply_to_section_database"
)
corrosion_application_mode = _cfg_batch(
    "corrosion_application_mode", "corrosion", "application_mode"
)
corrosion_drop_invalid_sections = _cfg_batch(
    "corrosion_drop_invalid_sections", "corrosion", "drop_invalid_sections"
)
corrosion_csv_path = _build_output_path(output_dir, "sections_corroded.csv")
corrosion_loss_cm = 0.0
corrosion_catalog_summary = None
if corrosion_enabled:
    corrosion_loss_cm, corrosion_curve_path = calculate_and_plot_three_period_corrosion(
        end_years=corrosion_period_end_years,
        categories=corrosion_categories,
        output_path=corrosion_curve_path,
        data_output_path=corrosion_curve_data_path,
    )
    if corrosion_apply_to_database:
        corrosion_catalog_summary = corrode_section_catalog_csv(
            input_csv_path=csv_path,
            output_csv_path=corrosion_csv_path,
            corrosion_loss_cm=corrosion_loss_cm,
            application_mode=corrosion_application_mode,
            drop_invalid=corrosion_drop_invalid_sections,
        )
        csv_path = corrosion_csv_path

s_1 = _cfg("section_and_material", "starter_section")
m_1 = tuple(_cfg("section_and_material", "materials", "m_1"))
m_2 = tuple(_cfg("section_and_material", "materials", "m_2"))
m_3 = tuple(_cfg("section_and_material", "materials", "m_3"))
m_4 = tuple(_cfg("section_and_material", "materials", "m_4"))
m_5 = tuple(_cfg("section_and_material", "materials", "m_5"))

#-----------------------------------------------------------LOADS AND CONSTRAINTS---------------------------------------------------------
ndm = _cfg("model", "ndm")
ndf = _cfg("model", "ndf")
c1 = tuple(_cfg("boundary_conditions", "c1"))
c2 = tuple(_cfg("boundary_conditions", "c2"))
c3 = tuple(_cfg("boundary_conditions", "c3"))
c4 = tuple(_cfg("boundary_conditions", "c4"))
c5 = tuple(_cfg("boundary_conditions", "c5"))

_q1_cfg = _cfg("loads", "q1")
q1 = (*_q1_cfg["values"], _q1_cfg["distribution"])

_q2_cfg = _cfg("loads", "q2")
q2 = (
    *_q2_cfg["symmetrical_values"],
    _q2_cfg["symmetrical_distribution"],
    *_q2_cfg["asymmetrical_values"],
    _q2_cfg["asymmetrical_distribution"],
    *_q2_cfg["asymmetrical_region"],
)

_active_load_case = str(_cfg("loads", "active_elastic_load_case")).strip().lower()
if _active_load_case == "q1":
    load_el = q1
elif _active_load_case == "q2":
    load_el = q2
else:
    raise ValueError("loads.active_elastic_load_case must be 'q1' or 'q2'.")

#----------------------------------------------------------LINEAR ELASTIC ANALYSIS--------------------------------------------------------
section_type = _cfg("linear_elastic", "section_type")
section_assignment_json = _build_output_path(output_dir, "section_assignment.json")
section_assignment_report = _build_output_path(output_dir, "section_assignment.txt")
section_selector_show_labels = bool(_cfg("linear_elastic", "section_selector_show_labels"))

#-----------------------------------------------------------REUSED STOCK MATCHING----------------------------------------------------------
reuse_matching_enabled = bool(_cfg("reuse", "matching_enabled"))
reuse_csv_path = _cfg("reuse", "stock_csv")
reuse_length_units_in = _cfg("reuse", "length_units_in")
reuse_max_cut_length = _cfg("reuse", "max_cut_length")
reuse_min_remainder_length = _cfg("reuse", "min_remainder_length")
reuse_group_sizes = tuple(_cfg("reuse", "group_sizes"))
reuse_remaining_csv = _build_output_path(output_dir, "reused_stock_remaining.csv")
reuse_report_txt = _build_output_path(output_dir, "reuse_matching_report.txt")
reuse_sections_used_csv = _build_output_path(output_dir, "sections_used.csv")
reuse_corrosion_apply_to_database = _cfg_batch(
    "reuse_corrosion_apply_to_database", "corrosion", "apply_to_reuse_database"
)
reuse_corrosion_csv_path = _build_output_path(output_dir, "Database_reuse_corroded.csv")
reuse_corrosion_catalog_summary = None
reuse_corrosion_database_plots = None
if corrosion_enabled and reuse_corrosion_apply_to_database:
    reuse_corrosion_catalog_summary = corrode_section_catalog_csv(
        input_csv_path=reuse_csv_path,
        output_csv_path=reuse_corrosion_csv_path,
        corrosion_loss_cm=corrosion_loss_cm,
        application_mode=corrosion_application_mode,
        drop_invalid=corrosion_drop_invalid_sections,
    )
    reuse_corrosion_database_plots = plot_corroded_database_fig12_style(
        input_csv_path=reuse_corrosion_csv_path,
        output_dir=output_dir,
        prefix="reuse_corroded_database",
    )
    reuse_csv_path = reuse_corrosion_csv_path

gamma_steel = _cfg("reuse", "gamma_steel")
ecf_new = _cfg_batch("ecf_new", "impact_and_cost", "ecf_new")
ecf_reused = _cfg_batch("ecf_reused", "impact_and_cost", "ecf_reused")
ecf_work = _cfg_batch("ecf_work", "impact_and_cost", "ecf_work")
ecf_scrap = _cfg_batch("ecf_scrap", "impact_and_cost", "ecf_scrap")
cost_new = _cfg_batch("cost_new", "impact_and_cost", "cost_new")
cost_reused = _cfg_batch("cost_reused", "impact_and_cost", "cost_reused")
cost_work = _cfg_batch("cost_work", "impact_and_cost", "cost_work")
cost_scrap = _cfg_batch("cost_scrap", "impact_and_cost", "cost_scrap")

#------------------------------------------------------REUSE + GENETIC GRID OPTIMIZATION----------------------------------------------------
reuse_genetic_opt = bool(_cfg("genetic", "enabled"))
genetic_matching_reuse = bool(_cfg("genetic", "matching_reuse_inside_loop"))
genetic_mech_solver = _cfg_batch("genetic_mech_solver", "genetic", "mech_solver")
genetic_opt_mech = _cfg_batch("genetic_opt_mech", "genetic", "opt_mech")
genetic_move_range_x = tuple(_cfg_batch("genetic_move_range_x", "genetic", "move_range_x"))
genetic_move_range_y = tuple(_cfg_batch("genetic_move_range_y", "genetic", "move_range_y"))
genetic_objective_mode = _cfg_batch("genetic_objective_mode", "genetic", "objective_mode")
genetic_obj_1 = _cfg_batch("genetic_obj_1", "genetic", "obj_1")
genetic_obj_2 = _cfg_batch("genetic_obj_2", "genetic", "obj_2")
genetic_obj_3 = _cfg_batch("genetic_obj_3", "genetic", "obj_3")
genetic_reference_mode = _cfg_batch("genetic_reference_mode", "genetic", "reference_mode")
_genetic_select_pareto_cfg = bool(_cfg("genetic", "select_pareto_for_nonlinear")) and (not batch_mode)
genetic_select_pareto_for_nl = _batch_overrides.get(
    "genetic_select_pareto_for_nl", _genetic_select_pareto_cfg
)
genetic_engine = _cfg_batch("genetic_engine", "genetic", "engine")
genetic_population = _cfg_batch("genetic_population", "genetic", "population")
genetic_generations = _cfg_batch("genetic_generations", "genetic", "generations")
genetic_seed = _cfg_batch("genetic_seed", "genetic", "seed")
genetic_verbose = _cfg_batch("genetic_verbose", "genetic", "verbose")
genetic_progress_heartbeat_s = _cfg_batch(
    "genetic_progress_heartbeat_s", "genetic", "progress_heartbeat_s"
)
genetic_candidate_stall_timeout_s = _cfg_batch(
    "genetic_candidate_stall_timeout_s", "genetic", "candidate_stall_timeout_s"
)
genetic_suppress_evaluation_output = _cfg_batch(
    "genetic_suppress_evaluation_output", "genetic", "suppress_evaluation_output"
)
genetic_terminal_progress_every = _cfg_batch(
    "genetic_terminal_progress_every", "genetic", "terminal_progress_every"
)
genetic_n_offsprings = _cfg_batch("genetic_n_offsprings", "genetic", "n_offsprings")
genetic_sampling = _cfg_batch("genetic_sampling", "genetic", "sampling")
genetic_eliminate_duplicates = _cfg_batch(
    "genetic_eliminate_duplicates", "genetic", "eliminate_duplicates"
)
genetic_crossover_prob = _cfg_batch(
    "genetic_crossover_prob", "genetic", "crossover", "probability"
)
genetic_crossover_eta = _cfg_batch(
    "genetic_crossover_eta", "genetic", "crossover", "eta"
)
genetic_mutation_prob = _cfg_batch(
    "genetic_mutation_prob", "genetic", "mutation", "probability"
)
genetic_mutation_eta = _cfg_batch(
    "genetic_mutation_eta", "genetic", "mutation", "eta"
)
genetic_n_workers = _cfg_batch("genetic_n_workers", "genetic", "n_workers")
genetic_mutation_scale = _cfg_batch(
    "genetic_mutation_scale", "genetic", "mutation", "scale"
)
genetic_buckling_mode = _cfg("genetic", "buckling_mode")
genetic_dcr_limit = _cfg_batch("genetic_dcr_limit", "genetic", "dcr_limit")
genetic_disp_limit = _cfg_batch("genetic_disp_limit", "genetic", "displacement_limit")
genetic_buckling_lambda_min = _cfg_batch(
    "genetic_buckling_lambda_min", "genetic", "buckling_lambda_min"
)
genetic_use_merchant_rankine_constraint = _cfg_batch(
    "genetic_use_merchant_rankine_constraint", "genetic", "merchant_rankine", "enabled"
)
genetic_merchant_rankine_min = _cfg_batch(
    "genetic_merchant_rankine_min", "genetic", "merchant_rankine", "minimum"
)
genetic_merchant_rankine_mode = _cfg_batch(
    "genetic_merchant_rankine_mode", "genetic", "merchant_rankine", "mode"
)
if genetic_merchant_rankine_mode is None:
    genetic_merchant_rankine_mode = genetic_buckling_mode

genetic_reuse_member_mask_enabled = _cfg_batch(
    "genetic_reuse_member_mask_enabled", "genetic", "reuse_member_mask", "enabled"
)
genetic_reuse_member_mask_threshold = _cfg_batch(
    "genetic_reuse_member_mask_threshold", "genetic", "reuse_member_mask", "threshold"
)
genetic_reuse_member_mask_mode = _cfg_batch(
    "genetic_reuse_member_mask_mode", "genetic", "reuse_member_mask", "mode"
)

_gc = _cfg("genetic", "controlled_variables")
genetic_control_ga_dcr_limit = _batch_overrides.get(
    "genetic_control_ga_dcr_limit", bool(_gc["ga_dcr_limit"]["enabled"])
)
genetic_control_ga_dcr_limit_range = _batch_overrides.get(
    "genetic_control_ga_dcr_limit_range", tuple(_gc["ga_dcr_limit"]["range"])
)
genetic_control_ga_disp_limit = _batch_overrides.get(
    "genetic_control_ga_disp_limit", bool(_gc["ga_displacement_limit"]["enabled"])
)
genetic_control_ga_disp_limit_range = _batch_overrides.get(
    "genetic_control_ga_disp_limit_range", tuple(_gc["ga_displacement_limit"]["range"])
)
genetic_control_ga_buckling_limit = _batch_overrides.get(
    "genetic_control_ga_buckling_limit", bool(_gc["ga_buckling_limit"]["enabled"])
)
genetic_control_ga_buckling_limit_range = _batch_overrides.get(
    "genetic_control_ga_buckling_limit_range", tuple(_gc["ga_buckling_limit"]["range"])
)
genetic_control_ga_mr_limit = _batch_overrides.get(
    "genetic_control_ga_mr_limit", bool(_gc["ga_merchant_rankine_limit"]["enabled"])
)
genetic_control_ga_mr_limit_range = _batch_overrides.get(
    "genetic_control_ga_mr_limit_range", tuple(_gc["ga_merchant_rankine_limit"]["range"])
)
genetic_control_reuse_max_cut_length = _batch_overrides.get(
    "genetic_control_reuse_max_cut_length", bool(_gc["reuse_max_cut_length"]["enabled"])
)
genetic_control_reuse_max_cut_length_range = _batch_overrides.get(
    "genetic_control_reuse_max_cut_length_range", tuple(_gc["reuse_max_cut_length"]["range"])
)
genetic_control_opt_dcr_limit = _batch_overrides.get(
    "genetic_control_opt_dcr_limit", bool(_gc["optimizer_dcr_limit"]["enabled"])
)
genetic_control_opt_dcr_limit_range = _batch_overrides.get(
    "genetic_control_opt_dcr_limit_range", tuple(_gc["optimizer_dcr_limit"]["range"])
)
genetic_control_opt_dlim = _batch_overrides.get(
    "genetic_control_opt_dlim", bool(_gc["optimizer_displacement_limit"]["enabled"])
)
genetic_control_opt_dlim_range = _batch_overrides.get(
    "genetic_control_opt_dlim_range", tuple(_gc["optimizer_displacement_limit"]["range"])
)
genetic_control_opt_buckling_limit = _batch_overrides.get(
    "genetic_control_opt_buckling_limit", bool(_gc["optimizer_buckling_limit"]["enabled"])
)
genetic_control_opt_buckling_limit_range = _batch_overrides.get(
    "genetic_control_opt_buckling_limit_range", tuple(_gc["optimizer_buckling_limit"]["range"])
)
genetic_control_opt_mr_limit = _batch_overrides.get(
    "genetic_control_opt_mr_limit", bool(_gc["optimizer_merchant_rankine_limit"]["enabled"])
)
genetic_control_opt_mr_limit_range = _batch_overrides.get(
    "genetic_control_opt_mr_limit_range", tuple(_gc["optimizer_merchant_rankine_limit"]["range"])
)

genetic_history_json = _build_output_path(output_dir, "reuse_genetic_history.json")
genetic_summary_json = _build_output_path(output_dir, "reuse_genetic_summary.json")
genetic_pareto_json = _build_output_path(output_dir, "reuse_genetic_pareto_front.json")
genetic_pareto_csv = _build_output_path(output_dir, "reuse_genetic_pareto_front.csv")
genetic_pareto_global_json = _build_output_path(output_dir, "reuse_genetic_pareto_front_global.json")
genetic_pareto_global_csv = _build_output_path(output_dir, "reuse_genetic_pareto_front_global.csv")
genetic_pareto_candidate_metrics_csv = _build_output_path(output_dir, "pareto_candidate_metrics.csv")
genetic_pareto_plot = _build_output_path(output_dir, "reuse_genetic_pareto_front.png")
genetic_pareto_only_plot = _build_output_path(output_dir, "reuse_genetic_pareto_only.png")
genetic_material_takeoff_dir = _build_output_path(output_dir, "pareto_material_takeoffs")
genetic_parallel_coords_plot = _build_output_path(output_dir, "reuse_genetic_parallel_coords.png")
genetic_pareto_fixed_parallel_coords_plot = _build_output_path(
    output_dir, "reuse_genetic_parallel_coords_pareto_fixed_base.png"
)
genetic_pareto_metrics_parallel_coords_plot = _build_output_path(
    output_dir, "reuse_genetic_parallel_coords_pareto_metrics.png"
)
genetic_fixed_reference_json = _build_output_path(output_dir, "genetic_fixed_reference_summary.json")
genetic_output_dir = _build_output_path(output_dir, "genetic_search")
genetic_resume_from_metrics = bool(
    _batch_overrides.get("genetic_resume_from_metrics", _cfg("genetic", "resume", "from_metrics"))
)
_genetic_resume_metrics_cfg = _cfg("genetic", "resume", "metrics_csv")
if _genetic_resume_metrics_cfg is None:
    _genetic_resume_metrics_cfg = _build_output_path(genetic_output_dir, "genetic_candidate_metrics.csv")
genetic_resume_metrics_csv = _batch_overrides.get(
    "genetic_resume_metrics_csv", _genetic_resume_metrics_cfg
)
genetic_headless_exit_after_ga = bool(
    _batch_overrides.get(
        "genetic_headless_exit_after_ga",
        _cfg("genetic", "resume", "headless_exit_after_ga"),
    )
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---EC3 CHECKS---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma_M0 = _cfg("ec3", "gamma_M0")
gamma_M1 = _cfg("ec3", "gamma_M1")
gamma_M2 = _cfg("ec3", "gamma_M2")
beta = _cfg("ec3", "beta")
compression_sign = _cfg("ec3", "compression_sign")
alpha_LT = _cfg("ec3", "alpha_LT")

#----------------------------------------------------------STRUCTURAL OPTIMIZATION---------------------------------------------------------
optimization = bool(_cfg("mechanical_optimization", "enabled"))
mecc_opt = _cfg("mechanical_optimization", "mode")
opt_dcr_limit = _cfg("mechanical_optimization", "dcr_limit")
check_disp = bool(_cfg("mechanical_optimization", "displacement", "check"))
opt_dlim = _cfg("mechanical_optimization", "displacement", "limit")
opt_check_buckling = bool(_cfg("mechanical_optimization", "buckling", "check"))
opt_buckling_lambda_min = _cfg("mechanical_optimization", "buckling", "lambda_min")
opt_buckling_mode = _cfg("mechanical_optimization", "buckling", "mode")
opt_check_merchant_rankine = bool(_cfg("mechanical_optimization", "merchant_rankine", "check"))
opt_merchant_rankine_min = _cfg("mechanical_optimization", "merchant_rankine", "minimum")
opt_merchant_rankine_mode = _cfg("mechanical_optimization", "merchant_rankine", "mode")
if opt_merchant_rankine_mode is None:
    opt_merchant_rankine_mode = opt_buckling_mode
opt_buckling_retries = _cfg("mechanical_optimization", "buckling", "retries")
opt_buckling_group_limit = _cfg("mechanical_optimization", "buckling", "group_limit")
opt_section_group_limit = _batch_overrides.get(
    "opt_section_group_limit", _cfg("mechanical_optimization", "legacy", "section_group_limit")
)
opt_section_symmetry = _batch_overrides.get(
    "opt_section_symmetry", _cfg("mechanical_optimization", "legacy", "section_symmetry")
)
opt_demand_grouping = _batch_overrides.get(
    "opt_demand_grouping", _cfg("mechanical_optimization", "legacy", "demand_grouping")
)
opt_demand_group_rel_tol = _batch_overrides.get(
    "opt_demand_group_rel_tol", _cfg("mechanical_optimization", "legacy", "demand_group_relative_tolerance")
)
opt_max_iter = _cfg("mechanical_optimization", "max_iterations")
opt_verbose = _cfg("mechanical_optimization", "verbose")
section_num = _cfg("mechanical_optimization", "start_section_row")

#-------------------------------------GA INTERNAL MECHANICAL OPTIMIZATION (CANDIDATE EVALUATIONS)----------------------------------------
genetic_opt_start_section_row = _batch_overrides.get(
    "genetic_opt_start_section_row", _cfg("genetic_mechanical_optimization", "start_section_row")
)
genetic_opt_max_iter = _batch_overrides.get(
    "genetic_opt_max_iter", _cfg("genetic_mechanical_optimization", "max_iterations")
)
genetic_opt_dcr_limit = _batch_overrides.get(
    "genetic_opt_dcr_limit", _cfg("genetic_mechanical_optimization", "dcr_limit")
)
genetic_opt_check_disp = _batch_overrides.get(
    "genetic_opt_check_disp", _cfg("genetic_mechanical_optimization", "displacement", "check")
)
genetic_opt_dlim = _batch_overrides.get(
    "genetic_opt_dlim", _cfg("genetic_mechanical_optimization", "displacement", "limit")
)
genetic_opt_check_buckling = _batch_overrides.get(
    "genetic_opt_check_buckling", _cfg("genetic_mechanical_optimization", "buckling", "check")
)
genetic_opt_buckling_lambda_min = _batch_overrides.get(
    "genetic_opt_buckling_lambda_min", _cfg("genetic_mechanical_optimization", "buckling", "lambda_min")
)
genetic_opt_buckling_mode = _batch_overrides.get(
    "genetic_opt_buckling_mode", _cfg("genetic_mechanical_optimization", "buckling", "mode")
)
genetic_opt_check_merchant_rankine = _batch_overrides.get(
    "genetic_opt_check_merchant_rankine",
    _cfg("genetic_mechanical_optimization", "merchant_rankine", "check"),
)
genetic_opt_merchant_rankine_min = _batch_overrides.get(
    "genetic_opt_merchant_rankine_min",
    _cfg("genetic_mechanical_optimization", "merchant_rankine", "minimum"),
)
genetic_opt_merchant_rankine_mode = _batch_overrides.get(
    "genetic_opt_merchant_rankine_mode",
    _cfg("genetic_mechanical_optimization", "merchant_rankine", "mode"),
)
if genetic_opt_merchant_rankine_mode is None:
    genetic_opt_merchant_rankine_mode = genetic_opt_buckling_mode
genetic_opt_buckling_retries = _batch_overrides.get(
    "genetic_opt_buckling_retries", _cfg("genetic_mechanical_optimization", "buckling", "retries")
)
genetic_opt_buckling_group_limit = _batch_overrides.get(
    "genetic_opt_buckling_group_limit", _cfg("genetic_mechanical_optimization", "buckling", "group_limit")
)
genetic_opt_section_group_limit = _batch_overrides.get(
    "genetic_opt_section_group_limit",
    _cfg("genetic_mechanical_optimization", "legacy", "section_group_limit"),
)
genetic_opt_section_symmetry = _batch_overrides.get(
    "genetic_opt_section_symmetry",
    _cfg("genetic_mechanical_optimization", "legacy", "section_symmetry"),
)
genetic_opt_demand_grouping = _batch_overrides.get(
    "genetic_opt_demand_grouping",
    _cfg("genetic_mechanical_optimization", "legacy", "demand_grouping"),
)
genetic_opt_demand_group_rel_tol = _batch_overrides.get(
    "genetic_opt_demand_group_rel_tol",
    _cfg("genetic_mechanical_optimization", "legacy", "demand_group_relative_tolerance"),
)

#------------------------------------------------------------NONLINEAR ANALYSIS------------------------------------------------------------
non_lin = bool(_cfg("nonlinear", "enabled"))
T_fiber_ele = _cfg("nonlinear", "fiber_element")
lobatto_n_integration_pts = _cfg("nonlinear", "lobatto_n_integration_points")
_hinge_lp_cfg = _cfg("nonlinear", "hinge_length_ratio")
if isinstance(_hinge_lp_cfg, str) and _hinge_lp_cfg.strip().lower() == "auto":
    hinge_lp_ratio = 0.50 if T_fiber_ele == "HingeRadauTwo" else 0.10
else:
    hinge_lp_ratio = float(_hinge_lp_cfg)
Analysis_control = _cfg("nonlinear", "analysis_control")
force_use_fallbacks = bool(_cfg("nonlinear", "force_control", "use_fallbacks"))
force_use_adaptive_step = bool(_cfg("nonlinear", "force_control", "use_adaptive_step"))
force_fallback_strategies = [
    {
        "algorithm": tuple(item["algorithm"]),
        "test": tuple(item["test"]),
    }
    for item in _cfg("nonlinear", "force_control", "fallback_strategies")
]
_force_solver_cfg = _cfg("nonlinear", "force_control", "solver_options")
force_solver_options = dict(
    adaptive_force_step=force_use_adaptive_step,
    force_min_increment=_force_solver_cfg["force_min_increment"],
    force_cutback_factor=_force_solver_cfg["force_cutback_factor"],
    force_easy_iterations=_force_solver_cfg["force_easy_iterations"],
    force_slow_iterations=_force_solver_cfg["force_slow_iterations"],
    fallback_growth_steps=_force_solver_cfg["fallback_growth_steps"],
    fallback_growth_factor=_force_solver_cfg["fallback_growth_factor"],
    force_max_attempts=_force_solver_cfg["force_max_attempts"],
)
force_solver_max_steps = _cfg("nonlinear", "force_control", "fixed_step_max_steps")
force_adaptive_max_steps = _cfg("nonlinear", "force_control", "adaptive_max_steps")
uz_limit = _cfg("nonlinear", "uz_limit")
local_inst_control = bool(_cfg("nonlinear", "local_instability_postprocessing"))
plot_fiber_sections = bool(_cfg("fibers", "plot_sections"))
plot_fiber_sections_show = bool(_cfg("fibers", "show_section_plots"))
fiber_section_plot_dir = _build_output_path(output_dir, "fiber_plots")
fiber_n_circ = _cfg("fibers", "n_circ")
fiber_n_rad = _cfg("fibers", "n_rad")
fiber_L = list(_cfg("fibers", "L"))
fiber_2L = list(_cfg("fibers", "2L"))
fiber_4L = list(_cfg("fibers", "4L"))

#---------------------------------------------------------------------------------------------------------------------------------------------
#==============================================================PLOT CONTROL PARAM=============================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
plot_geo = bool(_cfg("plotting", "geometry"))
pl_mesh = bool(_cfg("plotting", "mesh"))
plot_opvis = bool(_cfg("plotting", "opensees_model"))
plot_def = bool(_cfg("plotting", "deformed_shape"))
plot_buck = bool(_cfg("plotting", "buckling_shape"))
plot_buck_n_curve_points = _cfg("plotting", "buckling_curve_points")
plot_sec_plan = bool(_cfg("plotting", "section_plan"))
live_plot = bool(_cfg("plotting", "nonlinear_live_plot"))
save_load_map = bool(_cfg("plotting", "save_load_map"))
show_load_map = bool(_cfg("plotting", "show_load_map"))
writer_os = bool(_cfg("plotting", "write_opensees_code"))
diagram_force = bool(_cfg("plotting", "force_diagram"))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
# =================================================================MAINCODE===================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       
#---------------------------------------------------------------------------------------------------------------------------------------------
#==================================================================GEOMETRY===================================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
G = define_geometry(geom, B, L, f, nx, ny, grid)                                        # Point definition
if plot_geo:                                                                            # Toggle
    plot_geometry(G, show_ids=True)                                                     # Plot points
    plot_boundary_sets(G)                                                               # Plot point goups
genetic_grid_spec = None
if reuse_genetic_opt:
    genetic_grid_spec = build_quarter_mirror_move_spec(
        G,
        x_move_range=genetic_move_range_x,
        y_move_range=genetic_move_range_y,
    )
    if _IS_MAIN_PROCESS:
        print(f"# GENETIC GRID | {describe_move_spec(genetic_grid_spec)}")
if _IS_MAIN_PROCESS:
    write_nodes_to_txt(G, _build_output_path(output_dir, "nodes.txt"))                  # Write nodes
#---------------------------------------------------------------------------------------------------------------------------------------------
#=====================================================================MESH====================================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
M = define_mesh(G, grid)                                                                # Mesh definition
if pl_mesh:                                                                             # Toggle
    plot_mesh(G, M, show_nodes=True, show_boundary=True)                                # Plot Mesh
if _IS_MAIN_PROCESS:
    write_members_to_txt(G, M, _build_output_path(output_dir, "members.txt"))           #Write mesh
genetic_reuse_mask_spec = None
if reuse_genetic_opt and bool(genetic_matching_reuse) and bool(genetic_reuse_member_mask_enabled):
    genetic_reuse_mask_spec = build_reuse_member_mask_spec(
        G,
        M,
        mode=genetic_reuse_member_mask_mode,
    )
    if _IS_MAIN_PROCESS:
        print(
            f"# GENETIC REUSE MASK | mode={genetic_reuse_mask_spec['mode']} | "
            f"groups={len(genetic_reuse_mask_spec.get('groups', []))}"
        )
#---------------------------------------------------------------------------------------------------------------------------------------------
#=============================================================EXPORT GEOMETRICAL MODEL========================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
if _IS_MAIN_PROCESS:
    export_dxf(G, M, path=dxf_export, print_summary=False)                              # Export in dxf
#---------------------------------------------------------------------------------------------------------------------------------------------
#=================================================================SECTION IMPORT==============================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
catalog = load_section_catalog(csv_path, sep=";", encoding="utf-8", name_col_index=0)   # Load section catalogue
candidate_sections = get_section_properties(catalog,                                    # catalogue form .csv file
                                           units_in=units_in,                           # units in
                                           units_out=units_out,                         # units out
                                           print_props=False,                           # Print sections props
                                           print_all=False)                             # Print sections summary
#---------------------------------------------------------------------------------------------------------------------------------------------
#===================================================================MATERIAL==================================================================
#---------------------------------------------------------------------------------------------------------------------------------------------
mat1 = define_material(m_1, name="M1")                                                  # Material 1 - ELastic
mat2 = define_material(m_2, name="M2")                                                  # Material 2 - nonlinear steel
mat3 = define_material(m_3, name="M3")                                                  # Material 3 - Steel 02
mat4 = define_material(m_4, name="M4")                                                  # Material 4 - EPP
mat5 = define_material(m_5, name="M5")                                                  # Material 5 - Hardening
material_el = mat1                                                                      # Material for analysis 


def _candidate_run_folder_name(evaluation, pareto_index, run_index, solution_kind="reused"):
    candidate_id = evaluation.meta.get("candidate_id", "na")
    kind = str(solution_kind or "reused").strip().lower()
    return f"run_{int(run_index):03d}_{kind}_pareto_{int(pareto_index) + 1:03d}_candidate_{candidate_id}"


def _build_fixed_reference_evaluation(fixed_reference):
    if not isinstance(fixed_reference, dict):
        raise RuntimeError("Fixed reference data is not available.")
    G_ref = fixed_reference.get("G", None)
    M_ref = fixed_reference.get("M", None)
    opt_result_ref = fixed_reference.get("opt_result", None)
    summary_ref = dict(fixed_reference.get("summary", {}) or {})
    if G_ref is None or M_ref is None or not isinstance(opt_result_ref, dict):
        raise RuntimeError("Fixed reference data is incomplete.")
    return CandidateEvaluation(
        x=tuple(),
        feasible=bool(summary_ref.get("analysis_ok", True)),
        objectives={},
        constraints={},
        moved_G=G_ref,
        moved_M=M_ref,
        opt_result=opt_result_ref,
        reuse_result=None,
        final_state=summary_ref.get("final_state", None),
        section_by_member=None,
        lambda_ref=summary_ref.get("lambda_ref", None),
        meta={
            "candidate_id": "fixed_base",
            "comparison_reference_label": str(fixed_reference.get("label", "fixed_base_new")),
            "ga_disp_limit_used": fixed_reference.get("disp_limit", opt_dlim),
            "opt_dlim_used": fixed_reference.get("opt_dlim", opt_dlim),
        },
    )


def _evaluation_displacement_plot_limit(evaluation):
    meta = getattr(evaluation, "meta", {}) or {}
    for key in ("ga_disp_limit_used", "disp_limit", "opt_dlim_used"):
        try:
            value = float(meta.get(key))
        except Exception:
            continue
        if value > 1e-12:
            return value
    return float(opt_dlim)


def _takeoff_member_key(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _takeoff_member_lengths(G, M):
    nodes = G.get("nodes", {})
    out = {}
    for na, nb in M.get("members", []) or []:
        xa, ya, za = nodes[int(na)]
        xb, yb, zb = nodes[int(nb)]
        out[_takeoff_member_key(na, nb)] = ((xb - xa) ** 2 + (yb - ya) ** 2 + (zb - za) ** 2) ** 0.5
    return out


def _takeoff_float(value, default=""):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return value


def _takeoff_section_group_size(section):
    family = str((section or {}).get("section_family", "") or "").strip().upper().replace(" ", "")
    name = str((section or {}).get("name", "") or "").strip().upper().replace(" ", "")
    text = f"{family} {name}"
    if "4L" in text:
        return 4
    if "2L" in text:
        return 2
    return 1


def _takeoff_section_fields(section):
    section = dict(section or {})
    return {
        "section_name": section.get("name", ""),
        "section_family": section.get("section_family", ""),
        "fabrication": section.get("fabrication", ""),
        "section_class": section.get("section_class", ""),
        "w_kN_per_m": _takeoff_float(section.get("w_kN_per_m", "")),
        "A": _takeoff_float(section.get("A", "")),
        "Aeff": _takeoff_float(section.get("Aeff", "")),
        "Av": _takeoff_float(section.get("Av", "")),
        "Av_y": _takeoff_float(section.get("Av_y", "")),
        "Av_z": _takeoff_float(section.get("Av_z", "")),
        "Iy": _takeoff_float(section.get("Iy", "")),
        "Iz": _takeoff_float(section.get("Iz", "")),
        "Iuu": _takeoff_float(section.get("Iuu", "")),
        "Ivv": _takeoff_float(section.get("Ivv", "")),
        "It": _takeoff_float(section.get("It", "")),
        "Iw": _takeoff_float(section.get("Iw", "")),
        "iy": _takeoff_float(section.get("iy", "")),
        "iz": _takeoff_float(section.get("iz", "")),
        "Wel_y": _takeoff_float(section.get("Wel_y", "")),
        "Wel_z": _takeoff_float(section.get("Wel_z", "")),
        "Wpl_y": _takeoff_float(section.get("Wpl_y", "")),
        "Wpl_z": _takeoff_float(section.get("Wpl_z", "")),
        "Weff_y": _takeoff_float(section.get("Weff_y", "")),
        "Weff_z": _takeoff_float(section.get("Weff_z", "")),
        "h": _takeoff_float(section.get("h", "")),
        "b": _takeoff_float(section.get("b", "")),
        "tf": _takeoff_float(section.get("tf", "")),
        "tw": _takeoff_float(section.get("tw", "")),
        "r": _takeoff_float(section.get("r", "")),
        "dext": _takeoff_float(section.get("dext", section.get("Dext", ""))),
        "t": _takeoff_float(section.get("t", "")),
        "dint": _takeoff_float(section.get("dint", section.get("Dint", ""))),
    }


def _takeoff_fieldnames(extra_fields):
    return list(extra_fields) + [
        "section_name", "section_family", "fabrication", "section_class", "section_units",
        "w_kN_per_m", "A", "Aeff", "Av", "Av_y", "Av_z",
        "Iy", "Iz", "Iuu", "Ivv", "It", "Iw", "iy", "iz",
        "Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z",
        "h", "b", "tf", "tw", "r", "dext", "t", "dint",
    ]


def _write_takeoff_csv(path, fieldnames, rows):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _takeoff_folder_token(value):
    text = str(value).strip() or "na"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _export_pareto_material_takeoff_csvs(evaluation, output_dir, pareto_index=None, solution_kind="reused"):
    kind = str(solution_kind or "reused").strip().lower()
    G_run = evaluation.moved_G
    M_run = evaluation.moved_M
    if G_run is None or M_run is None:
        raise RuntimeError("Cannot export material takeoff without geometry and mesh.")
    os.makedirs(output_dir, exist_ok=True)

    if kind in {"new", "fixed_base"}:
        section_by_member = (
            evaluation.opt_result.get("section_by_member", {})
            if isinstance(evaluation.opt_result, dict)
            else {}
        )
        reuse_result = None
        reused_members = {}
    else:
        section_by_member = dict(evaluation.section_by_member or {})
        reuse_result = evaluation.reuse_result if isinstance(evaluation.reuse_result, dict) else None
        reused_members = {}
        for rec in (reuse_result or {}).get("reused_records", []) or []:
            member = rec.get("member", None)
            if member is not None:
                reused_members[_takeoff_member_key(member[0], member[1])] = rec

    lengths = _takeoff_member_lengths(G_run, M_run)
    candidate_id = evaluation.meta.get("candidate_id", "")
    rank = "" if pareto_index is None else int(pareto_index) + 1
    common = {
        "pareto_rank": rank,
        "candidate_id": candidate_id,
        "solution_kind": kind,
    }

    section_units = units_out
    member_fields = [
        "pareto_rank", "candidate_id", "solution_kind", "source",
        "member_i", "member_j", "member_length_m", "piece_count",
        "length_used_m", "stock_label", "stock_length_m", "cut_length_m",
    ]
    reused_rows = []
    new_rows = []
    summary = {}

    for mk, section in sorted(section_by_member.items()):
        mk = _takeoff_member_key(mk[0], mk[1])
        member_length = float(lengths.get(mk, 0.0))
        rec = reused_members.get(mk, None)
        piece_count = int(rec.get("group_size", 1)) if rec is not None else _takeoff_section_group_size(section)
        length_used = member_length * max(1, piece_count)
        row = dict(common)
        row.update(
            {
                "source": "reused" if rec is not None else "new",
                "member_i": mk[0],
                "member_j": mk[1],
                "member_length_m": member_length,
                "piece_count": piece_count,
                "length_used_m": length_used,
                "stock_label": "" if rec is None else rec.get("stock_label", ""),
                "stock_length_m": "" if rec is None else rec.get("stock_length", ""),
                "cut_length_m": "" if rec is None else max(0.0, float(rec.get("stock_length", 0.0) or 0.0) - member_length),
                "section_units": section_units,
            }
        )
        row.update(_takeoff_section_fields(section))
        if rec is not None:
            reused_rows.append(row)
        else:
            new_rows.append(row)

        key = (row["source"], str(row.get("section_name", "")), int(piece_count))
        item = summary.setdefault(
            key,
            {
                "source": row["source"],
                "section_name": row.get("section_name", ""),
                "piece_count": int(piece_count),
                "members_count": 0,
                "total_member_length_m": 0.0,
                "total_length_used_m": 0.0,
                "section_units": section_units,
            },
        )
        item["members_count"] += 1
        item["total_member_length_m"] += member_length
        item["total_length_used_m"] += length_used
        for prop_key, prop_value in _takeoff_section_fields(section).items():
            item.setdefault(prop_key, prop_value)

    scrap_fields = [
        "pareto_rank", "candidate_id", "solution_kind", "member_i", "member_j",
        "stock_label", "stock_length_m", "used_length_m", "scrap_length_m",
        "scrap_mass_kN", "piece_count",
    ]
    scrap_rows = []
    for rec in (reuse_result or {}).get("scrap_records", []) or []:
        member = rec.get("member", ("", ""))
        section = rec.get("section", {}) or {}
        row = dict(common)
        row.update(
            {
                "member_i": member[0] if len(member) > 0 else "",
                "member_j": member[1] if len(member) > 1 else "",
                "stock_label": rec.get("stock_label", ""),
                "stock_length_m": rec.get("stock_length", ""),
                "used_length_m": rec.get("used_length", ""),
                "scrap_length_m": rec.get("scrap_length", ""),
                "scrap_mass_kN": rec.get("scrap_mass", ""),
                "piece_count": rec.get("group_size", ""),
                "section_units": section_units,
            }
        )
        row.update(_takeoff_section_fields(section))
        scrap_rows.append(row)

    if not scrap_rows and reuse_result is not None and float(reuse_result.get("total_cut_length", 0.0) or 0.0) > 0.0:
        row = dict(common)
        row.update(
            {
                "member_i": "",
                "member_j": "",
                "stock_label": "aggregate_only",
                "stock_length_m": "",
                "used_length_m": "",
                "scrap_length_m": float(reuse_result.get("total_cut_length", 0.0) or 0.0),
                "scrap_mass_kN": float(reuse_result.get("total_scrap_mass", 0.0) or 0.0),
                "piece_count": "",
                "section_units": section_units,
            }
        )
        row.update(_takeoff_section_fields({}))
        scrap_rows.append(row)

    summary_fields = [
        "source", "section_name", "section_family", "fabrication", "section_class",
        "piece_count", "members_count", "total_member_length_m", "total_length_used_m", "section_units",
        "w_kN_per_m", "A", "Aeff", "Av", "Av_y", "Av_z",
        "Iy", "Iz", "Iuu", "Ivv", "It", "Iw", "iy", "iz",
        "Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z",
        "h", "b", "tf", "tw", "r", "dext", "t", "dint",
    ]

    paths = {
        "reused": _write_takeoff_csv(
            _build_output_path(output_dir, "sections_reused_used.csv"),
            _takeoff_fieldnames(member_fields),
            reused_rows,
        ),
        "new": _write_takeoff_csv(
            _build_output_path(output_dir, "sections_new_used.csv"),
            _takeoff_fieldnames(member_fields),
            new_rows,
        ),
        "scraps": _write_takeoff_csv(
            _build_output_path(output_dir, "scraps.csv"),
            _takeoff_fieldnames(scrap_fields),
            scrap_rows,
        ),
        "summary": _write_takeoff_csv(
            _build_output_path(output_dir, "sections_material_summary.csv"),
            summary_fields,
            [summary[key] for key in sorted(summary.keys())],
        ),
    }
    return paths


def _export_pareto_material_takeoff_set(evaluations, base_dir, solution_kind="reused"):
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    exported = 0
    failed = []
    for idx, evaluation in enumerate(evaluations or []):
        candidate_id = _takeoff_folder_token(evaluation.meta.get("candidate_id", "na"))
        folder = _build_output_path(
            base_dir,
            f"pareto_{idx + 1:03d}_candidate_{candidate_id}_{_takeoff_folder_token(solution_kind)}",
        )
        try:
            _export_pareto_material_takeoff_csvs(
                evaluation,
                folder,
                pareto_index=idx,
                solution_kind=solution_kind,
            )
            exported += 1
        except Exception as exc:
            failed.append(f"pareto_rank={idx + 1}, candidate_id={candidate_id}, error={exc}")
    if failed:
        with open(_build_output_path(base_dir, "material_takeoff_failures.txt"), "w", encoding="utf-8") as fobj:
            for row in failed:
                fobj.write(f"{row}\n")
    return {"exported": exported, "failed": len(failed), "output_dir": base_dir}


def _run_candidate_gmna_from_evaluation(evaluation, pareto_index, run_output_dir, solution_kind="reused"):
    kind = str(solution_kind or "reused").strip().lower()
    if evaluation.moved_G is None or evaluation.moved_M is None:
        raise RuntimeError("Selected Pareto evaluation does not contain geometry and mesh data.")

    os.makedirs(run_output_dir, exist_ok=True)
    G_run = evaluation.moved_G
    M_run = evaluation.moved_M
    run_warnings = []

    if kind in {"new", "fixed_base"}:
        if evaluation.opt_result is None or evaluation.opt_result.get("section_by_member", None) is None:
            raise RuntimeError("Selected Pareto evaluation does not contain the new-reference section assignment.")
        section_by_member_run = evaluation.opt_result["section_by_member"]
        props_global_run = evaluation.opt_result.get("props_global", None)
    else:
        if evaluation.section_by_member is None:
            raise RuntimeError("Selected Pareto evaluation does not contain the reused section assignment.")
        section_by_member_run = evaluation.section_by_member
        props_global_run = None
        if evaluation.opt_result is not None:
            props_global_run = evaluation.opt_result.get("props_global", None)

    _export_pareto_material_takeoff_csvs(
        evaluation,
        run_output_dir,
        pareto_index=pareto_index,
        solution_kind=kind,
    )

    write_nodes_to_txt(G_run, _build_output_path(run_output_dir, "nodes.txt"))
    write_members_to_txt(G_run, M_run, _build_output_path(run_output_dir, "members.txt"))
    export_dxf(G_run, M_run, path=_build_output_path(run_output_dir, "geom_mesh.dxf"), print_summary=False)

    plot_section_assignment_plan(
        G_run,
        M_run,
        section_by_member_run,
        show_nodes=False,
        show_labels=False,
        save_path=_build_output_path(run_output_dir, "section_assignment_plan.png"),
        show=False,
    )

    writer_el_local = PyWriter()
    writer_nl_local = PyWriter()

    os_start = init_model(ndm, ndf)
    if writer_os:
        writer_el_local.add(os_start)

    os_nodes, node_tags = define_nodes(G_run, ndf, set_mass=False)
    if writer_os:
        writer_el_local.add(os_nodes)

    os_fix, fixed = apply_costr(G_run, ndf, c1, c2, c3, c4, c5)
    if writer_os:
        writer_el_local.add(os_fix)

    os_ele, ele_data = el_ele(
        G=G_run,
        M=M_run,
        material=material_el,
        section=props_global_run,
        section_by_member=section_by_member_run,
        section_rotation_opts=_build_fiber_section_opts(
            fiber_n_circ,
            fiber_n_rad,
            fiber_L,
            fiber_2L,
            fiber_4L,
        ),
        transf_type="PDelta",
        subdivide=1,
        first_new_node_tag=2000,
        update_G=False,
        print_ops=False,
        print_mode="headtail",
        n_preview=3,
        print_summary=False,
    )
    if writer_os:
        writer_el_local.add(os_ele)

    q_ele, os_q_ele = apply_selfweight(
        G=G_run,
        ele_data=ele_data,
        mode="eleload",
        section=props_global_run,
        section_by_member=section_by_member_run,
        catalog=catalog,
        units_in=units_in,
        units_out=units_out,
        pattern_tag=21,
        ts_tag=21,
    )
    if writer_os:
        writer_el_local.add(os_q_ele)

    K0, _ = get_tangent_stiffness(
        system="FullGeneral",
        numberer="RCM",
        constraints="Plain",
        algorithm="Newton",
        print_ops=False,
    )

    loads, areas, os_loads = sf_load_as_nodal_xy_projection(
        G_run,
        M=M_run,
        q=load_el,
        direction="-Z",
        pattern_tag=2,
        ts_tag=2,
        ndf=6,
        area_mode="faces",
        dx=None,
        dy=None,
        round_decimals=8,
        print_ops=False,
        print_mode="all",
        n_preview=10,
        print_summary=False,
        print_components=False,
        plot_loads=save_load_map or show_load_map,
        plot_filename=(_build_output_path(run_output_dir, "applied_load_map_elastic.png") if save_load_map else None),
        plot_show=show_load_map,
        plot_title=None,
        plot_s=30,
        plot_show_labels=True,
        plot_label_fmt="{:.2f}",
        plot_label_offset=(0.01, 0.01),
        plot_label_fontsize=8,
        plot_label_nodes=None,
        plot_component="Fz",
        diag_check_uniformity=False,
        diag_tol_cv=0.02,
    )
    if writer_os:
        writer_el_local.add(os_loads)

    os_static, ok = run_elastic_static_analysis(
        load_control=1.0,
        analyze_steps=1,
        set_load_const=False,
        print_ops=False,
        print_mode="headtail",
        n_preview=10,
        print_summary=False,
    )
    if writer_os:
        writer_el_local.add(os_static)
        writer_el_local.save(_build_output_path(run_output_dir, "opensees_EL_model_export.py"), as_py=False)

    res_run = extract_results(
        G_run,
        M_run,
        ele_data=ele_data,
        ndf=ndf,
        uz_dof=3,
        uz_ref=uz_limit,
    )
    plot_nodal_displacement_plan(
        G_run,
        res_run.get("disp", {}),
        M=M_run,
        save_path=_build_output_path(run_output_dir, "nodal_displacement_plan.png"),
        show=False,
        component="mag",
        dlim=_evaluation_displacement_plot_limit(evaluation),
    )

    write_force_report(
        res_run["ele_forces"],
        filepath=_build_output_path(run_output_dir, "force_report.txt"),
        compression_sign=compression_sign,
        sort_by="N_comp_max",
        descending=True,
    )
    write_member_force_report(
        res_run["ele_forces"],
        ele_data=ele_data,
        filepath=_build_output_path(run_output_dir, "member_force_report.txt"),
        compression_sign=compression_sign,
        sort_by="N_comp_max",
        descending=True,
    )

    ec3_results_run, ec3_dcr_max_run, sections_by_key_run, lengths_by_key_run = run_ec3_check(
        G=G_run,
        ele_data=ele_data,
        ele_forces=res_run["ele_forces"],
        material=material_el,
        section=props_global_run,
        section_by_member=section_by_member_run,
        gamma_M0=gamma_M0,
        gamma_M1=gamma_M1,
        gamma_M2=gamma_M2,
        beta=beta,
        alpha_LT=alpha_LT,
        compression_sign=compression_sign,
        aggregate_by_member=True,
        print_summary=False,
        top_n=20,
        save_txt=_build_output_path(run_output_dir, "checks.txt"),
        plot_dcr=True,
        plot_mode="members",
        plot_show=False,
        plot_save_path=_build_output_path(run_output_dir, "DCR.png"),
        plot_n_colors=10,
    )
    plot_ec3_dcr_plan(
        G_run,
        M_run,
        ec3_results_run,
        save_path=_build_output_path(run_output_dir, "DCR_plan.png"),
        show=False,
        title=f"EC3 DCR - Plan View | max={ec3_dcr_max_run:.3f}",
    )

    lam_run, vecs_run = linear_buckling_sparse(
        K0,
        n_modes=5,
        drop_tol=1e-12,
        eig_tol=1e-12,
        tangent_kwargs=dict(
            system="FullGeneral",
            numberer="RCM",
            constraints="Plain",
            algorithm="Newton",
        ),
        print_ops=False,
        print_summary=False,
    )
    modes_run = vectors_to_modes(vecs_run, ndf, node_tags=node_tags)
    plot_buckling_modes(
        G_run,
        M_run,
        modes_run,
        lam=lam_run,
        mode_index=[1, 2, 3],
        scale=0.20,
        subplot=False,
        ncols=2,
        n_curve_points=plot_buck_n_curve_points,
        show=False,
        save=True,
        out_dir=run_output_dir,
        save_prefix="buckling_mode",
        dpi=150,
    )

    mr_factors_run, mr_lambdas_run = merchant_rankine_factors_from_results(
        dcr_max=ec3_dcr_max_run,
        lam=lam_run,
        n_modes=5,
    )
    with open(_build_output_path(run_output_dir, "Merchant_Rankine.txt"), "w", encoding="utf-8") as f:
        f.write(f"DCRmax = {ec3_dcr_max_run:.6f}\n")
        f.write(f"Linear buckling factor (first) = {mr_lambdas_run[0]:.6f}\n")
        f.write(f"Merchant-Rankine factor = {mr_factors_run[0]:.6f}\n")

    if not non_lin:
        return f"Elastic rerun completed in {run_output_dir}"

    os_start = init_model(ndm, ndf)
    if writer_os:
        writer_nl_local.add(os_start)
    os_nodes, node_tags = define_nodes(G_run, ndf, set_mass=False)
    if writer_os:
        writer_nl_local.add(os_nodes)
    os_fix, fixed = apply_costr(G_run, ndf, c1, c2, c3, c4, c5)
    if writer_os:
        writer_nl_local.add(os_fix)

    nl_material = mat2
    q_nl = load_el
    fy = nl_material["fyk"]
    E = nl_material["E"]
    control_node_data = res_run["max_vertical"].get("max_abs_nd") or res_run["max_vertical"]["min"]
    control_node = control_node_data["node"]

    if T_fiber_ele == "Lobatto":
        os_ele, ele_data = fb_L_ele(
            G_run,
            M_run,
            material=nl_material,
            section=props_global_run,
            section_by_member=section_by_member_run,
            transf_type="Corotational",
            subdivide=4,
            first_new_node_tag=2000,
            update_G=False,
            n_integration_pts=lobatto_n_integration_pts,
            fiber_opts=_build_fiber_section_opts(
                fiber_n_circ,
                fiber_n_rad,
                fiber_L,
                fiber_2L,
                fiber_4L,
            ),
            first_mat_tag=1,
            first_integ_tag=1,
            first_ele_tag=1,
            first_sec_tag=1,
            first_tr_tag=1,
            plot_fiber_sections=plot_fiber_sections,
            plot_fiber_sections_dir=_build_output_path(run_output_dir, "fiber_plots"),
            plot_fiber_sections_show=False,
            print_ops=False,
            print_mode="headtail",
            n_preview=3,
            print_summary=False,
        )
    else:
        os_ele, ele_data = fb_H_ele(
            G_run,
            M_run,
            material=nl_material,
            section=props_global_run,
            section_by_member=section_by_member_run,
            transf_type="Corotational",
            subdivide=4,
            first_new_node_tag=2000,
            update_G=False,
            lp_ratio=hinge_lp_ratio,
            integration=T_fiber_ele,
            fiber_opts=_build_fiber_section_opts(
                fiber_n_circ,
                fiber_n_rad,
                fiber_L,
                fiber_2L,
                fiber_4L,
            ),
            first_mat_tag=1,
            first_integ_tag=1,
            first_ele_tag=1,
            first_sec_tag=1,
            first_tr_tag=1,
            plot_fiber_sections=plot_fiber_sections,
            plot_fiber_sections_dir=_build_output_path(run_output_dir, "fiber_plots"),
            plot_fiber_sections_show=False,
            print_ops=False,
            print_mode="headtail",
            n_preview=3,
            print_summary=False,
        )
    if writer_os:
        writer_nl_local.add(os_ele)

    apply_selfweight(
        G=G_run,
        ele_data=ele_data,
        mode="eleload",
        section=props_global_run,
        section_by_member=section_by_member_run,
        catalog=catalog,
        units_in=units_in,
        units_out=units_out,
        pattern_tag=21,
        ts_tag=21,
    )

    loads, areas, os_loads = sf_load_as_nodal_xy_projection(
        G_run,
        M=M_run,
        q=q_nl,
        direction="-Z",
        pattern_tag=2,
        ts_tag=2,
        ndf=6,
        area_mode="faces",
        dx=None,
        dy=None,
        round_decimals=8,
        print_ops=False,
        print_mode="all",
        n_preview=10,
        print_summary=False,
        print_components=False,
        plot_loads=save_load_map or show_load_map,
        plot_filename=(_build_output_path(run_output_dir, "applied_load_map_nonlinear.png") if save_load_map else None),
        plot_show=show_load_map,
        plot_title=None,
        plot_s=30,
        plot_show_labels=True,
        plot_label_fmt="{:.2f}",
        plot_label_offset=(0.01, 0.01),
        plot_label_fontsize=8,
        plot_label_nodes=None,
        plot_component="Fz",
        diag_check_uniformity=False,
        diag_tol_cv=0.02,
    )
    loads_Fz = {nid: loads[nid][2] for nid in loads}

    monitor = None
    if Analysis_control == "Force":
        hist, summ = run_gmna_force_control(
            G_run,
            M_run,
            fy=fy,
            E=E,
            ele_data=ele_data,
            loads_dict=loads_Fz,
            ndf=ndf,
            pattern_tag=2,
            control_node=control_node,
            control_dof=3,
            elastic_res=res_run,
            control_node_prefer="minUz",
            dLambda=0.005,
            n_steps=(force_adaptive_max_steps if force_use_adaptive_step else force_solver_max_steps),
            constraints="Transformation",
            numberer="AMD",
            system="UmfPack",
            test=("NormUnbalance", 1e-5, 50),
            algorithm=("Newton",),
            fallback_strategies=(force_fallback_strategies if force_use_fallbacks else None),
            force_log_prefix=_build_output_path(run_output_dir, "GMNA_solver"),
            **force_solver_options,
            analysis="Static",
            monitor_fibers=True,
            fiber_ele_tags=ele_data["ele_tags"],
            fiber_nIP=ele_data["meta"]["n_ip"],
            fiber_sec_nums=None,
            T_fiber_ele=T_fiber_ele,
            fiber_strain_col=4,
            fiber_chunk_size=None,
            fiber_area_min=0.0,
            fiber_area_col=2,
            exclude_neutral_axis=True,
            neutral_axis_tol=1e-12,
            dump_fibers=False,
            dump_fibers_txt=_build_output_path(run_output_dir, "fiber_debug.txt"),
            dump_every=10,
            dump_include_full_row=True,
            print_ops=False,
            print_mode="headtail",
            n_preview=12,
            print_step_info=True,
            print_every=1,
            writer=writer_nl_local,
            live_plot=False,
            live_plot_every=1,
            live_monitor=monitor,
            keyboard_abort=True,
            ctrlc_abort=True,
            stop_keys=(b"\x1b", b"q", b"Q"),
            record_local_response=True,
            local_response_map=ele_data.get("local_instability", None),
            local_response_verbose=False,
        )
    elif Analysis_control == "Displacements":
        hist, summ = run_gmna_disp_control(
            G=G_run,
            M=M_run,
            fy=fy,
            E=E,
            ele_data=ele_data,
            loads_dict=loads_Fz,
            ndf=ndf,
            pattern_tag=2,
            control_node=control_node,
            control_dof=3,
            elastic_res=res_run,
            control_node_prefer="minUz",
            dU=-1e-3,
            n_steps=1000,
            U_max=1.5,
            constraints="Transformation",
            numberer="RCM",
            system="UmfPack",
            test=("NormUnbalance", 1e-5, 50),
            algorithm=("Newton",),
            analysis="Static",
            monitor_fibers=True,
            fiber_ele_tags=ele_data["ele_tags"],
            fiber_nIP=ele_data["meta"]["n_ip"],
            fiber_sec_nums=None,
            T_fiber_ele=T_fiber_ele,
            fiber_strain_col=4,
            fiber_chunk_size=None,
            fiber_area_min=0.0,
            fiber_area_col=2,
            exclude_neutral_axis=True,
            neutral_axis_tol=1e-12,
            dump_fibers=False,
            print_ops=False,
            print_mode="headtail",
            n_preview=12,
            print_step_info=True,
            print_every=1,
            live_plot=False,
            live_plot_every=1,
            live_monitor=monitor,
            record_local_response=True,
            local_response_map=ele_data.get("local_instability", None),
            local_response_verbose=False,
        )
    else:
        hist, summ = run_gmna_arclength_control(
            G=G_run,
            M=M_run,
            fy=fy,
            E=E,
            ele_data=ele_data,
            loads_dict=loads_Fz,
            ndf=ndf,
            pattern_tag=2,
            control_node=control_node,
            control_dof=3,
            elastic_res=res_run,
            control_node_prefer="minUz",
            s=0.1,
            alpha=1.0,
            n_steps=50,
            LF_max=None,
            stop_if_unstable=True,
            constraints="Transformation",
            numberer="RCM",
            system="UmfPack",
            test=("NormUnbalance", 1e-6, 80),
            algorithm=("NewtonLineSearch",),
            analysis="Static",
            integrator_name="ArcLength",
            fallback_strategies=[
                {"test": ("NormUnbalance", 1e-5, 150), "algorithm": ("NewtonLineSearch",), "s": 0.0025},
                {"test": ("NormUnbalance", 1e-5, 200), "algorithm": ("Newton",), "s": 0.001},
            ],
            monitor_fibers=True,
            fiber_ele_tags=ele_data["ele_tags"],
            fiber_nIP=ele_data["meta"]["n_ip"],
            fiber_sec_nums=None,
            T_fiber_ele=T_fiber_ele,
            fiber_strain_col=4,
            fiber_chunk_size=None,
            fiber_area_min=0.0,
            fiber_area_col=2,
            exclude_neutral_axis=True,
            neutral_axis_tol=1e-12,
            dump_fibers=False,
            print_ops=False,
            print_mode="headtail",
            n_preview=12,
            print_step_info=True,
            print_every=1,
            live_plot=False,
            live_plot_every=1,
            live_monitor=monitor,
            record_local_response=True,
            local_response_map=ele_data.get("local_instability", None),
            local_response_verbose=False,
        )

    if writer_os:
        writer_nl_local.save(_build_output_path(run_output_dir, "opensees_GMNA_model_export.py"), as_py=True)

    if local_inst_control:
        try:
            postprocess_local_instability(
                hist=hist,
                summary=summ,
                N_use="abs",
                drop_ratio=0.08,
                drop_consec=3,
                d_tol=1e-4,
                dU_eps=1e-5,
                energy_consec=4,
                decision_rule="either",
                active_methods=("pdrop_dend", "pdrop_dmid", "energy"),
                N_peak_min_abs=5.0,
                N_peak_min_ratio_global=0.05,
                d_event_min=1e-4,
                rb_tol=0.02,
                out_dir=run_output_dir,
                events_txt="local_instability_events_post.txt",
                curves_txt="local_instability_curves_post.txt",
                plot_member_curves=True,
                plot_member_curves_max=1,
                plot_member_curves_sort_by="step",
                plot_member_curves_normalized=True,
                plot_member_curves_add_raw=True,
                plot_energy_maps=False,
                energy_map_every=10,
                energy_map_show_deformed=False,
                energy_map_scale_factor=5.0,
                ele_data=ele_data,
                show_plots=False,
            )
        except Exception as exc:
            warning = f"local_instability_postprocess_failed: {exc}"
            run_warnings.append(warning)
            print(f"# GENETIC REUSE | warning | candidate_id={evaluation.meta.get('candidate_id', 'n/a')} | {warning}")

    import matplotlib.pyplot as plt

    plot_gmna_results(
        hist,
        u_lim=uz_limit,
        fy=fy,
        E=E,
        delta_key="Uz",
        delta_mode="abs",
        show=False,
        save_prefix="GMNA",
        out_dir=run_output_dir,
        show_local_nd_curves=True,
    )
    save_gmna_history_txt(
        hist,
        u_lim=uz_limit,
        filename=_build_output_path(run_output_dir, "GMNA_history.txt"),
        delta_mode="raw",
    )

    ax = plot_deformed_mesh_mpl(
        G_run,
        M_run,
        sf=20,
        mode="subelements",
        ele_data=ele_data,
        unstable_ele_tags=summ.get("unstable_ele_tags", []),
        plastic_onset_ele_tags=summ.get("plastic_onset_ele_tags", []),
        plastic_full_ele_tags=summ.get("plastic_full_ele_tags", []),
    )
    ax.figure.savefig(_build_output_path(run_output_dir, "deformed_shape.png"), dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_deformed_mesh_tension_compression_mpl(
        G_run,
        M_run,
        sf=20,
        ax=ax,
        mode="subelements",
        ele_data=ele_data,
        N0=1e-2,
    )
    ax.figure.savefig(_build_output_path(run_output_dir, "tension_compression_shape.png"), dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    plot_merchant_rankine_comparison(
        dcr_max=ec3_dcr_max_run,
        lam=lam_run,
        merchant_rankine_factors=mr_factors_run,
        lf_nonlinear=summ["LF_last"],
        show=False,
        save_path=_build_output_path(run_output_dir, "Merchant_Rankine_comparison.png"),
        title="MR1-MR2-MR3-MR4-MR5 vs 1/DCRmax vs nonlinear LF vs first 5 linear buckling modes",
    )
    try:
        plot_gmna_vs_local_buckling_normalized(
            hist=hist,
            n_members=3,
            delta_key="Uz",
            delta_mode="abs",
            sort_by="step",
            show=False,
            out_dir=run_output_dir,
            save_prefix="GMNA_local_summary",
            member_labels=True,
        )
    except Exception as exc:
        warning = f"gmna_local_summary_failed: {exc}"
        run_warnings.append(warning)
        print(f"# GENETIC REUSE | warning | candidate_id={evaluation.meta.get('candidate_id', 'n/a')} | {warning}")

    if run_warnings:
        with open(_build_output_path(run_output_dir, "run_warnings.txt"), "w", encoding="utf-8") as f:
            for warning in run_warnings:
                f.write(f"{warning}\n")
        return f"Nonlinear run saved in {run_output_dir} (with warnings)"
    return f"Nonlinear run saved in {run_output_dir}"


def _displayed_pareto_front(genetic_result):
    return list(genetic_result.pareto_front_global or genetic_result.pareto_front)


def _pareto_front_for_material_takeoffs(genetic_ctx, pareto_front):
    ready = []
    for evaluation in pareto_front or []:
        if evaluation.moved_G is not None and evaluation.moved_M is not None:
            ready.append(evaluation)
            continue
        rebuilt = evaluate_candidate_design(genetic_ctx, evaluation.x)
        for key in ("candidate_id", "generation"):
            if key in evaluation.meta:
                rebuilt.meta[key] = evaluation.meta[key]
        ready.append(rebuilt)
    return ready


def _export_genetic_run_outputs(genetic_ctx, genetic_settings, genetic_result, genetic_best):
    displayed_pareto_front = _displayed_pareto_front(genetic_result)
    with open(genetic_history_json, "w", encoding="utf-8") as fobj:
        json.dump(genetic_result.history, fobj, indent=2)
    with open(genetic_summary_json, "w", encoding="utf-8") as fobj:
        json.dump(
            {
                "method": genetic_result.method,
                "objective_mode": genetic_objective_mode,
                "objective_names": list(genetic_result.objective_names),
                "comparison_reference_mode": str(getattr(genetic_ctx, "reference_mode", "candidate_new")),
                "ga_constraints": {
                    "dcr_limit": float(genetic_dcr_limit),
                    "disp_limit": float(genetic_disp_limit),
                    "buckling_lambda_min": float(genetic_buckling_lambda_min),
                    "merchant_rankine_min": (
                        float(genetic_merchant_rankine_min) if genetic_use_merchant_rankine_constraint else None
                    ),
                },
                "genetic_algorithm_settings": {
                    "engine": str(genetic_engine),
                    "population_size": int(genetic_population),
                    "n_generations": int(genetic_settings.n_generations),
                    "n_offsprings": (None if genetic_n_offsprings is None else int(genetic_n_offsprings)),
                    "seed": (None if genetic_seed is None else int(genetic_seed)),
                    "sampling": str(genetic_sampling),
                    "eliminate_duplicates": bool(genetic_eliminate_duplicates),
                    "crossover_prob": float(genetic_crossover_prob),
                    "crossover_eta": float(genetic_crossover_eta),
                    "mutation_prob": float(genetic_mutation_prob),
                    "mutation_eta": float(genetic_mutation_eta),
                    "mutation_scale_builtin": float(genetic_mutation_scale),
                    "verbose": int(genetic_verbose),
                    "generation_offset": int(getattr(genetic_settings, "generation_offset", 0) or 0),
                },
                "n_evaluations": genetic_result.n_evaluations,
                "cache_hits": genetic_result.cache_hits,
                "selected_candidate_id": genetic_best.meta.get("candidate_id", None),
                "feasible": bool(genetic_best.feasible),
                "objectives": genetic_best.objectives,
                "constraints": genetic_best.constraints,
                "lambda_ref": genetic_best.lambda_ref,
                "meta": genetic_best.meta,
                "decision_vector": list(genetic_best.x),
                "pareto_front_size": len(displayed_pareto_front),
                "pareto_front_final_population_size": len(genetic_result.pareto_front),
                "pareto_front_global_size": len(genetic_result.pareto_front_global),
            },
            fobj,
            indent=2,
        )
        export_pareto_front_json(genetic_pareto_json, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_csv, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_json(genetic_pareto_global_json, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_global_csv, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_candidate_metrics_csv(genetic_pareto_candidate_metrics_csv, genetic_ctx, displayed_pareto_front)
        _export_pareto_material_takeoff_set(
            _pareto_front_for_material_takeoffs(genetic_ctx, displayed_pareto_front),
            genetic_material_takeoff_dir,
            solution_kind="reused",
        )
        save_pareto_metrics_parallel_coords_plot(
            pareto_front=displayed_pareto_front,
            save_path=genetic_pareto_metrics_parallel_coords_plot,
        )
        save_candidate_metrics_parallel_coords_plot(
            all_evaluations=genetic_result.all_evaluations,
            save_path=genetic_parallel_coords_plot,
        )
        if str(getattr(genetic_ctx, "reference_mode", "candidate_new")).strip().lower() == "fixed_base":
            save_pareto_fixed_base_parallel_coords_plot(
                pareto_front=displayed_pareto_front,
                fixed_reference=getattr(genetic_ctx, "fixed_reference", None),
                save_path=genetic_pareto_fixed_parallel_coords_plot,
            )
        plot_pareto_front(
            genetic_pareto_plot,
            displayed_pareto_front,
            genetic_result.objective_names,
            all_evaluations=genetic_result.all_evaluations,
        )


def _continue_live_genetic_run(genetic_ctx, genetic_settings, genetic_result, extra_generations):
    resumed_settings = build_resume_settings(genetic_result, genetic_settings, extra_generations)
    resumed_result = run_reuse_genetic_optimization(genetic_ctx, resumed_settings)
    resumed_best = resumed_result.best
    if resumed_best is None:
        raise RuntimeError("The continued genetic optimization did not return a valid representative solution.")
    _export_genetic_run_outputs(genetic_ctx, resumed_settings, resumed_result, resumed_best)
    print(
        f"# GENETIC REUSE | continue_generations={int(extra_generations)} | "
        f"evaluations={resumed_result.n_evaluations} | pareto_final={len(resumed_result.pareto_front)} | "
        f"pareto_global={len(resumed_result.pareto_front_global)}"
    )
    return resumed_settings, resumed_result, resumed_best

genetic_result = None
genetic_best = None
props_global = None
section_by_member = None
section_name_by_member = None
reuse_result = None
def _unitled_main():
    global G, M
    global genetic_result, genetic_best
    global props_global, section_by_member, section_name_by_member, reuse_result
    global catalog, candidate_sections
    use_genetic_solution = bool(reuse_genetic_opt)
    
    if use_genetic_solution:
        if not genetic_grid_spec:
            raise RuntimeError("reuse_genetic_opt is active but no genetic_grid_spec was built.")
    
        objective_mode = str(genetic_objective_mode).strip().lower()
        if objective_mode not in {"mono", "multi"}:
            raise ValueError("genetic_objective_mode must be 'mono' or 'multi'.")
        comparison_reference_mode = str(genetic_reference_mode).strip().lower()
        if comparison_reference_mode not in {"candidate_new", "fixed_base"}:
            raise ValueError("genetic_reference_mode must be 'candidate_new' or 'fixed_base'.")

        objective_names = [str(genetic_obj_1).strip()]
        if objective_mode == "multi":
            for raw_name in (genetic_obj_2, genetic_obj_3):
                if raw_name is None:
                    continue
                name = str(raw_name).strip()
                if name:
                    objective_names.append(name)
        objective_names = tuple(dict.fromkeys(objective_names))
        genetic_multi_obj = len(objective_names) > 1
        genetic_decision_controls = ReuseGeneticDecisionControls(
            ga_dcr_limit=_build_ga_variable_control(
                genetic_control_ga_dcr_limit,
                genetic_control_ga_dcr_limit_range,
                (genetic_dcr_limit, genetic_dcr_limit),
            ),
            ga_disp_limit=_build_ga_variable_control(
                genetic_control_ga_disp_limit,
                genetic_control_ga_disp_limit_range,
                (genetic_disp_limit, genetic_disp_limit),
            ),
            ga_buckling_lambda_min=_build_ga_variable_control(
                genetic_control_ga_buckling_limit,
                genetic_control_ga_buckling_limit_range,
                (genetic_buckling_lambda_min, genetic_buckling_lambda_min),
            ),
            ga_merchant_rankine_min=_build_ga_variable_control(
                genetic_control_ga_mr_limit,
                genetic_control_ga_mr_limit_range,
                (genetic_merchant_rankine_min, genetic_merchant_rankine_min),
            ),
            reuse_max_cut_length=_build_ga_variable_control(
                genetic_control_reuse_max_cut_length,
                genetic_control_reuse_max_cut_length_range,
                _parse_ga_range(genetic_control_reuse_max_cut_length_range, (0.0, 10.0)),
            ),
            opt_dcr_limit=_build_ga_variable_control(
                genetic_control_opt_dcr_limit,
                genetic_control_opt_dcr_limit_range,
                (genetic_opt_dcr_limit, genetic_opt_dcr_limit),
            ),
            opt_dlim=_build_ga_variable_control(
                genetic_control_opt_dlim,
                genetic_control_opt_dlim_range,
                (genetic_opt_dlim, genetic_opt_dlim),
            ),
            opt_buckling_lambda_min=_build_ga_variable_control(
                genetic_control_opt_buckling_limit,
                genetic_control_opt_buckling_limit_range,
                (genetic_opt_buckling_lambda_min, genetic_opt_buckling_lambda_min),
            ),
            opt_merchant_rankine_min=_build_ga_variable_control(
                genetic_control_opt_mr_limit,
                genetic_control_opt_mr_limit_range,
                (genetic_opt_merchant_rankine_min, genetic_opt_merchant_rankine_min),
            ),
        )
        genetic_ctx = ReuseGeneticEvaluationContext(
            base_G=G,
            move_spec=genetic_grid_spec,
            grid=grid,
            material=material_el,
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
            q_load=load_el,
            ndm=ndm,
            ndf=ndf,
            init_model=init_model,
            define_nodes=define_nodes,
            apply_costr=apply_costr,
            el_ele=el_ele,
            apply_loads=sf_load_as_nodal_xy_projection,
            run_static=run_elastic_static_analysis,
            section_rotation_opts=_build_fiber_section_opts(
                fiber_n_circ,
                fiber_n_rad,
                fiber_L,
                fiber_2L,
                fiber_4L,
            ),
            base_M=M,
            reuse_base_mesh_topology=True,
            candidate_sections=candidate_sections,
            mech_solver=genetic_mech_solver,
            mecc_opt=genetic_opt_mech,
            initial_section_name=s_1,
            start_from_first_section=False,
            start_section_row=genetic_opt_start_section_row,
            opt_max_iter=genetic_opt_max_iter,
            opt_dcr_limit=genetic_opt_dcr_limit,
            opt_dlim=genetic_opt_dlim,
            opt_check_disp=genetic_opt_check_disp,
            opt_check_buckling=genetic_opt_check_buckling,
            opt_buckling_lambda_min=genetic_opt_buckling_lambda_min,
            opt_buckling_mode_index=genetic_opt_buckling_mode,
            opt_check_merchant_rankine=genetic_opt_check_merchant_rankine,
            opt_merchant_rankine_min=genetic_opt_merchant_rankine_min,
            opt_merchant_rankine_mode_index=genetic_opt_merchant_rankine_mode,
            opt_buckling_retries=genetic_opt_buckling_retries,
            opt_buckling_group_limit=genetic_opt_buckling_group_limit,
            gamma_M0=gamma_M0,
            gamma_M1=gamma_M1,
            gamma_M2=gamma_M2,
            beta=beta,
            alpha_LT=alpha_LT,
            compression_sign=compression_sign,
            subdivide=1,
            first_new_node_tag=2000,
            update_G=False,
            area_mode="faces",
            gamma_steel=gamma_steel,
            ecf_new=ecf_new,
            ecf_reused=ecf_reused,
            ecf_work=ecf_work,
            ecf_scrap=ecf_scrap,
            cost_new=cost_new,
            cost_reused=cost_reused,
            cost_work=cost_work,
            cost_scrap=cost_scrap,
            reuse_enabled=bool(genetic_matching_reuse),
            reuse_csv_path=(reuse_csv_path if genetic_matching_reuse else None),
            reuse_length_units_in=reuse_length_units_in,
            reuse_max_cut_length=reuse_max_cut_length,
            reuse_min_remainder_length=reuse_min_remainder_length,
            reuse_group_sizes=reuse_group_sizes,
            reuse_member_mask_enabled=bool(genetic_matching_reuse and genetic_reuse_member_mask_enabled),
            reuse_member_mask_threshold=float(genetic_reuse_member_mask_threshold),
            reuse_member_mask_mode=str(genetic_reuse_member_mask_mode),
            reuse_member_mask_groups=tuple((genetic_reuse_mask_spec or {}).get("groups", [])),
            output_dir=genetic_output_dir,
            save_candidate_artifacts=True,
            reference_mode=comparison_reference_mode,
            constraints=ReuseGeneticConstraints(
                dcr_limit=genetic_dcr_limit,
                disp_limit=genetic_disp_limit,
                buckling_lambda_min=genetic_buckling_lambda_min,
                buckling_mode_index=genetic_buckling_mode,
                merchant_rankine_enabled=bool(genetic_use_merchant_rankine_constraint),
                merchant_rankine_min=float(genetic_merchant_rankine_min),
                merchant_rankine_mode_index=genetic_merchant_rankine_mode,
            ),
            objectives=ReuseGeneticObjectives(
                names=tuple(objective_names),
                multi_objective=bool(genetic_multi_obj),
            ),
            decision_controls=genetic_decision_controls,
        )
        if comparison_reference_mode == "fixed_base":
            print("# GENETIC REUSE | computing fixed-base reference design")
            fixed_reference_ctx = replace(
                genetic_ctx,
                mecc_opt=mecc_opt,
                start_section_row=section_num,
                opt_max_iter=opt_max_iter,
                opt_dcr_limit=opt_dcr_limit,
                opt_dlim=opt_dlim,
                opt_check_disp=check_disp,
                opt_check_buckling=opt_check_buckling,
                opt_buckling_lambda_min=opt_buckling_lambda_min,
                opt_buckling_mode_index=opt_buckling_mode,
                opt_check_merchant_rankine=opt_check_merchant_rankine,
                opt_merchant_rankine_min=opt_merchant_rankine_min,
                opt_merchant_rankine_mode_index=opt_merchant_rankine_mode,
                opt_buckling_retries=opt_buckling_retries,
                opt_buckling_group_limit=opt_buckling_group_limit,
            )
            fixed_reference = compute_fixed_reference_design(fixed_reference_ctx)
            genetic_ctx.fixed_reference = fixed_reference
            fixed_summary = dict(fixed_reference.get("summary", {}) or {})
            with open(genetic_fixed_reference_json, "w", encoding="utf-8") as fobj:
                json.dump(
                    {
                        "reference_mode": comparison_reference_mode,
                        "label": fixed_reference.get("label", "fixed_base_new"),
                        "mass": fixed_reference.get("mass", None),
                        "mass_tonnes": fixed_reference.get("mass_tonnes", None),
                        "ec": fixed_reference.get("ec", None),
                        "cost": fixed_reference.get("cost", None),
                        "summary": {
                            "analysis_ok": fixed_summary.get("analysis_ok", None),
                            "lambda_ref": fixed_summary.get("lambda_ref", None),
                            "merchant_rankine_lambda": fixed_summary.get("merchant_rankine_lambda", None),
                            "merchant_rankine_mode_index": fixed_summary.get("merchant_rankine_mode_index", None),
                            "merchant_rankine": fixed_summary.get("merchant_rankine", None),
                            "dcr_max": fixed_summary.get("dcr_max", None),
                            "dcr_mean": fixed_summary.get("dcr_mean", None),
                            "disp_ratio": fixed_summary.get("disp_ratio", None),
                            "max_abs_uz": fixed_summary.get("max_abs_uz", None),
                        },
                    },
                    fobj,
                    indent=2,
                )
        genetic_settings = ReuseGeneticAlgorithmSettings(
            population_size=genetic_population,
            n_generations=genetic_generations,
            crossover_prob=genetic_crossover_prob,
            mutation_prob=genetic_mutation_prob,
            mutation_scale=genetic_mutation_scale,
            seed=genetic_seed,
            multi_objective=bool(genetic_multi_obj),
            engine=genetic_engine,
            verbose=genetic_verbose,
            n_offsprings=genetic_n_offsprings,
            sampling=genetic_sampling,
            eliminate_duplicates=bool(genetic_eliminate_duplicates),
            crossover_eta=genetic_crossover_eta,
            mutation_eta=genetic_mutation_eta,
            n_workers=genetic_n_workers,
            progress_heartbeat_s=genetic_progress_heartbeat_s,
            candidate_stall_timeout_s=genetic_candidate_stall_timeout_s,
            suppress_evaluation_output=bool(genetic_suppress_evaluation_output),
            terminal_progress_every=genetic_terminal_progress_every,
        )
        genetic_result = run_reuse_genetic_optimization(genetic_ctx, genetic_settings)
        genetic_best = genetic_result.best
        displayed_pareto_front = _displayed_pareto_front(genetic_result)
        export_pareto_front_json(genetic_pareto_json, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_csv, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_json(genetic_pareto_global_json, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_global_csv, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_candidate_metrics_csv(genetic_pareto_candidate_metrics_csv, genetic_ctx, displayed_pareto_front)
        pareto_metrics_parallel_plot = save_pareto_metrics_parallel_coords_plot(
            pareto_front=displayed_pareto_front,
            save_path=genetic_pareto_metrics_parallel_coords_plot,
        )
        candidate_parallel_plot = save_candidate_metrics_parallel_coords_plot(
            all_evaluations=genetic_result.all_evaluations,
            save_path=genetic_parallel_coords_plot,
        )
        fixed_parallel_plot = None
        if comparison_reference_mode == "fixed_base":
            fixed_parallel_plot = save_pareto_fixed_base_parallel_coords_plot(
                pareto_front=displayed_pareto_front,
                fixed_reference=getattr(genetic_ctx, "fixed_reference", None),
                save_path=genetic_pareto_fixed_parallel_coords_plot,
            )
        if pareto_metrics_parallel_plot:
            print(f"# GENETIC REUSE | pareto_metrics_parallel_coords_plot={pareto_metrics_parallel_plot}")
        if candidate_parallel_plot:
            print(f"# GENETIC REUSE | parallel_coords_plot={candidate_parallel_plot}")
        if fixed_parallel_plot:
            print(f"# GENETIC REUSE | fixed_parallel_coords_plot={fixed_parallel_plot}")
        if genetic_headless_exit_after_ga:
            print(
                f"# GENETIC | headless_exit_after_ga | evaluations={genetic_result.n_evaluations} | "
                f"pareto_final={len(genetic_result.pareto_front)} | pareto_global={len(genetic_result.pareto_front_global)}"
            )
            return
        _export_pareto_material_takeoff_set(
            _pareto_front_for_material_takeoffs(genetic_ctx, displayed_pareto_front),
            genetic_material_takeoff_dir,
            solution_kind="reused",
        )
        selected_pareto_index = None
        pareto_nl_runs_dir = _build_output_path(output_dir, "pareto_nonlinear_runs")
        pareto_nl_run_state = {"next_id": 1}
    
        def _launch_pareto_nonlinear_run(evaluation, pareto_index, solution_kind="reused"):
            run_index = int(pareto_nl_run_state["next_id"])
            pareto_nl_run_state["next_id"] = run_index + 1
            folder_name = _candidate_run_folder_name(evaluation, pareto_index, run_index, solution_kind=solution_kind)
            run_output_dir = _build_output_path(pareto_nl_runs_dir, folder_name)
            message = _run_candidate_gmna_from_evaluation(
                evaluation,
                pareto_index,
                run_output_dir,
                solution_kind=solution_kind,
            )
            print(
                f"# GENETIC REUSE | nonlinear_run=single | "
                f"solution_kind={solution_kind} | "
                f"pareto_rank={int(pareto_index) + 1} | "
                f"candidate_id={evaluation.meta.get('candidate_id', 'n/a')} | "
                f"out_dir={run_output_dir}"
            )
            return message
        def _launch_all_pareto_nonlinear_runs(evaluations, solution_kind="reused"):
            total = len(evaluations)
            completed = 0
            failed = 0
            failed_runs = []
            for idx, evaluation in enumerate(evaluations):
                candidate_id = evaluation.meta.get("candidate_id", "n/a")
                try:
                    _launch_pareto_nonlinear_run(evaluation, idx, solution_kind=solution_kind)
                    completed += 1
                except Exception as exc:
                    failed += 1
                    failed_runs.append(
                        f"solution_kind={solution_kind}, pareto_rank={idx + 1}, candidate_id={candidate_id}, error={exc}"
                    )
                    print(
                        f"# GENETIC REUSE | nonlinear_run_all_failed | "
                        f"solution_kind={solution_kind} | pareto_rank={idx + 1} | candidate_id={candidate_id} | error={exc}"
                    )
                print(
                    f"# GENETIC REUSE | nonlinear_run_all_progress={idx + 1}/{total} | "
                    f"solution_kind={solution_kind} | candidate_id={candidate_id} | completed={completed} | failed={failed}"
                )
            if failed_runs:
                with open(
                    _build_output_path(pareto_nl_runs_dir, f"run_all_failures_{solution_kind}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for row in failed_runs:
                        f.write(f"{row}\n")
            if failed:
                return (
                    f"Completed {completed}/{total} {solution_kind} nonlinear runs in {pareto_nl_runs_dir}. "
                    f"Failed runs: {failed}."
                )
            return f"Completed nonlinear runs for {total} Pareto {solution_kind} solutions in {pareto_nl_runs_dir}"
    
        def _launch_pareto_new_reference_run(evaluation, pareto_index):
            return _launch_pareto_nonlinear_run(evaluation, pareto_index, solution_kind="new")
    
        def _launch_all_pareto_new_reference_runs(evaluations):
            return _launch_all_pareto_nonlinear_runs(evaluations, solution_kind="new")

        def _launch_fixed_base_reference_run():
            fixed_reference = getattr(genetic_ctx, "fixed_reference", None)
            if not isinstance(fixed_reference, dict):
                raise RuntimeError("Fixed-base reference is not available for nonlinear analysis.")
            run_index = int(pareto_nl_run_state["next_id"])
            pareto_nl_run_state["next_id"] = run_index + 1
            fixed_evaluation = _build_fixed_reference_evaluation(fixed_reference)
            folder_name = _candidate_run_folder_name(
                fixed_evaluation,
                pareto_index=0,
                run_index=run_index,
                solution_kind="fixed_base",
            )
            run_output_dir = _build_output_path(pareto_nl_runs_dir, folder_name)
            message = _run_candidate_gmna_from_evaluation(
                fixed_evaluation,
                pareto_index=0,
                run_output_dir=run_output_dir,
                solution_kind="fixed_base",
            )
            print(
                f"# GENETIC REUSE | nonlinear_run=fixed_base | "
                f"solution_kind=fixed_base | out_dir={run_output_dir}"
            )
            return message
    
        displayed_pareto_front = _displayed_pareto_front(genetic_result)
        if (
            bool(genetic_multi_obj)
            and bool(genetic_select_pareto_for_nl)
            and len(displayed_pareto_front) > 1
        ):
            while True:
                displayed_pareto_front = _displayed_pareto_front(genetic_result)
                pareto_selection = select_pareto_solution_gui(
                    displayed_pareto_front,
                    genetic_result.all_evaluations,
                    genetic_result.objective_names,
                    title="Select Pareto solution for final analysis",
                    on_run_selected=_launch_pareto_nonlinear_run,
                    on_run_all=_launch_all_pareto_nonlinear_runs,
                    on_run_reference_selected=_launch_pareto_new_reference_run,
                    on_run_reference_all=_launch_all_pareto_new_reference_runs,
                    on_run_fixed_base=(
                        _launch_fixed_base_reference_run
                        if comparison_reference_mode == "fixed_base"
                        else None
                    ),
                    save_pareto_only_path=genetic_pareto_only_plot,
                    save_parallel_coords_path=genetic_parallel_coords_plot,
                )
                if pareto_selection is None:
                    raise RuntimeError("Pareto selection was cancelled by the user.")
                if pareto_selection.continue_requested:
                    extra_generations = int(pareto_selection.continue_generations or 1)
                    try:
                        genetic_settings, genetic_result, genetic_best = _continue_live_genetic_run(
                            genetic_ctx,
                            genetic_settings,
                            genetic_result,
                            extra_generations,
                        )
                    except Exception as exc:
                        print(
                            "# GENETIC REUSE | continue_generations=failed | "
                            f"error={type(exc).__name__}: {exc} | reopening_previous_gui=1",
                            flush=True,
                        )
                    continue
                if pareto_selection.run_all_executed:
                    run_kind = str(getattr(pareto_selection, "run_all_solution_kind", "reused") or "reused").strip().lower()
                    if run_kind == "new":
                        message = _launch_all_pareto_new_reference_runs(displayed_pareto_front)
                    else:
                        message = _launch_all_pareto_nonlinear_runs(displayed_pareto_front)
                    print(
                        f"# GENETIC REUSE | nonlinear_run_all=completed | "
                        f"solution_kind={run_kind} | "
                        f"message={message}"
                    )
                    continue
                genetic_best = pareto_selection.evaluation
                selected_pareto_index = int(pareto_selection.index)
                print(
                    f"# GENETIC REUSE | nonlinear_selection=pareto_gui | "
                    f"pareto_rank={selected_pareto_index + 1} | "
                    f"candidate_id={genetic_best.meta.get('candidate_id', 'n/a')}"
                )
                break
        if (
            genetic_best is None
            or genetic_best.moved_G is None
            or genetic_best.moved_M is None
            or genetic_best.section_by_member is None
        ):
            raise RuntimeError("Genetic optimization did not produce a valid final design.")
    
        G = genetic_best.moved_G
        M = genetic_best.moved_M
    
        section_by_member = {mk: dict(sec) for mk, sec in genetic_best.section_by_member.items()}
        section_name_by_member = {
            mk: str(sec.get("name", ""))
            for mk, sec in section_by_member.items()
        }
        props_global = None
        if genetic_best.opt_result is not None:
            props_global = genetic_best.opt_result.get("props_global", None)
        if len(section_by_member) == len(M.get("members", [])) and len({name for name in section_name_by_member.values() if name}) <= 1:
            first_sec = next(iter(section_by_member.values()), None)
            if first_sec is not None:
                props_global = dict(first_sec)
        reuse_result = genetic_best.reuse_result if genetic_matching_reuse else None
    
        if genetic_matching_reuse and reuse_result is not None:
            write_sections_used_catalog(
                section_by_member,
                csv_path=reuse_sections_used_csv,
                gamma_steel=gamma_steel,
                units_out=units_out,
            )
            print(
                f"# GENETIC REUSE | using_selected_candidate_model=1 | "
                f"candidate_id={genetic_best.meta.get('candidate_id', 'n/a')} | "
                f"matched={len(reuse_result.get('reused_records', []) or [])} | "
                f"unmatched={len(reuse_result.get('unmatched_members', []) or [])} | "
                f"remaining_stock={reuse_result.get('remaining_stock_count', 'n/a')}"
            )
        else:
            reuse_result = None
    
        write_nodes_to_txt(G, _build_output_path(output_dir, "nodes.txt"))
        write_members_to_txt(G, M, _build_output_path(output_dir, "members.txt"))
        export_dxf(G, M, path=dxf_export, print_summary=False)
    
        with open(genetic_history_json, "w", encoding="utf-8") as fobj:
            json.dump(genetic_result.history, fobj, indent=2)
        with open(genetic_summary_json, "w", encoding="utf-8") as fobj:
            json.dump(
                {
                    "method": genetic_result.method,
                    "objective_mode": genetic_objective_mode,
                    "objective_names": list(genetic_result.objective_names),
                    "sustainability_factors": {
                        "ecf_new": float(ecf_new),
                        "ecf_reused": float(ecf_reused),
                        "ecf_work": float(ecf_work),
                        "ecf_scrap": float(ecf_scrap),
                    },
                    "cost_factors": {
                        "cost_new": float(cost_new),
                        "cost_reused": float(cost_reused),
                        "cost_work": float(cost_work),
                        "cost_scrap": float(cost_scrap),
                    },
                    "ga_constraint_defaults": {
                        "dcr_limit": float(genetic_dcr_limit),
                        "disp_limit": float(genetic_disp_limit),
                        "buckling_lambda_min": float(genetic_buckling_lambda_min),
                        "merchant_rankine_min": (
                            float(genetic_merchant_rankine_min) if genetic_use_merchant_rankine_constraint else None
                        ),
                    },
                    "base_mechanical_optimizer": {
                        "mecc_opt": int(mecc_opt),
                        "start_section_row": (None if section_num is None else int(section_num)),
                        "opt_max_iter": int(opt_max_iter),
                        "opt_dcr_limit": float(opt_dcr_limit),
                        "opt_check_disp": bool(check_disp),
                        "opt_dlim": float(opt_dlim),
                        "opt_check_buckling": bool(opt_check_buckling),
                        "opt_buckling_lambda_min": float(opt_buckling_lambda_min),
                        "opt_buckling_mode_index": int(opt_buckling_mode),
                        "opt_check_merchant_rankine": bool(opt_check_merchant_rankine),
                        "opt_merchant_rankine_min": (
                            float(opt_merchant_rankine_min) if opt_check_merchant_rankine else None
                        ),
                        "opt_merchant_rankine_mode_index": (
                            int(opt_merchant_rankine_mode) if opt_merchant_rankine_mode is not None else int(opt_buckling_mode)
                        ),
                        "opt_buckling_retries": int(opt_buckling_retries),
                        "opt_buckling_group_limit": (
                            None if opt_buckling_group_limit is None else int(opt_buckling_group_limit)
                        ),
                        "opt_section_group_limit": (
                            None if opt_section_group_limit is None else int(opt_section_group_limit)
                        ),
                        "opt_section_symmetry": opt_section_symmetry,
                        "opt_demand_grouping": bool(opt_demand_grouping),
                        "opt_demand_group_rel_tol": float(opt_demand_group_rel_tol),
                    },
                    "ga_mechanical_optimizer": {
                        "mecc_opt": int(genetic_opt_mech),
                        "start_section_row": (
                            None if genetic_opt_start_section_row is None else int(genetic_opt_start_section_row)
                        ),
                        "opt_max_iter": int(genetic_opt_max_iter),
                        "opt_dcr_limit": float(genetic_opt_dcr_limit),
                        "opt_check_disp": bool(genetic_opt_check_disp),
                        "opt_dlim": float(genetic_opt_dlim),
                        "opt_check_buckling": bool(genetic_opt_check_buckling),
                        "opt_buckling_lambda_min": float(genetic_opt_buckling_lambda_min),
                        "opt_buckling_mode_index": int(genetic_opt_buckling_mode),
                        "opt_check_merchant_rankine": bool(genetic_opt_check_merchant_rankine),
                        "opt_merchant_rankine_min": (
                            float(genetic_opt_merchant_rankine_min) if genetic_opt_check_merchant_rankine else None
                        ),
                        "opt_merchant_rankine_mode_index": (
                            int(genetic_opt_merchant_rankine_mode)
                            if genetic_opt_merchant_rankine_mode is not None
                            else int(genetic_opt_buckling_mode)
                        ),
                        "opt_buckling_retries": int(genetic_opt_buckling_retries),
                        "opt_buckling_group_limit": (
                            None
                            if genetic_opt_buckling_group_limit is None
                            else int(genetic_opt_buckling_group_limit)
                        ),
                        "opt_section_group_limit": (
                            None
                            if genetic_opt_section_group_limit is None
                            else int(genetic_opt_section_group_limit)
                        ),
                        "opt_section_symmetry": genetic_opt_section_symmetry,
                        "opt_demand_grouping": bool(genetic_opt_demand_grouping),
                        "opt_demand_group_rel_tol": float(genetic_opt_demand_group_rel_tol),
                    },
                    "ga_controlled_ranges": {
                        "ga_dcr_limit": {
                            "enabled": bool(genetic_control_ga_dcr_limit),
                            "range": list(_parse_ga_range(genetic_control_ga_dcr_limit_range, (genetic_dcr_limit, genetic_dcr_limit))),
                        },
                        "ga_disp_limit": {
                            "enabled": bool(genetic_control_ga_disp_limit),
                            "range": list(_parse_ga_range(genetic_control_ga_disp_limit_range, (genetic_disp_limit, genetic_disp_limit))),
                        },
                        "ga_buckling_lambda_min": {
                            "enabled": bool(genetic_control_ga_buckling_limit),
                            "range": list(_parse_ga_range(genetic_control_ga_buckling_limit_range, (genetic_buckling_lambda_min, genetic_buckling_lambda_min))),
                        },
                        "ga_merchant_rankine_min": {
                            "enabled": bool(genetic_control_ga_mr_limit),
                            "range": list(_parse_ga_range(genetic_control_ga_mr_limit_range, (genetic_merchant_rankine_min, genetic_merchant_rankine_min))),
                        },
                        "reuse_max_cut_length": {
                            "enabled": bool(genetic_control_reuse_max_cut_length),
                            "range": list(_parse_ga_range(genetic_control_reuse_max_cut_length_range, (0.0, 10.0))),
                        },
                        "opt_dcr_limit": {
                            "enabled": bool(genetic_control_opt_dcr_limit),
                            "range": list(_parse_ga_range(genetic_control_opt_dcr_limit_range, (genetic_opt_dcr_limit, genetic_opt_dcr_limit))),
                        },
                        "opt_dlim": {
                            "enabled": bool(genetic_control_opt_dlim),
                            "range": list(_parse_ga_range(genetic_control_opt_dlim_range, (genetic_opt_dlim, genetic_opt_dlim))),
                        },
                        "opt_buckling_lambda_min": {
                            "enabled": bool(genetic_control_opt_buckling_limit),
                            "range": list(_parse_ga_range(genetic_control_opt_buckling_limit_range, (genetic_opt_buckling_lambda_min, genetic_opt_buckling_lambda_min))),
                        },
                        "opt_merchant_rankine_min": {
                            "enabled": bool(genetic_control_opt_mr_limit),
                            "range": list(_parse_ga_range(genetic_control_opt_mr_limit_range, (genetic_opt_merchant_rankine_min, genetic_opt_merchant_rankine_min))),
                        },
                    },
                    "merchant_rankine_constraint_enabled": bool(genetic_use_merchant_rankine_constraint),
                    "merchant_rankine_min": (float(genetic_merchant_rankine_min) if genetic_use_merchant_rankine_constraint else None),
                    "merchant_rankine_mode_index": (
                        int(genetic_merchant_rankine_mode) if genetic_merchant_rankine_mode is not None else int(genetic_buckling_mode)
                    ),
                    "genetic_algorithm_settings": {
                        "engine": str(genetic_engine),
                        "population_size": int(genetic_population),
                        "n_generations": int(genetic_generations),
                        "n_offsprings": (None if genetic_n_offsprings is None else int(genetic_n_offsprings)),
                        "seed": (None if genetic_seed is None else int(genetic_seed)),
                        "sampling": str(genetic_sampling),
                        "eliminate_duplicates": bool(genetic_eliminate_duplicates),
                        "crossover_prob": float(genetic_crossover_prob),
                        "crossover_eta": float(genetic_crossover_eta),
                        "mutation_prob": float(genetic_mutation_prob),
                        "mutation_eta": float(genetic_mutation_eta),
                        "mutation_scale_builtin": float(genetic_mutation_scale),
                        "verbose": int(genetic_verbose),
                    },
                    "n_evaluations": genetic_result.n_evaluations,
                    "cache_hits": genetic_result.cache_hits,
                    "selected_from_pareto_gui": bool(selected_pareto_index is not None),
                    "selected_pareto_rank": (None if selected_pareto_index is None else int(selected_pareto_index + 1)),
                    "selected_candidate_id": genetic_best.meta.get("candidate_id", None),
                    "feasible": bool(genetic_best.feasible),
                    "objectives": genetic_best.objectives,
                    "constraints": genetic_best.constraints,
                    "lambda_ref": genetic_best.lambda_ref,
                    "meta": genetic_best.meta,
                    "decision_vector": list(genetic_best.x),
                    "pareto_front_size": len(_displayed_pareto_front(genetic_result)),
                    "pareto_front_final_population_size": len(genetic_result.pareto_front),
                    "pareto_front_global_size": len(genetic_result.pareto_front_global),
                },
                fobj,
                indent=2,
            )
        displayed_pareto_front = _displayed_pareto_front(genetic_result)
        export_pareto_front_json(genetic_pareto_json, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_csv, displayed_pareto_front, genetic_result.objective_names)
        export_pareto_front_json(genetic_pareto_global_json, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_pareto_front_csv(genetic_pareto_global_csv, genetic_result.pareto_front_global, genetic_result.objective_names)
        export_candidate_metrics_csv(genetic_pareto_candidate_metrics_csv, genetic_ctx, displayed_pareto_front)
        pareto_metrics_parallel_plot = save_pareto_metrics_parallel_coords_plot(
            pareto_front=displayed_pareto_front,
            save_path=genetic_pareto_metrics_parallel_coords_plot,
        )
        candidate_parallel_plot = save_candidate_metrics_parallel_coords_plot(
            all_evaluations=genetic_result.all_evaluations,
            save_path=genetic_parallel_coords_plot,
        )
        fixed_parallel_plot = None
        if comparison_reference_mode == "fixed_base":
            fixed_parallel_plot = save_pareto_fixed_base_parallel_coords_plot(
                pareto_front=displayed_pareto_front,
                fixed_reference=getattr(genetic_ctx, "fixed_reference", None),
                save_path=genetic_pareto_fixed_parallel_coords_plot,
            )
        takeoff_export = _export_pareto_material_takeoff_set(
            _pareto_front_for_material_takeoffs(genetic_ctx, displayed_pareto_front),
            genetic_material_takeoff_dir,
            solution_kind="reused",
        )
        pareto_plot_written = plot_pareto_front(
            genetic_pareto_plot,
            displayed_pareto_front,
            genetic_result.objective_names,
            all_evaluations=genetic_result.all_evaluations,
        )
        print(
            f"# GENETIC REUSE | method={genetic_result.method} | "
            f"mode={genetic_objective_mode} | "
            f"objectives={','.join(genetic_result.objective_names)} | "
            f"evaluations={genetic_result.n_evaluations} | "
            f"pareto_final={len(genetic_result.pareto_front)} | "
            f"pareto_global={len(genetic_result.pareto_front_global)} | "
            f"feasible={bool(genetic_best.feasible)} | "
            f"mass={float(genetic_best.objectives.get('mass', float('inf'))):.6f} | "
            f"cost={float(genetic_best.objectives.get('cost', float('inf'))):.6f} | "
            f"reuse_ratio={float(genetic_best.objectives.get('reuse_ratio', 0.0)):.6f}"
        )
        if pareto_plot_written:
            print(f"# GENETIC REUSE | pareto_plot={genetic_pareto_plot}")
        if pareto_metrics_parallel_plot:
            print(f"# GENETIC REUSE | pareto_metrics_parallel_coords_plot={pareto_metrics_parallel_plot}")
        if candidate_parallel_plot:
            print(f"# GENETIC REUSE | parallel_coords_plot={candidate_parallel_plot}")
        if fixed_parallel_plot:
            print(f"# GENETIC REUSE | fixed_parallel_coords_plot={fixed_parallel_plot}")
        print(f"# GENETIC REUSE | pareto_final_json={genetic_pareto_json}")
        print(f"# GENETIC REUSE | pareto_final_csv={genetic_pareto_csv}")
        print(f"# GENETIC REUSE | pareto_global_json={genetic_pareto_global_json}")
        print(f"# GENETIC REUSE | pareto_global_csv={genetic_pareto_global_csv}")
        print(
            f"# GENETIC REUSE | material_takeoffs={takeoff_export['output_dir']} | "
            f"exported={takeoff_export['exported']} | failed={takeoff_export['failed']}"
        )
        if genetic_matching_reuse and reuse_result is not None:
            print(
                f"# REUSE | matched={len(reuse_result['reused_records'])} | "
                f"unmatched={len(reuse_result['unmatched_members'])} | "
                f"remaining_stock={reuse_result['remaining_stock_count']}"
            )
    #---------------------------------------------------------------------------------------------------------------------------------------------       
    #==============================================================SECTION ASSIGNMENT=============================================================
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------structural optimization-----------------------------------------------------------
    if not use_genetic_solution and optimization:
        opt_result = optimize_sections_ec3(G=G,
                                            M=M,
                                            mecc_opt=mecc_opt,
                                            material=material_el,
                                            catalog=catalog,
                                            units_in=units_in,
                                            units_out=units_out,
                                            c1=c1,
                                            c2=c2,
                                            c3=c3,
                                            c4=c4,
                                            c5=c5,
                                            q_load=load_el,
                                            ndm=ndm,
                                            ndf=ndf,
                                            init_model=init_model,
                                            define_nodes=define_nodes,
                                            apply_costr=apply_costr,
                                            el_ele=el_ele,
                                            apply_loads=sf_load_as_nodal_xy_projection,
                                            run_static=run_elastic_static_analysis,
                                            initial_section_name=s_1,
                                            start_from_first_section=False,
                                            start_section_row=section_num,
                                            dcr_limit=opt_dcr_limit,
                                            dlim=opt_dlim,
                                            check_disp=check_disp,
                                            check_buckling=opt_check_buckling,
                                            buckling_lambda_min=opt_buckling_lambda_min,
                                            buckling_mode_index=opt_buckling_mode,
                                            check_merchant_rankine=opt_check_merchant_rankine,
                                            merchant_rankine_min=opt_merchant_rankine_min,
                                            merchant_rankine_mode_index=opt_merchant_rankine_mode,
                                            buckling_max_retries=opt_buckling_retries,
                                            buckling_group_limit=opt_buckling_group_limit,
                                            gamma_M0=gamma_M0,
                                            gamma_M1=gamma_M1,
                                            gamma_M2=gamma_M2,
                                            beta=beta,
                                            alpha_LT=alpha_LT,
                                            compression_sign=compression_sign,
                                            max_iter=opt_max_iter,
                                            subdivide=1,
                                            first_new_node_tag=2000,
                                            update_G=False,
                                            area_mode="faces",
                                            opt_verbose=opt_verbose,)
        props_global = opt_result.get("props_global", None)
        section_by_member = opt_result["section_by_member"]
        section_name_by_member = opt_result["section_name_by_member"]
        print(
            f"# SECTION SETUP | mode=optimize | mecc_opt={mecc_opt} | "
            f"DCR_limit={opt_dcr_limit} | "
            f"Uz_limit={opt_dlim} | "
            f"buckling_check={opt_check_buckling} | "
            f"lambda_min={opt_buckling_lambda_min} | "
            f"buckling_mode={opt_buckling_mode} | "
            f"MR_check={opt_check_merchant_rankine} | "
            f"MR_min={opt_merchant_rankine_min if opt_check_merchant_rankine else 'off'} | "
            f"MR_mode={(opt_merchant_rankine_mode if opt_merchant_rankine_mode is not None else opt_buckling_mode)}")
    #---------------------------------------------------------------------------------------------------------------------------------------------   
    elif not use_genetic_solution:
        props_global, section_by_member, section_name_by_member = resolve_section_definition(G=G,                                                                                            
                                                                                            M=M,                                                                                              
                                                                                            catalog=catalog,                                                                               
                                                                                            section_type=section_type,                                                                    
                                                                                            s_1=s_1,                                                                                            
                                                                                            units_in=units_in,                                                                              
                                                                                            units_out=units_out,                                                                             
                                                                                            assignment_json_path=section_assignment_json,                                                     
                                                                                            assignment_report_path=section_assignment_report,                                                  
                                                                                            selector_title="Section assignment by member",                                                      
                                                                                            show_member_labels=section_selector_show_labels,                                                    
                                                                                            print_summary=True,)
    
    if section_by_member is None and props_global is not None:
        section_by_member = {
            ((int(na), int(nb)) if int(na) < int(nb) else (int(nb), int(na))): dict(props_global)
            for (na, nb) in M.get("members", [])}
        section_name_by_member = {
            mk: str(props_global.get("name", s_1))
            for mk in section_by_member}
    #---------------------------------------------------------------------------------------------------------------------------------------------
    if not use_genetic_solution and reuse_matching_enabled:
        reuse_result = match_reused_members(G=G,
                                            M=M,
                                            section_by_member=section_by_member,
                                            reuse_csv_path=reuse_csv_path,
                                            section_units_in=units_in,
                                            units_out=units_out,
                                            length_units_in=reuse_length_units_in,
                                            max_cut_length=reuse_max_cut_length,
                                            min_remainder_length=reuse_min_remainder_length,
                                            group_sizes=reuse_group_sizes,
                                            remaining_csv_path=reuse_remaining_csv,
                                            report_txt_path=reuse_report_txt,
                                            verbose=True,)
        
        section_by_member = reuse_result["section_by_member"]
        section_name_by_member = reuse_result["section_name_by_member"]
    
        write_sections_used_catalog(section_by_member,
                                    csv_path=reuse_sections_used_csv,
                                    gamma_steel=gamma_steel,
                                    units_out=units_out,)
        catalog = load_section_catalog(reuse_sections_used_csv, sep=";", encoding="utf-8", name_col_index=0)
        section_by_member = {
            mk: get_section_properties(catalog=catalog,
                                        name=sec_name,
                                        units_in=units_in,
                                        units_out=units_out,
                                        print_props=False,)
            for mk, sec_name in section_name_by_member.items()}
        candidate_sections = get_section_properties(catalog,
                                                    units_in=units_in,
                                                    units_out=units_out,
                                                    print_props=False,
                                                    print_all=False,)
        props_global = None
        print(
            f"# REUSE | matched={len(reuse_result['reused_records'])} | "
            f"unmatched={len(reuse_result['unmatched_members'])} | "
            f"remaining_stock={reuse_result['remaining_stock_count']}")
    
    props_ref = props_global if props_global is not None else next(iter(section_by_member.values()))        #Reference section for quick checks
    A  = props_ref.get("A")                                                                                 #Area
    Iy = props_ref.get("Iy")                                                                                #Second moment y
    Iz = props_ref.get("Iz")                                                                                #Second moment z
    Wy = props_ref.get("Wy")                                                                                #Elastic module y
    Wz = props_ref.get("Wz")                                                                                #Elastic module z
    Wply = props_ref.get("Wply")                                                                            #Plastic module y
    Wplz = props_ref.get("Wplz")                                                                            #Plastic module z
    J  = props_ref.get("J")                                                                                 #Torsional constant
    #---------------------------------------------------------------------------------------------------------------------------------------------
    section_by_member_plot = section_by_member
    if section_by_member_plot is None and props_global is not None:
        section_by_member_plot = {
            ((int(na), int(nb)) if int(na) < int(nb) else (int(nb), int(na))): dict(props_global)
            for (na, nb) in M.get("members", [])}
    
    plot_section_assignment_plan(G,
                                M,
                                section_by_member_plot,
                                show_nodes=False,
                                show_labels=False,
                                save_path=_build_output_path(output_dir, "section_assignment_plan.png"),
                                show=plot_sec_plan,)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #======================================================1st LINEAR ELASTIC STRUCTURAL MODEL====================================================
    #---------------------------------------------------------------------------------------------------------------------------------------------
    os_start = init_model(ndm, ndf)                                                          # Start model
    if writer_os:                                                                            # Toggle
        W_el.add(os_start)                                                                   # Writer
    #--------------------------------------------------------------Structural nodes---------------------------------------------------------------
    os_nodes, node_tags = define_nodes(G, ndf, set_mass=False)                               # Node definition
    if writer_os:                                                                            # Toggle
        W_el.add(os_nodes)                                                                   # Writer
    #-----------------------------------------------------------Structural constraints------------------------------------------------------------
    os_fix, fixed = apply_costr(G, ndf, c1, c2, c3, c4, c5)                                  # Constraints definition
    if writer_os:                                                                            # Toggle
        W_el.add(os_fix)                                                                     # Writer
    #---------------------------------------------------------Structural Elastic elements---------------------------------------------------------
    os_ele, ele_data = el_ele(G=G,                                                           # Geometry
                              M=M,                                                           # Mesh                       
                              material=material_el,                                          # Associated material
                              section=props_global,                                          # Associated sections (1 section)
                              section_by_member=section_by_member,                           # Associated sections (Multiple section)
                              section_rotation_opts=_build_fiber_section_opts(               # Section builder
                                  fiber_n_circ,                                              # info for circ
                                  fiber_n_rad,                                               # info for circ
                                  fiber_L,                                                   # Rotation for L
                                  fiber_2L,                                                  # info for 2L
                                  fiber_4L,),                                                # info for 4L
                              transf_type="PDelta",                                          # Associated transf
                              subdivide = 1,                                                 # nÂ° of subdvision x ele
                              first_new_node_tag = 2000,                                     # new node id start
                              update_G = False,                                              # Update node id
                              print_ops = False,                                             # Print code
                              print_mode = "headtail",                                       # print all/headtail
                              n_preview=3,                                                   # number of headtail row to be printed
                              print_summary=False)                                           # Summary
    if writer_os:                                                                            # Toggle
        W_el.add(os_ele)                                                                     # Writer
    #-----------------------------------------------------------Plot structural model-------------------------------------------------------------
    if plot_opvis:                                                                           # Opvis-openseespy
        plot_model_3d(node_labels=0,                                                         # Node labels default off
                      element_labels=0,                                                      # Element labels default offf
                      local_axes=True,                                                       # Local axes default off
                      node_supports=True,)                                                   # Node support default on 
    #-------------------------------------------------------------Apply self weigth---------------------------------------------------------------
    q_ele, os_q_ele = apply_selfweight(G=G,                                                   # Geometry
                                       ele_data=ele_data,                                     # Elements
                                       mode="eleload",                                        # element
                                       section=props_global,                                  # None se section_type="user"
                                       section_by_member=section_by_member,                   # None se section_type="1forall"
                                       catalog=catalog,                                       # catalogue
                                       units_in=units_in,                                     # units
                                       units_out=units_out,                                   # units
                                       pattern_tag=21,                                        # PatternTag
                                       ts_tag=21,)                                            # Timeseries
    if writer_os:                                                                             # toggle
        W_el.add(os_q_ele)                                                                    # Writer 
    #---------------------------------------------------------------Calculate k0------------------------------------------------------------------
    K0, os_K0 = get_tangent_stiffness(system="FullGeneral",                                   # System
                                      numberer="RCM",                                         # Numberer
                                      constraints="Plain",                                    # Constraints
                                      algorithm="Newton",                                     # Algorithm
                                      print_ops=False)                                        # Print code
    #-------------------------------------------------------------1st load condition--------------------------------------------------------------
    loads, areas, os_loads = sf_load_as_nodal_xy_projection(G,                                # Geometry
                                                            M=M,                              # Mesh
                                                            q=load_el,                        # load
                                                            direction="-Z",                   # Global Direction
                                                            pattern_tag=2,                    # Pattern tag
                                                            ts_tag=2,                         # TimeSeriesTag
                                                            ndf=6,                            # degreeoffreedom
                                                            area_mode="faces",                # use projected triangular faces for geom=4
                                                            dx=None,                          # dx
                                                            dy=None,                          # dy
                                                            round_decimals=8,                 # Decimals
                                                            print_ops=False,                  # Print code preview
                                                            print_mode="all",                 # "headtail"
                                                            n_preview=10,                     # nÂ° preview rows
                                                            print_summary=True,               # Summary
                                                            print_components=False,           # Summary components
                                                            plot_loads=save_load_map or show_load_map, # Print loads
                                                            plot_filename=(_build_output_path(output_dir, "applied_load_map_elastic.png") if save_load_map else None), # Plot name
                                                            plot_show=show_load_map,          # Plot loads
                                                            plot_title=None,                  # Plot title
                                                            plot_s=30,                        # Plot dim
                                                            plot_show_labels=False,           # Plot show
                                                            plot_label_fmt="{:.2f}",          # Plot labels
                                                            plot_label_offset=(0.01, 0.01),   # Plot off
                                                            plot_label_fontsize=8,            # Plot font
                                                            plot_label_nodes=None,            # Plot label nodes
                                                            plot_component="Fz",              # Fx,Fy,Fz
                                                            diag_check_uniformity=False,      # Check
                                                            diag_tol_cv=0.02,)                # Tol
    if writer_os:                                                                             # Toggle
        W_el.add(os_loads)                                                                    # Writer
    #-----------------------------------------------------------Elastic Static Analysis-----------------------------------------------------------
    os_static, ok = run_elastic_static_analysis(load_control=1.0,                                             # Default = 1
                                                    analyze_steps=1,                                          # Default = 1
                                                    set_load_const=False,                                     # Default = True
                                                    print_ops=False,                                          # print opensees.py code
                                                    print_mode="headtail",                                    # all/headtail
                                                    n_preview=10,                                             # nÂ° headtail row
                                                    print_summary=False)                                      # summary
    if writer_os:                                                                                             # Toggle
            W_el.add(os_static)                                                                               # Writer opensees.py code
            W_el.save(_build_output_path(output_dir, "opensees_EL_model_export.py"), as_py=False)             #Save writer opensees.py code
    #-------------------------------------------------------Elastic Static Analysis Results-------------------------------------------------------
    res = extract_results(G,
                        M,
                        ele_data=ele_data,
                        ndf=ndf,
                        uz_dof=3,
                        uz_ref=uz_limit,)
    plot_nodal_displacement_plan(G,
                                res.get("disp", {}),
                                M=M,
                                save_path=_build_output_path(output_dir, "nodal_displacement_plan.png"),
                                show=False,
                                component="mag",
                                dlim=opt_dlim,)
    
    write_force_report(res["ele_forces"],                                         #Results
                                filepath=_build_output_path(output_dir, "force_report.txt"),                  #output file name
                                compression_sign=compression_sign,                    #sign compression
                                sort_by="N_comp_max",                                 #sorted by
                                descending=True,)                                     #descending or not
    
    write_member_force_report(res["ele_forces"],                                  #Results
                                ele_data=ele_data,                                    #data
                                filepath=_build_output_path(output_dir, "member_force_report.txt"),           #output file name
                                compression_sign=compression_sign,                    #sign compression
                                sort_by="N_comp_max",                                 #sorted by
                                descending=True,)                                     #descending or not
    
    print_results_summary(res,                                                    #results
                            uz_dof=3,                                                 #direction for displ.
                            print_summary=True)                                       #print results summary
    
    if plot_def:                                                                  #Toggle
            plot_deformed_mesh_mpl(G,                                                 #Geometry
                                M,                                                    #Mesh
                                sf=200,                                               #scale factor
                                mode="members",                                       #members, subelements
                                show_undeformed=True,                                 #show undeformed shape
                                ele_data=ele_data,)                                   #element
    
    if diagram_force:                                                             #Toggle
            write_elastic_force_report(res["ele_forces"],                             #results
                                    filepath=_build_output_path(output_dir, "elastic_force_report.txt"),      #path
                                    sort_by="element",)                               #sorted by
    
            plot_elastic_force_suite(res["ele_forces"],                               #Results
                                    components=("N", "My", "Mz", "Vy", "Vz", "T"),    #Components
                                    sort_by="element",                                #sorted by
                                    annotate_top_n=0,                                 #annotate
                                    show=False,                                       #show
                                    out_dir=output_dir,                               #direction
                                    save_prefix="Elastic_force",)                     #name      
    #----------------------------------------------------------------EC3 check--------------------------------------------------------------------
    ec3_results, ec3_dcr_max, sections_by_key, lengths_by_key = run_ec3_check(G=G,                                          #Geometry
                                                                ele_data=ele_data,                                          #Elements
                                                                ele_forces=res["ele_forces"],                               #Results list
                                                                material=material_el,                                       #material props
                                                                section=props_global,                                       #section props
                                                                section_by_member=section_by_member,                        #section props
                                                                gamma_M0=gamma_M0,                                          #gammaM0
                                                                gamma_M1=gamma_M1,                                          #gammaM1
                                                                gamma_M2=gamma_M2,                                          #gammaM2
                                                                beta=beta,                                                  #beta factor
                                                                alpha_LT=alpha_LT,                                          #alfaLT
                                                                compression_sign=compression_sign,                          #compression sign
                                                                aggregate_by_member=True,                                   #aggregate
                                                                print_summary=False,                                        #summary
                                                                top_n=20,                                                   #nÂ°summ
                                                                save_txt=_build_output_path(output_dir, "checks.txt"),      #name
                                                                plot_dcr=True,                                              #plot
                                                                plot_mode="members",                                        #members,subelements
                                                                plot_show=False,                                            #show plot
                                                                plot_save_path=_build_output_path(output_dir, "DCR.png"),   #name plot
                                                                plot_n_colors=10,)                                          #nÂ°color scale
    plot_ec3_dcr_plan(G,
                    M,
                    ec3_results,
                    save_path=_build_output_path(output_dir, "DCR_plan.png"),
                    show=False,
                    title=f"EC3 DCR - Plan View | max={ec3_dcr_max:.3f}",)
    print("# SUMMARY | EC3 DCR max =", ec3_dcr_max)                                                                         #DCR summary
    #----------------------------------------------------------Elastic Buckling Analysis----------------------------------------------------------
    lam, vecs = linear_buckling_sparse(K0,                                                         #Initial stiffness
                                        n_modes=5,                                                 #Number of calculated modes
                                        drop_tol=1e-12,                                            #Default value
                                        eig_tol=1e-12,                                             #Default value
                                        tangent_kwargs=dict(system="FullGeneral",                  #Default value
                                                            numberer="RCM",                        #Default value
                                                            constraints="Plain",                   #Default value
                                                            algorithm="Newton"),                   #Default value
                                        print_ops=False,                                           #print opensees.py code
                                        print_summary=True)                                        #print summary
    modes = vectors_to_modes(vecs, ndf, node_tags=node_tags)                                       #save buckling modes
    if plot_buck or use_genetic_solution:                                                          #Toggle
            plot_buckling_modes(G,                                                                 #Geometry
                                M,                                                                 #Mesh
                                modes,                                                             #buckling modes
                                lam=lam,                                                           #buckling lam
                                mode_index=([1, 2, 3] if use_genetic_solution else [1, 2, 3, 4, 5]),     #mode index to be defined
                                scale=0.20,                                                        #scale
                                subplot=False,                                                     #True for subplotting
                                ncols=2,                                                           #subplotting columns
                                n_curve_points=plot_buck_n_curve_points,                           #graphical refinement per member
                                show=False,                                                        #show plots
                                save=True,                                                         #Save plot
                                out_dir=output_dir,                                                #output
                                save_prefix="buckling_mode",                                       #name
                                dpi=150)                                                           #dpi     
    #--------------------------------------------------------------Merchant Rankine---------------------------------------------------------------                                                                        
    mr_factors, mr_lambdas = merchant_rankine_factors_from_results(dcr_max=ec3_dcr_max,             #dcr max
                                                                    lam=lam,                        #lam bguckling analysis
                                                                    n_modes=5,)                     #nÂ° modes
    print(f"# SUMMARY | MR1 = {mr_factors[0]:.6f}  using lambda1 = {mr_lambdas[0]:.6f}")            #MerchantRankine 1
    print(f"# SUMMARY | MR2 = {mr_factors[1]:.6f}  using lambda2 = {mr_lambdas[1]:.6f}")            #MerchantRankine 2
    print(f"# SUMMARY | MR3 = {mr_factors[2]:.6f}  using lambda3 = {mr_lambdas[2]:.6f}")            #MerchantRankine 3
    print(f"# SUMMARY | MR4 = {mr_factors[3]:.6f}  using lambda4 = {mr_lambdas[3]:.6f}")            #MerchantRankine 4
    print(f"# SUMMARY | MR5 = {mr_factors[4]:.6f}  using lambda5 = {mr_lambdas[4]:.6f}")            #MerchantRankine 5
    with open(_build_output_path(output_dir, "Merchant_Rankine.txt"), "w", encoding="utf-8") as f:  #Output 
        f.write(f"DCRmax = {ec3_dcr_max:.6f}\n")                                                    #DCR max
        f.write(f"Linear buckling factor (first) = {mr_lambdas[0]:.6f}\n")                          #Linear buck
        f.write(f"Merchant-Rankine factor = {mr_factors[0]:.6f}\n")                                 #Merchant Rankine  
    #--------------------------------------------------------------------------------------------------------------------------------------------- 
    #==========================================================NON LINEAR ANALYSIS (PUSHDOWN)=====================================================
    #--------------------------------------------------------------------------------------------------------------------------------------------- 
    if non_lin:
        os_start = init_model(ndm, ndf)                                                          #Start model
        if writer_os:                                                                            #Toggle
            W_non_lin.add(os_start)                                                              #Writer
    
    #----------------------------------------------------------------Structural nodes-------------------------------------------------------------
    
        os_nodes, node_tags = define_nodes(G, ndf, set_mass=False)                               #Node definition
        if writer_os:                                                                            #Toggle
            W_non_lin.add(os_nodes)                                                              #Writer
    
    #------------------------------------------------------------Structural constraints-----------------------------------------------------------
    
        os_fix, fixed = apply_costr(G, ndf, c1, c2, c3, c4, c5)                                  #Constraints definition
        if writer_os:                                                                            #Toggle
            W_non_lin.add(os_fix)                                                                #Writer
    
    #--------------------------------------------------------Structural non linear elements-------------------------------------------------------
       
        nl_material = mat2                                                                        #Material for nonlinear model                
        q_nl = load_el                                                                            #Load for nonlinear model    
        fy = nl_material["fyk"]                                                                   #Control node                      
        E  = nl_material["E"]                                                                     #Control node                      
        control_node_data = res["max_vertical"].get("max_abs_nd") or res["max_vertical"]["min"]
        control_node = control_node_data["node"]
                        
    
        if T_fiber_ele == "Lobatto":
            os_ele, ele_data = fb_L_ele(G, M,                                                     # Geometry and mesh
                                      material=nl_material,                                       # Material
                                      section=props_global,
                                      section_by_member=section_by_member,                                      #Single section
                                      transf_type="Corotational",                                 # Geometric transformation
                                      subdivide=4,                                                # Number of sub-elements for each member
                                      first_new_node_tag=2000,                                    # Start tag for generated internal nodes
                                      update_G=False,                                             # Keep original G unchanged
                                      n_integration_pts=lobatto_n_integration_pts,                 # Lobatto integration points
                                      fiber_opts=_build_fiber_section_opts(                       # Fiber mesh/rotation options by family
                                          fiber_n_circ,
                                          fiber_n_rad,
                                          fiber_L,
                                          fiber_2L,
                                          fiber_4L,
                                      ),
                                      first_mat_tag=1,                                            # First uniaxial material tag
                                      first_integ_tag=1,                                          # First beamIntegration tag
                                      first_ele_tag=1,                                            # First element tag
                                      first_sec_tag=1,                                            # First section tag
                                      first_tr_tag=1,                                             # First geometric transformation tag
                                      plot_fiber_sections=plot_fiber_sections,                    # Save fiber section plots
                                      plot_fiber_sections_dir=fiber_section_plot_dir,             # Plot output directory
                                      plot_fiber_sections_show=plot_fiber_sections_show,          # Interactive show
                                      print_ops=False,                                            # Print generated OpenSees commands
                                      print_mode="headtail",                                      # all/headtail
                                      n_preview=3,                                                # Number of preview lines
                                      print_summary=False)                                        # Print summary
    
        elif T_fiber_ele in {"HingeRadau", "HingeRadauTwo"}:
            os_ele, ele_data = fb_H_ele(G, M,                                                     #Geometry and Mesh
                                            material = nl_material,                               #Material
                                            section=props_global,
                                            section_by_member=section_by_member,                                      #Single section
                                            transf_type = "Corotational",                         #Transf type
                                            subdivide = 4,                                        #NÂ° of element subdivision
                                            first_new_node_tag = 2000,                            #New node tag
                                            update_G = False,                                     #Update nodes list (for loads)
                                            lp_ratio = hinge_lp_ratio,                            # ratio at each end of the subelement
                                            integration = T_fiber_ele,
                                            fiber_opts=_build_fiber_section_opts(                 # Fiber mesh/rotation options by family
                                                fiber_n_circ,
                                                fiber_n_rad,
                                                fiber_L,
                                                fiber_2L,
                                                fiber_4L,
                                            ),
                                            first_mat_tag = 1,                                    #material tag
                                            first_integ_tag = 1,                                  #integration tag
                                            first_ele_tag = 1,                                    #first ele tag
                                            first_sec_tag = 1,                                    #first section tag
                                            first_tr_tag = 1,                                     #first transf tag
                                            plot_fiber_sections=plot_fiber_sections,              # Save fiber section plots
                                            plot_fiber_sections_dir=fiber_section_plot_dir,       # Plot output directory
                                            plot_fiber_sections_show=plot_fiber_sections_show,    # Interactive show
                                            print_ops=False,                                      #Print code
                                            print_mode="headtail",                                #all/headtail row
                                            n_preview=3,                                          #nÂ° of headtail rows
                                            print_summary=False)                                  #print summary
            
        if writer_os:                                                                             #Toggle
            W_non_lin.add(os_ele)                                                                 #Writer
    
    #---------------------------------------------------------------Apply self weigth-------------------------------------------------------------
    
        q_ele = apply_selfweight(G=G,
                                ele_data=ele_data,
                                mode="eleload",
                                section=props_global,
                                section_by_member=section_by_member,
                                catalog=catalog,
                                units_in=units_in,
                                units_out=units_out,
                                pattern_tag=21,
                                ts_tag=21,)
    
    #---------------------------------------------------------------1st load condition------------------------------------------------------------
    
        loads, areas, os_loads = sf_load_as_nodal_xy_projection(G,                                           #Geometry
                                                                M=M,                                         #Mesh
                                                                q=q_nl,                                      #loads
                                                                direction="-Z",                              #Z,X,Y
                                                                pattern_tag=2,                               #Pattern tag
                                                                ts_tag=2,                                    #TimeSeries tag
                                                                ndf=6,                                       #degree of freedom
                                                                area_mode="faces",                           #use projected triangular faces for geom=4
                                                                dx=None,                                     #dx
                                                                dy=None,                                     #dy
                                                                round_decimals=8,                            #decimals
                                                                print_ops=False,                             #print code
                                                                print_mode="all",                            #headtail, all
                                                                n_preview=10,                                #nÂ° of row
                                                                print_summary=False,                         #summary
                                                                print_components=False,                      #components
                                                                plot_loads=save_load_map or show_load_map,   #plot loads
                                                                plot_filename=(_build_output_path(output_dir, "applied_load_map_nonlinear.png") if save_load_map else None), #file name
                                                                plot_show=show_load_map,                     #show plot
                                                                plot_title=None,                             #plot title
                                                                plot_s=30,                                   #plot dim
                                                                plot_show_labels=False,                      #plot lables
                                                                plot_label_fmt="{:.2f}",                     #labels
                                                                plot_label_offset=(0.01, 0.01),              #offsets x loads
                                                                plot_label_fontsize=8,                       #font
                                                                plot_label_nodes=None,                       #plot nodes labels
                                                                plot_component="Fz",                         #Fx, Fy, Fz
                                                                diag_check_uniformity=False,                 #check
                                                                diag_tol_cv=0.02,)                           #tol
        
        loads_Fz = {nid: loads[nid][2] for nid in loads}                                                     #Fz total load
    
    #-----------------------------------------------------------Live plot and monitor------------------------------------------------------------
    
        monitor = None                                                             # monitor control toggle
        if live_plot:
            monitor = LivePushoverMonitor(G, M, fy=fy, E=E,
                                          ele_data=ele_data,                       # Geometry, Mesh and material
                                          mode_right="members",                    #members,subelements
                                          show_undeformed=False,
                                          delta_comp="Uz",                         # Delta comp: default Uz
                                          delta_mode="raw",                        # Delta mode 
                                          make_eps_raw_fig=False,                  # Figure only epsilon: default False
                                          eps_abs_max_key="eps_abs_max",           # default
                                          sf_fixed=50,
                                          eps_abs_minmax_key="eps_abs_minmax",)    # default
    
    
        loads_Fz = {nid: loads[nid][2] for nid in loads}                           # Fz total load
    
    #---------------------------------------------------------Analysis control - Force------------------------------------------------------------
    
        if Analysis_control == "Force":
            hist, summ = run_gmna_force_control(G, M,
                                                fy=fy,
                                                E=E,
                                                ele_data=ele_data,
                                                loads_dict=loads_Fz,
                                                ndf=ndf,
                                                pattern_tag=2,
                                                control_node=control_node,
                                                control_dof=3,
                                                elastic_res=res,
                                                control_node_prefer="minUz",
                                                dLambda=0.01,
                                                n_steps=(force_adaptive_max_steps if force_use_adaptive_step else force_solver_max_steps),
                                                constraints="Transformation",
                                                numberer="AMD",
                                                system="UmfPack",
                                                test=("NormUnbalance", 1e-5, 50),
                                                algorithm=("Newton",),
                                                fallback_strategies=(force_fallback_strategies if force_use_fallbacks else None),
                                                force_log_prefix=_build_output_path(output_dir, "GMNA_solver"),
                                                **force_solver_options,
                                                analysis="Static",
                                                monitor_fibers=True,
                                                fiber_ele_tags=ele_data["ele_tags"],
                                                fiber_nIP=ele_data["meta"]["n_ip"],
                                                fiber_sec_nums=None,
                                                T_fiber_ele=T_fiber_ele,
                                                fiber_strain_col=4,
                                                fiber_chunk_size=None,
                                                fiber_area_min=0.0,
                                                fiber_area_col=2,
                                                exclude_neutral_axis=True,
                                                neutral_axis_tol=1e-12,
                                                dump_fibers=False,
                                                dump_fibers_txt=_build_output_path(output_dir, "fiber_debug.txt"),
                                                dump_every=10,
                                                dump_include_full_row=True,
                                                print_ops=False,
                                                print_mode="headtail",
                                                n_preview=12,
                                                print_step_info=True,
                                                print_every=1,
                                                writer=W_non_lin,
                                                live_plot=live_plot,
                                                live_plot_every=1,
                                                live_monitor=monitor,
                                                keyboard_abort=True,
                                                ctrlc_abort=True,
                                                stop_keys=(b"\x1b", b"q", b"Q"),
                                                record_local_response=True,
                                                local_response_map=ele_data.get("local_instability", None),
                                                local_response_verbose=False,)
    
    #--------------------------------------------------------Analysis control - Displacements-----------------------------------------------------
       
        elif Analysis_control == "Displacements":
            hist, summ = run_gmna_disp_control(G=G, M=M,
                                               fy=fy,
                                               E=E,
                                               ele_data=ele_data,
                                               loads_dict=loads_Fz,
                                               ndf=ndf,
                                               pattern_tag=2,
                                               control_node=control_node,
                                               control_dof=3,
                                               elastic_res=res,
                                               control_node_prefer="minUz",
                                               dU=-1e-3,
                                               n_steps=1000,
                                               U_max=1.5,
                                               constraints="Transformation",
                                               numberer="RCM",
                                               system="UmfPack",
                                               test=("NormUnbalance", 1e-5, 50),
                                               algorithm=("Newton",),
                                               analysis="Static",
                                               monitor_fibers=True,
                                               fiber_ele_tags=ele_data["ele_tags"],
                                               fiber_nIP=ele_data["meta"]["n_ip"],
                                               fiber_sec_nums=None,
                                               T_fiber_ele=T_fiber_ele,
                                               fiber_strain_col=4,
                                               fiber_chunk_size=None,
                                               fiber_area_min=0.0,
                                               fiber_area_col=2,
                                               exclude_neutral_axis=True,
                                               neutral_axis_tol=1e-12,
                                               dump_fibers=False,
                                               print_ops=False,
                                               print_mode="headtail",
                                               n_preview=12,
                                               print_step_info=True,
                                               print_every=1,
                                               live_plot=live_plot,
                                               live_plot_every=1,
                                               live_monitor=monitor,
                                               record_local_response=True,
                                               local_response_map=ele_data.get("local_instability", None),
                                               local_response_verbose=False,)
             
    #---------------------------------------------------------Analysis control - Archlength-------------------------------------------------------
       
        elif Analysis_control == "ArcLength":
            hist, summ = run_gmna_arclength_control(G=G, M=M,
                                                    fy=fy,
                                                    E=E,
                                                    ele_data=ele_data,
                                                    loads_dict=loads_Fz,
                                                    ndf=ndf,
                                                    pattern_tag=2,
                                                    control_node=control_node,
                                                    control_dof=3,
                                                    elastic_res=res,
                                                    control_node_prefer="minUz",
                                                    s=0.1,
                                                    alpha=1.0,
                                                    n_steps=50,
                                                    LF_max=None,
                                                    stop_if_unstable=True,
                                                    constraints="Transformation",
                                                    numberer="RCM",
                                                    system="UmfPack",
                                                    test=("NormUnbalance", 1e-6, 80),
                                                    algorithm=("NewtonLineSearch",),
                                                    analysis="Static",
                                                    integrator_name="ArcLength",
                                                    fallback_strategies=[
                                                        {"test": ("NormUnbalance", 1e-5, 150), "algorithm": ("NewtonLineSearch",), "s": 0.0025},
                                                        {"test": ("NormUnbalance", 1e-5, 200), "algorithm": ("Newton",), "s": 0.001},
                                                        ],
                                                    monitor_fibers=True,
                                                    fiber_ele_tags=ele_data["ele_tags"],
                                                    fiber_nIP=ele_data["meta"]["n_ip"],
                                                    fiber_sec_nums=None,
                                                    T_fiber_ele=T_fiber_ele,
                                                    fiber_strain_col=4,
                                                    fiber_chunk_size=None,
                                                    fiber_area_min=0.0,
                                                    fiber_area_col=2,
                                                    exclude_neutral_axis=True,
                                                    neutral_axis_tol=1e-12,
                                                    dump_fibers=False,
                                                    print_ops=True,
                                                    print_mode="headtail",
                                                    n_preview=12,
                                                    print_step_info=True,
                                                    print_every=1,
                                                    live_plot=live_plot,
                                                    live_plot_every=1,
                                                    live_monitor=monitor,
                                                    record_local_response=True,
                                                    local_response_map=ele_data.get("local_instability", None),
                                                    local_response_verbose=False,)
        if writer_os:
            W_non_lin.save(_build_output_path(output_dir, "opensees_GMNA_model_export.py"), as_py=True)
    
    
        data_loc = [h for h in hist if h.get("ok", 0) == 0 and isinstance(h.get("local_response"), dict) and h["local_response"]]
        print("n step con local_response =", len(data_loc))
    
        if data_loc:
            h0 = data_loc[min(5, len(data_loc)-1)]
            mid = next(iter(h0["local_response"]))
            rec = h0["local_response"][mid]
            print("member sample =", mid)
            for k in ["N_member_raw", "N_member_abs", "N_member_comp", "d_end", "d_mid", "dU_local", "U_local"]:
                print(k, "=", rec.get(k))
        print("local_response_recorded =", summ.get("local_response_recorded"))
        print("local_response_n_members =", summ.get("local_response_n_members"))
    
        conv = [h for h in hist if h.get("ok", 0) == 0]
        print("n converged =", len(conv))
        if conv:
            lr0 = conv[min(1, len(conv)-1)].get("local_response", None)
            print("type(local_response sample) =", type(lr0).__name__)
            if isinstance(lr0, dict):
                print("len(local_response sample) =", len(lr0))
                print("sample keys =", list(lr0.keys())[:10])
    #---------------------------------------------------------------------------------------------------------------------------------------------          
        
        if local_inst_control:    
            local_pp = postprocess_local_instability(
                hist=hist,
                summary=summ,
                N_use="abs",
                drop_ratio=0.08,
                drop_consec=3,
                d_tol=1e-4,
                dU_eps=1e-5,
                energy_consec=4,
                decision_rule="either",
                active_methods=("pdrop_dend", "pdrop_dmid", "energy"),
                N_peak_min_abs=5.0,
                N_peak_min_ratio_global=0.05,
                d_event_min=1e-4,
                rb_tol=0.02,
                out_dir=output_dir,
                events_txt="local_instability_events_post.txt",
                curves_txt="local_instability_curves_post.txt",
                plot_member_curves=True,
                plot_member_curves_max=1,              # oppure None
                plot_member_curves_sort_by="step",                                  #"step", "U_max", "member",N_peak
                plot_member_curves_normalized=True,
                plot_member_curves_add_raw=True,
                plot_energy_maps=False,
                energy_map_every=10,
                energy_map_show_deformed=False,
                energy_map_scale_factor=5.0,
                ele_data=ele_data,
                show_plots=not batch_mode,
            )
            print("n events =", len(local_pp["events"]))
            print("Rb =", local_pp.get("Rb"))
            print("pattern =", local_pp.get("instability_pattern"))
            if local_pp["events"]:
                print(local_pp["events"][:5])
    
    #------------------------------------------------------------Results plotting-----------------------------------------------------------------
    
        #if hasattr(monitor, "close"):
        #    monitor.close()
        
    
        import os
        import matplotlib.pyplot as plt
        figA, figB = plot_gmna_results(hist,
                                       u_lim=uz_limit,   # dlim
                                       fy=fy,
                                       E=E,
                                       delta_key="Uz",   # oppure "u" se vuoi il control disp â€œuâ€
                                       delta_mode="abs", # o "raw"
                                       show=False,
                                       save_prefix="GMNA",
                                       out_dir=output_dir,
                                       show_local_nd_curves=True)
    
        save_gmna_history_txt(hist,
                              u_lim=uz_limit,
                              filename=_build_output_path(output_dir, "GMNA_history.txt"),
                              delta_mode="raw",)
    
        ax = plot_deformed_mesh_mpl(G, M, sf=20,
                                    mode="subelements",#"members",#subelements
                                    ele_data=ele_data,
                                    unstable_ele_tags=summ.get("unstable_ele_tags", []),
                                    plastic_onset_ele_tags=summ.get("plastic_onset_ele_tags", []),
                                    plastic_full_ele_tags=summ.get("plastic_full_ele_tags", []),)
    
        ax.figure.savefig(_build_output_path(output_dir, "deformed_shape.png"), dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
    
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_deformed_mesh_tension_compression_mpl(G, M,
                                                   sf=20,
                                                   ax=ax,
                                                   mode="subelements",#subelements
                                                   ele_data=ele_data,   # deve contenere ele_tags
                                                   N0=1e-2)              # soglia: regola in base alle unitÃ  (kN))plt.show()
        ax.figure.savefig(_build_output_path(output_dir, "tension_compression_shape.png"), dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        if figA is not None:
            plt.close(figA)
        if figB is not None:
            plt.close(figB)
        print("GMNA runtime [s] =", summ["runtime_s"])
        print("LF_last =", summ["LF_last"], "u_last =", summ["u_last"])
    
        fig_mr, ax_mr, mr_data = plot_merchant_rankine_comparison(
            dcr_max=ec3_dcr_max,
            lam=lam,
            merchant_rankine_factors=mr_factors,
            lf_nonlinear=summ["LF_last"],
            show=False,
            save_path=_build_output_path(output_dir, "Merchant_Rankine_comparison.png"),
            title="MR1-MR2-MR3-MR4-MR5 vs 1/DCRmax vs nonlinear LF vs first 5 linear buckling modes",
        )
    
        fig_mid, fig_end, meta = plot_gmna_vs_local_buckling_normalized(
            hist=hist,
            n_members=3,          # 10, 20, ecc.
            delta_key="Uz",
            delta_mode="abs",
            sort_by="step",        # oppure "n_peak" o "member"
            show=False,
            out_dir=output_dir,
            save_prefix="GMNA_local_summary",
            member_labels=True,
        )

if __name__ == "__main__":
    _unitled_main()
