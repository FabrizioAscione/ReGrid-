from __future__ import annotations

import copy
import csv
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from openpyxl import Workbook


# =============================================================================
# ReGrid GUI
# =============================================================================
APP_TITLE = "ReGrid"
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
MAIN_PATH = PROJECT_DIR / "main.py"
GUI_STATE_DIR = PROJECT_DIR / ".regrid_gui"
GUI_INPUT_DIR = GUI_STATE_DIR / "inputs"
GUI_STATE_DIR.mkdir(exist_ok=True)
GUI_INPUT_DIR.mkdir(exist_ok=True)

GEOMETRIES = {
    "Barrel vault": 1,
    "Parabolic dome": 2,
    "Hyperbolic paraboloid (hypar)": 3,
    "Kiewitt-8 spherical dome": 4,
}
GRID_TYPES = {"Quadrangular": 1, "Triangular": 2}
WORKFLOWS = ["Analysis", "Mechanical design", "Reuse-based design", "Optimisation"]
OBJECTIVES = [
    "pi_mass",
    "pi_ec",
    "pi_mr",
    "pi_cost",
    "mass",
    "cost",
    "reuse_ratio",
    "reuse_efficiency",
    "reuse_ratio_members",
    "dcr_max",
    "lscrap",
    "n_cuts",
    "max_abs_uz",
    "lambda_ref",
    "ec",
]
CORROSION_CATEGORIES = ["C1", "C2", "C3", "C4", "C5", "CX"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_EXTENSIONS = {".csv", ".json", ".txt", ".xlsx", ".dxf"}


# =============================================================================
# General helpers
# =============================================================================
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a YAML mapping at the top level.")
    return data


def nested_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def nested_set(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    target = data
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def index_for_value(mapping: dict[str, Any], value: Any, fallback: int = 0) -> int:
    values = list(mapping.values())
    try:
        return values.index(value)
    except ValueError:
        return fallback


def list_index(values: list[Any], value: Any, fallback: int = 0) -> int:
    try:
        return values.index(value)
    except ValueError:
        return fallback


def project_csv_files() -> list[str]:
    paths = list(PROJECT_DIR.glob("*.csv")) + list(GUI_INPUT_DIR.glob("*.csv"))
    result: list[str] = []
    for path in sorted(set(paths), key=lambda p: p.name.lower()):
        try:
            result.append(path.relative_to(PROJECT_DIR).as_posix())
        except ValueError:
            result.append(str(path))
    return result


def save_uploaded_file(uploaded_file) -> str:
    safe_name = Path(uploaded_file.name).name
    target = GUI_INPUT_DIR / safe_name
    target.write_bytes(uploaded_file.getbuffer())
    return target.relative_to(PROJECT_DIR).as_posix()


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = PROJECT_DIR / "outputs" / f"gui_run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = PROJECT_DIR / "outputs" / f"gui_run_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def relative_project_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


def scan_result_files(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        [p for p in run_dir.rglob("*") if p.is_file()],
        key=lambda p: (str(p.parent).lower(), p.name.lower()),
    )


def discover_runs() -> list[Path]:
    outputs = PROJECT_DIR / "outputs"
    if not outputs.exists():
        return []
    runs = [p for p in outputs.glob("gui_run_*") if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def readable_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def zip_directory_bytes(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in scan_result_files(run_dir):
            zf.write(path, arcname=path.relative_to(run_dir))
    return buffer.getvalue()


def safe_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", name)
    cleaned = cleaned[:31] or "Sheet"
    candidate = cleaned
    number = 2
    while candidate in used:
        suffix = f"_{number}"
        candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
        number += 1
    used.add(candidate)
    return candidate


def csv_bundle_to_xlsx(csv_paths: list[Path], run_dir: Path) -> bytes:
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    used: set[str] = set()

    for path in csv_paths:
        relative_name = path.relative_to(run_dir).as_posix().replace("/", "_")
        ws = wb.create_sheet(safe_sheet_name(Path(relative_name).stem, used))
        ws.freeze_panes = "A2"
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    ws.append(row)
        except UnicodeDecodeError:
            with path.open("r", encoding="latin-1", newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    ws.append(row)

    if not wb.sheetnames:
        wb.create_sheet("README")["A1"] = "No CSV result files were available."

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def preview_csv(path: Path, max_rows: int = 100) -> tuple[list[str], list[list[str]]]:
    def _read(encoding: str):
        with path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.reader(fh)
            rows = []
            header = next(reader, [])
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
            return header, rows

    try:
        return _read("utf-8-sig")
    except UnicodeDecodeError:
        return _read("latin-1")


def file_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name




def apply_theme(theme: str) -> None:
    """Apply a lightweight runtime theme without changing Streamlit's global config."""
    if theme == "Light":
        bg = "#FFFFFF"
        surface = "#F7F7F9"
        surface_alt = "#EFEFF3"
        text = "#111827"
        muted = "#5F6673"
        border = "#D9DCE3"
        accent = "#111827"
        button_hover = "#2B3444"
        button_disabled = "#6B7280"
        button_text = "#FFFFFF"
    else:
        bg = "#0E1117"
        surface = "#171B22"
        surface_alt = "#20252E"
        text = "#FAFAFA"
        muted = "#AAB2C0"
        border = "#343A46"
        accent = "#111827"
        button_hover = "#2B3444"
        button_disabled = "#4B5563"
        button_text = "#FFFFFF"

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {'light' if theme == 'Light' else 'dark'};
        }}

        [data-testid="stAppViewContainer"] {{
            background: {bg};
            color: {text};
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: {surface};
            border-right: 1px solid {border};
        }}

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {{
            color: {text};
        }}

        h1, h2, h3, h4, h5, h6, p, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {{
            color: {text};
        }}

        [data-testid="stCaptionContainer"] p {{
            color: {muted};
        }}

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {{
            background: {surface};
            color: {text};
            border-color: {border};
        }}

        [data-baseweb="select"] span,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            color: {text} !important;
        }}

        div[data-baseweb="popover"] > div,
        ul[role="listbox"] {{
            background: {surface};
            color: {text};
        }}

        div[data-baseweb="popover"] li,
        ul[role="listbox"] li {{
            color: {text};
        }}

        [data-testid="stExpander"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 0.5rem;
        }}

        [data-testid="stTabs"] button {{
            color: {text};
        }}

        [data-testid="stMetric"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 0.5rem;
            padding: 0.75rem;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background: {surface};
        }}

        pre, code {{
            background: {surface_alt} !important;
            color: {text} !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: {accent} !important;
            color: {button_text} !important;
            border: 1px solid {accent} !important;
        }}

        .stButton > button *,
        .stDownloadButton > button * {{
            color: {button_text} !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: {button_hover} !important;
            border-color: {button_hover} !important;
            color: {button_text} !important;
            opacity: 1;
        }}

        .stButton > button:disabled,
        .stDownloadButton > button:disabled {{
            background: {button_disabled} !important;
            border-color: {button_disabled} !important;
            color: #F3F4F6 !important;
        }}

        .stButton > button:disabled *,
        .stDownloadButton > button:disabled * {{
            color: #F3F4F6 !important;
        }}

        hr {{
            border-color: {border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Configuration assembled from the GUI
# =============================================================================
def apply_gui_settings(base_cfg: dict[str, Any], ui: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # Geometry and files
    nested_set(cfg, ("geometry", "geom"), ui["geom"])
    nested_set(cfg, ("geometry", "B"), ui["B"])
    nested_set(cfg, ("geometry", "L"), ui["L"])
    nested_set(cfg, ("geometry", "f"), ui["f"])
    nested_set(cfg, ("geometry", "nx"), ui["nx"])
    nested_set(cfg, ("geometry", "ny"), ui["ny"])
    nested_set(cfg, ("geometry", "grid"), ui["grid"])
    nested_set(cfg, ("files", "output_dir"), relative_project_path(run_dir))
    nested_set(cfg, ("files", "section_catalogue"), ui["section_catalogue"])

    # Loads
    nested_set(cfg, ("loads", "active_elastic_load_case"), ui["load_case"])
    nested_set(cfg, ("loads", "q1", "values"), ui["q1_values"])
    nested_set(cfg, ("loads", "q2", "symmetrical_values"), ui["q2_sym_values"])
    nested_set(cfg, ("loads", "q2", "asymmetrical_values"), ui["q2_asym_values"])
    nested_set(cfg, ("loads", "q2", "asymmetrical_region"), ui["q2_region"])

    workflow = ui["workflow"]
    is_analysis = workflow == "Analysis"
    is_mech = workflow == "Mechanical design"
    is_reuse_design = workflow == "Reuse-based design"
    is_optim = workflow == "Optimisation"

    # Workflow gates
    nested_set(cfg, ("mechanical_optimization", "enabled"), bool(is_mech or is_reuse_design or is_optim))
    nested_set(cfg, ("reuse", "matching_enabled"), bool(is_reuse_design))
    nested_set(cfg, ("genetic", "enabled"), bool(is_optim))
    nested_set(cfg, ("nonlinear", "enabled"), bool(ui["nonlinear_enabled"]))

    # Mechanical design/check settings
    nested_set(cfg, ("mechanical_optimization", "mode"), ui["mech_mode"])
    nested_set(cfg, ("mechanical_optimization", "dcr_limit"), ui["dcr_limit"])
    nested_set(cfg, ("mechanical_optimization", "displacement", "check"), ui["check_disp"])
    nested_set(cfg, ("mechanical_optimization", "displacement", "limit"), ui["disp_limit"])
    nested_set(cfg, ("mechanical_optimization", "buckling", "check"), ui["check_buckling"])
    nested_set(cfg, ("mechanical_optimization", "buckling", "lambda_min"), ui["buckling_min"])
    nested_set(cfg, ("mechanical_optimization", "merchant_rankine", "check"), ui["check_mr"])
    nested_set(cfg, ("mechanical_optimization", "merchant_rankine", "minimum"), ui["mr_min"])

    # Reuse and corrosion
    reuse_active = bool(is_reuse_design or (is_optim and ui["optim_use_reuse"]))
    nested_set(cfg, ("reuse", "stock_csv"), ui["stock_csv"])
    nested_set(cfg, ("reuse", "max_cut_length"), ui["max_cut_length"])
    nested_set(cfg, ("reuse", "min_remainder_length"), ui["min_remainder_length"])
    nested_set(cfg, ("reuse", "group_sizes"), ui["group_sizes"])
    nested_set(cfg, ("corrosion", "enabled"), bool(reuse_active and ui["corrosion_enabled"]))
    nested_set(cfg, ("corrosion", "period_end_years"), ui["corrosion_years"])
    nested_set(cfg, ("corrosion", "categories"), ui["corrosion_categories"])
    nested_set(cfg, ("corrosion", "apply_to_reuse_database"), bool(reuse_active and ui["corrosion_apply_stock"]))

    # Genetic optimisation
    nested_set(cfg, ("genetic", "matching_reuse_inside_loop"), bool(is_optim and ui["optim_use_reuse"]))
    nested_set(cfg, ("genetic", "objective_mode"), ui["objective_mode"])
    nested_set(cfg, ("genetic", "obj_1"), ui["obj_1"])
    nested_set(cfg, ("genetic", "obj_2"), ui["obj_2"])
    nested_set(cfg, ("genetic", "obj_3"), ui["obj_3"])
    nested_set(cfg, ("genetic", "population"), ui["population"])
    nested_set(cfg, ("genetic", "generations"), ui["generations"])
    nested_set(cfg, ("genetic", "n_workers"), ui["n_workers"])
    nested_set(cfg, ("genetic", "move_range_x"), ui["move_range_x"])
    nested_set(cfg, ("genetic", "move_range_y"), ui["move_range_y"])
    nested_set(cfg, ("genetic", "diagonal_topology", "enabled"), bool(is_optim and ui["diagonal_topology_enabled"]))
    nested_set(cfg, ("genetic", "diagonal_topology", "threshold"), float(ui["diagonal_topology_threshold"]))
    nested_set(cfg, ("genetic", "diagonal_topology", "pattern"), str(ui["diagonal_topology_pattern"]))
    nested_set(cfg, ("genetic", "select_pareto_for_nonlinear"), bool(ui["pareto_selector"]))

    # GA structural constraints
    nested_set(cfg, ("genetic", "dcr_limit"), ui["ga_dcr_limit"])
    nested_set(cfg, ("genetic", "displacement_limit"), ui["ga_disp_limit"])
    nested_set(cfg, ("genetic", "buckling_lambda_min"), ui["ga_buckling_min"])
    nested_set(cfg, ("genetic", "merchant_rankine", "enabled"), ui["ga_mr_enabled"])
    nested_set(cfg, ("genetic", "merchant_rankine", "minimum"), ui["ga_mr_min"])

    # Internal mechanical optimiser during the GA
    nested_set(cfg, ("genetic_mechanical_optimization", "dcr_limit"), ui["dcr_limit"])
    nested_set(cfg, ("genetic_mechanical_optimization", "displacement", "check"), ui["check_disp"])
    nested_set(cfg, ("genetic_mechanical_optimization", "displacement", "limit"), ui["disp_limit"])
    nested_set(cfg, ("genetic_mechanical_optimization", "buckling", "check"), ui["check_buckling"])
    nested_set(cfg, ("genetic_mechanical_optimization", "buckling", "lambda_min"), ui["buckling_min"])
    nested_set(cfg, ("genetic_mechanical_optimization", "merchant_rankine", "check"), ui["check_mr"])
    nested_set(cfg, ("genetic_mechanical_optimization", "merchant_rankine", "minimum"), ui["mr_min"])

    # Nonlinear analysis
    nested_set(cfg, ("nonlinear", "fiber_element"), ui["fiber_element"])
    nested_set(cfg, ("nonlinear", "analysis_control"), ui["analysis_control"])
    nested_set(cfg, ("nonlinear", "uz_limit"), ui["uz_limit"])

    # Results-oriented plotting. Interactive plotting remains disabled because
    # Streamlit displays the saved files in the Results page.
    nested_set(cfg, ("plotting", "geometry"), False)
    nested_set(cfg, ("plotting", "mesh"), False)
    nested_set(cfg, ("plotting", "opensees_model"), False)
    nested_set(cfg, ("plotting", "deformed_shape"), False)
    nested_set(cfg, ("plotting", "buckling_shape"), False)
    nested_set(cfg, ("plotting", "section_plan"), False)
    nested_set(cfg, ("plotting", "nonlinear_live_plot"), False)
    nested_set(cfg, ("plotting", "show_load_map"), False)
    nested_set(cfg, ("plotting", "save_load_map"), True)

    # Keep user/manual section assignment out of the web GUI for v1.
    nested_set(cfg, ("linear_elastic", "section_type"), "1forall")

    return cfg


# =============================================================================
# ReGrid execution
# =============================================================================
def execute_regrid(run_cfg: dict[str, Any], run_dir: Path) -> tuple[int, str]:
    original_config_bytes = CONFIG_PATH.read_bytes()
    run_config_path = run_dir / "config_used.yaml"
    log_path = run_dir / "run.log"
    yaml_text = yaml.safe_dump(run_cfg, sort_keys=False, allow_unicode=True)
    run_config_path.write_text(yaml_text, encoding="utf-8")

    live_box = st.empty()
    lines: list[str] = []
    return_code = -1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"

    try:
        # main.py currently reads config.yaml from its own directory. Use the
        # run configuration only while the subprocess is active, then restore
        # the user's original file byte-for-byte.
        CONFIG_PATH.write_text(yaml_text, encoding="utf-8")

        process = subprocess.Popen(
            [sys.executable, str(MAIN_PATH)],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                lines.append(line.rstrip())
                if len(lines) > 250:
                    lines = lines[-250:]
                live_box.code("\n".join(lines), language="text")

        return_code = process.wait()
    finally:
        CONFIG_PATH.write_bytes(original_config_bytes)

    return return_code, "\n".join(lines)


# =============================================================================
# UI components
# =============================================================================
def render_python_status() -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] == (3, 10):
        st.caption(f"Python {version} · compatible with the validated OpenSeesPy environment")
    else:
        st.warning(
            f"This GUI is running with Python {version}. ReGrid is currently validated with Python 3.10 "
            "because of the pinned OpenSeesPy version."
        )


def render_design_page(base_cfg: dict[str, Any]) -> None:
    st.title("ReGrid")
    st.caption("Structural analysis, reuse-based design and stock-constrained optimisation of steel gridshells")
    render_python_status()

    workflow = st.radio(
        "What do you want to do?",
        WORKFLOWS,
        index=list_index(WORKFLOWS, st.session_state.get("workflow_choice", "Mechanical design"), 1),
        horizontal=True,
        key="workflow_choice",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    st.subheader("1 · Geometry and mesh")
    c1, c2, c3 = st.columns(3)
    current_geom = nested_get(base_cfg, "geometry", "geom", default=3)
    geom_name = c1.selectbox(
        "Geometry",
        list(GEOMETRIES.keys()),
        index=index_for_value(GEOMETRIES, current_geom, 2),
    )
    current_grid = nested_get(base_cfg, "geometry", "grid", default=1)
    grid_name = c2.selectbox(
        "Grid",
        list(GRID_TYPES.keys()),
        index=index_for_value(GRID_TYPES, current_grid, 0),
    )
    section_catalogues = project_csv_files()
    default_section = str(nested_get(base_cfg, "files", "section_catalogue", default="sections_1.csv"))
    if default_section not in section_catalogues:
        section_catalogues.insert(0, default_section)
    section_catalogue = c3.selectbox(
        "Section catalogue",
        section_catalogues,
        index=list_index(section_catalogues, default_section),
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    B = c1.number_input("B [m]", min_value=0.01, value=float(nested_get(base_cfg, "geometry", "B", default=30.0)), step=0.5)
    L = c2.number_input("L [m]", min_value=0.01, value=float(nested_get(base_cfg, "geometry", "L", default=30.0)), step=0.5)
    f = c3.number_input("Rise f [m]", min_value=0.0, value=float(nested_get(base_cfg, "geometry", "f", default=3.75)), step=0.25)
    nx = c4.number_input("nx", min_value=2, value=int(nested_get(base_cfg, "geometry", "nx", default=11)), step=1)
    ny = c5.number_input("ny", min_value=2, value=int(nested_get(base_cfg, "geometry", "ny", default=11)), step=1)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    with st.expander("Loads", expanded=False):
        active = str(nested_get(base_cfg, "loads", "active_elastic_load_case", default="q1"))
        load_case = st.radio("Active load case", ["q1", "q2"], index=0 if active == "q1" else 1, horizontal=True)

        q1_default = nested_get(base_cfg, "loads", "q1", "values", default=[0.0, 0.0, 1.8])
        st.caption("q1 · symmetrical load components")
        q1c = st.columns(3)
        q1_values = [
            q1c[0].number_input("q1 Fx", value=float(q1_default[0])),
            q1c[1].number_input("q1 Fy", value=float(q1_default[1])),
            q1c[2].number_input("q1 Fz", value=float(q1_default[2])),
        ]

        q2s_default = nested_get(base_cfg, "loads", "q2", "symmetrical_values", default=[0.0, 0.0, 0.6])
        q2a_default = nested_get(base_cfg, "loads", "q2", "asymmetrical_values", default=[0.0, 0.0, 1.2])
        q2r_default = nested_get(base_cfg, "loads", "q2", "asymmetrical_region", default=[1, 4])
        st.caption("q2 · symmetrical component")
        q2c = st.columns(3)
        q2_sym_values = [
            q2c[0].number_input("q2 sym Fx", value=float(q2s_default[0])),
            q2c[1].number_input("q2 sym Fy", value=float(q2s_default[1])),
            q2c[2].number_input("q2 sym Fz", value=float(q2s_default[2])),
        ]
        st.caption("q2 · asymmetrical component")
        q2c2 = st.columns(3)
        q2_asym_values = [
            q2c2[0].number_input("q2 asym Fx", value=float(q2a_default[0])),
            q2c2[1].number_input("q2 asym Fy", value=float(q2a_default[1])),
            q2c2[2].number_input("q2 asym Fz", value=float(q2a_default[2])),
        ]
        q2rc = st.columns(2)
        q2_region = [
            q2rc[0].number_input("Asymmetrical region start", value=int(q2r_default[0]), step=1),
            q2rc[1].number_input("Asymmetrical region end", value=int(q2r_default[1]), step=1),
        ]

    # ------------------------------------------------------------------
    # Mechanical design settings
    # ------------------------------------------------------------------
    mech_relevant = workflow in {"Mechanical design", "Reuse-based design", "Optimisation"}
    if mech_relevant:
        st.subheader("2 · Structural design checks")
        m1, m2, m3, m4 = st.columns(4)
        mech_mode = m1.selectbox(
            "Mechanical optimisation",
            [1, 2],
            index=0 if int(nested_get(base_cfg, "mechanical_optimization", "mode", default=2)) == 1 else 1,
            format_func=lambda x: "1 section for all members" if x == 1 else "Multiple sections",
        )
        dcr_limit = m2.number_input(
            "DCR limit",
            min_value=0.01,
            value=float(nested_get(base_cfg, "mechanical_optimization", "dcr_limit", default=1.0)),
            step=0.05,
        )
        check_disp = m3.checkbox(
            "Displacement check",
            value=bool(nested_get(base_cfg, "mechanical_optimization", "displacement", "check", default=True)),
        )
        disp_limit = m4.number_input(
            "|Uz| limit [m]",
            min_value=0.0,
            value=float(nested_get(base_cfg, "mechanical_optimization", "displacement", "limit", default=0.15)),
            step=0.01,
            disabled=not check_disp,
        )
        m1, m2, m3, m4 = st.columns(4)
        check_buckling = m1.checkbox(
            "Buckling check",
            value=bool(nested_get(base_cfg, "mechanical_optimization", "buckling", "check", default=True)),
        )
        buckling_min = m2.number_input(
            "Minimum λcr",
            min_value=0.0,
            value=float(nested_get(base_cfg, "mechanical_optimization", "buckling", "lambda_min", default=1.0)),
            step=0.1,
            disabled=not check_buckling,
        )
        check_mr = m3.checkbox(
            "Merchant–Rankine",
            value=bool(nested_get(base_cfg, "mechanical_optimization", "merchant_rankine", "check", default=False)),
        )
        mr_min = m4.number_input(
            "Minimum MR factor",
            min_value=0.0,
            value=float(nested_get(base_cfg, "mechanical_optimization", "merchant_rankine", "minimum", default=1.0)),
            step=0.1,
            disabled=not check_mr,
        )
    else:
        mech_mode = int(nested_get(base_cfg, "mechanical_optimization", "mode", default=2))
        dcr_limit = float(nested_get(base_cfg, "mechanical_optimization", "dcr_limit", default=1.0))
        check_disp = bool(nested_get(base_cfg, "mechanical_optimization", "displacement", "check", default=True))
        disp_limit = float(nested_get(base_cfg, "mechanical_optimization", "displacement", "limit", default=0.15))
        check_buckling = bool(nested_get(base_cfg, "mechanical_optimization", "buckling", "check", default=True))
        buckling_min = float(nested_get(base_cfg, "mechanical_optimization", "buckling", "lambda_min", default=1.0))
        check_mr = bool(nested_get(base_cfg, "mechanical_optimization", "merchant_rankine", "check", default=False))
        mr_min = float(nested_get(base_cfg, "mechanical_optimization", "merchant_rankine", "minimum", default=1.0))

    # ------------------------------------------------------------------
    # Reuse settings
    # ------------------------------------------------------------------
    optim_use_reuse = False
    if workflow == "Optimisation":
        st.subheader("3 · Optimisation")
        optim_use_reuse = st.toggle(
            "Include reclaimed steel in the optimisation",
            value=bool(nested_get(base_cfg, "genetic", "matching_reuse_inside_loop", default=False)),
        )

    reuse_relevant = workflow == "Reuse-based design" or (workflow == "Optimisation" and optim_use_reuse)
    stock_csv = str(nested_get(base_cfg, "reuse", "stock_csv", default="Database_reuse_2.csv"))
    max_cut_length = nested_get(base_cfg, "reuse", "max_cut_length", default=10.0)
    min_remainder_length = float(nested_get(base_cfg, "reuse", "min_remainder_length", default=3.0))
    group_sizes = list(nested_get(base_cfg, "reuse", "group_sizes", default=[1, 2, 4]))
    corrosion_enabled = False
    corrosion_apply_stock = False
    corrosion_years = list(nested_get(base_cfg, "corrosion", "period_end_years", default=[20, 36, 50]))
    corrosion_categories = list(nested_get(base_cfg, "corrosion", "categories", default=["C3", "C3", "C3"]))

    if reuse_relevant:
        heading_no = "3" if workflow == "Reuse-based design" else "4"
        st.subheader(f"{heading_no} · Reclaimed steel stock")
        csv_options = project_csv_files()
        if stock_csv not in csv_options:
            csv_options.insert(0, stock_csv)
        s1, s2 = st.columns([2, 1])
        stock_csv = s1.selectbox("Reclaimed stock database", csv_options, index=list_index(csv_options, stock_csv))
        uploaded = s2.file_uploader("Upload stock CSV", type=["csv"])
        if uploaded is not None:
            stock_csv = save_uploaded_file(uploaded)
            s2.success(f"Using {Path(stock_csv).name}")

        r1, r2, r3 = st.columns(3)
        max_cut_length = r1.number_input(
            "Maximum cut length [m]",
            min_value=0.0,
            value=float(max_cut_length if max_cut_length is not None else 10.0),
            step=0.5,
        )
        min_remainder_length = r2.number_input(
            "Minimum residual length [m]",
            min_value=0.0,
            value=min_remainder_length,
            step=0.5,
        )
        group_sizes = r3.multiselect("Allowed L assemblies", [1, 2, 4], default=[x for x in group_sizes if x in {1, 2, 4}])
        if not group_sizes:
            st.warning("Select at least one allowed L-profile assembly. ReGrid will otherwise have no reuse grouping option.")

        with st.expander("Corrosion scenario", expanded=False):
            corrosion_enabled = st.toggle(
                "Enable corrosion",
                value=bool(nested_get(base_cfg, "corrosion", "enabled", default=False)),
            )
            corrosion_apply_stock = st.checkbox(
                "Apply corrosion loss to the reclaimed stock database",
                value=bool(nested_get(base_cfg, "corrosion", "apply_to_reuse_database", default=False)),
                disabled=not corrosion_enabled,
            )
            if corrosion_enabled:
                cols = st.columns(3)
                years_new: list[int] = []
                cats_new: list[str] = []
                for i in range(3):
                    with cols[i]:
                        years_new.append(
                            int(
                                st.number_input(
                                    f"Period {i+1} end year",
                                    min_value=1,
                                    value=int(corrosion_years[i] if i < len(corrosion_years) else (20 + 15 * i)),
                                    step=1,
                                    key=f"corr_year_{i}",
                                )
                            )
                        )
                        cat_default = corrosion_categories[i] if i < len(corrosion_categories) else "C3"
                        cats_new.append(
                            st.selectbox(
                                f"Period {i+1} category",
                                CORROSION_CATEGORIES,
                                index=list_index(CORROSION_CATEGORIES, cat_default, 2),
                                key=f"corr_cat_{i}",
                            )
                        )
                corrosion_years = years_new
                corrosion_categories = cats_new

    # ------------------------------------------------------------------
    # Genetic optimisation settings
    # ------------------------------------------------------------------
    objective_mode = str(nested_get(base_cfg, "genetic", "objective_mode", default="multi"))
    obj_1 = str(nested_get(base_cfg, "genetic", "obj_1", default="pi_mass"))
    obj_2 = str(nested_get(base_cfg, "genetic", "obj_2", default="pi_ec"))
    obj_3 = str(nested_get(base_cfg, "genetic", "obj_3", default="pi_cost"))
    population = int(nested_get(base_cfg, "genetic", "population", default=1500))
    generations = int(nested_get(base_cfg, "genetic", "generations", default=10))
    n_workers = int(nested_get(base_cfg, "genetic", "n_workers", default=1))
    move_range_x = list(nested_get(base_cfg, "genetic", "move_range_x", default=[-0.5, 0.5]))
    move_range_y = list(nested_get(base_cfg, "genetic", "move_range_y", default=[-0.5, 0.5]))
    diagonal_topology_enabled = bool(nested_get(base_cfg, "genetic", "diagonal_topology", "enabled", default=False))
    diagonal_topology_threshold = float(nested_get(base_cfg, "genetic", "diagonal_topology", "threshold", default=0.5))
    diagonal_topology_pattern = str(nested_get(base_cfg, "genetic", "diagonal_topology", "pattern", default="alternating"))
    pareto_selector = False
    ga_dcr_limit = float(nested_get(base_cfg, "genetic", "dcr_limit", default=1.0))
    ga_disp_limit = float(nested_get(base_cfg, "genetic", "displacement_limit", default=0.15))
    ga_buckling_min = float(nested_get(base_cfg, "genetic", "buckling_lambda_min", default=1.0))
    ga_mr_enabled = bool(nested_get(base_cfg, "genetic", "merchant_rankine", "enabled", default=True))
    ga_mr_min = float(nested_get(base_cfg, "genetic", "merchant_rankine", "minimum", default=1.0))

    if workflow == "Optimisation":
        o1, o2, o3 = st.columns(3)
        objective_mode = o1.radio(
            "Objective mode",
            ["mono", "multi"],
            index=0 if objective_mode == "mono" else 1,
            format_func=lambda x: "Single-objective" if x == "mono" else "Multi-objective",
        )
        obj_1 = o2.selectbox("Objective 1", OBJECTIVES, index=list_index(OBJECTIVES, obj_1))
        if objective_mode == "multi":
            obj_2 = o3.selectbox("Objective 2", OBJECTIVES, index=list_index(OBJECTIVES, obj_2))
            obj_3 = st.selectbox("Objective 3", OBJECTIVES, index=list_index(OBJECTIVES, obj_3))
            chosen = [obj_1, obj_2, obj_3]
            if len(set(chosen)) != len(chosen):
                st.warning("The selected multi-objective functions are not unique.")

        g1, g2, g3 = st.columns(3)
        population = g1.number_input("Population", min_value=2, value=population, step=10)
        generations = g2.number_input("Generations", min_value=1, value=generations, step=1)
        n_workers = g3.number_input("Parallel workers", min_value=1, value=n_workers, step=1)

        with st.expander("Geometry and topology search domain", expanded=False):
            st.markdown("**Shape optimisation**")
            gx1, gx2, gy1, gy2 = st.columns(4)
            xmin = gx1.number_input("X movement min [m]", value=float(move_range_x[0]))
            xmax = gx2.number_input("X movement max [m]", value=float(move_range_x[1]))
            ymin = gy1.number_input("Y movement min [m]", value=float(move_range_y[0]))
            ymax = gy2.number_input("Y movement max [m]", value=float(move_range_y[1]))
            move_range_x = [min(xmin, xmax), max(xmin, xmax)]
            move_range_y = [min(ymin, ymax), max(ymin, ymax)]

            st.markdown("**Diagonal topology optimisation**")
            quadrangular_grid = GRID_TYPES[grid_name] == 1 and GEOMETRIES[geom_name] != 4
            diagonal_topology_enabled = st.toggle(
                "Use grid diagonals as genetic design variables",
                value=bool(diagonal_topology_enabled) if quadrangular_grid else False,
                disabled=not quadrangular_grid,
                help=(
                    "Each variable controls one symmetry-linked group of diagonals in the +x/+y quarter. "
                    "A binary value of 1 keeps/adds the diagonal; 0 removes it. The choice is mirrored "
                    "over the complete grid before structural analysis and section optimisation."
                ),
            )
            if not quadrangular_grid:
                st.caption("Diagonal topology optimisation is currently available for quadrilateral face meshes only (barrel vault, parabolic dome and hypar).")
            if diagonal_topology_enabled:
                pattern_options = ["alternating", "slash", "backslash"]
                diagonal_topology_pattern = st.selectbox(
                    "Candidate diagonal pattern",
                    pattern_options,
                    index=list_index(pattern_options, diagonal_topology_pattern, 0),
                    format_func=lambda value: {
                        "alternating": "Alternating / checkerboard",
                        "slash": "Direction A",
                        "backslash": "Direction B",
                    }[value],
                    help=(
                        "The pattern defines which diagonal of each quadrilateral is eligible. "
                        "The GA then decides independently, by symmetry group, whether that diagonal is present."
                    ),
                )
                diagonal_topology_threshold = st.slider(
                    "Binary activation threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=float(diagonal_topology_threshold),
                    step=0.05,
                    help="GA values greater than or equal to this threshold are interpreted as an active diagonal.",
                )

        with st.expander("GA structural constraints", expanded=False):
            gc1, gc2, gc3, gc4 = st.columns(4)
            ga_dcr_limit = gc1.number_input("GA DCR limit", min_value=0.01, value=ga_dcr_limit, step=0.05)
            ga_disp_limit = gc2.number_input("GA |Uz| limit [m]", min_value=0.0, value=ga_disp_limit, step=0.01)
            ga_buckling_min = gc3.number_input("GA minimum λcr", min_value=0.0, value=ga_buckling_min, step=0.1)
            ga_mr_enabled = gc4.checkbox("GA Merchant–Rankine", value=ga_mr_enabled)
            ga_mr_min = st.number_input("GA minimum MR factor", min_value=0.0, value=ga_mr_min, step=0.1, disabled=not ga_mr_enabled)

    # ------------------------------------------------------------------
    # Nonlinear analysis
    # ------------------------------------------------------------------
    section_number = 2
    if workflow == "Optimisation":
        section_number = 5 if reuse_relevant else 4
    elif workflow == "Reuse-based design":
        section_number = 4
    elif workflow == "Mechanical design":
        section_number = 3

    st.subheader(f"{section_number} · Post-analysis")
    nonlinear_default = bool(nested_get(base_cfg, "nonlinear", "enabled", default=False))
    nonlinear_enabled = st.toggle("Run nonlinear analysis after the selected workflow", value=nonlinear_default)
    fiber_element = str(nested_get(base_cfg, "nonlinear", "fiber_element", default="HingeRadauTwo"))
    analysis_control = str(nested_get(base_cfg, "nonlinear", "analysis_control", default="Force"))
    uz_limit = float(nested_get(base_cfg, "nonlinear", "uz_limit", default=1.0))

    if nonlinear_enabled:
        nl1, nl2, nl3 = st.columns(3)
        fiber_options = ["Lobatto", "HingeRadau", "HingeRadauTwo"]
        fiber_element = nl1.selectbox("Fiber element", fiber_options, index=list_index(fiber_options, fiber_element, 2))
        control_options = ["Force", "Displacements", "ArcLength"]
        analysis_control = nl2.selectbox("Analysis control", control_options, index=list_index(control_options, analysis_control, 0))
        uz_limit = nl3.number_input("Nonlinear |Uz| stop limit [m]", min_value=0.01, value=uz_limit, step=0.1)
        if workflow == "Optimisation" and objective_mode == "multi":
            pareto_selector = st.checkbox(
                "Open the existing desktop Pareto selector before the final nonlinear analysis",
                value=False,
                help=(
                    "When disabled, ReGrid continues with the automatically selected final solution. "
                    "When enabled, the existing desktop Pareto-selection window is opened by main.py."
                ),
            )

    # ------------------------------------------------------------------
    # Advanced/info and run
    # ------------------------------------------------------------------
    with st.expander("Advanced configuration", expanded=False):
        st.caption(
            "Parameters not exposed in this simplified interface are preserved from config.yaml. "
            "The GUI does not delete or reset them."
        )
        config_bytes = CONFIG_PATH.read_bytes()
        st.download_button(
            "Download current config.yaml",
            data=config_bytes,
            file_name="config.yaml",
            mime="text/yaml",
        )
        st.code(
            "main.py remains fully usable without the GUI:\n\npython main.py",
            language="text",
        )

    ui = {
        "workflow": workflow,
        "geom": GEOMETRIES[geom_name],
        "grid": GRID_TYPES[grid_name],
        "B": float(B),
        "L": float(L),
        "f": float(f),
        "nx": int(nx),
        "ny": int(ny),
        "section_catalogue": section_catalogue,
        "load_case": load_case,
        "q1_values": [float(x) for x in q1_values],
        "q2_sym_values": [float(x) for x in q2_sym_values],
        "q2_asym_values": [float(x) for x in q2_asym_values],
        "q2_region": [int(x) for x in q2_region],
        "mech_mode": int(mech_mode),
        "dcr_limit": float(dcr_limit),
        "check_disp": bool(check_disp),
        "disp_limit": float(disp_limit),
        "check_buckling": bool(check_buckling),
        "buckling_min": float(buckling_min),
        "check_mr": bool(check_mr),
        "mr_min": float(mr_min),
        "optim_use_reuse": bool(optim_use_reuse),
        "stock_csv": stock_csv,
        "max_cut_length": float(max_cut_length) if max_cut_length is not None else None,
        "min_remainder_length": float(min_remainder_length),
        "group_sizes": [int(x) for x in group_sizes],
        "corrosion_enabled": bool(corrosion_enabled),
        "corrosion_apply_stock": bool(corrosion_apply_stock),
        "corrosion_years": [int(x) for x in corrosion_years],
        "corrosion_categories": list(corrosion_categories),
        "objective_mode": objective_mode,
        "obj_1": obj_1,
        "obj_2": obj_2,
        "obj_3": obj_3,
        "population": int(population),
        "generations": int(generations),
        "n_workers": int(n_workers),
        "move_range_x": [float(x) for x in move_range_x],
        "move_range_y": [float(x) for x in move_range_y],
        "diagonal_topology_enabled": bool(diagonal_topology_enabled),
        "diagonal_topology_threshold": float(diagonal_topology_threshold),
        "diagonal_topology_pattern": str(diagonal_topology_pattern),
        "pareto_selector": bool(pareto_selector),
        "ga_dcr_limit": float(ga_dcr_limit),
        "ga_disp_limit": float(ga_disp_limit),
        "ga_buckling_min": float(ga_buckling_min),
        "ga_mr_enabled": bool(ga_mr_enabled),
        "ga_mr_min": float(ga_mr_min),
        "nonlinear_enabled": bool(nonlinear_enabled),
        "fiber_element": fiber_element,
        "analysis_control": analysis_control,
        "uz_limit": float(uz_limit),
    }

    st.divider()
    run_col, info_col = st.columns([1, 3])
    run_clicked = run_col.button("▶ Run ReGrid", type="primary", use_container_width=True)
    info_col.caption(
        "A new output folder is created for every run. The original config.yaml is restored automatically after execution."
    )

    if run_clicked:
        if not MAIN_PATH.exists():
            st.error("main.py was not found in the same directory as app.py.")
            return
        if reuse_relevant and not ui["group_sizes"]:
            st.error("Select at least one allowed L-profile assembly before running.")
            return
        if ui["diagonal_topology_enabled"] and ui["grid"] != 1:
            st.error("Diagonal topology optimisation currently requires a quadrangular grid.")
            return
        if ui["workflow"] == "Optimisation" and ui["objective_mode"] == "multi":
            if len({ui["obj_1"], ui["obj_2"], ui["obj_3"]}) != 3:
                st.error("Select three different objectives for multi-objective optimisation.")
                return

        run_dir = make_run_dir()
        run_cfg = apply_gui_settings(base_cfg, ui, run_dir)
        st.session_state["last_run_dir"] = str(run_dir)
        st.session_state["last_workflow"] = workflow
        st.info(f"Running {workflow} · results → {relative_project_path(run_dir)}")
        with st.spinner("ReGrid is running. The terminal output is shown below."):
            try:
                return_code, final_log = execute_regrid(run_cfg, run_dir)
            except Exception as exc:
                st.session_state["last_return_code"] = -1
                st.error(f"Failed to start or complete ReGrid: {type(exc).__name__}: {exc}")
                return

        st.session_state["last_return_code"] = return_code
        if return_code == 0:
            st.success("ReGrid completed successfully. Open Results from the sidebar.")
        else:
            st.error(f"ReGrid ended with exit code {return_code}. Open Results to inspect the log and generated files.")
        if final_log:
            with st.expander("Final run log", expanded=return_code != 0):
                st.code(final_log, language="text")


def render_summary_json(run_dir: Path) -> None:
    summary_path = run_dir / "reuse_genetic_summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return

    meta = summary.get("meta") or {}
    objectives = summary.get("objectives") or {}
    cols = st.columns(5)
    metrics = [
        ("Feasible", "Yes" if summary.get("feasible") else "No"),
        ("Mass", objectives.get("mass")),
        ("Cost", objectives.get("cost")),
        ("Reuse ratio", objectives.get("reuse_ratio")),
        ("λcr", summary.get("lambda_ref")),
    ]
    for col, (label, value) in zip(cols, metrics):
        if isinstance(value, float):
            col.metric(label, f"{value:.4g}")
        elif value is not None:
            col.metric(label, str(value))


def render_results_page() -> None:
    st.title("ReGrid · Results")
    runs = discover_runs()
    last_run_raw = st.session_state.get("last_run_dir")
    last_run = Path(last_run_raw) if last_run_raw else None
    if last_run is not None and last_run.exists() and last_run not in runs:
        runs.insert(0, last_run)

    if not runs:
        st.info("No GUI runs were found yet. Run an analysis from the Design page first.")
        return

    labels = [p.name for p in runs]
    default_index = 0
    if last_run is not None:
        try:
            default_index = runs.index(last_run)
        except ValueError:
            default_index = 0

    selected_label = st.selectbox("Run", labels, index=default_index)
    run_dir = runs[labels.index(selected_label)]
    files = scan_result_files(run_dir)
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    jsons = [p for p in files if p.suffix.lower() == ".json"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(files))
    c2.metric("Figures", len(images))
    c3.metric("CSV tables", len(csvs))
    c4.metric("Run size", readable_size(sum(p.stat().st_size for p in files)))

    render_summary_json(run_dir)
    st.caption(f"Folder: {run_dir}")

    tab_images, tab_data, tab_downloads, tab_log = st.tabs(["Figures", "Data", "Downloads", "Run log"])

    with tab_images:
        if not images:
            st.info("No saved figures were found for this run.")
        else:
            for i in range(0, len(images), 2):
                cols = st.columns(2)
                for j, path in enumerate(images[i : i + 2]):
                    with cols[j]:
                        st.image(str(path), caption=file_label(path, run_dir), use_container_width=True)

    with tab_data:
        preview_candidates = csvs + jsons
        if not preview_candidates:
            st.info("No CSV or JSON result files were found.")
        else:
            selected = st.selectbox(
                "Preview result file",
                [file_label(p, run_dir) for p in preview_candidates],
            )
            path = preview_candidates[[file_label(p, run_dir) for p in preview_candidates].index(selected)]
            if path.suffix.lower() == ".csv":
                header, rows = preview_csv(path)
                if header:
                    records = [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows]
                    st.dataframe(records, use_container_width=True, hide_index=True)
                    if len(rows) >= 100:
                        st.caption("Preview limited to the first 100 data rows.")
                else:
                    st.info("The CSV file is empty.")
            else:
                try:
                    st.json(json.loads(path.read_text(encoding="utf-8")), expanded=False)
                except Exception:
                    st.code(path.read_text(encoding="utf-8", errors="replace")[:30000], language="json")

    with tab_downloads:
        if csvs:
            excel_bytes = csv_bundle_to_xlsx(csvs, run_dir)
            st.download_button(
                "Download all CSV results as Excel workbook",
                data=excel_bytes,
                file_name=f"{run_dir.name}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        total_size = sum(p.stat().st_size for p in files)
        if total_size <= 150 * 1024 * 1024:
            zip_bytes = zip_directory_bytes(run_dir)
            st.download_button(
                "Download complete run (.zip)",
                data=zip_bytes,
                file_name=f"{run_dir.name}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        else:
            st.warning("The run is larger than 150 MB. Use the individual file downloads below to avoid loading a very large ZIP into memory.")

        st.markdown("#### Individual files")
        downloadable = [p for p in files if p.suffix.lower() in DATA_EXTENSIONS or p.suffix.lower() in IMAGE_EXTENSIONS]
        for path in downloadable:
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.write(file_label(path, run_dir))
            c2.caption(readable_size(path.stat().st_size))
            c3.download_button(
                "Download",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime_type(path),
                key=f"dl_{run_dir.name}_{file_label(path, run_dir)}",
            )

    with tab_log:
        log_path = run_dir / "run.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            st.code(text[-60000:], language="text")
            st.download_button("Download run.log", data=text, file_name="run.log", mime="text/plain")
        else:
            st.info("run.log was not found for this run.")


# =============================================================================
# App entry point
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="ReGrid", page_icon="🕸️", layout="wide")

    try:
        base_cfg = load_config()
    except Exception as exc:
        st.error(f"Unable to load config.yaml: {type(exc).__name__}: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown("# ReGrid")
        st.caption("Research interface")
        dark_mode = st.toggle(
            "Dark mode",
            value=bool(st.session_state.get("dark_mode", False)),
            key="dark_mode",
            help="Switch between the light and dark ReGrid interface.",
        )
        theme = "Dark" if dark_mode else "Light"
        page = st.radio("Navigation", ["Design", "Results"], key="nav")
        st.divider()
        st.caption("Engine")
        st.code("main.py", language="text")
        st.caption("Configuration")
        st.code("config.yaml", language="text")
        if st.session_state.get("last_run_dir"):
            st.caption("Latest run")
            st.code(Path(st.session_state["last_run_dir"]).name, language="text")

    apply_theme(theme)

    if page == "Design":
        render_design_page(base_cfg)
    else:
        render_results_page()


if __name__ == "__main__":
    main()
