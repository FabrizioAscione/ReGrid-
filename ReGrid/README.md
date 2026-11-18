# ReGrid

**ReGrid** is a Python research toolkit for the parametric modelling, structural analysis, and reuse-based stock-constrained optimisation of steel gridshell structures. The framework uses **OpenSeesPy** for structural analysis and combines structural verification, section sizing, reclaimed-member matching, corrosion modelling, and genetic / multi-objective optimisation within a configurable workflow.

The code was developed as part of research on the direct reuse of structural steel components and the design of gridshells constrained by the characteristics of available reclaimed-element inventories.

## Main features

### Parametric geometry
Generate different gridshell geometries from a limited set of geometric parameters, including:

- barrel vaults;
- parabolic domes;
- hyperbolic paraboloids (hypar surfaces);
- Kiewitt-8 domes.

### Mesh generation
Create quadrangular or triangular structural grids and automatically identify nodes, members, boundary sets, and geometric subsets required by the analysis and optimisation routines.

### Structural analysis
Use OpenSeesPy to perform:

- linear elastic static analysis;
- linear buckling analysis;
- nonlinear load-path analysis;
- force-control, displacement-control, and arc-length nonlinear solution strategies.

### Design checks and section sizing
Assign steel sections to grid members and evaluate structural demand through dedicated verification routines. The mechanical optimisation workflow includes Eurocode 3 member checks and can search a steel-section catalogue while accounting for criteria such as:

- maximum demand-to-capacity ratio (DCR);
- maximum displacement;
- linear buckling load multiplier;
- Merchant–Rankine-based performance constraints.

### Stock-constrained optimisation
Use genetic and multi-objective optimisation procedures to explore geometry and section-design alternatives under reclaimed-stock constraints. The framework can compare competing objectives such as:

- structural mass / structural efficiency;
- embodied carbon;
- economic cost;
- reuse-related performance measures.

Pareto-front utilities are included to support the interpretation and comparison of alternative design solutions.

### Reuse and corrosion
The reuse workflow can:

- match structural members with available reclaimed steel elements;
- consider geometric and mechanical compatibility;
- manage cutting operations and residual material;
- update the available stock during the matching process;
- model corrosion-related cross-section loss;
- generate modified reclaimed-stock databases under selected corrosion scenarios.

### Results and export
The project includes utilities for:

- structural and optimisation plots;
- CSV and text reports;
- Pareto-front visualisation;
- DXF geometry export;
- saving and reloading optimisation results;
- batch execution of multiple optimisation cases.

## Workflow

The general workflow starts from user-defined geometry, mesh, material, section, load, and boundary-condition parameters. The generated structural model is then analysed and checked according to the selected analysis settings.

Depending on the active options, ReGrid can perform a conventional structural analysis, mechanical section optimisation, or a reuse-based stock-constrained optimisation. When reuse is enabled, the structural solution is matched against a database of reclaimed elements, optionally including corrosion-related degradation and residual-stock management. Multi-objective procedures can then explore trade-offs among structural, environmental, and economic performance.

The code is organised as a configurable research environment: available analyses and optional workflows depend on the selected settings and input data.

## Repository structure

The current release intentionally retains the original research-code organisation so that internal imports and file paths remain unchanged.

| Component | Main files |
| --- | --- |
| Main workflow and runtime configuration | `main.py`, `main_helpers.py` |
| Geometry and mesh generation | `geometry.py`, `mesh.py` |
| Materials and steel-section data | `materials.py`, `import_section.py`, `sections_1.csv` |
| OpenSees model and analyses | `OS_*.py` |
| Structural checks and simplified performance | `Structural_check.py`, `Merchant_Rankine.py` |
| Reclaimed-stock matching | `reuse_matching.py` |
| Genetic / multi-objective optimisation | `reuse_genetic_opt.py`, `reuse_genetic_evaluator.py`, `genetic_grid_helpers.py` |
| Corrosion modelling | `corrosion.py`, `section_corrosion.py` |
| Reclaimed-element databases | `Database_reuse_1.csv`, `Database_reuse_2.csv` |
| Batch and post-processing utilities | `run_ga_batch.py`, `run_ga_headless.py`, Pareto and export scripts |

Additional notes on the mechanical optimisation workflow are provided in [`OPT_MECH_2_EXPLANATION.txt`](OPT_MECH_2_EXPLANATION.txt).

## Special Features

ReGrid includes two graphical interfaces designed to simplify both the definition of the analysis workflow and the post-processing of optimisation results.

### ReGrid Graphical User Interface

A dedicated Streamlit-based graphical user interface is provided through `app.py`. The interface allows users to configure and run ReGrid without directly modifying the source code or the `config.yaml` file.

Different workflows can be selected depending on the purpose of the analysis, including:

- structural analysis;
- mechanical design and section optimisation;
- reuse-based design using reclaimed steel inventories;
- single- and multi-objective structural optimisation, with or without steel reuse.

The interface dynamically displays only the parameters relevant to the selected workflow, including geometry, loading conditions, structural constraints, reclaimed-stock properties, corrosion scenarios, optimisation settings and nonlinear analysis options.

After completion of the analysis, the interface also provides direct access to the generated results, including figures, CSV files, Excel workbooks, DXF files and complete result packages.

The GUI can be launched with:

```bash
python -m streamlit run app.py
```

or, on Windows, through the provided `run_gui.bat` file.

### Interactive Pareto-Front Selection

For multi-objective optimisation problems, ReGrid also includes a dedicated Pareto-front graphical interface for the interactive exploration of non-dominated solutions.

Once the optimisation process is completed, the interface allows the user to inspect the Pareto front and select individual solutions according to the preferred trade-off between the considered objectives, such as structural performance, embodied carbon and economic cost.

From the Pareto-selection interface, the user can also:

- select a preferred Pareto-optimal solution for further investigation;
- launch nonlinear structural analysis directly on the selected solution;
- perform nonlinear analyses on multiple Pareto solutions;
- continue the optimisation process by launching additional generations;
- compare selected reuse-based solutions with reference structural configurations.

This feature allows the optimisation process to remain interactive, enabling the user to move from the exploration of alternative design solutions to detailed nonlinear verification without restarting the complete ReGrid workflow.
## Requirements

The current environment was developed and tested with **Python 3.10** on Windows. The exact package versions used in the development environment are listed in [`requirements.txt`](requirements.txt).

Main dependencies include:

- OpenSeesPy;
- NumPy;
- SciPy;
- Matplotlib;
- pymoo;
- openpyxl.

> **Platform note:** `requirements.txt` reproduces the tested Windows environment and includes `openseespywin`. Users on other operating systems may need to adapt the OpenSeesPy installation accordingly.

## Installation

Clone or download the repository, then create a dedicated Python environment. For example, on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The local `.venv` directory is intentionally excluded from version control.

## Running ReGrid

The current main research script is:

```bash
python main.py
```

The main model and analysis options are defined in the **USER INPUT** section near the beginning of `main.py`. These include geometry, mesh, section catalogue, corrosion settings, loading, reclaimed-stock matching, and optimisation parameters.

By default, generated results are written to an `outputs/` directory. Generated outputs are excluded from Git version control.

For automated studies, the repository also contains batch and headless execution utilities such as `run_ga_batch.py` and `run_ga_headless.py`.

## Input data

The repository includes the steel-section catalogues and reclaimed-element databases required by the current research workflow. In particular:

- `sections_1.csv` is used as the default steel-section catalogue in the main workflow;
- `Database_reuse_2.csv` is used as the default reclaimed-stock database when reuse matching is activated.

Users replacing these datasets should preserve the expected column structure and units used by the corresponding import and matching routines.

## Research use and citation

If you use this code, or part of the methodology implemented in this repository, in your research, please cite the associated publications listed in [`CITATION.md`](CITATION.md).

The two principal references are:

1. **Ascione, F., Esposito, F., Faiella, D., Mele, E. (2026).** *Reuse Based Design of Steel Gridshells Through Multi-Objective Optimization.* In: Zanon, R., Nastri, E., Simoncelli, M., Paolacci, F. (eds), Proceedings of the XXX Congress CTA 2026. Lecture Notes in Civil Engineering, vol. 942. Springer, Cham. https://doi.org/10.1007/978-3-032-39505-4_10

2. **Ascione, F., Esposito, F., Faiella, D., Mele, E. (2025).** *Steel reuse for sustainable design of large-span structures: The case of gridshells.* Journal of Constructional Steel Research, 234, 109743. https://doi.org/10.1016/j.jcsr.2025.109743

## Research software notice

ReGrid is research software developed for scientific investigation of structural optimisation and structural-steel reuse. It is **not a certified structural-design package**. Results intended for engineering applications should be independently verified and checked against the applicable design standards, modelling assumptions, and project-specific requirements.

## Authors

**Fabrizio Ascione**, **Francesco Esposito**, **Diana Faiella**, and **Elena Mele**  
Department of Structures for Engineering and Architecture, University of Naples Federico II, Naples, Italy.
