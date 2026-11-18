# ReGrid graphical interface

ReGrid can be used either from the command line through `main.py` or through the optional Streamlit graphical interface in `app.py`.

## Start the interface

ReGrid is currently validated with **Python 3.10** because of the pinned OpenSeesPy version.

After creating and activating the virtual environment and installing `requirements.txt`, start the GUI with:

```bash
python -m streamlit run app.py
```

On Windows, `run_gui.bat` can also be used after the `.venv` environment has been created.

## Available workflows

The interface provides four guided workflows:

- **Analysis** — structural analysis of a user-defined gridshell geometry.
- **Mechanical design** — EC3-based section optimisation followed, optionally, by nonlinear analysis.
- **Reuse-based design** — mechanical design combined with reclaimed-stock matching, optional corrosion scenarios, cutting constraints and optional nonlinear analysis.
- **Optimisation** — single- or multi-objective genetic optimisation, with or without reclaimed steel. Structural, environmental, cost and reuse-related objectives can be selected from the objectives implemented in ReGrid.

Only parameters relevant to the selected workflow are shown. Parameters not exposed in the simplified GUI are preserved from `config.yaml`.

## Configuration handling

The GUI does **not** permanently overwrite the repository `config.yaml` during a run. It temporarily writes the selected settings so that `main.py` can use them, restores the original file after execution, and saves the exact run configuration as:

```text
outputs/gui_run_YYYYMMDD_HHMMSS/config_used.yaml
```

This keeps the command-line and GUI workflows independent and reproducible.

## Results

Each GUI run receives its own output directory. The **Results** page automatically detects saved figures and result files and provides:

- figure gallery;
- CSV and JSON previews;
- automatic Excel workbook containing the CSV result tables;
- individual file downloads;
- complete run ZIP download;
- execution log (`run.log`).

The scientific analysis and optimisation routines remain in `main.py` and the existing ReGrid modules. `app.py` acts only as the user interface and run manager.
