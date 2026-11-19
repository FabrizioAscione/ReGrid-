"""Create a filtered Excel summary for the nine selected nonlinear runs."""

import csv
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs_9_soluzioni_ottimali"
OUTPUT_PATH = DATA_DIR / "riepilogo_non_lineari_9_soluzioni.xlsx"

GMNA_FIELDS = (
    "step", "ok", "LF", "Ftot", "u", "u_abs", "u_over_ulim",
    "eps_abs_max", "eps_abs_minmax", "pl_onset_any", "pl_full_any",
    "n_pl_onset_new", "n_pl_full_new", "local_any", "local_reason",
    "buckling_mode",
)
GMNA_INT_FIELDS = {
    "step", "ok", "pl_onset_any", "pl_full_any", "n_pl_onset_new",
    "n_pl_full_new", "local_any",
}
EVENT_FIELDS = (
    "member_i", "member_j", "step", "LF", "u", "Uz", "methods",
    "buckling_mode", "N_peak", "N_event", "d_end", "d_mid",
)
EVENT_PATTERN = re.compile(
    r"^\s*\((\d+),\s*(\d+)\)\s+(\d+)\s+(\S+)\s+(\S+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)
HEADER_FILL = PatternFill("solid", fgColor="24445A")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(size=16, bold=True, color="24445A")
NOTE_FONT = Font(italic=True, color="666666")


def _number(value, as_int=False):
    if value.lower() in {"nan", "none", "-"}:
        return None
    parsed = float(value)
    return int(parsed) if as_int else parsed


def read_gmna_history(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        if len(values) < 14:
            continue
        values.extend(["-"] * (len(GMNA_FIELDS) - len(values)))
        record = {}
        for field, value in zip(GMNA_FIELDS, values):
            record[field] = value if field not in GMNA_INT_FIELDS and field not in {
                "LF", "Ftot", "u", "u_abs", "u_over_ulim", "eps_abs_max", "eps_abs_minmax"
            } else _number(value, as_int=field in GMNA_INT_FIELDS)
        rows.append(record)
    return rows


def read_local_events(path):
    events = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = EVENT_PATTERN.match(line)
        if not match:
            continue
        values = match.groups()
        events.append({
            "member_i": int(values[0]),
            "member_j": int(values[1]),
            "step": int(values[2]),
            "LF": float(values[3]),
            "u": float(values[4]),
            "Uz": float(values[5]),
            "methods": values[6],
            "buckling_mode": values[7],
            "N_peak": float(values[8]),
            "N_event": float(values[9]),
            "d_end": float(values[10]),
            "d_mid": float(values[11]),
        })
    return events


def style_table(ws, headers, rows, start_row, start_col, table_name):
    for offset, value in enumerate(headers):
        cell = ws.cell(start_row, start_col + offset, value)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row_offset, values in enumerate(rows, start=1):
        for col_offset, value in enumerate(values):
            ws.cell(start_row + row_offset, start_col + col_offset, value)

    end_row = start_row + max(1, len(rows))
    end_col = start_col + len(headers) - 1
    if not rows:
        for column in range(start_col, end_col + 1):
            ws.cell(start_row + 1, column, None)
    table = Table(
        displayName=table_name,
        ref=f"{get_column_letter(start_col)}{start_row}:{get_column_letter(end_col)}{end_row}",
    )
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False,
        showLastColumn=False, showRowStripes=True, showColumnStripes=False,
    )
    ws.add_table(table)
    return end_row


def solution_summary(history, events):
    converged = [row for row in history if row["ok"] == 0]
    final = converged[-1] if converged else {}
    plastic_steps = [
        row for row in history
        if (row["n_pl_onset_new"] or 0) > 0 or (row["n_pl_full_new"] or 0) > 0
    ]
    event_steps = {event["step"] for event in events}
    return {
        "converged_steps": len(converged),
        "final": final,
        "plastic_steps": plastic_steps,
        "plastic_onset_new": sum(row["n_pl_onset_new"] or 0 for row in history),
        "plastic_full_new": sum(row["n_pl_full_new"] or 0 for row in history),
        "local_events": len(events),
        "local_event_steps": len(event_steps),
    }


def add_solution_sheet(workbook, index_ws, sheet_index, record):
    case_name = record["case"]
    optimum = record["optimum"]
    candidate_id = record["candidate_id"]
    run_dir = DATA_DIR / record["copy"]
    sheet_name = f"{case_name}_{optimum}_{candidate_id}"
    ws = workbook.create_sheet(sheet_name)
    history = read_gmna_history(run_dir / "GMNA_history.txt")
    events = read_local_events(run_dir / "local_instability_events_post.txt")
    summary = solution_summary(history, events)
    final = summary["final"]

    ws.merge_cells("A1:U1")
    ws["A1"] = f"{case_name} | {optimum} | candidato {candidate_id}"
    ws["A1"].font = TITLE_FONT
    details = (
        ("Case study", case_name), ("Ottimo", optimum),
        ("Candidate ID", candidate_id), ("Passi GMNA convergenti", summary["converged_steps"]),
        ("Ultimo LF convergente", final.get("LF")),
        ("Ultima forza totale [kN]", final.get("Ftot")),
        ("Ultimo spostamento di controllo [m]", final.get("u")),
        ("Passi con nuove plasticizzazioni", len(summary["plastic_steps"])),
        ("Nuovi elementi a onset (somma per passi)", summary["plastic_onset_new"]),
        ("Nuovi elementi full (somma per passi)", summary["plastic_full_new"]),
        ("Eventi di instabilità locale", summary["local_events"]),
        ("Passi con instabilità locale", summary["local_event_steps"]),
    )
    for row_number, (label, value) in enumerate(details, start=3):
        ws.cell(row_number, 1, label).font = Font(bold=True)
        ws.cell(row_number, 2, value)
    ws["D3"] = "Nota plasticizzazioni"
    ws["D3"].font = Font(bold=True)
    ws["E3"] = "La history salva i conteggi per passo, non gli ID dei singoli elementi."
    ws["E3"].font = NOTE_FONT
    ws["E3"].alignment = Alignment(wrap_text=True)

    ws["A17"] = "Plasticizzazioni | deformata"
    ws["J17"] = "Instabilità locali | deformata"
    for cell in (ws["A17"], ws["J17"]):
        cell.font = Font(bold=True, color="24445A")
    for filename, anchor in (
        ("mappa_plasticizzazioni_deformata.png", "A18"),
        ("mappa_instabilita_locale_deformata.png", "J18"),
    ):
        picture = ExcelImage(run_dir / filename)
        picture.width = 360
        picture.height = 360
        ws.add_image(picture, anchor)

    plastic_headers = (
        "step", "LF", "u [m]", "eps_abs_max", "eps_abs_minmax",
        "nuove aste onset", "nuove aste full",
    )
    plastic_rows = [
        [row[field] for field in (
            "step", "LF", "u", "eps_abs_max", "eps_abs_minmax",
            "n_pl_onset_new", "n_pl_full_new",
        )]
        for row in summary["plastic_steps"]
    ]
    if not plastic_rows:
        plastic_rows = [["Nessun evento registrato", None, None, None, None, 0, 0]]
    event_rows = [[event[field] for field in EVENT_FIELDS] for event in events]
    if not event_rows:
        event_rows = [["Nessun evento registrato"] + [None] * (len(EVENT_FIELDS) - 1)]

    plastic_end = style_table(
        ws, plastic_headers, plastic_rows, 43, 1, f"Plastic{sheet_index:02d}"
    )
    local_end = style_table(
        ws, EVENT_FIELDS, event_rows, 43, 10, f"Local{sheet_index:02d}"
    )
    history_start = max(plastic_end, local_end) + 3
    ws.cell(history_start - 1, 1, "Storia GMNA completa").font = Font(bold=True, size=12)
    history_rows = [[row[field] for field in GMNA_FIELDS] for row in history]
    style_table(
        ws, GMNA_FIELDS, history_rows, history_start,
        1, f"GMNA{sheet_index:02d}",
    )

    ws.freeze_panes = f"A{history_start + 1}"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 17
    for col in range(3, 22):
        ws.column_dimensions[get_column_letter(col)].width = 15
    ws.column_dimensions["E"].width = 72
    for col in ("G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"):
        ws.column_dimensions[col].width = 16

    index_ws.cell(index_ws.max_row + 1, 1, case_name)
    index_ws.cell(index_ws.max_row, 2, optimum)
    index_ws.cell(index_ws.max_row, 3, candidate_id)
    index_ws.cell(index_ws.max_row, 4, summary["converged_steps"])
    index_ws.cell(index_ws.max_row, 5, final.get("LF"))
    index_ws.cell(index_ws.max_row, 6, final.get("Ftot"))
    index_ws.cell(index_ws.max_row, 7, final.get("u"))
    index_ws.cell(index_ws.max_row, 8, len(summary["plastic_steps"]))
    index_ws.cell(index_ws.max_row, 9, summary["plastic_onset_new"])
    index_ws.cell(index_ws.max_row, 10, summary["plastic_full_new"])
    index_ws.cell(index_ws.max_row, 11, summary["local_events"])
    index_ws.cell(index_ws.max_row, 12, summary["local_event_steps"])
    link = index_ws.cell(index_ws.max_row, 13, "Apri scheda")
    link.hyperlink = f"#'{sheet_name}'!A1"
    link.style = "Hyperlink"


def main():
    manifest_path = DATA_DIR / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter=";"))
    if len(manifest) != 9:
        raise ValueError(f"Expected nine selected runs, found {len(manifest)}")

    workbook = Workbook()
    index = workbook.active
    index.title = "Indice"
    headers = (
        "Caso", "Ottimo", "Candidate ID", "Passi convergenti", "LF finale",
        "Forza finale [kN]", "Spostamento finale [m]", "Passi con onset/full",
        "Onset (eventi sommati sui passi)", "Full (eventi sommati sui passi)", "Eventi locali",
        "Passi con eventi locali", "Scheda",
    )
    index.append(headers)
    for cell in index[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for number, record in enumerate(manifest, start=1):
        add_solution_sheet(workbook, index, number, record)

    index.freeze_panes = "A2"
    index.auto_filter.ref = f"A1:M{index.max_row}"
    index.sheet_view.showGridLines = False
    for column, width in enumerate((15, 12, 15, 17, 14, 19, 20, 20, 18, 18, 16, 21, 16), start=1):
        index.column_dimensions[get_column_letter(column)].width = width
    workbook.save(OUTPUT_PATH)
    print(f"Workbook written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()