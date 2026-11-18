import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import colors as mcolors
import tkinter as tk
from tkinter import messagebox, ttk

from import_section import get_section_properties

MemberKey = Tuple[int, int]


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _member_key_to_str(key: MemberKey) -> str:
    a, b = _member_key(*key)
    return f"{a}-{b}"


def _str_to_member_key(text: str) -> MemberKey:
    parts = str(text).replace("(", "").replace(")", "").replace(",", "-").split("-")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid member key string: {text!r}")
    return _member_key(int(parts[0]), int(parts[1]))


def extract_section_names_from_catalog(
    catalog,
    units_in: str = "cm",
    units_out: str = "m",
) -> List[str]:
    """
    Best-effort extraction of section names from the section catalog.
    Works with common dict/list formats and falls back to get_section_properties(...).
    """
    names: List[str] = []

    if isinstance(catalog, Mapping):
        if all(not isinstance(v, (str, bytes)) for v in catalog.values()):
            names = [str(k) for k in catalog.keys()]

    if not names and isinstance(catalog, (list, tuple)):
        for item in catalog:
            if isinstance(item, Mapping):
                for key in ("name", "section_name", "profile", "nome"):
                    if key in item and item[key] is not None:
                        names.append(str(item[key]))
                        break

    if not names:
        try:
            all_sections = get_section_properties(
                catalog=catalog,
                units_in=units_in,
                units_out=units_out,
                print_props=False,
                print_all=False,
            )
            if isinstance(all_sections, Mapping):
                names = [str(k) for k in all_sections.keys()]
            elif isinstance(all_sections, (list, tuple)):
                for item in all_sections:
                    if isinstance(item, Mapping):
                        for key in ("name", "section_name", "profile", "nome"):
                            if key in item and item[key] is not None:
                                names.append(str(item[key]))
                                break
                    else:
                        names.append(str(item))
        except Exception:
            pass

    # unique preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_section_props_by_member(
    section_name_by_member: Mapping[MemberKey, str],
    catalog,
    units_in: str = "cm",
    units_out: str = "m",
    print_props: bool = False,
) -> Dict[MemberKey, dict]:
    cache: Dict[str, dict] = {}
    out: Dict[MemberKey, dict] = {}
    for key, name in section_name_by_member.items():
        mk = _member_key(*key)
        if name not in cache:
            cache[name] = get_section_properties(
                catalog=catalog,
                name=name,
                units_in=units_in,
                units_out=units_out,
                print_props=print_props,
            )
        out[mk] = dict(cache[name])
    return out


def _distinct_section_colors(names: Iterable[str]) -> Dict[str, Tuple[float, float, float, float]]:
    uniq = list(dict.fromkeys(str(n) for n in names))
    n = max(1, len(uniq))
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for i, name in enumerate(uniq):
        hue = (float(i) / float(n)) % 1.0
        sat = 0.72 if (i % 2 == 0) else 0.58
        val = 0.88 if (i % 3 != 0) else 0.75
        out[name] = mcolors.to_rgba(mcolors.hsv_to_rgb((hue, sat, val)))
    return out


def save_section_assignment_json(path: str, section_name_by_member: Mapping[MemberKey, str]) -> None:
    data = {_member_key_to_str(k): str(v) for k, v in section_name_by_member.items()}
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def load_section_assignment_json(path: str) -> Dict[MemberKey, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {_str_to_member_key(k): str(v) for k, v in raw.items()}



def write_section_assignment_report(
    path: str,
    section_name_by_member: Mapping[MemberKey, str],
) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    counts: Dict[str, int] = {}
    for sec_name in section_name_by_member.values():
        counts[sec_name] = counts.get(sec_name, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SECTION ASSIGNMENT REPORT\n")
        f.write(f"total_members = {len(section_name_by_member)}\n\n")
        f.write("## BY SECTION\n")
        for sec_name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{sec_name}: {count}\n")
        f.write("\n## BY MEMBER\n")
        for key, sec_name in sorted(section_name_by_member.items()):
            f.write(f"{_member_key_to_str(key)}: {sec_name}\n")


@dataclass
class _MemberArtist:
    key: MemberKey
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    line: any


class SectionAssignmentApp:
    def __init__(
        self,
        G,
        M,
        section_names: Iterable[str],
        title: str = "Section assignment by member",
        initial_assignments: Optional[Mapping[MemberKey, str]] = None,
        default_section_name: Optional[str] = None,
        require_all_assigned: bool = True,
        save_json_path: Optional[str] = None,
        show_member_labels: bool = False,
    ):
        self.G = G
        self.M = M
        self.title = title
        self.section_names = list(section_names)
        if not self.section_names:
            raise ValueError("No section names available for the selector.")
        self.require_all_assigned = bool(require_all_assigned)
        self.save_json_path = save_json_path

        self.member_keys = sorted({_member_key(*m) for m in M.get("members", [])})
        self.assignments: Dict[MemberKey, Optional[str]] = {mk: None for mk in self.member_keys}
        if default_section_name is not None:
            for mk in self.member_keys:
                self.assignments[mk] = default_section_name
        if initial_assignments:
            for k, v in initial_assignments.items():
                self.assignments[_member_key(*k)] = str(v)

        self.selected: set[MemberKey] = set()
        self.member_artists: List[_MemberArtist] = []
        self.result: Optional[Dict[MemberKey, str]] = None

        self._section_color = _distinct_section_colors(self.section_names)
        self._unassigned_color = (0.75, 0.75, 0.75, 1.0)

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1400x850")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui(show_member_labels=show_member_labels)
        self._refresh_plot()
        self._refresh_summary()

    def _build_ui(self, show_member_labels: bool = False) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(outer, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        left_tools = ttk.Frame(left)
        left_tools.pack(side="left", fill="y", padx=(0, 8))

        left_plot = ttk.Frame(left)
        left_plot.pack(side="left", fill="both", expand=True)

        plot_frame = ttk.Frame(left_plot)
        plot_frame.pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(left_plot)
        toolbar_frame.pack(fill="x", pady=(4, 0))

        ttk.Label(left_tools, text="View", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Button(left_tools, text="Home", command=self._nav_home, width=10).pack(fill="x", pady=2)
        ttk.Button(left_tools, text="Back", command=self._nav_back, width=10).pack(fill="x", pady=2)
        ttk.Button(left_tools, text="Forward", command=self._nav_forward, width=10).pack(fill="x", pady=2)
        ttk.Button(left_tools, text="Pan", command=self._nav_pan, width=10).pack(fill="x", pady=(10, 2))
        ttk.Button(left_tools, text="Zoom", command=self._nav_zoom, width=10).pack(fill="x", pady=2)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Mesh plan view (click to select, drag rectangle to multi-select)")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)

        nodes = self.G.get("nodes", {})
        xs_all = []
        ys_all = []
        for mk in self.member_keys:
            na, nb = mk
            x1, y1, _ = nodes[na]
            x2, y2, _ = nodes[nb]
            line, = self.ax.plot([x1, x2], [y1, y2], picker=5)
            line._member_key = mk
            self.member_artists.append(_MemberArtist(mk, (x1, y1), (x2, y2), line))
            xs_all.extend([x1, x2])
            ys_all.extend([y1, y2])
            if show_member_labels:
                xm = 0.5 * (x1 + x2)
                ym = 0.5 * (y1 + y2)
                self.ax.text(xm, ym, _member_key_to_str(mk), fontsize=6, ha="center", va="center")

        if xs_all and ys_all:
            dx = max(xs_all) - min(xs_all)
            dy = max(ys_all) - min(ys_all)
            pad = 0.05 * max(dx, dy, 1.0)
            self.ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
            self.ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x", expand=True)

        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.rect_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=False,
            button=[1],
            minspanx=0.001,
            minspany=0.001,
            spancoords="data",
            interactive=False,
        )

        ttk.Label(right, text="Section selection", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.combo_var = tk.StringVar(value=self.section_names[0])
        self.combo = ttk.Combobox(right, values=self.section_names, textvariable=self.combo_var, state="readonly")
        self.combo.pack(fill="x", pady=(0, 12))

        self.selection_label = ttk.Label(right, text="Selected members: 0")
        self.selection_label.pack(anchor="w", pady=(0, 8))

        btn_row1 = ttk.Frame(right)
        btn_row1.pack(fill="x", pady=4)
        ttk.Button(btn_row1, text="Assign to selection", command=self._assign_current_section).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btn_row1, text="Fill unassigned", command=self._fill_unassigned).pack(side="left", fill="x", expand=True)

        btn_row2 = ttk.Frame(right)
        btn_row2.pack(fill="x", pady=4)
        ttk.Button(btn_row2, text="Clear selection", command=self._clear_selection).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btn_row2, text="Remove assignment", command=self._remove_assignment_from_selection).pack(side="left", fill="x", expand=True)

        btn_row3 = ttk.Frame(right)
        btn_row3.pack(fill="x", pady=4)
        ttk.Button(btn_row3, text="Reset all", command=self._reset_all).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btn_row3, text="Save and close", command=self._commit_and_close).pack(side="left", fill="x", expand=True)

        help_text = (
            "Mouse:\n"
            "• click line = toggle member\n"
            "• drag rectangle = add many members\n"
            "• toolbar pan/zoom still available\n\n"
            "Workflow:\n"
            "1) select members on the left\n"
            "2) choose section on the right\n"
            "3) assign\n"
            "4) save and close"
        )
        ttk.Label(right, text=help_text, justify="left").pack(anchor="w", pady=(12, 10))

        ttk.Label(right, text="Assigned sections", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(6, 4))
        self.summary_container = ttk.Frame(right)
        self.summary_container.pack(fill="both", expand=True)

    def _get_line_color(self, mk: MemberKey):
        sec = self.assignments.get(mk)
        return self._section_color.get(sec, self._unassigned_color)

    def _refresh_plot(self) -> None:
        for item in self.member_artists:
            color = self._get_line_color(item.key)
            item.line.set_color(color)
            item.line.set_alpha(1.0 if item.key in self.selected else 0.85)
            item.line.set_linewidth(3.2 if item.key in self.selected else 1.8)
            item.line.set_zorder(5 if item.key in self.selected else 2)
        self.selection_label.configure(text=f"Selected members: {len(self.selected)}")
        self.canvas.draw_idle()

    def _refresh_summary(self) -> None:
        for child in self.summary_container.winfo_children():
            child.destroy()

        counts: Dict[str, int] = {}
        for sec in self.assignments.values():
            if sec is not None:
                counts[sec] = counts.get(sec, 0) + 1

        total_assigned = sum(counts.values())
        total_members = len(self.assignments)
        total_unassigned = total_members - total_assigned

        header = ttk.Label(
            self.summary_container,
            text=f"Assigned: {total_assigned}/{total_members} | Unassigned: {total_unassigned}",
            font=("TkDefaultFont", 9, "bold"),
        )
        header.pack(anchor="w", pady=(0, 6))

        for sec_name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            row = ttk.Frame(self.summary_container)
            row.pack(fill="x", pady=2)
            color_hex = mcolors.to_hex(self._section_color[sec_name])
            swatch = tk.Canvas(row, width=24, height=14, highlightthickness=0)
            swatch.create_rectangle(0, 0, 24, 14, fill=color_hex, outline="black")
            swatch.pack(side="left", padx=(0, 6))
            ttk.Label(row, text=f"{sec_name}").pack(side="left", anchor="w")
            ttk.Label(row, text=f"{count} members").pack(side="right", anchor="e")

    def _nav_home(self) -> None:
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.home()
            self.canvas.draw_idle()

    def _nav_back(self) -> None:
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.back()
            self.canvas.draw_idle()

    def _nav_forward(self) -> None:
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.forward()
            self.canvas.draw_idle()

    def _nav_pan(self) -> None:
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.pan()
            self.canvas.draw_idle()

    def _nav_zoom(self) -> None:
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.zoom()
            self.canvas.draw_idle()

    def _segment_intersects_box(self, p1, p2, xmin, xmax, ymin, ymax) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        if (xmin <= x1 <= xmax and ymin <= y1 <= ymax) or (xmin <= x2 <= xmax and ymin <= y2 <= ymax):
            return True
        if max(x1, x2) < xmin or min(x1, x2) > xmax or max(y1, y2) < ymin or min(y1, y2) > ymax:
            return False
        return True

    def _on_pick(self, event) -> None:
        artist = event.artist
        mk = getattr(artist, "_member_key", None)
        if mk is None:
            return
        if mk in self.selected:
            self.selected.remove(mk)
        else:
            self.selected.add(mk)
        self._refresh_plot()

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:
            return
        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        ymin = min(eclick.ydata, erelease.ydata)
        ymax = max(eclick.ydata, erelease.ydata)
        for item in self.member_artists:
            if self._segment_intersects_box(item.p1, item.p2, xmin, xmax, ymin, ymax):
                self.selected.add(item.key)
        self._refresh_plot()

    def _assign_current_section(self) -> None:
        if not self.selected:
            messagebox.showinfo("Section assignment", "Select at least one member first.")
            return
        sec_name = self.combo_var.get().strip()
        if not sec_name:
            messagebox.showinfo("Section assignment", "Choose a section first.")
            return
        for mk in self.selected:
            self.assignments[mk] = sec_name
        self._refresh_plot()
        self._refresh_summary()

    def _fill_unassigned(self) -> None:
        sec_name = self.combo_var.get().strip()
        if not sec_name:
            messagebox.showinfo("Section assignment", "Choose a section first.")
            return
        for mk, val in list(self.assignments.items()):
            if val is None:
                self.assignments[mk] = sec_name
        self._refresh_plot()
        self._refresh_summary()

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._refresh_plot()

    def _remove_assignment_from_selection(self) -> None:
        if not self.selected:
            messagebox.showinfo("Section assignment", "Select at least one member first.")
            return
        for mk in self.selected:
            self.assignments[mk] = None
        self._refresh_plot()
        self._refresh_summary()

    def _reset_all(self) -> None:
        if not messagebox.askyesno("Reset all", "Remove all current assignments?"):
            return
        for mk in self.assignments:
            self.assignments[mk] = None
        self.selected.clear()
        self._refresh_plot()
        self._refresh_summary()

    def _get_unassigned_members(self) -> List[MemberKey]:
        return [mk for mk, val in self.assignments.items() if val is None]

    def _commit_and_close(self) -> None:
        unassigned = self._get_unassigned_members()
        if self.require_all_assigned and unassigned:
            preview = ", ".join(_member_key_to_str(k) for k in unassigned[:10])
            more = "" if len(unassigned) <= 10 else f" ... (+{len(unassigned)-10} more)"
            messagebox.showwarning(
                "Incomplete assignment",
                f"There are still {len(unassigned)} unassigned members:\n{preview}{more}\n\nAssign all members before closing.",
            )
            return

        self.result = {mk: sec for mk, sec in self.assignments.items() if sec is not None}
        if self.save_json_path:
            save_section_assignment_json(self.save_json_path, self.result)
        self.root.destroy()

    def _on_close(self) -> None:
        if self.result is None:
            if messagebox.askokcancel("Close", "Close the selector without saving assignments?"):
                self.result = None
                self.root.destroy()
        else:
            self.root.destroy()

    def run(self) -> Optional[Dict[MemberKey, str]]:
        self.root.mainloop()
        return self.result



def launch_section_assignment_gui(
    G,
    M,
    section_names: Iterable[str],
    title: str = "Section assignment by member",
    initial_assignments: Optional[Mapping[MemberKey, str]] = None,
    default_section_name: Optional[str] = None,
    require_all_assigned: bool = True,
    save_json_path: Optional[str] = None,
    show_member_labels: bool = False,
) -> Optional[Dict[MemberKey, str]]:
    app = SectionAssignmentApp(
        G=G,
        M=M,
        section_names=section_names,
        title=title,
        initial_assignments=initial_assignments,
        default_section_name=default_section_name,
        require_all_assigned=require_all_assigned,
        save_json_path=save_json_path,
        show_member_labels=show_member_labels,
    )
    return app.run()



def resolve_section_definition(
    G,
    M,
    catalog,
    section_type: str,
    s_1: str,
    units_in: str = "cm",
    units_out: str = "m",
    assignment_json_path: Optional[str] = None,
    assignment_report_path: Optional[str] = None,
    selector_title: str = "Section assignment by member",
    show_member_labels: bool = False,
    print_summary: bool = True,
    clear_previous_assignment: bool = True,
):
    """
    Returns
    -------
    props_global : dict | None
    section_by_member : dict | None
    section_name_by_member : dict[MemberKey, str]
    """
    mode = str(section_type).strip().lower()
    if mode not in {"1forall", "user"}:
        raise ValueError("section_type must be '1forall' or 'user'.")

    members = sorted({_member_key(*m) for m in M.get("members", [])})

    if mode == "1forall":
        props_global = get_section_properties(
            catalog=catalog,
            name=s_1,
            units_in=units_in,
            units_out=units_out,
            print_props=False,
        )
        section_name_by_member = {mk: s_1 for mk in members}

        if assignment_json_path:
            save_section_assignment_json(assignment_json_path, section_name_by_member)
        if assignment_report_path:
            write_section_assignment_report(assignment_report_path, section_name_by_member)

        if print_summary:
            print(f"# SECTION SETUP | mode=1forall | section='{s_1}' | n_members={len(members)}")

        return props_global, None, section_name_by_member

    section_names = extract_section_names_from_catalog(
        catalog,
        units_in=units_in,
        units_out=units_out,
    )
    if s_1 and s_1 not in section_names:
        section_names = [s_1] + section_names

    initial_assignments = None

    # NEW: in modalità user cancella il vecchio json prima di aprire la GUI
    if clear_previous_assignment and assignment_json_path:
        try:
            if os.path.exists(assignment_json_path):
                os.remove(assignment_json_path)
                if print_summary:
                    print(f"# SECTION SETUP | deleted old assignment file: {assignment_json_path}")
        except Exception as e:
            print(f"# SECTION SETUP | warning: could not delete old assignment file: {e}")

    # carica solo se NON hai chiesto la pulizia preventiva
    if (not clear_previous_assignment) and assignment_json_path and os.path.exists(assignment_json_path):
        try:
            initial_assignments = load_section_assignment_json(assignment_json_path)
        except Exception:
            initial_assignments = None

    section_name_by_member = launch_section_assignment_gui(
        G=G,
        M=M,
        section_names=section_names,
        title=selector_title,
        initial_assignments=initial_assignments,
        default_section_name=None,
        require_all_assigned=True,
        save_json_path=assignment_json_path,
        show_member_labels=show_member_labels,
    )

    if not section_name_by_member:
        raise RuntimeError("Section assignment was cancelled or no members were assigned.")

    section_by_member = build_section_props_by_member(
        section_name_by_member=section_name_by_member,
        catalog=catalog,
        units_in=units_in,
        units_out=units_out,
        print_props=False,
    )

    if assignment_report_path:
        write_section_assignment_report(assignment_report_path, section_name_by_member)

    if print_summary:
        used = sorted(set(section_name_by_member.values()))
        print(
            f"# SECTION SETUP | mode=user | assigned_members={len(section_name_by_member)} | "
            f"used_sections={used}"
        )

    return None, section_by_member, section_name_by_member
