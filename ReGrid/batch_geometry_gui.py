import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_DIR, "outputs", "batch_runs")


class BatchGeometryApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch geometry runs")
        self.root.geometry("1100x760")

        self.run_count_var = tk.IntVar(value=3)
        self.output_root_var = tk.StringVar(value=DEFAULT_OUTPUT_ROOT)
        self.status_var = tk.StringVar(value="Ready")
        self.run_button = None
        self.rows_container = None
        self.canvas = None
        self.rows = []

        self._build_ui()
        self._rebuild_rows()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill="x")

        ttk.Label(controls, text="N run").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=100,
            textvariable=self.run_count_var,
            width=8,
            command=self._rebuild_rows,
        ).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Button(controls, text="Update rows", command=self._rebuild_rows).grid(row=0, column=2, sticky="w")

        ttk.Label(controls, text="Output root").grid(row=0, column=3, sticky="w", padx=(18, 0))
        ttk.Entry(controls, textvariable=self.output_root_var, width=55).grid(row=0, column=4, sticky="ew", padx=(6, 0))
        controls.columnconfigure(4, weight=1)

        hint = (
            "Compila geom, B, L, f, nx, ny e grid per ogni run. "
            "Ogni analisi verra lanciata in sequenza e salvera i risultati in una sottocartella dedicata."
        )
        ttk.Label(outer, text=hint, wraplength=980, justify="left").pack(anchor="w", pady=(10, 8))

        table_wrap = ttk.Frame(outer)
        table_wrap.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(table_wrap, highlightthickness=0)
        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.canvas.yview)
        self.rows_container = ttk.Frame(self.canvas)

        self.rows_container.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.rows_container, anchor="nw")
        self.canvas.configure(yscrollcommand=yscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self._build_header()

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(10, 6))

        ttk.Button(buttons, text="Fill defaults", command=self._fill_defaults).pack(side="left")
        ttk.Button(buttons, text="Save batch table", command=self._save_batch_table).pack(side="left", padx=(8, 0))
        self.run_button = ttk.Button(buttons, text="Run batch", command=self._start_batch)
        self.run_button.pack(side="right")

        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(2, 8))

        self.log = tk.Text(outer, height=14, wrap="word")
        self.log.pack(fill="both", expand=False)

    def _build_header(self):
        headers = ["Run", "Label", "geom", "B", "L", "f", "nx", "ny", "grid"]
        widths = [8, 16, 8, 10, 10, 10, 8, 8, 8]
        for col, (title, width) in enumerate(zip(headers, widths)):
            ttk.Label(
                self.rows_container,
                text=title,
                width=width,
                anchor="center",
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=col, padx=4, pady=(0, 6), sticky="ew")

    def _rebuild_rows(self):
        target = max(1, int(self.run_count_var.get()))

        while len(self.rows) < target:
            idx = len(self.rows) + 1
            row_index = len(self.rows) + 1

            row = {
                "label": tk.StringVar(value=f"run_{idx:02d}"),
                "geom": tk.StringVar(value="1"),
                "B": tk.StringVar(value="30"),
                "L": tk.StringVar(value="30"),
                "f": tk.StringVar(value="3.75"),
                "nx": tk.StringVar(value="11"),
                "ny": tk.StringVar(value="11"),
                "grid": tk.StringVar(value="1"),
            }

            ttk.Label(self.rows_container, text=f"{idx}", width=8, anchor="center").grid(
                row=row_index, column=0, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["label"], width=16).grid(
                row=row_index, column=1, padx=4, pady=2
            )
            ttk.Combobox(self.rows_container, textvariable=row["geom"], values=("1", "2", "3", "4"), width=8, state="readonly").grid(
                row=row_index, column=2, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["B"], width=10).grid(
                row=row_index, column=3, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["L"], width=10).grid(
                row=row_index, column=4, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["f"], width=10).grid(
                row=row_index, column=5, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["nx"], width=8).grid(
                row=row_index, column=6, padx=4, pady=2
            )
            ttk.Entry(self.rows_container, textvariable=row["ny"], width=8).grid(
                row=row_index, column=7, padx=4, pady=2
            )
            ttk.Combobox(self.rows_container, textvariable=row["grid"], values=("1", "2"), width=8, state="readonly").grid(
                row=row_index, column=8, padx=4, pady=2
            )
            self.rows.append(row)

        while len(self.rows) > target:
            row_index = len(self.rows)
            for widget in self.rows_container.grid_slaves(row=row_index):
                widget.destroy()
            self.rows.pop()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fill_defaults(self):
        for idx, row in enumerate(self.rows, start=1):
            row["label"].set(f"run_{idx:02d}")
            row["geom"].set("1")
            row["B"].set("30")
            row["L"].set("30")
            row["f"].set("3.75")
            row["nx"].set("11")
            row["ny"].set("11")
            row["grid"].set("1")

    def _collect_runs(self):
        runs = []
        for idx, row in enumerate(self.rows, start=1):
            label = row["label"].get().strip() or f"run_{idx:02d}"
            safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
            run = {
                "label": safe_label,
                "geom": int(row["geom"].get()),
                "B": float(row["B"].get()),
                "L": float(row["L"].get()),
                "f": float(row["f"].get()),
                "nx": int(row["nx"].get()),
                "ny": int(row["ny"].get()),
                "grid": int(row["grid"].get()),
            }
            runs.append(run)
        return runs

    def _save_batch_table(self):
        try:
            runs = self._collect_runs()
        except Exception as exc:
            messagebox.showerror("Invalid batch", str(exc))
            return

        output_root = self.output_root_var.get().strip() or DEFAULT_OUTPUT_ROOT
        os.makedirs(output_root, exist_ok=True)
        batch_path = os.path.join(output_root, "batch_table.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump({"runs": runs}, f, indent=2)
        self._append_log(f"Saved batch table: {batch_path}")

    def _start_batch(self):
        try:
            runs = self._collect_runs()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        output_root = self.output_root_var.get().strip() or DEFAULT_OUTPUT_ROOT
        os.makedirs(output_root, exist_ok=True)

        self.run_button.configure(state="disabled")
        self.status_var.set(f"Running {len(runs)} analyses...")
        self._append_log("")
        self._append_log(f"Batch start | output root = {output_root}")

        worker = threading.Thread(target=self._run_batch_worker, args=(runs, output_root), daemon=True)
        worker.start()

    def _run_batch_worker(self, runs, output_root):
        ok_count = 0
        fail_count = 0

        for idx, run in enumerate(runs, start=1):
            run_dir = os.path.join(output_root, run["label"])
            os.makedirs(run_dir, exist_ok=True)

            config_path = os.path.join(run_dir, "batch_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(run, f, indent=2)

            cmd = [sys.executable, "unitled.py", "--batch-config", config_path, "--output-dir", run_dir]
            self._ui_log(f"[{idx}/{len(runs)}] Launching {run['label']} -> {run_dir}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._ui_log(line.rstrip())
                code = proc.wait()
            except Exception as exc:
                code = -1
                self._ui_log(f"Execution error for {run['label']}: {exc}")

            if code == 0:
                ok_count += 1
                self._ui_log(f"[{idx}/{len(runs)}] Completed {run['label']}")
            else:
                fail_count += 1
                self._ui_log(f"[{idx}/{len(runs)}] Failed {run['label']} | exit code {code}")

        self.root.after(0, lambda: self._finish_batch(ok_count, fail_count))

    def _finish_batch(self, ok_count, fail_count):
        self.run_button.configure(state="normal")
        self.status_var.set(f"Completed | ok={ok_count} fail={fail_count}")
        if fail_count == 0:
            messagebox.showinfo("Batch complete", f"All runs completed successfully: {ok_count}")
        else:
            messagebox.showwarning("Batch complete", f"Completed with failures. ok={ok_count}, fail={fail_count}")

    def _append_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _ui_log(self, text):
        self.root.after(0, lambda: self._append_log(text))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    BatchGeometryApp().run()
