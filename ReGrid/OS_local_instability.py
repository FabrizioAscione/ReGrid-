# file: OS_local_instability.py
import os
import math
import numpy as np
import openseespy.opensees as ops


# =========================================================================================
# small geometry helpers
# =========================================================================================
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def delta_mid(node_i, node_m, node_j):
    """Distanza del punto M dalla corda IJ, in configurazione deformata."""
    xi = ops.nodeCoord(int(node_i))
    xj = ops.nodeCoord(int(node_j))
    xm = ops.nodeCoord(int(node_m))
    ui = ops.nodeDisp(int(node_i))
    uj = ops.nodeDisp(int(node_j))
    um = ops.nodeDisp(int(node_m))

    Xi = (xi[0] + ui[0], xi[1] + ui[1], xi[2] + ui[2])
    Xj = (xj[0] + uj[0], xj[1] + uj[1], xj[2] + uj[2])
    Xm = (xm[0] + um[0], xm[1] + um[1], xm[2] + um[2])

    IJ = _sub(Xj, Xi)
    den = _norm(IJ)
    if den < 1e-18:
        return 0.0

    MI = _sub(Xm, Xi)
    MJ = _sub(Xm, Xj)
    num = _norm(_cross(MI, MJ))
    return num / den


def axial_N_forceBeamColumn(eleTag):
    """Per forceBeamColumn: basicForce[0] = N."""
    try:
        bf = ops.eleResponse(int(eleTag), "basicForce")
        if bf is not None and len(bf) > 0:
            return float(bf[0])
    except Exception:
        pass
    return None


def _basic_deformation_safe(eleTag: int):
    """
    Ritorna la deformazione del sistema basic dell'elemento beam-column.
    Fallback su eleResponse('basicDeformation') se serve.
    """
    try:
        v = ops.basicDeformation(int(eleTag))
        if v is not None:
            return np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        pass

    try:
        v = ops.eleResponse(int(eleTag), "basicDeformation")
        if v is not None:
            return np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        pass

    return None


def _basic_stiffness_safe(eleTag: int):
    """
    Ritorna la matrice di rigidezza tangent (basic system) in forma 2D numpy.
    """
    try:
        k = ops.basicStiffness(int(eleTag))
        if k is None:
            return None
        k = np.asarray(k, dtype=float).reshape(-1)
        n = int(round(math.sqrt(len(k))))
        if n * n != len(k):
            return None
        return k.reshape((n, n))
    except Exception:
        return None


# =========================================================================================
# Local Instability Monitor
# =========================================================================================
class LocalInstabilityMonitor:
    """
    Monitor locale per membro.

    Criteri disponibili (combinabili):
      (A) Energia: ΔU_member < -|dU_eps| (per energy_consec step) => instabile live
          ΔU_e = 0.5 * Δδ_e^T * Kt_e * Δδ_e   (basic system)
          ΔU_member = somma(ΔU_e) sui sub-elementi del membro

      (B) Buckling "P-peak then drop" su N(dmid)/N(dend):
          >>> in questa versione NON flagga live: registra SOLO curve.
          Il post-process legge curves_txt e decide se c'è davvero stata riduzione post-picco.

    mode (combinazione tra energia e P-drop):
      - "energy_only": live solo energia
      - "either": (live energia) OR (post pdrop)  -> in pratica: energia live, pdrop solo post
      - "both": energia live e pdrop post (il merge lo fai in OS_pushdown)
    """

    def __init__(
        self,
        member_map,
        events_txt="outputs/local_instability_events.txt",
        curves_txt="outputs/local_instability_curves.txt",

        # ---- P-drop (buckling) ----
        pdrop_ratio=0.02,       # drop relativo rispetto al picco: N < Nmax*(1-pdrop_ratio)
        pdrop_consec=2,         # passi consecutivi con drop in post-process
        N_use="comp_pos",       # "raw" | "abs" | "comp_pos" (compressione positiva)
        d_tol=1e-6,             # tolleranza per decidere ↑/↓ su dmid/dend in post-process
        use_dmid=True,
        use_dend=True,

        # ---- energy ΔU ----
        mode="either",
        U_eps=None,
        dU_eps=None,
        dd_basic_min=1e-10,
        energy_consec=2,

        verbose=False,
    ):
        self.member_map = member_map if member_map is not None else {}
        self.events_txt = str(events_txt) if events_txt is not None else None
        self.curves_txt = str(curves_txt) if curves_txt is not None else None

        # pdrop config (USED IN POSTPROCESS)
        self.pdrop_ratio = float(pdrop_ratio)
        self.pdrop_consec = int(pdrop_consec)
        self.N_use = str(N_use).lower().strip()
        self.d_tol = float(d_tol)
        self.use_dmid = bool(use_dmid)
        self.use_dend = bool(use_dend)

        # energy config (USED LIVE)
        self.mode = str(mode).lower().strip() if mode is not None else "either"
        if self.mode not in ("energy_only", "either", "both"):
            self.mode = "either"
        if dU_eps is None:
            dU_eps = 0.0 if U_eps is None else float(U_eps)
        self.dU_eps = float(dU_eps)
        self.dd_basic_min = float(dd_basic_min)
        self.energy_consec = int(energy_consec)

        self.verbose = bool(verbose)

        # stato per energia: contatore consecutivi per membro
        self.energy_drop_count = {k: 0 for k in self.member_map}

        # cache deformazioni basic e stiffness (prev/curr) per energia
        self._basic_def_cache_prev = {}
        self._basic_def_cache_now = {}
        self._basic_stiff_cache_now = {}

        # flagged (solo energia live, in questa versione)
        self.flagged = {}  # key -> dict(step, LF, Ftot, reason, ...)

        # last-step per mappe energetiche: eleTag -> dUe
        self.last_step_DU_ele_map = {}

        # init files
        if self.events_txt:
            self._ensure_parent_dir(self.events_txt)
            with open(self.events_txt, "w", encoding="utf-8") as f:
                f.write("# step\tmember\tLF\tFtot\treason\tbuckling_mode\tNraw\tNuse\tdmid\tdend\tDU_member\n")

        if self.curves_txt:
            self._ensure_parent_dir(self.curves_txt)
            with open(self.curves_txt, "w", encoding="utf-8") as f:
                f.write("# step\tmember\tLF\tFtot\tNraw\tNuse\tdmid\tdend\tDU_member\n")

    # ---------------------------------------------------------------------
    # utilities
    # ---------------------------------------------------------------------
    def _ensure_parent_dir(self, path: str):
        try:
            if path is None:
                return
            d = os.path.dirname(str(path))
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    def _append_line(self, path: str, line: str):
        if path is None:
            return
        try:
            with open(str(path), "a", encoding="utf-8") as f:
                f.write(line.rstrip("\n") + "\n")
        except Exception:
            pass

    def _N_transform(self, Nraw: float) -> float:
        if Nraw is None:
            return None
        if self.N_use == "raw":
            return float(Nraw)
        if self.N_use == "comp_pos":
            # compressione positiva assumendo Nraw<0 in compressione
            return float(-Nraw)
        # "abs"
        return float(abs(Nraw))

    def _dend(self, node_i: int, node_j: int) -> float:
        """
        Misura semplice di 'dend': norma della differenza di spostamento tra i due end nodes.
        """
        try:
            ui = ops.nodeDisp(int(node_i))
            uj = ops.nodeDisp(int(node_j))
            if ui is None or uj is None:
                return 0.0
            ui = np.asarray(ui, dtype=float).reshape(-1)
            uj = np.asarray(uj, dtype=float).reshape(-1)
            n = min(len(ui), len(uj))
            if n <= 0:
                return 0.0
            du = ui[:n] - uj[:n]
            return float(np.linalg.norm(du))
        except Exception:
            return 0.0

    # ---------------------------------------------------------------------
    # ENERGY: cache per step
    # ---------------------------------------------------------------------
    def _cache_step_basic_quantities(self):
        self._basic_def_cache_prev = dict(self._basic_def_cache_now)
        self._basic_def_cache_now = {}
        self._basic_stiff_cache_now = {}

        for key, info in self.member_map.items():
            subE = [int(e) for e in info.get("subEles", [])]
            for e in subE:
                if e not in self._basic_def_cache_now:
                    self._basic_def_cache_now[e] = _basic_deformation_safe(e)
                if e not in self._basic_stiff_cache_now:
                    self._basic_stiff_cache_now[e] = _basic_stiffness_safe(e)

    def _compute_member_energy_increment(self, subE, dd_basic_min=1e-10):
        """
        Calcola DU_member e DU_by_ele (per mappe), usando Δδ e Kt nel basic system.
        """
        DU_member = 0.0
        DU_by_ele = {}

        for e in subE:
            e = int(e)
            d_now = self._basic_def_cache_now.get(e, None)
            d_prev = self._basic_def_cache_prev.get(e, None)
            Kt = self._basic_stiff_cache_now.get(e, None)
            if d_now is None or d_prev is None or Kt is None:
                DU_by_ele[e] = 0.0
                continue

            dd = (d_now - d_prev).reshape(-1)
            if float(np.linalg.norm(dd)) < float(dd_basic_min):
                DU_by_ele[e] = 0.0
                continue

            try:
                v = float(0.5 * dd.T.dot(Kt).dot(dd))
            except Exception:
                v = 0.0

            DU_by_ele[e] = float(v)
            DU_member += float(v)

        return float(DU_member), DU_by_ele

    # ---------------------------------------------------------------------
    # MAIN UPDATE (live): energy flag + always record curves
    # ---------------------------------------------------------------------
    def update(self, step: int, LF: float = None, Ftot: float = None, delta_control: float = None):
        """
        Live update:
          - calcola energia ΔU e (se abilitato) flagga live per energia
          - registra SEMPRE su curves_txt: Nraw/Nuse/dmid/dend/DU_member (serve per post)
        """
        step = int(step)

        # cache basic quantities (energia)
        self._cache_step_basic_quantities()

        # reset last-step ele map
        self.last_step_DU_ele_map = {}

        newly_flagged = []

        for key, info in self.member_map.items():
            already_flagged = (key in self.flagged)

            # required geometry keys
            try:
                di = int(info["i"])
                dj = int(info["j"])
                dm = int(info["m"])
            except Exception:
                continue

            subE = [int(e) for e in info.get("subEles", [])]
            if not subE:
                continue

            # --- dmid / dend (always computed; used in post) ---
            try:
                dmid = float(delta_mid(di, dm, dj))
            except Exception:
                dmid = 0.0

            dend = float(self._dend(di, dj))

            # --- member axial N (average on sub-elements) ---
            Ns = []
            for e in subE:
                n = axial_N_forceBeamColumn(e)
                if n is not None:
                    Ns.append(float(n))

            Nraw = None
            Nuse = None
            if Ns:
                Nraw = float(sum(Ns) / len(Ns))
                Nuse = self._N_transform(Nraw)

            # --- energy increment ---
            DU_member, DU_by_ele = self._compute_member_energy_increment(subE, dd_basic_min=self.dd_basic_min)

            # store per-element for plotting (sub-elements) ALWAYS
            for eTag, dUe in DU_by_ele.items():
                self.last_step_DU_ele_map[int(eTag)] = float(dUe)

            # record curves ALWAYS (for post buckling)
            if self.curves_txt is not None:
                self._append_line(
                    self.curves_txt,
                    f"{step}\t{key}\t"
                    f"{(float(LF) if LF is not None else 0.0):.12g}\t"
                    f"{(float(Ftot) if Ftot is not None else 0.0):.12g}\t"
                    f"{(float(Nraw) if Nraw is not None else float('nan')):.12g}\t"
                    f"{(float(Nuse) if Nuse is not None else float('nan')):.12g}\t"
                    f"{float(dmid):.12g}\t{float(dend):.12g}\t{float(DU_member):.12g}"
                )

            # if already flagged, do not re-trigger logic (but we DID record + energy map)
            if already_flagged:
                continue

            # ---------------- live ENERGY criterion ----------------
            energy_trigger = False
            if self.mode in ("energy_only", "either", "both"):
                if float(DU_member) < -abs(float(self.dU_eps)):
                    self.energy_drop_count[key] = int(self.energy_drop_count.get(key, 0) + 1)
                else:
                    self.energy_drop_count[key] = 0

                if int(self.energy_drop_count.get(key, 0)) >= int(self.energy_consec):
                    energy_trigger = True
            else:
                self.energy_drop_count[key] = 0

            if energy_trigger:
                self.flagged[key] = dict(
                    step=int(step),
                    LF=None if LF is None else float(LF),
                    Ftot=None if Ftot is None else float(Ftot),
                    reason="energy",
                    buckling_mode="unknown",
                    Nraw=Nraw,
                    Nuse=Nuse,
                    dmid=dmid,
                    dend=dend,
                    DU_member=DU_member,
                )
                newly_flagged.append(key)

                if self.events_txt is not None:
                    self._append_line(
                        self.events_txt,
                        f"{step}\t{key}\t"
                        f"{(float(LF) if LF is not None else 0.0):.12g}\t"
                        f"{(float(Ftot) if Ftot is not None else 0.0):.12g}\t"
                        f"energy\tunknown\t"
                        f"{(float(Nraw) if Nraw is not None else float('nan')):.12g}\t"
                        f"{(float(Nuse) if Nuse is not None else float('nan')):.12g}\t"
                        f"{float(dmid):.12g}\t{float(dend):.12g}\t{float(DU_member):.12g}"
                    )

        return newly_flagged

    # ---------------------------------------------------------------------
    # helper: return unstable sub-elements for plotting (energy or post)
    # ---------------------------------------------------------------------
    def unstable_sub_elements(self):
        tags = set()
        for _, info in self.member_map.items():
            for e in info.get("subEles", []):
                tags.add(int(e))
        # NOT filtered here; filtering is done by caller if needed
        # (historically you used unstable tags union; keep behaviour simple)
        flagged_members = set(self.flagged.keys())
        unstable = set()
        for k in flagged_members:
            info = self.member_map.get(k, None)
            if not info:
                continue
            for e in info.get("subEles", []):
                unstable.add(int(e))
        return unstable

    # ---------------------------------------------------------------------
    # POST-PROCESS: P-drop buckling from curves file
    # ---------------------------------------------------------------------
    def postprocess_pdrop_from_curves(
        self,
        drop_ratio=None,
        drop_consec=None,
        local_mode=None,
        use_dmid=None,
        use_dend=None,
        d_tol=None,
        out_events_txt=None,
    ):
        """
        Legge curves_txt (salvato live) e valuta buckling P-drop in post.
        Restituisce:
          - post_events: lista dict(evento)
          - nd_curves: dict con curve N/Nmax-dmid e N/Nmax-dend per primo/ultimo membro instabilizzato

        NOTE:
          - Qui assumiamo Nuse già "compressione positiva" se N_use="comp_pos".
          - Se Nuse è NaN o Nmax <=0 -> skip.
        """
        curves_path = self.curves_txt
        if curves_path is None:
            return [], None

        drop_ratio = float(self.pdrop_ratio if drop_ratio is None else drop_ratio)
        drop_consec = int(self.pdrop_consec if drop_consec is None else drop_consec)
        d_tol = float(self.d_tol if d_tol is None else d_tol)
        use_dmid = bool(self.use_dmid if use_dmid is None else use_dmid)
        use_dend = bool(self.use_dend if use_dend is None else use_dend)

        # read curves
        data = []  # list of (step, member, LF, Ftot, Nraw, Nuse, dmid, dend, DU)
        try:
            with open(curves_path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) < 9:
                        continue
                    step = int(float(parts[0]))
                    member = parts[1]
                    LF = float(parts[2])
                    Ftot = float(parts[3])
                    Nraw = float(parts[4])
                    Nuse = float(parts[5])
                    dmid_v = float(parts[6])
                    dend_v = float(parts[7])
                    DU = float(parts[8])
                    data.append((step, member, LF, Ftot, Nraw, Nuse, dmid_v, dend_v, DU))
        except Exception:
            return [], None

        # group by member
        by_m = {}
        for row in data:
            by_m.setdefault(row[1], []).append(row)

        post_events = []

        def _mode_from(dmid_at_peak, dend_at_peak, dmid_now, dend_now):
            up_mid = (dmid_now - dmid_at_peak) > d_tol
            up_end = (dend_now - dend_at_peak) > d_tol
            if up_end and (not up_mid):
                return "1"
            if up_mid and (not up_end):
                return "2"
            if up_mid and up_end:
                return "3"
            return "unknown"

        # detect event per member
        for m, rows in by_m.items():
            rows = sorted(rows, key=lambda r: r[0])
            Nuse_arr = np.array([r[5] for r in rows], dtype=float)
            if len(Nuse_arr) < max(5, drop_consec + 2):
                continue

            # skip if all nan
            if not np.isfinite(Nuse_arr).any():
                continue

            # Nmax over finite values
            Nmax = float(np.nanmax(Nuse_arr))
            if (not np.isfinite(Nmax)) or (Nmax <= 0.0):
                continue

            # peak index (first occurrence of max)
            peak_idx = int(np.nanargmax(Nuse_arr))
            dmid_peak = float(rows[peak_idx][6])
            dend_peak = float(rows[peak_idx][7])

            thr = (1.0 - float(drop_ratio)) * float(Nmax)

            # search post-peak for drop_consec consecutive steps below thr
            consec = 0
            event_idx = None
            for i in range(peak_idx + 1, len(rows)):
                Ni = float(rows[i][5])
                if (np.isfinite(Ni) and (Ni < thr)):
                    consec += 1
                else:
                    consec = 0

                if consec >= int(drop_consec):
                    # candidate event at i (end of run)
                    # but require kinematic growth condition on dmid/dend (at least one enabled)
                    dmid_i = float(rows[i][6])
                    dend_i = float(rows[i][7])

                    kin_ok = False
                    if use_dmid and (dmid_i - dmid_peak) > d_tol:
                        kin_ok = True
                    if use_dend and (dend_i - dend_peak) > d_tol:
                        kin_ok = True

                    if kin_ok:
                        event_idx = i
                        break

            if event_idx is None:
                continue

            step, _, LF, Ftot, Nraw, Nuse, dmid_i, dend_i, DU = rows[event_idx]
            mode_id = _mode_from(dmid_peak, dend_peak, dmid_i, dend_i)

            ev = dict(
                step=int(step),
                member=m,
                LF=float(LF),
                Ftot=float(Ftot),
                reason="pdrop_post",
                buckling_mode=str(mode_id),
                Nmax=float(Nmax),
                Nraw=float(Nraw),
                Nuse=float(Nuse),
                dmid=float(dmid_i),
                dend=float(dend_i),
                DU_member=float(DU),
            )
            post_events.append(ev)

        # sort events by step
        post_events.sort(key=lambda e: (int(e.get("step", 10**12)), str(e.get("member", ""))))

        # write/append events file
        if out_events_txt is None:
            out_events_txt = self.events_txt
        if out_events_txt:
            self._ensure_parent_dir(out_events_txt)
            # append (non riscriviamo: potresti avere già eventi energia live)
            for ev in post_events:
                self._append_line(
                    out_events_txt,
                    f"{int(ev['step'])}\t{ev['member']}\t"
                    f"{float(ev['LF']):.12g}\t{float(ev['Ftot']):.12g}\t"
                    f"pdrop_post\t{ev.get('buckling_mode','unknown')}\t"
                    f"{float(ev.get('Nraw', float('nan'))):.12g}\t"
                    f"{float(ev.get('Nuse', float('nan'))):.12g}\t"
                    f"{float(ev.get('dmid', 0.0)):.12g}\t{float(ev.get('dend', 0.0)):.12g}\t"
                    f"{float(ev.get('DU_member', 0.0)):.12g}"
                )

        # build nd_curves for first/last buckling member (for plotting)
        nd_curves = None
        if post_events:
            first_m = post_events[0]["member"]
            last_m = post_events[-1]["member"]

            def _make_curves(member_id):
                rows = sorted(by_m.get(member_id, []), key=lambda r: r[0])
                Nuse_arr = np.array([r[5] for r in rows], dtype=float)
                Nmax = float(np.nanmax(Nuse_arr)) if np.isfinite(Nuse_arr).any() else None
                if Nmax is None or (not np.isfinite(Nmax)) or Nmax <= 0.0:
                    return None

                dmid_arr = np.array([r[6] for r in rows], dtype=float)
                dend_arr = np.array([r[7] for r in rows], dtype=float)
                Nn = Nuse_arr / float(Nmax)

                return dict(
                    step=np.array([r[0] for r in rows], dtype=int),
                    LF=np.array([r[2] for r in rows], dtype=float),
                    Nnorm=Nn,
                    dmid=dmid_arr,
                    dend=dend_arr,
                )

            nd_curves = dict(
                first_member=first_m,
                last_member=last_m,
                first=_make_curves(first_m),
                last=_make_curves(last_m),
            )

        return post_events, nd_curves
