def define_material(mdef, name=None):
    """
    Normalizza una definizione materiale dell'utente.

    Input supportati (tuple/list):
      - ("elastic", fyk, E, G)
      - ("steel01", fyk, E, G, b)
      - ("PPmat",  fyk, E, G, epsyP, [epsyN], [eps0])
      - ("steel02", fyk, E, G, b)

    Output: dict con campi standardizzati.
    Nota: OpenSees userà solo un sottoinsieme dei campi, ma qui li teniamo tutti
    perché servono anche a sezioni/verifiche.
    """

    if not isinstance(mdef, (tuple, list)) or len(mdef) == 0:
        raise TypeError("Material definition must be a non-empty tuple/list, e.g. ('elastic', fyk, E, G).")

    mtype = str(mdef[0]).strip()
    mkey = mtype.lower()

    def _as_float(x, label):
        try:
            return float(x)
        except Exception:
            raise TypeError(f"Material '{mtype}': parameter '{label}' must be a number, got {x!r}")

    # -------------------- ELASTIC --------------------
    if mkey == "elastic":
        if len(mdef) != 4:
            raise ValueError("Elastic material must be: ('elastic', fyk, E, G)")
        _, fyk, E, G = mdef

        fyk = _as_float(fyk, "fyk")
        E   = _as_float(E,   "E")
        G   = _as_float(G,   "G")

        epsy = (fyk / E) if (E != 0.0) else None

        return {
            "name": name,
            "type": "elastic",
            "fyk": fyk,
            "E": E,
            "G": G,
            "epsy": epsy,       # utile per verifiche
            # campi "opzionali" uniformati
            "b": None,
            "epsyP": None,
            "epsyN": None,
            "eps0": 0.0,
        }

    # -------------------- STEEL01 --------------------
    if mkey == "steel01":
        if len(mdef) != 5:
            raise ValueError("Steel01 material must be: ('steel01', fyk, E, G, b)")
        _, fyk, E, G, b = mdef

        fyk = _as_float(fyk, "fyk")
        E   = _as_float(E,   "E")
        G   = _as_float(G,   "G")
        b   = _as_float(b,   "b")

        if b < 0.0:
            raise ValueError("Steel02: 'b' must be >= 0.0 (0.0 = perfectly plastic).")

        epsy = (fyk / E) if (E != 0.0) else None

        return {
            "name": name,
            "type": "steel01",
            "fyk": fyk,
            "E": E,
            "G": G,
            "b": b,
            "epsy": epsy,
            # campi non usati da Steel01 ma mantenuti per coerenza
            "epsyP": None,
            "epsyN": None,
            "eps0": 0.0,
        }

    # -------------------- ELASTIC PERFECTLY PLASTIC (ElasticPP) --------------------
    # Utente lo chiama "PPmat"
    if mkey == "ppmat":
        # forme accettate:
        # ("PPmat", fyk, E, G, epsyP)
        # ("PPmat", fyk, E, G, epsyP, epsyN)
        # ("PPmat", fyk, E, G, epsyP, epsyN, eps0)
        if len(mdef) not in (5, 6, 7):
            raise ValueError(
                "PPmat must be: ('PPmat', fyk, E, G, epsyP, [epsyN], [eps0])"
            )

        _, fyk, E, G, epsyP, *rest = mdef
        fyk  = _as_float(fyk,  "fyk")
        E    = _as_float(E,    "E")
        G    = _as_float(G,    "G")
        epsyP = _as_float(epsyP, "epsyP")

        epsyN = epsyP
        eps0  = 0.0
        if len(rest) >= 1 and rest[0] is not None:
            epsyN = _as_float(rest[0], "epsyN")
        if len(rest) >= 2 and rest[1] is not None:
            eps0 = _as_float(rest[1], "eps0")

        # epsy teorico (da snervamento) lo teniamo comunque
        epsy = (fyk / E) if (E != 0.0) else None

        return {
            "name": name,
            "type": "ppmat",     # tuo alias
            "fyk": fyk,
            "E": E,
            "G": G,
            "epsy": epsy,
            "epsyP": epsyP,
            "epsyN": epsyN,
            "eps0": eps0,
            "b": None,
        }
# -------------------- STEEL02 --------------------
    if mkey == "steel02":
        if len(mdef) != 5:
            raise ValueError("Steel02 material must be: ('steel02', fyk, E, G, b)")
        _, fyk, E, G, b = mdef

        fyk = _as_float(fyk, "fyk")
        E   = _as_float(E,   "E")
        G   = _as_float(G,   "G")
        b   = _as_float(b,   "b")

        if b < 0.0:
            raise ValueError("Steel02: 'b' must be >= 0.0 (0.0 = perfectly plastic).")

        epsy = (fyk / E) if (E != 0.0) else None

        return {
            "name": name,
            "type": "steel02",
            "fyk": fyk,
            "E": E,
            "G": G,
            "b": b,
            "epsy": epsy,
            # campi non usati da Steel02 ma mantenuti per coerenza
            "epsyP": None,
            "epsyN": None,
            "eps0": 0.0,
        }
# -------------------- Hardening --------------------
    if mkey == "hardening":
        if len(mdef) != 5:
            raise ValueError("Hardening material must be: ('Hardening', E, fyk, H_iso, H_kin)")
        _, E, fyk, H_iso, H_kin = mdef

        E     = _as_float(E, "E")
        fyk   = _as_float(fyk, "fyk")
        H_iso = _as_float(H_iso, "H_iso")
        H_kin = _as_float(H_kin, "H_kin")

        if H_kin < 0.0:
            raise ValueError("Hardening: 'H_kin' must be >= 0.0.")

        epsy = (fyk / E) if (E != 0.0) else None

        return {
            "name": name,
            "type": "hardening",
            "fyk": fyk,
            "E": E,
            "H_iso": H_iso,
            "H_kin": H_kin,
            "epsy": epsy,
            "epsyP": None,
            "epsyN": None,
            "eps0": 0.0,
            "G": 80000000,
        }

    # -------------------- UNKNOWN --------------------
    raise ValueError(f"Unknown material type '{mtype}'. Supported: 'elastic', 'steel01', 'PPmat', 'Hardening'.")
