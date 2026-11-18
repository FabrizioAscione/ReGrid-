class PyWriter:
    def __init__(self):
        self._lines = []

    def add(self, text=""):
        """Aggiunge una o più righe (stringa con \n ammessi)."""
        if text is None:
            return
        s = str(text)
        if not s.strip():
            self._lines.append("")
            return
        self._lines.extend(s.splitlines())

    def blank(self):
        self._lines.append("")

    def header(self, title):
        self._lines.append("")
        self._lines.append(f"# {'='*20} {title} {'='*20}")

    def text(self):
        return "\n".join(self._lines)

    def save(self, filepath, as_py=True):
        """
        Salva su file. Se as_py=True scrive anche import openseespy.opensees as ops in testa (se manca).
        """
        lines = list(self._lines)

        if as_py:
            has_ops_import = any("import openseespy.opensees as ops" in ln for ln in lines)
            if not has_ops_import:
                lines = ["import openseespy.opensees as ops", ""] + lines
    

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            print("MODEL SAVED")

