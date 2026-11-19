import openseespy.opensees as ops


def _normalize_mass(mass, ndf):
    if mass is None:
        return [0.0] * ndf
    if isinstance(mass, (int, float)):
        return [float(mass)] * ndf
    if len(mass) != ndf:
        raise ValueError(f"mass must have length ndf={ndf}. Got len(mass)={len(mass)}.")
    return [float(m) for m in mass]



def define_nodes(G, ndf, set_mass=False, mass=None):
    """
    Define OpenSees nodes from the geometry dictionary `G`.

    Returns
    -------
    py_text : str
        Text of the executed OpenSees commands.
    node_tags : list[int]
        Created node tags, sorted increasingly.
    """
    nodes = G.get("nodes", {})
    if not nodes:
        raise ValueError("G['nodes'] is empty or missing.")

    ndf = int(ndf)
    if ndf <= 0:
        raise ValueError("ndf must be a positive integer.")

    mass_vec = _normalize_mass(mass, ndf) if set_mass else None

    py_lines = []
    node_tags = []

    for tag, coords in sorted(nodes.items(), key=lambda t: int(t[0])):
        if len(coords) != 3:
            raise ValueError(f"Node {tag} must have exactly 3 coordinates. Got {coords}.")

        tag = int(tag)
        x, y, z = (float(coords[0]), float(coords[1]), float(coords[2]))

        ops.node(tag, x, y, z)
        py_lines.append(f"ops.node({tag}, {x}, {y}, {z})")
        node_tags.append(tag)

        if set_mass:
            ops.mass(tag, *mass_vec)
            py_lines.append(f"ops.mass({tag}, {', '.join(str(m) for m in mass_vec)})")

    return "\n".join(py_lines), node_tags

