def export_dxf(
    G,
    M=None,
    path="geometry.dxf",
    export_nodes=True,
    export_members=True,
    export_faces=True,
    export_boundary=True,
    print_summary=True,          
):
    """
    Export geometry/mesh to a simple ASCII DXF (R12).
    - nodes  -> POINT entities (layer: NODES)
    - members -> LINE entities (layer: MEMBERS)
    - faces -> 3DFACE entities (layer: FACES)  [if provided]
    - boundary_edges -> LINE entities (layer: BOUNDARY)
    """

    nodes = G["nodes"]
    members = []
    faces = []
    boundary_edges = []

    if M is not None:
        members = M.get("members", []) or []
        faces = M.get("faces", []) or []
        boundary_edges = M.get("boundary_edges", []) or []

    def _dxf_header():
        return [
            "0","SECTION",
            "2","HEADER",
            "9","$ACADVER",
            "1","AC1009",  # AutoCAD R12
            "0","ENDSEC",
            "0","SECTION",
            "2","TABLES",
            "0","ENDSEC",
            "0","SECTION",
            "2","ENTITIES",
        ]

    def _dxf_footer():
        return ["0","ENDSEC","0","EOF"]

    def _layer(ent_lines, layer_name):
        # Insert layer name (group code 8)
        # For each entity, add: 8, <layer>
        out = []
        it = iter(ent_lines)
        for code in it:
            out.append(code)
            val = next(it)
            out.append(val)
            # After entity type line: 0, <TYPE> insert layer
            if code == "0":
                out += ["8", layer_name]
        return out

    def _point(x, y, z):
        return [
            "0","POINT",
            "10", f"{x:.6f}",
            "20", f"{y:.6f}",
            "30", f"{z:.6f}",
        ]

    def _line(x1, y1, z1, x2, y2, z2):
        return [
            "0","LINE",
            "10", f"{x1:.6f}",
            "20", f"{y1:.6f}",
            "30", f"{z1:.6f}",
            "11", f"{x2:.6f}",
            "21", f"{y2:.6f}",
            "31", f"{z2:.6f}",
        ]

    def _3dface(p1, p2, p3, p4):
        # DXF 3DFACE uses 10/20/30, 11/21/31, 12/22/32, 13/23/33
        # If triangle: set p4 = p3
        (x1,y1,z1) = p1
        (x2,y2,z2) = p2
        (x3,y3,z3) = p3
        (x4,y4,z4) = p4
        return [
            "0","3DFACE",
            "10", f"{x1:.6f}", "20", f"{y1:.6f}", "30", f"{z1:.6f}",
            "11", f"{x2:.6f}", "21", f"{y2:.6f}", "31", f"{z2:.6f}",
            "12", f"{x3:.6f}", "22", f"{y3:.6f}", "32", f"{z3:.6f}",
            "13", f"{x4:.6f}", "23", f"{y4:.6f}", "33", f"{z4:.6f}",
        ]

    out = []
    out += _dxf_header()

    # ---------- NODES ----------
    if export_nodes:
        ents = []
        for nid in sorted(nodes.keys()):
            x, y, z = nodes[nid]
            ents += _point(x, y, z)
        out += _layer(ents, "NODES")

    # ---------- MEMBERS ----------
    if export_members and members:
        ents = []
        for (a, b) in members:
            x1, y1, z1 = nodes[a]
            x2, y2, z2 = nodes[b]
            ents += _line(x1, y1, z1, x2, y2, z2)
        out += _layer(ents, "MEMBERS")

    # ---------- BOUNDARY ----------
    if export_boundary and boundary_edges:
        ents = []
        for (a, b) in boundary_edges:
            x1, y1, z1 = nodes[a]
            x2, y2, z2 = nodes[b]
            ents += _line(x1, y1, z1, x2, y2, z2)
        out += _layer(ents, "BOUNDARY")

    # ---------- FACES ----------
    # faces in your barrel: (eid, n1, n2, n3, n4)
    if export_faces and faces:
        ents = []
        for face in faces:
            if len(face) == 5:
                _, n1, n2, n3, n4 = face
                ents += _3dface(nodes[n1], nodes[n2], nodes[n3], nodes[n4])
            elif len(face) == 4:
                # triangle: (eid, n1, n2, n3)
                _, n1, n2, n3 = face
                ents += _3dface(nodes[n1], nodes[n2], nodes[n3], nodes[n3])
        out += _layer(ents, "FACES")

    out += _dxf_footer()

    # write file
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # ------------------ PRINT SUMMARY ------------------
    if print_summary:
        n_nodes = len(nodes)

        n_member_lines = len(members) if export_members else 0
        n_boundary_lines = len(boundary_edges) if export_boundary else 0
        n_lines = n_member_lines + n_boundary_lines

        print(f"DXF export completed: {path}")
        print(f"  nodes generated : {n_nodes}")
        print(f"  lines generated : {n_lines}")


    return path
