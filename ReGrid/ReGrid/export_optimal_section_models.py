"""Render the best saved PM, PC and Ps designs using their assigned sections."""

import ast
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from fiber_section_shapes import build_fiber_patch_specs
from saved_pareto_loader import load_saved_pareto_front


ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.grid": False})
EXPORT_DPI = 600
AREA_TO_CM2 = 1e4
SOURCE_CMAPS = {"new": ("Aste nuove", "Blues"), "reused": ("Aste riusate", "Oranges")}


def area_cm2(row):
    return float(row["A"]) * AREA_TO_CM2


def area_normalizer(rows):
    areas = [area_cm2(row) for row in rows]
    low, high = min(areas), max(areas)
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 0.1)
        low, high = low - padding, high + padding
    return matplotlib.colors.PowerNorm(gamma=0.7, vmin=low, vmax=high)


def area_colormap(source):
    base = plt.get_cmap(SOURCE_CMAPS[source][1])
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        f"{source}_area", (base(0.1), base(0.99))
    )


def member_color(row, area_norm):
    source = row["source"].strip().lower()
    if source not in SOURCE_CMAPS:
        raise ValueError(f"Provenienza asta non riconosciuta: {row['source']!r}")
    return area_colormap(source)(area_norm(area_cm2(row)))


def area_legend_html(area_norm, sources=None):
    low, high = area_norm.vmin, area_norm.vmax
    entries = []
    sources = SOURCE_CMAPS if sources is None else sources
    for source in sources:
        label, _ = SOURCE_CMAPS[source]
        cmap = area_colormap(source)
        start = matplotlib.colors.to_hex(cmap(0.0))
        end = matplotlib.colors.to_hex(cmap(1.0))
        entries.append(
            f'<div class="key"><strong>{label}</strong>'
            f'<span class="gradient" style="background:linear-gradient(90deg,{start},{end})"></span>'
            f'<span class="range">{low:.1f}–{high:.1f} cm²</span></div>'
        )
    return "".join(entries)


def section_name(row):
    name = row["section_name"]
    if row["section_family"] == "CHS":
        name = f"CHS {float(row['dext'])*1000:g} x {float(row['t'])*1000:g}"
    return name


def patch_polygons(section):
    polygons = []
    for patch in build_fiber_patch_specs(section):
        if patch["kind"] == "rect":
            y1, y2, z1, z2 = (patch[k] for k in ("y1", "y2", "z1", "z2"))
            polygons.append(np.array([(y1,z1),(y2,z1),(y2,z2),(y1,z2)]))
        elif patch["kind"] == "quad":
            polygons.append(np.array(patch["points"]))
        elif patch["kind"] == "circ":
            # Annular sectors preserve the bore, including at both member ends.
            angles = np.linspace(0, 2*np.pi, 33)
            center = np.array([patch["yC"], patch["zC"]])
            outer = center + patch["rO"]*np.column_stack([np.cos(angles),np.sin(angles)])
            inner = center + patch["rI"]*np.column_stack([np.cos(angles),np.sin(angles)])
            for i in range(32):
                polygons.append(np.array([outer[i],outer[i+1],inner[i+1],inner[i]]))
        else:
            raise ValueError(patch["kind"])
    return polygons


def read_model(folder):
    nodes = {int(r[0]):r[1:4] for r in np.loadtxt(folder/"nodes.txt")}
    members = np.loadtxt(folder/"members.txt")
    rows = []
    for filename in ("sections_reused_used.csv", "sections_new_used.csv"):
        with (folder/filename).open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream, delimiter=";"))
    sections = {tuple(sorted((int(r["member_i"]),int(r["member_j"])))):r for r in rows}
    assert len(sections) == len(rows) == len(members), "Incomplete/duplicate member assignments"
    transforms = {}
    for statement in ast.parse((folder/"opensees_EL_model_export.py").read_text(encoding="utf-8-sig")).body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "geomTransf":
                args = [ast.literal_eval(a) for a in call.args]
                transforms[args[1]] = np.array(args[2:5], dtype=float)
    return nodes, members, sections, transforms


def build_model(folder, area_norm):
    nodes, members, sections, transforms = read_model(folder)
    faces, face_colors, segments, line_colors, section_records = [], [], [], [], []
    for member in members:
        mid, ni, nj = map(int, member[:3])
        a,b = nodes[ni],nodes[nj]
        row = sections[tuple(sorted((ni,nj)))]
        assert row["section_units"] == "m"
        assert abs(np.linalg.norm(b-a)-float(row["member_length_m"])) < 1e-5
        name = section_name(row)
        color = member_color(row, area_norm)
        sec = {k:float(row[k]) for k in ("h","b","tf","tw","dext","t","dint") if row.get(k)}
        sec["section_family"] = row["section_family"]
        x = (b-a)/np.linalg.norm(b-a)
        y = np.cross(transforms[mid],x)
        y /= np.linalg.norm(y)
        z = np.cross(x,y)
        assert np.allclose(np.cross(x,y),z) and abs(np.dot(x,y)) < 1e-10
        polygons = patch_polygons(sec)
        area = sum(abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))/2 for p in polygons)
        assert np.isclose(area,float(row["A"]),rtol=0.008), (name,area,row["A"])
        for polygon in polygons:
            offsets = polygon[:,0,None]*y + polygon[:,1,None]*z
            start,end = a+offsets,b+offsets
            faces.extend([start[::-1],end])
            for i in range(len(polygon)):
                j = (i+1)%len(polygon)
                faces.append(np.array([start[i],start[j],end[j],end[i]]))
            face_colors.extend([color]*(len(polygon)+2))
        segments.append([a,b])
        line_colors.append(color)
        section_records.append({"member_id":mid,"node_i":ni,"node_j":nj,"section":name,
                    "source":row["source"],"area_cm2":area_cm2(row),
                    "length_m":float(row["member_length_m"])})
    return dict(faces=faces, face_colors=face_colors, segments=np.array(segments),
                line_colors=line_colors, sections=section_records, nodes=np.array(list(nodes.values())))


def decorate_3d(ax, model):
    points = model["nodes"]
    low,high = points.min(axis=0),points.max(axis=0)
    ax.set(xlim=(low[0]-.5,high[0]+.5),ylim=(low[1]-.5,high[1]+.5),zlim=(low[2]-.3,high[2]+.3),
           xlabel="X [m]",ylabel="Y [m]",zlabel="Z [m]")
    ax.set_box_aspect(high-low)
    ax.view_init(elev=28,azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.line.set_linewidth(.5)
        axis.set_pane_color((1,1,1,0))


def save_plot(model, title, path, area_norm, solid=False, plan=False):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111) if plan else fig.add_subplot(111,projection="3d")
    if plan:
        ax.add_collection(LineCollection(model["segments"][:,:,:2],colors=model["line_colors"],linewidths=3.0))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(xlabel="X [m]",ylabel="Y [m]")
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
    else:
        if solid:
            ax.add_collection3d(Poly3DCollection(model["faces"],facecolors=model["face_colors"],
                                                edgecolors=model["face_colors"],linewidths=0,shade=True,zsort="average"))
        else:
            ax.add_collection3d(Line3DCollection(model["segments"],colors=model["line_colors"],linewidths=3.0))
        decorate_3d(ax,model)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_position([0.01,0.04,0.64,0.90])
    sources = [source for source in SOURCE_CMAPS
               if any(row["source"].strip().lower() == source for row in model["sections"])]
    for index, source in enumerate(sources):
        label, _ = SOURCE_CMAPS[source]
        colorbar_ax = fig.add_axes([0.70, 0.63 - index * 0.29, 0.27, 0.04])
        scalar = matplotlib.cm.ScalarMappable(norm=area_norm, cmap=area_colormap(source))
        colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_ticks(np.linspace(area_norm.vmin, area_norm.vmax, 4))
        colorbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
        colorbar.ax.set_title(f"{label} · Area [cm²]", loc="left", fontsize=9, pad=6)
    fig.suptitle(title,fontsize=10,y=0.97)
    fig.savefig(path.with_suffix(".png"),dpi=EXPORT_DPI,transparent=True)
    fig.savefig(path.with_suffix(".pdf"),transparent=True)
    plt.close(fig)


def save_section_catalog(sections, area_norm, output_dir):
    entries = sorted(sections.items())
    fig,axes = plt.subplots(int(np.ceil(len(entries)/6)),6,figsize=(12,2.1*int(np.ceil(len(entries)/6))))
    for ax,((name,source),row) in zip(axes.flat,entries):
        sec = {k:float(row[k]) for k in ("h","b","tf","tw","dext","t","dint") if row.get(k)}
        sec["section_family"] = row["section_family"]
        for polygon in patch_polygons(sec):
            ax.add_patch(Polygon(polygon*1000,facecolor=member_color(row,area_norm),edgecolor="none"))
        ax.autoscale()
        ax.margins(.2)
        ax.set_aspect("equal")
        ax.set_title(f"{source}: {name}",fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("mm",fontsize=7)
        ax.set_axis_off()
    for ax in list(axes.flat)[len(entries):]:
        ax.set_axis_off()
    fig.suptitle("Sezioni nominali — blu: nuove; verde: riusate; intensità: area",fontsize=10)
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(output_dir/"catalogo_sezioni.png",dpi=EXPORT_DPI)
    fig.savefig(output_dir/"catalogo_sezioni.pdf")
    plt.close(fig)


def save_details(folder, title, path, area_norm):
    _,_,rows,_ = read_model(folder)
    representatives=[]
    for family in ("CHS","L","2L","4L"):
        candidates=[r for r in rows.values() if r["section_family"] == family]
        if candidates:
            representatives.append(max(candidates,key=lambda r:float(r["A"])))
    fig=plt.figure(figsize=(8,6))
    for index,row in enumerate(representatives,start=1):
        ax=fig.add_subplot(2,2,index,projection="3d")
        sec={k:float(row[k]) for k in ("h","b","tf","tw","dext","t","dint") if row.get(k)}
        sec["section_family"]=row["section_family"]
        faces=[]
        for polygon in patch_polygons(sec):
            a=np.column_stack([np.zeros(len(polygon)),polygon])
            b=a+np.array([.6,0,0])
            faces.extend([a[::-1],b])
            for i in range(len(polygon)):
                j=(i+1)%len(polygon)
                faces.append(np.array([a[i],a[j],b[j],b[i]]))
        color=member_color(row,area_norm)
        ax.add_collection3d(Poly3DCollection(faces,facecolors=[color]*len(faces),
                                            edgecolors=[color]*len(faces),linewidths=0,shade=True))
        vertices=np.vstack(faces)
        low,high=vertices.min(axis=0),vertices.max(axis=0)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1]-.01,high[1]+.01),zlim=(low[2]-.01,high[2]+.01))
        ax.set_box_aspect(high-low+np.array([0,.02,.02]))
        ax.view_init(elev=20,azim=-35)
        ax.set_axis_off()
        ax.set_title(section_name(row),fontsize=9)
    fig.suptitle(title+" — dettagli dei profili, tratti da 0,60 m",fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=EXPORT_DPI,transparent=True)
    fig.savefig(path.with_suffix(".pdf"),transparent=True)
    plt.close(fig)


def save_obj(model,path):
    with path.open("w",encoding="utf-8") as stream:
        stream.write("# Nominal model sections. Coordinates in metres.\n")
        index=1
        for face in model["faces"]:
            for vertex in face:
                stream.write("v "+" ".join(f"{value:.8f}" for value in vertex)+"\n")
            stream.write("f "+" ".join(str(index+i) for i in range(len(face)))+"\n")
            index+=len(face)


VIEWER = r'''<!doctype html><html lang="it"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:14px Arial;margin:0;background:#f7f7f7;color:#222}header{padding:16px;background:white}h1{font-size:20px;margin:0 0 8px}button{padding:8px 14px;margin:5px;cursor:pointer}canvas{display:block;width:100%;height:72vh;touch-action:none}details{padding:16px;background:white}.key{display:grid;grid-template-columns:130px minmax(120px,320px) auto;align-items:center;gap:12px;margin:12px 0}.gradient{height:16px;border:1px solid #888}.range{font-variant-numeric:tabular-nums}@media(max-width:600px){.key{grid-template-columns:1fr;gap:5px}}</style>
<header><h1>__TITLE__</h1><p>__METRICS__</p><p>Trascina per ruotare · Rotella per ingrandire · Profili e coordinate in scala reale (metri)</p>
<button onclick="reset()">Vista 3D</button><button onclick="az=0;el=Math.PI/2;draw()">Pianta</button><button onclick="solid=!solid;draw()">Solidi / aste</button>
<p>Forme nominali del modello, tubi cavi; raccordi, bulloni e giunti non modellati.</p></header><canvas id="c"></canvas><details open><summary>Legenda delle sezioni (numero di aste)</summary>__LEGEND__</details>
<script>const D=__DATA__,c=document.getElementById('c'),ctx=c.getContext('2d');let az=-.96,el=.49,zoom=1,solid=true,drag=null;
function reset(){az=-.96;el=.49;zoom=1;draw()}
function rotate(p){let x=p[0]-D.center[0],y=p[1]-D.center[1],z=p[2]-D.center[2];let u=Math.cos(az)*x+Math.sin(az)*y,v=-Math.sin(az)*x+Math.cos(az)*y;return [u,-Math.sin(el)*v+Math.cos(el)*z,Math.cos(el)*v+Math.sin(el)*z]}
function draw(){const ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*ratio;c.height=h*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);const scale=Math.min(w,h)*.78/D.span*zoom;const project=p=>{let q=rotate(p);return [w/2+q[0]*scale,h/2-q[1]*scale,q[2]]};
if(solid){let fs=D.faces.map((f,i)=>{let p=f.map(project);return {p,col:D.colors[i],d:p.reduce((s,q)=>s+q[2],0)/p.length}});fs.sort((a,b)=>a.d-b.d);for(let f of fs){ctx.beginPath();f.p.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();let a=f.p[0],b=f.p[1],e=f.p[2],ux=b[0]-a[0],uy=b[1]-a[1],uz=(b[2]-a[2])*scale,vx=e[0]-a[0],vy=e[1]-a[1],vz=(e[2]-a[2])*scale;let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx,n=Math.hypot(nx,ny,nz)||1;let light=.55+.45*Math.abs((nx*.3+ny*.4+nz*.86)/n);ctx.fillStyle='rgb('+f.col.map(v=>Math.round(v*light)).join(',')+')';ctx.fill()}}
else{D.lines.forEach((l,i)=>{let a=project(l[0]),b=project(l[1]);ctx.strokeStyle='rgb('+D.lineColors[i].join(',')+')';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()})}
ctx.fillStyle='#333';ctx.font='13px Arial';ctx.fillText('Scala sezioni 1:1 rispetto alla geometria',20,h-18)}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.008;el=Math.max(-1.56,Math.min(1.57,el+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};c.onpointerup=()=>drag=null;c.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(30,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;draw();</script></html>'''


def save_viewer(model,title,metrics,path,area_norm):
    points=model["nodes"]
    data={"faces":[np.round(f,7).tolist() for f in model["faces"]],
          "colors":[[round(255*v) for v in col[:3]] for col in model["face_colors"]],
          "lines":model["segments"].tolist(),
          "lineColors":[[round(255*v) for v in col[:3]] for col in model["line_colors"]],
          "center":((points.min(axis=0)+points.max(axis=0))/2).tolist(),"span":float(np.ptp(points,axis=0).max())}
    sources = [source for source in SOURCE_CMAPS
               if any(row["source"].strip().lower() == source for row in model["sections"])]
    legend=area_legend_html(area_norm, sources)
    content=VIEWER.replace("__TITLE__",html.escape(title)).replace("__METRICS__",html.escape(metrics)).replace("__LEGEND__",legend).replace("__DATA__",json.dumps(data,separators=(",",":")))
    path.write_text(content,encoding="utf-8")


def main(source_dir=ROOT/"outputs"):
    source_dir=Path(source_dir)
    if not source_dir.is_absolute():
        source_dir=ROOT/source_dir
    output_dir=source_dir/"optimal_section_models"
    output_dir.mkdir(parents=True,exist_ok=True)
    pareto,all_evaluations,_ = load_saved_pareto_front(output_dir=str(source_dir))
    feasible=[e for e in all_evaluations if e.feasible]
    selected=[]
    section_catalog={}
    for label,objective in (("PM","pi_mass"),("PC","pi_cost"),("Ps","pi_ec")):
        best=max(pareto,key=lambda e:e.objectives[objective])
        assert np.isclose(best.objectives[objective],max(e.objectives[objective] for e in feasible))
        cid=int(best.meta["candidate_id"])
        folder=next((source_dir/"pareto_nonlinear_runs").glob(f"*reused*candidate_{cid}"))
        _,_,sections,_=read_model(folder)
        for row in sections.values():
            source=row["source"].strip().lower()
            section_catalog[(section_name(row),source)]=row
        selected.append((label,objective,best,folder,area_normalizer(sections.values())))
    catalog_area_norm=area_normalizer(section_catalog.values())
    summary=[]
    for label,objective,best,folder,area_norm in selected:
        cid=int(best.meta["candidate_id"])
        stem=f"{label}_candidate_{cid}"
        title=f"Ottimo {label} — candidato {cid}"
        model=build_model(folder,area_norm)
        metrics=" · ".join(f"{k} = {best.objectives[k]:.6f}" for k in ("pi_mass","pi_cost","pi_ec"))
        save_plot(model,title,output_dir/(stem+"_sezioni_3D"),area_norm,solid=True)
        save_plot(model,title,output_dir/(stem+"_aste_3D"),area_norm)
        save_plot(model,title,output_dir/(stem+"_pianta"),area_norm,plan=True)
        save_details(folder,title,output_dir/(stem+"_dettagli_profili"),area_norm)
        save_obj(model,output_dir/(stem+"_sezioni_3D.obj"))
        save_viewer(model,title,metrics,output_dir/(stem+"_interattivo.html"),area_norm)
        with (output_dir/(stem+"_assegnazioni.csv")).open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(model["sections"][0]),delimiter=";")
            writer.writeheader();writer.writerows(model["sections"])
        summary.append({"optimum":label,"objective":objective,"candidate_id":cid,
                        **{k:best.objectives.get(k,best.meta.get(k)) for k in ("pi_mass","pi_cost","pi_ec","mass","cost","ec")},
                        "members":len(model["segments"]),"sections":len({r["section"] for r in model["sections"]}),
                        "source":str(folder.relative_to(ROOT))})
        print(stem,metrics,"members",len(model["segments"]),"faces",len(model["faces"]),flush=True)
    save_section_catalog(section_catalog,catalog_area_norm,output_dir)
    with (output_dir/"soluzioni_ottimali.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),delimiter=";")
        writer.writeheader();writer.writerows(summary)
    (output_dir/"soluzioni_ottimali.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    links="".join(f'<li><a href="{label}_candidate_{int(best.meta["candidate_id"])}_interattivo.html">Ottimo {label} — candidato {int(best.meta["candidate_id"])}</a></li>' for label,_,best,_,_ in selected)
    (output_dir/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Soluzioni ottimali</title><style>body{font:16px Arial;margin:40px}li{margin:20px}</style><h1>Soluzioni ottimali PM, PC e Ps</h1><p>Modelli 3D ruotabili con sezioni nominali in scala reale, disponibili anche offline.</p><ul>'+links+'</ul><p><a href="catalogo_sezioni.png">Catalogo sezioni e legenda per area/provenienza</a></p>',encoding="utf-8")
    (output_dir/"LEGGIMI.txt").write_text("PM = massimo pi_mass; PC = massimo pi_cost; Ps = massimo pi_ec.\nSelezione verificata su tutte le soluzioni ammissibili salvate.\nGeometria indeformata dai nodes.txt delle analisi salvate; sezioni e dimensioni SI dai CSV della stessa analisi.\nAssi locali dalle geomTransf del modello OpenSees esportato.\nSezioni nominali estruse senza amplificazione; CHS cavi con 32 lati, L/2L/4L dalle patch del modello.\nNon sono rappresentati raccordi, saldature, bulloni, giunti o perdita locale di spessore per corrosione.\n2L/4L rappresentano la sezione composta definita dal modello, non dettagli di assemblaggio.\nColori per area e provenienza: ogni soluzione usa il proprio intervallo di area, comune alle aste nuove e riusate; blu per aste nuove e arancione per aste riusate. Il catalogo usa l'intervallo complessivo.\nI file HTML sono autonomi e funzionano offline: trascinare per ruotare, rotella per zoom.\nPNG e PDF: vista con profili, vista ad aste, pianta. Font Arial 9.\n",encoding="utf-8")


def export_new_solution(source_dir):
    source_dir = Path(source_dir)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    output_dir = source_dir / "optimal_section_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = list((source_dir / "pareto_nonlinear_runs").glob("run_*_fixed_base_pareto_*_candidate_fixed_base"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one fixed-base model in {source_dir}, found {len(candidates)}")
    folder = candidates[0]
    _, _, sections, _ = read_model(folder)
    sources = {row["source"].strip().lower() for row in sections.values()}
    if sources != {"new"}:
        raise ValueError(f"Expected a new-only fixed-base model in {folder}, found sources {sources}")
    area_norm = area_normalizer(sections.values())
    model = build_model(folder, area_norm)
    title = f"Soluzione nuova — {source_dir.name}"
    stem = output_dir / "soluzione_nuova_fixed_base"
    save_plot(model, title, stem.with_name(stem.name + "_sezioni_3D"), area_norm, solid=True)
    save_plot(model, title, stem.with_name(stem.name + "_aste_3D"), area_norm)
    save_plot(model, title, stem.with_name(stem.name + "_pianta"), area_norm, plan=True)
    print(f"{source_dir.name}: exported new-only area maps for {len(model['segments'])} members", flush=True)


if __name__ == "__main__":
    import argparse

    parser=argparse.ArgumentParser(description="Esporta i modelli delle soluzioni ottimali salvate.")
    parser.add_argument("output_dirs",nargs="*",default=["outputs"],
                        help="Cartelle sorgente dei risultati (default: outputs).")
    parser.add_argument("--new-only", action="store_true",
                        help="Esporta le mappe delle sole soluzioni fixed-base con aste nuove.")
    args=parser.parse_args()
    for output_dir in args.output_dirs:
        if args.new_only:
            export_new_solution(output_dir)
        else:
            main(output_dir)
