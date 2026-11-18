"""Interactive editor for the saved four-panel Pareto figure. No analyses run."""

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import is_color_like
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
import numpy as np

from generate_pareto_3d_views import build_combined_pareto
from pareto_plot_style import PARETO_AXIS_TICKS
from reuse_genetic_opt import objective_coordinate_matrix
from saved_pareto_loader import load_saved_pareto_front


ROOT = Path(__file__).resolve().parent


def default_settings(names):
    panels=[]
    for indices in ((0,1,2),(0,2),(0,1),(1,2)):
        panels.append({"axes":[{"min":PARETO_AXIS_TICKS.get(names[i],(0,1))[0],
                                "max":PARETO_AXIS_TICKS.get(names[i],(0,1))[-1],
                                "step":PARETO_AXIS_TICKS.get(names[i],(0,1))[1]-PARETO_AXIS_TICKS.get(names[i],(0,1))[0],
                                "label":""} for i in indices],
                       "visible":True,"elev":20,"azim":-120,"texts":[]})
    return {"width":12,"height":9,"dpi":600,"font":9,"axis_width":.5,
            "pareto_size":7,"feasible_size":5,"pareto_color":"#C00000",
            "feasible_color":"#d9d9d9","edge_color":"black","transparent":True,
            "left":.10,"right":.975,"bottom":.055,"top":.98,"wspace":.32,"hspace":.14,
            "panels":panels}


def panel_rectangle(settings, index):
    """Normalized panel footprint; older configurations use the automatic grid."""
    custom=settings["panels"][index].get("rectangle")
    if custom is not None:
        return list(custom)
    width=(settings["right"]-settings["left"])/(2+settings["wspace"])
    height=(settings["top"]-settings["bottom"])/(2+settings["hspace"])
    row,col=divmod(index,2)
    return [settings["left"]+col*width*(1+settings["wspace"]),
            settings["top"]-height-row*height*(1+settings["hspace"]),width,height]


def validate_settings(settings):
    for key in ("width","height","dpi","font","axis_width","pareto_size","feasible_size"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key}: inserire un numero positivo.")
    if settings["width"]>50 or settings["height"]>50 or settings["dpi"]>1200:
        raise ValueError("Massimo 50 cm per lato e 1200 dpi.")
    if not (0<=settings["left"]<settings["right"]<=1 and 0<=settings["bottom"]<settings["top"]<=1):
        raise ValueError("Margini: usare valori fra 0 e 1; sinistra < destra, basso < alto.")
    if not all(math.isfinite(settings[k]) and 0<=settings[k]<=3 for k in ("wspace","hspace")):
        raise ValueError("Le distanze tra pannelli devono essere fra 0 e 3.")
    for key in ("pareto_color","feasible_color","edge_color"):
        if not is_color_like(settings[key]):
            raise ValueError(f"Colore non valido: {settings[key]}")
    if len(settings["panels"]) != 4:
        raise ValueError("Servono quattro pannelli.")
    for index,panel in enumerate(settings["panels"]):
        rectangle=panel.get("rectangle")
        if rectangle is not None:
            if len(rectangle)!=4 or not all(math.isfinite(v) for v in rectangle):
                raise ValueError("Dimensioni del pannello non valide.")
            left,bottom,width,height=rectangle
            if left<0 or bottom<0 or width<=0 or height<=0 or left+width>1+1e-9 or bottom+height>1+1e-9:
                raise ValueError("Il pannello deve avere dimensioni positive e rimanere dentro l'immagine.")
        if not math.isfinite(panel.get("zoom",1.15)) or not 0.1<=panel.get("zoom",1.15)<=3:
            raise ValueError("Ingrandimento 3D: scegliere un valore fra 0,1 e 3.")
        if len(panel["axes"]) != (3 if index==0 else 2):
            raise ValueError("Numero di assi non valido.")
        for axis in panel["axes"]:
            lo,hi,step=(axis[k] for k in ("min","max","step"))
            if not all(math.isfinite(v) for v in (lo,hi,step)) or lo>=hi or step<=0 or (hi-lo)/step>100:
                raise ValueError("Assi: minimo < massimo, passo positivo, massimo 100 intervalli.")


def apply_settings(fig, settings):
    validate_settings(settings)
    fig.set_size_inches(settings["width"]/2.54,settings["height"]/2.54)
    fig.subplots_adjust(**{k:settings[k] for k in ("left","right","bottom","top","wspace","hspace")})
    for index,(ax,panel) in enumerate(zip(fig.axes,settings["panels"])):
        if panel.get("rectangle") is not None:
            ax.set_position(panel["rectangle"])
        setters=(ax.set_xlim,ax.set_ylim,ax.set_zlim) if index==0 else (ax.set_xlim,ax.set_ylim)
        axes=(ax.xaxis,ax.yaxis,ax.zaxis) if index==0 else (ax.xaxis,ax.yaxis)
        for axis_index,(axis,setter,values) in enumerate(zip(axes,setters,panel["axes"])):
            lo,hi,step=(values[k] for k in ("min","max","step"))
            setter(lo,hi)
            ticks=lo+np.arange(int(np.floor((hi-lo)/step+1e-8))+1)*step
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_label_text(values["label"])
            axis.labelpad=2
            if index==0:
                axis.line.set_linewidth(settings["axis_width"])
                axis.set_label_text("")
                position=values.get("label_position",((.8,-.08),(.1,.02),(-.12,.55))[axis_index])
                label=ax.text2D(*position,values["label"],transform=ax.transAxes,
                                ha="center",va="center",picker=True)
            else:
                label=axis.label
                label.set_picker(True)
                if "label_position" in values:
                    axis.set_label_coords(*values["label_position"])
            label._pareto_axis_label=(index,axis_index)
        for spine in ax.spines.values():
            spine.set_linewidth(settings["axis_width"])
        ax.tick_params(axis="both",labelsize=settings["font"],width=settings["axis_width"])
        if index==0:
            ax.view_init(elev=panel["elev"],azim=panel["azim"])
            ax.set_box_aspect((4,4,3),zoom=panel.get("zoom",1.15))
            ax.tick_params(axis="z",labelsize=settings["font"],width=settings["axis_width"])
        ax.set_axis_on() if panel["visible"] else ax.set_axis_off()
        for j,collection in enumerate(ax.collections):
            pareto=j==len(ax.collections)-1
            collection.set_facecolor(settings["pareto_color"] if pareto else settings["feasible_color"])
            collection.set_edgecolor(settings["edge_color"] if pareto else "none")
            collection.set_sizes([(settings["pareto_size"] if pareto else settings["feasible_size"])**2])
        for text_index,entry in enumerate(panel["texts"]):
            writer=ax.text2D if index==0 else ax.text
            text=writer(entry["x"],entry["y"],entry["text"],transform=ax.transAxes,
                        ha="center",va="center",rotation=entry.get("rotation",0),picker=True)
            text._pareto_text=(index,text_index)
        for axis in axes:
            axis.get_ticklabels()
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontsize(settings["font"])


class ParetoEditor:
    def __init__(self,root,output):
        self.root,self.output=root,Path(output)
        pareto,evaluations,self.names=load_saved_pareto_front(output_dir=str(output))
        self.feasible=objective_coordinate_matrix([e for e in evaluations if e.feasible],self.names)
        self.pareto=objective_coordinate_matrix(pareto,self.names)
        self.settings=default_settings(self.names)
        self.fig=self.canvas=self.toolbar=None
        self.drag=None
        self.panel_index=0
        self.vars={}
        root.title("Editor grafici Pareto")
        root.geometry("1240x850")
        sidebar=ttk.Frame(root,width=330)
        sidebar.pack(side="left",fill="y")
        tabs=ttk.Notebook(sidebar)
        tabs.pack(fill="both",expand=True,padx=8,pady=8)
        general,axes_tab,layout,panel_tab=(ttk.Frame(tabs,padding=8) for _ in range(4))
        tabs.add(general,text="Stile");tabs.add(axes_tab,text="Assi / testi");tabs.add(layout,text="Spazi")
        tabs.add(panel_tab,text="Pannello")
        for key,label in (("width","Larghezza [cm]"),("height","Altezza [cm]"),("dpi","DPI esportazione"),
                          ("font","Arial [pt]"),("axis_width","Spessore assi [pt]"),
                          ("pareto_size","Marker Pareto [pt]"),("feasible_size","Altri marker [pt]")):
            self.field(general,key,label)
        for key,label in (("pareto_color","Colore Pareto"),("feasible_color","Altri punti"),("edge_color","Bordo Pareto")):
            self.field(general,key,label)
            ttk.Button(general,text="Scegli colore…",command=lambda k=key:self.choose_color(k)).pack(anchor="e",pady=(0,6))
        self.transparent=tk.BooleanVar(value=True)
        ttk.Checkbutton(general,text="Sfondo trasparente",variable=self.transparent).pack(anchor="w")
        self.panel=tk.StringVar(value="3D")
        choices=["3D",f"{self.names[0]} / {self.names[2]}",f"{self.names[0]} / {self.names[1]}",f"{self.names[1]} / {self.names[2]}"]
        self.choices=choices
        selector=ttk.Combobox(axes_tab,textvariable=self.panel,values=choices,state="readonly",width=28)
        selector.pack(fill="x",pady=(0,10));selector.bind("<<ComboboxSelected>>",self.select_panel)
        panel_selector=ttk.Combobox(panel_tab,textvariable=self.panel,values=choices,state="readonly",width=28)
        panel_selector.pack(fill="x",pady=(0,10))
        panel_selector.bind("<<ComboboxSelected>>",self.select_panel)
        self.manual_rectangle=tk.BooleanVar(value=False)
        ttk.Checkbutton(panel_tab,text="Dimensioni e posizione personalizzate",
                        variable=self.manual_rectangle).pack(anchor="w",pady=8)
        self.panel_vars={}
        for key,label in (("x","Distanza da sinistra [cm]"),("y","Distanza dal basso [cm]"),
                          ("width","Larghezza grafico [cm]"),("height","Altezza grafico [cm]"),
                          ("zoom","Ingrandimento vista 3D")):
            row=ttk.Frame(panel_tab);row.pack(fill="x",pady=5)
            ttk.Label(row,text=label).pack(side="left")
            self.panel_vars[key]=tk.StringVar()
            ttk.Entry(row,textvariable=self.panel_vars[key],width=9).pack(side="right")
        ttk.Label(panel_tab,text="Seleziona il grafico, attiva le dimensioni personalizzate e premi Applica.\n\n"
                  "Le misure descrivono l'area del grafico, senza le scritte esterne.\n\n"
                  "Per il 3D, usa anche Ingrandimento: valori maggiori ingrandiscono la vista nel riquadro.\n\n"
                  "Disattiva l'opzione per tornare alla griglia automatica.",wraplength=300).pack(pady=12)
        self.axis_vars=[]
        for letter in "XYZ":
            frame=ttk.LabelFrame(axes_tab,text=f"Asse {letter}",padding=5)
            frame.pack(fill="x",pady=3)
            values={}
            for key,label in (("min","Min"),("max","Max"),("step","Passo"),("label","Nome asse")):
                var=tk.StringVar();values[key]=var
                row=ttk.Frame(frame);row.pack(fill="x")
                ttk.Label(row,text=label,width=10).pack(side="left")
                ttk.Entry(row,textvariable=var,width=17).pack(side="right")
            self.axis_vars.append(values)
        self.visible=tk.BooleanVar(value=True)
        ttk.Checkbutton(axes_tab,text="Mostra assi e numeri",variable=self.visible).pack(anchor="w",pady=4)
        for key,label in (("elev","Elevazione 3D [°]"),("azim","Azimut 3D [°]")):
            self.vars[key]=tk.StringVar()
            row=ttk.Frame(axes_tab);row.pack(fill="x")
            ttk.Label(row,text=label).pack(side="left")
            ttk.Entry(row,textvariable=self.vars[key],width=10).pack(side="right")
        self.annotation=tk.StringVar()
        ttk.Label(axes_tab,text="Testo libero (trascinabile con il mouse)").pack(anchor="w",pady=(12,0))
        ttk.Entry(axes_tab,textvariable=self.annotation).pack(fill="x")
        ttk.Button(axes_tab,text="Aggiungi al pannello",command=self.add_text).pack(fill="x",pady=3)
        ttk.Button(axes_tab,text="Rimuovi ultimo testo",command=self.remove_text).pack(fill="x")
        for key,label in (("left","Margine sinistro"),("right","Limite destro"),("bottom","Margine basso"),
                          ("top","Limite alto"),("wspace","Distanza orizzontale"),("hspace","Distanza verticale")):
            self.field(layout,key,label)
        ttk.Label(layout,text="Margini: frazioni dell'immagine, da 0 a 1.",wraplength=280).pack(pady=10)
        for label,command in (("Applica / aggiorna anteprima",self.update),
                              ("Esporta PNG / PDF…",self.export),
                              ("Salva configurazione…",self.save_config),
                              ("Carica configurazione…",self.load_config)):
            ttk.Button(sidebar,text=label,command=command).pack(fill="x",padx=10,pady=3)
        self.status=tk.StringVar(value="I testi liberi si trascinano. Nel 3D trascina lo sfondo per ruotare.")
        ttk.Label(sidebar,textvariable=self.status,wraplength=310).pack(padx=10,pady=10)
        self.preview=ttk.Frame(root);self.preview.pack(side="right",fill="both",expand=True)
        self.load_fields();self.render()

    def field(self,parent,key,label):
        row=ttk.Frame(parent);row.pack(fill="x",pady=3)
        ttk.Label(row,text=label).pack(side="left")
        self.vars[key]=tk.StringVar(value=str(self.settings[key]))
        ttk.Entry(row,textvariable=self.vars[key],width=14).pack(side="right")

    def load_fields(self):
        for key,var in self.vars.items():
            if key in self.settings:
                var.set(str(self.settings[key]))
        self.transparent.set(self.settings["transparent"])
        panel=self.settings["panels"][self.panel_index]
        self.manual_rectangle.set(panel.get("rectangle") is not None)
        left,bottom,width,height=panel_rectangle(self.settings,self.panel_index)
        for key,value in (("x",left*self.settings["width"]),("y",bottom*self.settings["height"]),
                          ("width",width*self.settings["width"]),("height",height*self.settings["height"]),
                          ("zoom",panel.get("zoom",1.15))):
            self.panel_vars[key].set(f"{value:.8g}")
        for i,values in enumerate(self.axis_vars):
            for key,var in values.items():
                var.set(str(panel["axes"][i][key]) if i<len(panel["axes"]) else "")
        self.visible.set(panel["visible"])
        self.vars["elev"].set(str(panel["elev"]))
        self.vars["azim"].set(str(panel["azim"]))

    def capture(self):
        # Work on a copy so invalid fields never corrupt the last valid settings.
        settings=json.loads(json.dumps(self.settings))
        for key,var in self.vars.items():
            if key in settings:
                settings[key]=var.get() if "color" in key else float(var.get().replace(",","."))
        settings["transparent"]=self.transparent.get()
        panel=settings["panels"][self.panel_index]
        if self.manual_rectangle.get():
            if settings["width"]<=0 or settings["height"]<=0:
                raise ValueError("Le dimensioni dell'immagine devono essere positive.")
            dims={k:float(v.get().replace(",",".")) for k,v in self.panel_vars.items()}
            panel["rectangle"]=[dims["x"]/settings["width"],dims["y"]/settings["height"],
                                dims["width"]/settings["width"],dims["height"]/settings["height"]]
        else:
            panel["rectangle"]=None
        panel["zoom"]=float(self.panel_vars["zoom"].get().replace(",","."))
        for values,variables in zip(panel["axes"],self.axis_vars):
            for key,var in variables.items():
                values[key]=var.get() if key=="label" else float(var.get().replace(",","."))
        panel["visible"]=self.visible.get()
        panel["elev"]=float(self.vars["elev"].get().replace(",","."))
        panel["azim"]=float(self.vars["azim"].get().replace(",","."))
        validate_settings(settings)
        self.settings=settings

    def select_panel(self,event=None):
        try:
            self.capture()
        except (ValueError,KeyError) as error:
            messagebox.showerror("Valori non validi",str(error));self.panel.set(self.choices[self.panel_index]);return
        self.panel_index=self.choices.index(self.panel.get());self.load_fields()

    def choose_color(self,key):
        color=colorchooser.askcolor(color=self.vars[key].get(),parent=self.root)[1]
        if color:self.vars[key].set(color)

    def render(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy();self.toolbar.destroy();plt.close(self.fig)
        self.fig=build_combined_pareto(self.feasible,self.pareto,self.names)
        apply_settings(self.fig,self.settings)
        self.fig.set_dpi(150)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.preview)
        self.toolbar=NavigationToolbar2Tk(self.canvas,self.preview,pack_toolbar=False)
        self.toolbar.pack(side="bottom",fill="x")
        self.canvas.get_tk_widget().pack(expand=True)
        self.canvas.mpl_connect("pick_event",self.pick)
        self.canvas.mpl_connect("motion_notify_event",self.motion)
        self.canvas.mpl_connect("button_release_event",self.release)
        self.canvas.draw()

    def update(self):
        try:
            self.capture();self.render();self.status.set("Anteprima aggiornata.")
            return True
        except (ValueError,KeyError) as error:
            messagebox.showerror("Valori non validi",str(error));return False

    def add_text(self):
        if not self.annotation.get().strip() or not self.update():return
        self.settings["panels"][self.panel_index]["texts"].append(
            {"text":self.annotation.get(),"x":.5,"y":.92,"rotation":0})
        self.render()

    def remove_text(self):
        if not self.update():return
        texts=self.settings["panels"][self.panel_index]["texts"]
        if texts:texts.pop();self.render()

    def pick(self,event):
        if (hasattr(event.artist,"_pareto_text") or hasattr(event.artist,"_pareto_axis_label")) and not self.toolbar.mode:
            self.drag=event.artist
            # Avoid rotating the 3D axes while a caption is dragged.
            if event.artist.axes.name=="3d":event.artist.axes.disable_mouse_rotation()

    def motion(self,event):
        if self.drag is None or event.x is None or event.y is None:return
        x,y=self.drag.axes.transAxes.inverted().transform((event.x,event.y))
        if hasattr(self.drag,"_pareto_axis_label") and self.drag.axes.name!="3d":
            panel,axis_index=self.drag._pareto_axis_label
            axis=(self.drag.axes.xaxis,self.drag.axes.yaxis)[axis_index]
            axis.set_label_coords(x,y)
        else:
            self.drag.set_position((x,y))
        self.canvas.draw_idle()

    def release(self,event):
        if self.drag is not None:
            x,y=self.drag.get_position()
            if hasattr(self.drag,"_pareto_axis_label"):
                panel,index=self.drag._pareto_axis_label
                self.settings["panels"][panel]["axes"][index]["label_position"]=[float(x),float(y)]
            else:
                panel,index=self.drag._pareto_text
                self.settings["panels"][panel]["texts"][index].update(x=float(x),y=float(y))
            if self.drag.axes.name=="3d":self.drag.axes.mouse_init()
            self.drag=None
        ax=self.fig.axes[0]
        self.settings["panels"][0].update(elev=float(ax.elev),azim=float(ax.azim))
        if self.panel_index==0:
            self.vars["elev"].set(f"{ax.elev:g}");self.vars["azim"].set(f"{ax.azim:g}")
        # Retain limits changed with the Matplotlib zoom/pan toolbar on export.
        for index,ax in enumerate(self.fig.axes):
            limits=(ax.get_xlim(),ax.get_ylim(),ax.get_zlim()) if index==0 else (ax.get_xlim(),ax.get_ylim())
            for axis_index,(values,(lo,hi)) in enumerate(zip(self.settings["panels"][index]["axes"],limits)):
                if lo<hi:
                    values.update(min=float(lo),max=float(hi))
                    if index==self.panel_index:
                        self.axis_vars[axis_index]["min"].set(f"{lo:g}")
                        self.axis_vars[axis_index]["max"].set(f"{hi:g}")

    def export(self):
        if not self.update():return
        destination=self.output/"pareto_3d_views"
        destination.mkdir(exist_ok=True)
        path=filedialog.asksaveasfilename(parent=self.root,initialdir=destination,
                initialfile="pareto_personalizzato.png",defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf")])
        if not path:return
        try:
            self.fig.set_size_inches(self.settings["width"]/2.54,self.settings["height"]/2.54)
            self.fig.savefig(path,dpi=int(self.settings["dpi"]),transparent=self.settings["transparent"],
                             facecolor="none" if self.settings["transparent"] else "white",bbox_inches=None)
            self.status.set(f"Esportato: {Path(path).name}")
            self.canvas.draw_idle()
        except (OSError,ValueError) as error:messagebox.showerror("Esportazione",str(error))

    def save_config(self):
        if not self.update():return
        path=filedialog.asksaveasfilename(parent=self.root,initialdir=self.output,
                initialfile="pareto_stile.json",defaultextension=".json",filetypes=[("JSON","*.json")])
        if path:
            try:Path(path).write_text(json.dumps(self.settings,indent=2),encoding="utf-8")
            except OSError as error:messagebox.showerror("Salvataggio",str(error))

    def load_config(self):
        path=filedialog.askopenfilename(parent=self.root,initialdir=self.output,filetypes=[("JSON","*.json")])
        if not path:return
        try:
            settings=json.loads(Path(path).read_text(encoding="utf-8"))
            validate_settings(settings)
            self.settings=settings;self.load_fields();self.render()
        except (OSError,ValueError,KeyError,TypeError) as error:messagebox.showerror("Configurazione",str(error))


def main():
    root=tk.Tk()
    try:
        editor=ParetoEditor(root,ROOT/"outputs")
        root.protocol("WM_DELETE_WINDOW",lambda:(plt.close(editor.fig),root.destroy()))
        root.mainloop()
    except Exception:
        root.destroy()
        raise


if __name__=="__main__":
    main()
