"""Export five Pareto camera angles and a combined 3D/2D publication figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter

from pareto_plot_style import PARETO_AXIS_TICKS, save_pareto_figure
from reuse_genetic_opt import objective_coordinate_matrix
from saved_pareto_loader import load_saved_pareto_front


def build_combined_pareto(feasible_coords, pareto_coords, names):
    """A compact 3D view and three equal 2D panels on a 12 x 9 cm canvas."""
    fig = plt.figure(figsize=(12 / 2.54, 9 / 2.54))
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.975, bottom=0.055,
                            top=0.98, wspace=0.32, hspace=0.14)
    for index, pair in enumerate((None, (0, 2), (0, 1), (1, 2))):
        ax = fig.add_subplot(grid[index // 2, index % 2],
                             projection="3d" if pair is None else None)
        for coords, is_pareto in ((feasible_coords, False), (pareto_coords, True)):
            if not len(coords):
                continue
            style = dict(s=7**2 if is_pareto else 5**2,
                         c="#C00000" if is_pareto else "#d9d9d9",
                         edgecolors="black" if is_pareto else "none",
                         linewidths=0.6 if is_pareto else 0,
                         alpha=1, zorder=3 if is_pareto else 2)
            if pair is None:
                ax.scatter(*coords.T, depthshade=False, **style)
            else:
                ax.scatter(coords[:, pair[0]], coords[:, pair[1]], **style)
        if pair is None:
            ax.view_init(elev=20, azim=-120)
            ax.set_proj_type("persp")
            ax.set_box_aspect((4, 4, 3), zoom=1.15)
            ax.computed_zorder = False
            for axis, setter, name in zip((ax.xaxis,ax.yaxis,ax.zaxis),
                                          (ax.set_xlim, ax.set_ylim, ax.set_zlim), names):
                ticks = PARETO_AXIS_TICKS.get(name)
                if ticks is not None:
                    setter(ticks[0], ticks[-1])
                    axis.set_major_locator(FixedLocator(ticks))
                axis.line.set_linewidth(0.5)
                axis.set_pane_color((1,1,1,0))
                axis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.tick_params(axis="both", labelsize=9, width=0.5, pad=-3)
            ax.tick_params(axis="z", labelsize=9, width=0.5, pad=-3)
        else:
            for axis, setter, name in ((ax.xaxis, ax.set_xlim, names[pair[0]]),
                                       (ax.yaxis, ax.set_ylim, names[pair[1]])):
                ticks = PARETO_AXIS_TICKS.get(name)
                if ticks is not None:
                    setter(ticks[0], ticks[-1])
                    axis.set_major_locator(FixedLocator(ticks))
                axis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=9, width=0.5, length=2, pad=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        ax.grid(False)
        ax.get_xticklabels()
        ax.get_yticklabels()
        if pair is None:
            ax.get_zticklabels()
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontsize(9)
    return fig


def save_combined_pareto(destination, feasible_coords, pareto_coords, names):
    fig = build_combined_pareto(feasible_coords, pareto_coords, names)
    path = destination / "pareto_3D_2D_12x9cm"
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches=None, transparent=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=None, transparent=True)
    plt.close(fig)
    print(path.with_suffix(".png"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", action="store_true", help="Apri l'editor grafico interattivo")
    if parser.parse_args().edit:
        from edit_pareto_plots import main as edit_main
        edit_main()
        return
    output = Path(__file__).resolve().parent / "outputs"
    destination = output / "pareto_3d_views"
    destination.mkdir(exist_ok=True)
    pareto, evaluations, names = load_saved_pareto_front(output_dir=str(output))
    feasible_coords = objective_coordinate_matrix(
        [evaluation for evaluation in evaluations if evaluation.feasible], names
    )
    pareto_coords = objective_coordinate_matrix(pareto, names)
    save_combined_pareto(destination, feasible_coords, pareto_coords, names)
    views = [(20, -120), (30, -80), (25, -30), (25, 30), (35, 135)]
    images = []
    for index, (elev, azim) in enumerate(views, start=1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if len(feasible_coords):
            ax.scatter(*feasible_coords.T)
        ax.scatter(*pareto_coords.T)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        path = destination / f"pareto_3d_{index:02d}_elev_{elev}_azim_{azim}.png"
        save_pareto_figure(fig, path, is_3d=True, view=(elev, azim))
        plt.close(fig)
        images.append((path, elev, azim))
        print(path)

    overview, axes = plt.subplots(2, 3, figsize=(10, 6.7), dpi=150)
    for index, (ax, (path, elev, azim)) in enumerate(zip(axes.flat, images), start=1):
        ax.imshow(plt.imread(path))
        ax.set_title(f"{index} — elev. {elev}°, azimut {azim}°", fontfamily="Arial", fontsize=9)
        ax.set_axis_off()
    axes.flat[-1].set_axis_off()
    overview.tight_layout()
    overview.savefig(destination / "confronto_5_viste.png", facecolor="white", transparent=False)
    plt.close(overview)


if __name__ == "__main__":
    main()
