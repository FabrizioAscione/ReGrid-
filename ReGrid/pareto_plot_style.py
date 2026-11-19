"""Physical sizes and typography shared by Pareto image exports."""

from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, MaxNLocator


PARETO_3D_SIZE = (6.5 / 2.54, 6.5 / 2.54)
PARETO_2D_SIZE = (4.5 / 2.54, 2.0 / 2.54)
PARETO_AXIS_TICKS = {
    "pi_mass": (0, 0.25, 0.5, 0.75, 1),
    "pi_ec": (0, 1, 2, 3, 4, 5),
    "pi_cost": (0, 0.5, 1, 1.5),
}


def save_pareto_figure(fig, path, *, is_3d=False, view=None):
    """Keep the full canvas at the requested size, including axis labels."""
    fig.set_size_inches(PARETO_3D_SIZE if is_3d else PARETO_2D_SIZE)
    if is_3d:
        _save_styled_3d(fig, path, view=view)
        return
    for ax in fig.axes:
        ax.grid(False)
        ax.set_title("")
        if ax.get_label() == "<colorbar>":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.tick_params(labelsize=9, pad=1, length=2)
            continue
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
            axis.labelpad = 1
        ax.tick_params(axis="both", labelsize=9, pad=1, length=2)
        if getattr(ax, "name", "") == "3d":
            ax.zaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
            ax.zaxis.labelpad = 1
            ax.tick_params(axis="z", labelsize=9, pad=0)
            ax.set_box_aspect((1, 1, 1), zoom=0.78)
        elif ax.get_label() != "<colorbar>":
            ax.xaxis.set_label_coords(0.5, -0.85)
            ax.yaxis.set_label_coords(-0.27, 0.5)
        for collection in ax.collections:
            if hasattr(collection, "set_sizes") and len(collection.get_sizes()):
                collection.set_sizes([12 if is_3d else 5])
                collection.set_linewidths([0.25])
    if is_3d:
        fig.subplots_adjust(left=0.02, right=0.80 if len(fig.axes) > 1 else 0.98,
                            bottom=0.12, top=0.98)
        if len(fig.axes) > 1:
            fig.axes[0].set_position([0.00, 0.12, 0.70, 0.82])
            fig.axes[1].set_position([0.80, 0.24, 0.035, 0.55])
    else:
        fig.subplots_adjust(left=0.30, right=0.97, bottom=0.53, top=0.96)
    # Materialize tick labels before applying typography.
    for ax in fig.axes:
        ax.get_xticklabels()
        ax.get_yticklabels()
        if getattr(ax, "name", "") == "3d":
            ax.get_zticklabels()
    for label in fig.findobj(match=Text):
        label.set_fontfamily("Arial")
        label.set_fontsize(9)
    fig.savefig(path, dpi=300, bbox_inches=None)


def _save_styled_3d(fig, path, *, view=None):
    """Apply reference marker colors while retaining the standard 3D axes."""
    ax = next(ax for ax in fig.axes if getattr(ax, "name", "") == "3d")
    for extra_ax in list(fig.axes):
        if extra_ax is not ax:
            fig.delaxes(extra_ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title("")
    ax.set_axis_on()
    ax.grid(False)
    elev, azim = view if view is not None else (30, -60)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp")
    ax.set_box_aspect((4, 4, 3), zoom=0.85)
    ax.set_position([0.02, 0.14, 0.84, 0.82])
    if view is not None:
        ax.set_position([0.10, 0.14, 0.78, 0.78])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        ticks = PARETO_AXIS_TICKS.get(axis.get_label_text())
        if ticks is not None:
            axis.set_view_interval(ticks[0], ticks[-1], ignore=True)
            axis.set_major_locator(FixedLocator(ticks))
        else:
            axis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        axis.line.set_linewidth(0.5)
        axis.labelpad = 1
        axis.set_pane_color((1, 1, 1, 0))
    ax.tick_params(axis="both", labelsize=9, pad=1, length=2, width=0.5)
    ax.tick_params(axis="z", labelsize=9, pad=1, width=0.5)
    ax.computed_zorder = False
    for index, collection in enumerate(ax.collections):
        is_pareto = index == len(ax.collections) - 1
        collection.set_array(None)
        collection.set_facecolor("#C00000" if is_pareto else "#d9d9d9")
        collection.set_edgecolor("black" if is_pareto else "none")
        collection.set_alpha(1)
        collection.set_depthshade(False)
        # scatter sizes are areas in pt²; marker diameters are 7 and 5 pt.
        collection.set_sizes([7 ** 2 if is_pareto else 5 ** 2])
        collection.set_linewidths([0.6 if is_pareto else 0])
        collection.set_zorder(3 if is_pareto else 2)
    ax.get_xticklabels()
    ax.get_yticklabels()
    ax.get_zticklabels()
    # Keep the labels at the shared lower corner from touching.
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.get_yticklabels()[0].set_horizontalalignment("left")
    if view is not None:
        if azim < -90:
            ax.get_xticklabels()[0].set_horizontalalignment("left")
            ax.get_yticklabels()[0].set_horizontalalignment("right")
        elif azim > 90:
            ax.get_xticklabels()[0].set_horizontalalignment("right")
            ax.get_yticklabels()[-1].set_horizontalalignment("left")
    for label in fig.findobj(match=Text):
        label.set_fontfamily("Arial")
        label.set_fontsize(9)
    fig.savefig(path, dpi=300, bbox_inches=None, transparent=True)
