def configure_transparent_matplotlib_saves() -> None:
    """Save Matplotlib figures with transparent backgrounds by default."""
    import matplotlib as mpl

    mpl.rcParams["savefig.transparent"] = True
    mpl.rcParams["savefig.facecolor"] = "none"
    mpl.rcParams["savefig.edgecolor"] = "none"
