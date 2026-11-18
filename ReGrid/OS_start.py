import openseespy.opensees as ops


def init_model(ndm=3, ndf=6):
    """
    Initialize a clean OpenSeesPy model and return the executed Python call.

    Parameters
    ----------
    ndm : int
        Number of spatial dimensions.
    ndf : int
        Number of DOFs per node.
    """
    ndm = int(ndm)
    ndf = int(ndf)
    if ndm <= 0:
        raise ValueError(f"ndm must be > 0. Got {ndm}.")
    if ndf <= 0:
        raise ValueError(f"ndf must be > 0. Got {ndf}.")

    ops.wipe()
    ops.model("basic", "-ndm", ndm, "-ndf", ndf)

    return f'ops.model("basic", "-ndm", {ndm}, "-ndf", {ndf})'

