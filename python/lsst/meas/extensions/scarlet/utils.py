import numpy as np


def footprintsToNumpy(catalog, shape, xy0=None):
    """Convert all of the footprints in a catalog into a boolean array

    Parameters
    ----------
    catalog : `lsst.afw.SourceCatalog`
        The source catalog containing the footprints.
        This is typically a mergeDet catalog, or a full source catalog
        with the parents removed.
    shape : `tuple` of `int`
        The final shape of the output array.
    xy0 : `tuple` of `int`
        The lower-left corner of the array that will contain the spans.

    Returns
    -------
    result : `numpy.ndarray`
        The array with pixels contained in `spans` marked as `True`.
    """
    if xy0 is None:
        offset = (0, 0)
    else:
        offset = (-xy0[0], -xy0[1])

    result = np.zeros(shape, dtype=bool)
    for src in catalog:
        spans = src.getFootprint().spans
        yidx, xidx = spans.shiftedBy(*offset).indices()
        result[yidx, xidx] = 1
    return result
