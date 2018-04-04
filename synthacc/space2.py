"""
The 'space2' module.
"""


from numba import jit
import numpy as np


@jit(nopython=True)
def _distance(x1, y1, x2, y2):
    """
    """
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


def distance(x1, y1, x2, y2, validate=True):
    """
    """
    if validate is True:
        pass

    return _distance(x1, y1, x2, y2)
