"""
The 'source.rupture.surface' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import scipy.interpolate

from ...apy import is_string
from ... import space2


class Distribution(ABC, space2.DiscretizedRectangularSurface):
    """
    """

    def __init__(self, w, l, nw, nl, validate=True):
        """
        """
        super().__init__(w, l, nw, nl, validate=validate)

        self._values = None

    @property
    @abstractmethod
    def LABEL(self):
        """
        """
        raise NotImplementedError

    @property
    def values(self):
        """
        """
        return self._values

    @property
    def avg(self):
        """
        """
        return float(self._values.mean())

    @property
    def min(self):
        """
        """
        return float(self._values.min())

    @property
    def max(self):
        """
        """
        return float(self._values.max())

    def interpolate(self, xs, ys, validate=True):
        """
        """
        if validate is True:
            assert(xs[-1] <= self.w)
            assert(ys[-1] <= self.l)

        i = scipy.interpolate.RectBivariateSpline(
            self.xs, self.ys, self._values)

        return i(xs, ys)

    def plot(self, contours=False, cmap=None, title=None, size=None, filespec=None, validate=True):
        """
        """
        if validate is True:
            assert(title is None or is_string(title))

        fig, ax = plt.subplots(figsize=size)

        extent = [0, self.l/1000, self.w/1000, 0]
        p = ax.imshow(
            self._values, interpolation='bicubic', cmap=cmap, extent=extent)

        if contours is True:
            ax.contour(self.ygrid/1000, self.xgrid/1000, self._values,
                extent=extent, colors='gray')

        ax.axis('scaled')

        xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        cbar = fig.colorbar(p)
        cbar.set_label(self.LABEL)

        if title is not None:
            plt.title(title)

        if filespec is not None:
            plt.savefig(filespec, bbox_inches='tight')
        else:
            plt.show()
