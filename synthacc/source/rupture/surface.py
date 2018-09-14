"""
The 'source.rupture.surface' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.interpolate

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

    def plot(self, contours=False, cmap=None, size=None, png_filespec=None, validate=True):
        """
        """
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

        cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=0.25)
        cbar = fig.colorbar(p, cax=cax)
        cbar.set_label(self.LABEL)

        if png_filespec is not None:
            plt.savefig(png_filespec, bbox_inches='tight')
        else:
            plt.show()
