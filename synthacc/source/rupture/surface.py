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

    @property
    @abstractmethod
    def LABEL(self):
        """
        """
        pass

    @property
    def avg(self):
        """
        """
        return self._values.mean()

    @property
    def min(self):
        """
        """
        return self._values.min()

    @property
    def max(self):
        """
        """
        return self._values.max()

    @property
    def surface(self):
        """
        """
        s = space2.DiscretizedRectangularSurface(
            self.w, self.l, self.dw, self.dl, validate=False)

        return s

    def interpolate(self, xs, ys, validate=True):
        """
        """
        if validate is True:
            assert(xs[-1] <= self.surface.w)
            assert(ys[-1] <= self.surface.l)

        i = scipy.interpolate.RectBivariateSpline(
            self.surface.xs,
            self.surface.ys,
            self._values)

        return i(xs, ys)

    def plot(self, contours=False, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        extent = [0, self.l/1000, self.w/1000, 0]
        p = ax.imshow(self._values, interpolation='bicubic', extent=extent)

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

        plt.tight_layout()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()
