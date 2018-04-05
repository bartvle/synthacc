"""
The 'space2' module.
"""


from abc import ABC, abstractmethod

from numba import jit
import numpy as np

from .apy import Object, is_pos_number


class DiscretizedRectangularSurface(Object):
    """
    """

    def __init__(self, w, l, dw, dl, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw) and dw <= w/2)
            assert(is_pos_number(dl) and dl <= l/2)

        nw = round(w / dw)
        nl = round(l / dl)

        dw = w / nw
        dl = l / nl

        xs = np.linspace(0+dl/2, l-dl/2, nl)
        ys = np.linspace(0+dw/2, w-dw/2, nw)
        self._grid = np.dstack(np.meshgrid(xs, ys))

        self._w = w
        self._l = l
        self._dw = dw
        self._dl = dl

    def __len__(self):
        """
        """
        return np.prod(self.shape)

    @property
    def w(self):
        """
        """
        return self._w

    @property
    def l(self):
        """
        """
        return self._l

    @property
    def dw(self):
        """
        """
        return self._dw

    @property
    def dl(self):
        """
        """
        return self._dl

    @property
    def shape(self):
        """
        """
        return self._grid.shape[:2]

    @property
    def nw(self):
        """
        """
        return self.shape[0]

    @property
    def nl(self):
        """
        """
        return self.shape[1]

    @property
    def xs(self):
        """
        return: 1d numerical array
        """
        return self._grid[0,:,0]

    @property
    def ys(self):
        """
        return: 1d numerical array
        """
        return self._grid[:,0,1]

    @property
    def xgrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,0]

    @property
    def ygrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,1]

    @property
    def area(self):
        """
        """
        return self.w * self.l


class ACF(ABC, Object):
    """
    An autocorrelation function.
    """

    @abstractmethod
    def __call__(self):
        """
        """
        pass

    @abstractmethod
    def get_psd(self, k, a=1):
        """
        a is product of a of each dimension.
        """
        pass


class GaussianACF(ACF):
    """
    The Gaussian autocorrelation function. See Mai & Beroza (2002).
    """

    def __call__(self, r, a=1):
        """
        """
        return np.exp(-(r/a)**2)

    def get_psd(self, k, a=1):
        """
        """
        return 0.5*a*np.exp(-0.25*k**2)


class ExponentialACF(ACF):
    """
    The exponential autocorrelation function. See Mai & Beroza (2002).
    """

    def __call__(self, r, a=1):
        """
        """
        return np.exp(-r/a)

    def get_psd(self, k, a=1):
        """
        """
        return a/(1+k**2)**1.5


class VonKarmanACF(ACF):
    """
    The von Karman autocorrelation function. See Mai & Beroza (2002).
    """

    def __init__(self, h, validate=True):
        """
        h: Hurst exponent
        """
        if validate is True:
            assert(0 <= h <= 1)

        self._h = h

    @property
    def h(self):
        """
        """
        return self._h

    def __call__(self, r, a=1):
        """
        """
        r = r + (r[1]-r[0])
        acf = r**self.h * scipy.special.kv(self.h, r/a)
        acf /= acf[0]
        return acf

    def get_psd(self, k, a=1):
        """
        """
        return a/(1+k**2)**(self.h+1)


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
