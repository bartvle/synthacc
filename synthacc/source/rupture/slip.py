"""
The 'source.rupture.slip' module.
"""


import matplotlib.pyplot as plt
from numba import jit
import numpy as np

from ...apy import (Object, is_number, is_pos_number, is_pos_integer,
    is_2d_numeric_array)
from ... import space2
from ...data import Histogram
from ..faults import RIGIDITY
from ..moment import mw_to_m0
from .surface import Distribution


class SlipDistribution(Distribution):
    """
    A spatial slip distribution.
    """

    LABEL = 'Slip (m)'

    def __init__(self, w, l, slip, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(slip))
            assert(np.all(slip >= 0))

        dw = w / slip.shape[0]
        dl = l / slip.shape[1]

        super().__init__(w, l, dw, dl, validate=False)

        self._values = slip

    @property
    def slip(self):
        """
        """
        return self._values

    @property
    def histogram(self):
        """
        """
        return Histogram(self._values.flatten(), positive=True)

    @property
    def top(self):
        """
        """
        return SlipSection(self.l, self._values[0])


class SlipSection(Object):
    """
    """

    def __init__(self, length, values, validate=True):
        """
        """
        if validate is True:
            pass

        self._length = length
        self._values = values

    @property
    def length(self):
        """
        """
        return self._length

    @property
    def distances(self):
        """
        """
        return np.linspace(0, self.length, len(self._values))

    @property
    def values(self):
        """
        """
        return self._values.copy()

    def plot(self, size=None, png_filespec=None, validate=True):
        """
        """
        f, ax = plt.subplots(figsize=size)

        ax.plot(self.distances / 1000, self._values)

        ax.grid()
        xlabel, ylabel = 'Length (km)', 'Slip (m)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class RFSlipDistribution(SlipDistribution):
    """
    Random field slip distribution.
    """

    def __init__(self, w, l, slip, validate=True):
        """
        """
        super().__init__(w, l, slip, validate=validate)


class RFSlipDistributionGenerator(Object):
    """
    Random field slip distribution generator.
    """

    def __init__(self, w, l, dw, dl, acf, aw, al, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw))
            assert(is_pos_number(dl))
            assert(round(w / dw) % 2 == 1)
            assert(round(l / dl) % 2 == 1)

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, dw, dl, validate=False)

        self._srfg = space2.SpatialRandomFieldGenerator(
            self._surface.nw, self._surface.nl,
            self._surface.dw, self._surface.dl,
            acf, aw, al, validate=validate)

    def __call__(self, magnitude, rigidity=RIGIDITY, std=1, seed=None, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        field = self.srfg(seed, validate=validate)

        mean_slip = mw_to_m0(magnitude) / (self.surface.area * rigidity)

        # slip = mean_slip + field * 0,85
        slip = mean_slip * (1 + field * std)
        slip[slip < 0] = 0

        sd = RFSlipDistribution(self.surface.w, self.surface.l, slip)

        return sd

    @property
    def surface(self):
        """
        """
        return self._surface

    @property
    def srfg(self):
        """
        """
        return self._srfg


class FCSlipDistribution(SlipDistribution):
    """
    Fractal composite slip distribution.
    """

    def __init__(self, w, l, slip, sources, validate=True):
        """
        """
        super().__init__(w, l, slip, validate=validate)

        self._sources = sources

    def __len__(self):
        """
        """
        return len(self._sources)

    @property
    def radii(self):
        """
        """
        return self._sources[:,-1]


@jit(nopython=True)
def _calc_sources(n, p, d, rmax, l, w):
    """
    """
    randoms = np.random.random((3, n))
    radii = (randoms[0]*n*d/p + rmax**(-d))**(-1/d)
    sources = np.zeros((n, 3))
    sources[:,0] = radii + randoms[1] * (l-2*radii)
    sources[:,1] = radii + randoms[2] * (w-2*radii)
    sources[:,2] = radii
    return sources


@jit(nopython=True)
def _calc(input):
    """
    """
    return np.sqrt(input)


class FCSlipDistributionGenerator(Object):
    """
    Fractal composite slip distribution generator.
    """

    def __init__(self, w, l, d, rmin=2**(0.5), rmax=0.5, dimension=2, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(dimension))

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, d, d, validate=validate)

        self._rmin = rmin * d
        self._rmax = rmax * w
        self._dimension = dimension

        self._p = self._calc_p()
        self._n = self._calc_n()

    def __call__(self, magnitude, rigidity=RIGIDITY, seed=None, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))
            if seed is not None:
                assert(is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        w = self.surface.w
        l = self.surface.l

        sources = _calc_sources(self.n, self.p, self.dimension, self.rmax, l, w)

        # slip = np.zeros((self.surface.nw, self.surface.nl))

        x1 = np.tile(self.surface.xgrid[(Ellipsis,None)], (1,1,len(sources)))
        y1 = np.tile(self.surface.ygrid[(Ellipsis,None)], (1,1,len(sources)))

        distances = space2.distance(x1, y1, sources[:,0], sources[:,1])

        constant = ((1.5 / np.pi) * (mw_to_m0(magnitude) /
            (np.sqrt(self.surface.l*self.surface.w/np.pi)**3 * rigidity)))

        int = np.zeros_like(distances)
        res = sources[:,2]**2-distances**2
        indices = res > 0
        int[indices] = np.sqrt(res[indices])

        slip = constant * np.sum(int, axis=2)

        sd = FCSlipDistribution(self.surface.w, self.surface.l, slip, sources)

        return sd

    def _calc_p(self):
        """
        """
        rmax = self.rmax**(3-self.dimension)
        rmin = self.rmin**(3-self.dimension)
        p = (self.surface.area/np.pi)**(3/2) * (3-self.dimension) / (rmax-rmin)

        return p

    def _calc_n(self):
        """
        """
        rmin = self.rmin**-self.dimension
        rmax = self.rmax**-self.dimension
        n = int((self.p/self.dimension) * (rmin - rmax))

        return n

    @property
    def surface(self):
        """
        """
        return self._surface

    @property
    def rmin(self):
        """
        """
        return self._rmin

    @property
    def rmax(self):
        """
        """
        return self._rmax

    @property
    def dimension(self):
        """
        """
        return self._dimension

    @property
    def p(self):
        """
        """
        return self._p

    @property
    def n(self):
        """
        """
        return self._n


class MASlipDistributionGenerator(Object):
    """
    Multiple asperity slip distribution generator.
    """
    pass


class LiuEtAl2006NormalizedSlipRateGenerator(Object):
    """
    Normalized slip rate generator of Liu et al. (2006).
    """

    def __init__(self, time_delta, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(time_delta))

        self._time_delta = time_delta

    def __call__(self, rise_time, validate=True):
        """
        See Liu et al. (2006) p. 2121 eq. 7a and 7b.
        """
        if validate is True:
            assert(is_pos_number(rise_time))

        t1 = 0.13*rise_time
        t2 = rise_time - t1
        cn = np.pi / (1.4*np.pi*t1 + 1.2*t1 + 0.3*np.pi*t2)

        times = self.time_delta * np.arange(
            np.round(rise_time / self.time_delta) + 1)

        i1 = times < t1
        i3 = times >= 2*t1
        i2 = ~(i1 | i3)

        f = np.zeros_like(times)
        f[i1] = (0.7 - 0.7*np.cos(np.pi*times[i1]/t1) +
                    0.6*np.sin(np.pi*times[i1]/(2.*t1)))
        f[i2] = (1.0 - 0.7*np.cos(np.pi*times[i2]/t1) +
                    0.3*np.cos(np.pi*(times[i2]-t1)/t2))
        f[i3] = (0.3 + 0.3*np.cos(np.pi*(times[i3]-t1)/t2))

        return cn * f

    @property
    def time_delta(self):
        """
        """
        return self._time_delta
