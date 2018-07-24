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

        self._values = slip

        super().__init__(w, l, dw, dl, validate=False)

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

    def __init__(self, w, l, slip, acf, aw, al, validate=True):
        """
        """
        super().__init__(w, l, slip, validate=validate)

        self._acf = acf
        self._aw = aw
        self._al = al

    @property
    def acf(self):
        """
        """
        return self._acf

    @property
    def aw(self):
        """
        """
        return self._aw

    @property
    def al(self):
        """
        """
        return self._al


class RFSlipDistributionCalculator(Object):
    """
    Random field slip distribution calculator.
    """

    def __init__(self, sd, acf, aw, al, std=1, validate=True):
        """
        """            
        self._srfc = space2.SpatialRandomFieldCalculator(
            sd, sd, acf, aw, al, validate=validate)
    
        self._std = std

    def __call__(self, surface, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        sdg = RFSlipDistributionGenerator(
            surface.width, surface.length, magnitude, rigidity, self._srfc.dw,
            self._srfc.dl, self._srfc.acf, self._srfc.aw, self._srfc.al,
            self._std, validate=False)

        return sdg


class MaiBeroza2002RFSDC(RFSlipDistributionCalculator):
    """
    """

    def __init__(self, sd, std=1, validate=True):
        """
        """
        if validate is True:
            pass

        self._sd = sd
        self._std = std

    def __call__(self, surface, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        acf = space2.VonKarmanACF(h=0.75)
        aw = 10**(1/3*magnitude-1.6) * 1000
        al = 10**(1/2*magnitude-2.5) * 1000

        sdg = RFSlipDistributionGenerator(surface.width, surface.length,
            magnitude, rigidity, self._sd, self._sd, acf, aw, al, self._std,
            validate=False)

        return sdg


class RFSlipDistributionGenerator(Object):
    """
    Random field slip distribution generator.
    """

    def __init__(self, w, l, magnitude, rigidity, dw, dl, acf, aw, al, std, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))
        
        if not (round(w / dw) % 2 == 1):
            nw = round(w / (2*dw)) *2 + 1
            dw = w / nw

        if not (round(l / dl) % 2 == 1):
            nl = round(l / (2*dl)) * 2 + 1
            dl = l / nl

        self._srfc = space2.SpatialRandomFieldCalculator(
            dw, dl, acf, aw, al, validate=validate)
    
        self._std = std

        self._magnitude = magnitude
        self._rigidity = rigidity

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, dw, dl, validate=False)

        self._srfg = space2.SpatialRandomFieldGenerator(
            self._surface.nw, self._surface.nl,
            self._surface.dw, self._surface.dl,
            acf, aw, al, validate=validate)

    def __call__(self, seed=None, validate=True):
        """
        """
        field = self._srfg(seed, validate=validate)

        mean_slip = (mw_to_m0(self._magnitude) /
            (self.surface.area * self._rigidity))
        slip = mean_slip * (1 + field * self._std)
        slip[slip < 0] = 0

        sd = RFSlipDistribution(self.surface.w, self.surface.l, slip,
            self._srfg.acf, self._srfg.aw, self._srfg.al)

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


class CSSlipDistribution(SlipDistribution):
    """
    Composite source slip distribution.
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

    def plot_sources(self, n=1000, size=None, png_filespec=None, validate=True):
        """
        """
        _, ax = plt.subplots(figsize=size)

        for i in np.random.randint(0, len(self), n):
            c = plt.Circle(tuple([
                self._sources[i][1]/1000,
                self._sources[i][0]/1000,
                ]), radius=self._sources[i][2]/1000, fill=False)
            ax.add_patch(c)

        ax.axis('scaled')
        ax.set_xlim(0, self.l/1000)
        ax.set_ylim(0, self.w/1000)

        xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if png_filespec is not None:
            plt.savefig(png_filespec, bbox_inches='tight')
        else:
            plt.show()


class CSSlipDistributionCalculator(Object):
    """
    Composite source slip distribution calculator.
    """

    def __init__(self, d, rminf=2**(0.5), rmaxf=0.5, dimension=2, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(dimension))

        self._d = d
        self._rminf = rminf
        self._rmaxf = rmaxf
        self._dimension = dimension

    def __call__(self, surface, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        sdg = CSSlipDistributionGenerator(surface.width, surface.length,
            magnitude, rigidity, self._d, self._rminf, self._rmaxf,
            self._dimension, validate=False)

        return sdg

    @property
    def d(self):
        """
        """
        return self._d

    @property
    def rminf(self):
        """
        """
        return self._rminf

    @property
    def rmaxf(self):
        """
        """
        return self._rmaxf

    @property
    def dimension(self):
        """
        """
        return self._dimension


@jit(nopython=True)
def _calc_sources(n, p, d, rmax, w, l):
    """
    """
    randoms = np.random.random((3, n))
    radii = (randoms[0]*n*d/p + rmax**(-d))**(-1/d)
    sources = np.zeros((n, 3))
    sources[:,0] = radii + randoms[1] * (w-2*radii)
    sources[:,1] = radii + randoms[2] * (l-2*radii)
    sources[:,2] = radii

    return sources


class CSSlipDistributionGenerator(CSSlipDistributionCalculator):
    """
    Composite source slip distribution generator.
    """

    def __init__(self, w, l, magnitude, rigidity, d, rminf, rmaxf, dimension, validate=True):
        """
        """
        super().__init__(d, rminf, rmaxf, dimension, validate=validate)

        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, d, d, validate=validate)

        self._magnitude = magnitude
        self._rigidity = rigidity

        self._rmin = self._rminf * d
        self._rmax = self._rmaxf * w

        self._p = self._calc_p()
        self._n = self._calc_n()
        self._c = self._calc_c()

        self._xs = np.tile(self.surface.xgrid[(Ellipsis,None)], (1,1,self._n))
        self._ys = np.tile(self.surface.ygrid[(Ellipsis,None)], (1,1,self._n))

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            if seed is not None:
                assert(is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        sources = _calc_sources(self.n, self.p, self.dimension, self.rmax,
            self.surface.w, self.surface.l)

        distances = space2.distance(self._xs, self._ys, sources[:,0], sources[:,1])

        int = np.zeros_like(distances)
        res = sources[:,2]**2-distances**2
        indices = res > 0
        int[indices] = np.sqrt(res[indices])

        slip = self._c * np.sum(int, axis=2)

        sd = CSSlipDistribution(self.surface.w, self.surface.l, slip, sources)

        return sd

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
    def p(self):
        """
        """
        return self._p

    @property
    def n(self):
        """
        """
        return self._n

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

    def _calc_c(self):
        """
        """
        c = ((1.5 / np.pi) * (mw_to_m0(self._magnitude) /
            (np.sqrt(self.surface.area/np.pi)**3 * self._rigidity)))

        return c


class LiuEtAl2006NormalizedSlipRateCalculator(Object):
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


class GP2010SlipRateCalculator(Object):
    """
    """

    def __init__(self, time_delta, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(time_delta))

        self._time_delta = time_delta
        self._nsrf_g = LiuEtAl2006NormalizedSlipRateCalculator(time_delta)

    def __call__(self, surface, magnitude, sd):
        """
        """
        surface = surface.get_discretized(shape=sd.shape)
        depths = np.rollaxis(surface.centers, 2)[-1]

        rise_times = np.interp(depths, [5000, 8000], [2, 1]) * (sd.slip/100)**(1/2)
        average = (np.interp(surface.dip, [45, 60], [0.82, 1]) * 1.6 * 10**-9 *
            (10**7*mw_to_m0(magnitude))**(1/3))

        rise_times = rise_times * (average / rise_times.mean())
        #print(surface.shape, (np.round(rise_times / self._time_delta).astype(np.int).max()+1,))
        slip_rates = np.zeros(surface.shape + (np.round(rise_times / self._time_delta).astype(np.int).max()+1,))

        

        for i in np.ndindex(surface.shape):
            t = rise_times[i]
            if t != 0:
                sr = self._nsrf_g(float(t)) * sd.slip[i]
                slip_rates[i][:len(sr)] = sr

        return GP2010SlipRateGenerator(slip_rates)


class GP2010SlipRateGenerator(Object):
    """
    """

    def __init__(self, slip_rates):
        """
        """
        self._slip_rates = slip_rates

    def __call__(self):
        """
        """
        return self._slip_rates
