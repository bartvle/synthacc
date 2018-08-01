"""
The 'source.rupture.slip' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import scipy.stats

from ...apy import (Object, is_number, is_pos_number, is_pos_integer,
    is_2d_numeric_array)
from ... import space2
from ...data import Histogram
from ...earth import flat as earth
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
        _, ax = plt.subplots(figsize=size)

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


class RandomFieldSD(SlipDistribution):
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


class SlipDistributionCalculator(ABC, Object):
    """
    Slip distribution calculator.
    """

    @abstractmethod
    def __call__(self, segment, magnitude, rigidity=RIGIDITY, validate=True):
        """
        return: 'source.rupture.slip.SlipDistributionGenerator' instance
        """
        pass


class SlipDistributionGenerator(ABC, Object):
    """
    Slip distribution generator.
    """

    @abstractmethod
    def __call__(self, seed=None, validate=True):
        """
        return: 'source.rupture.slip.SlipDistribution' instance
        """
        pass


class RandomFieldSDC(SlipDistributionCalculator, ABC):
    """
    Random field slip distribution calculator.
    """

    @abstractmethod
    def get_acf(self):
        """
        """
        pass

    @abstractmethod
    def get_aw(self):
        """
        """
        pass

    @abstractmethod
    def get_al(self):
        """
        """
        pass


class MaiBeroza2002RFSDC(RandomFieldSDC):
    """
    Random field slip distribution calculator based on Mai & Beroza (2002).
    """

    def __init__(self, dw, dl, sd=1, validate=True):
        """
        """
        if validate is True:
            pass

        self._dw = dw
        self._dl = dl
        self._sd = sd

    def __call__(self, segment, magnitude, rigidity=RIGIDITY, validate=True):
        """
        return: 'source.rupture.slip.RandomFieldSDG' instance
        """
        sdg = RandomFieldSDG(
            self, segment, magnitude, rigidity, validate=validate)

        return sdg

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
    def sd(self):
        """
        """
        return self._sd

    def get_acf(self):
        """
        """
        mean, sd = 0.75, 0.22
        a = (0.5 - mean) / sd
        b = (1.0 - mean) / sd

        h = float(scipy.stats.truncnorm.rvs(a, b, mean, sd))

        return space2.VonKarmanACF(h=h)

    def get_aw(self, w):
        """
        """
        mean, sd = 0.55 + 0.35 * w / 1000, 0.78

        return float(scipy.stats.truncnorm.rvs(-2, +2, mean, sd) * 1000)

    def get_al(self, l):
        """
        """
        mean, sd = 1.54 + 0.33 * l / 1000, 2.31

        return float(scipy.stats.truncnorm.rvs(-2, +2, mean, sd) * 1000)


class RandomFieldSDG(SlipDistributionGenerator):
    """
    Random field slip distribution generator.
    """

    def __init__(self, calculator, segment, magnitude, rigidity, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(calculator, RandomFieldSDC))
            assert(type(segment) is earth.Rectangle)
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))
        
        w, l = segment.width, segment.length
        
        if not (round(w / calculator.dw) % 2 == 1):
            nw = round(w / (2*calculator.dw)) *2 + 1
            dw = w / nw
        else:
            dw = calculator.dw

        if not (round(l / calculator.dl) % 2 == 1):
            nl = round(l / (2*calculator.dl)) * 2 + 1
            dl = l / nl
        else:
            dl = calculator.dl

        self._calculator = calculator
        self._magnitude = magnitude
        self._rigidity = rigidity

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, dw, dl, validate=False)

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            if seed is not None:
                assert(is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        acf = self._calculator.get_acf()

        aw = self._calculator.get_aw(self._surface.w) ## Scipy uses np seed
        al = self._calculator.get_al(self._surface.l) ## Scipy uses np seed

        srfg = space2.SpatialRandomFieldGenerator(
            self._surface.nw, self._surface.nl,
            self._surface.dw, self._surface.dl, acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        mean_slip = (mw_to_m0(self._magnitude) /
            (self._surface.area * self._rigidity))
        slip = mean_slip * (1 + field * self._calculator.sd)
        slip[slip < 0] = 0

        sd = RandomFieldSD(self._surface.w, self._surface.l, slip, acf, aw, al)

        return sd


class CompositeSourceSD(SlipDistribution):
    """
    Composite source slip distribution.
    """

    def __init__(self, w, l, slip, sources, validate=True):
        """
        """
        super().__init__(w, l, slip, validate=validate)

        if validate is True:
            assert(is_2d_numeric_array(sources) and sources.shape[1] == 3)

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


class CompositeSourceSDC(SlipDistributionCalculator):
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

    def __call__(self, segment, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        sdg = CompositeSourceSDG(
            self, segment, magnitude, rigidity, validate=validate)

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

    def get_p(self, rmin, rmax, surface):
        """
        """
        rmin = rmin**(3-self._dimension)
        rmax = rmax**(3-self._dimension)

        p = (surface.area/np.pi)**(3/2) * (3-self.dimension) / (rmax - rmin)

        return p

    def get_n(self, rmin, rmax, p):
        """
        """
        rmin = rmin**(-self.dimension)
        rmax = rmax**(-self.dimension)

        n = int((p/self.dimension) * (rmin - rmax))

        return n

    def get_sources(self, n, p, rmax, surface):
        """
        """
        return _calc_sources(n, p, self.dimension, rmax, surface.w, surface.l)


class CompositeSourceSDG(SlipDistributionGenerator):
    """
    Composite source slip distribution generator.
    """

    def __init__(self, calculator, segment, magnitude, rigidity, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(calculator, CompositeSourceSDC))
            assert(type(segment) is earth.Rectangle)
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        d, w, l = calculator.d, segment.width, segment.length

        surface = space2.DiscretizedRectangularSurface(
            w, l, d, d, validate=validate)

        rmin = calculator.rminf * d
        rmax = calculator.rmaxf * w

        p = calculator.get_p(rmin, rmax, surface)
        n = calculator.get_n(rmin, rmax, p)

        c = ((1.5 / np.pi) * (mw_to_m0(magnitude) /
            (np.sqrt(surface.area/np.pi)**3 * rigidity)))

        xs = np.tile(surface.xgrid[(Ellipsis,None)], (1,1,n))
        ys = np.tile(surface.ygrid[(Ellipsis,None)], (1,1,n))

        self._calculator = calculator
        self._surface = surface
        self._rmin = rmin
        self._rmax = rmax
        self._p = p
        self._n = n
        self._c = c
        self._xs = xs
        self._ys = ys

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            if seed is not None:
                assert(is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        sources = self._calculator.get_sources(
            self._n, self._p, self._rmax, self._surface)

        distances = space2.distance(
            self._xs, self._ys, sources[:,0], sources[:,1])

        con = np.zeros_like(distances)
        res = sources[:,2]**2 - distances**2
        indices = res > 0
        con[indices] = np.sqrt(res[indices])
        slip = self._c * np.sum(con, axis=2)

        sd = CompositeSourceSD(self._surface.w, self._surface.l, slip, sources)

        return sd


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

        times = self._time_delta * np.arange(
            np.round(rise_time / self._time_delta) + 1)

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


class SlipRateCalculator(ABC):
    """
    """

    @abstractmethod
    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        pass


class SlipRateGenerator(ABC):
    """
    """

    @abstractmethod
    def __call__(self, seed=None, validate=True):
        """
        """
        pass


class GravesPitarka2010SRC(SlipRateCalculator):
    """
    Slip rate calculator of Graves & Pitarka (2010).
    """

    def __init__(self, time_delta, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(time_delta))

        self._time_delta = time_delta
        self._nsrc = LiuEtAl2006NormalizedSlipRateCalculator(time_delta)

    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        g = GravesPitarka2010SRG(
            self, segment, magnitude, sd, validate=validate)

        return g

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def nsrc(self):
        """
        """
        return self._nsrc

    def get_rise_times(self, depths, slip):
        """
        See Graves & Pitarka (2010) p. 2098 eq. 7.
        """
        return np.interp(depths, [5000, 8000], [2, 1]) * (slip/100)**(1/2)

    def get_average_rise_time(self, dip, moment):
        """
        See Graves & Pitarka (2010) p. 2099 eq. 8 and 9. Adjusted for moment in
        Nm instead of dyn-cm.
        """
        art = np.interp(dip, [45, 60], [0.82, 1]) * 1.6 * 10**-9 * (10**7*moment)**(1/3)

        return art


class GravesPitarka2010SRG(SlipRateGenerator):
    """
    """
    def __init__(self, calculator, segment, magnitude, sd, validate=True):
        """
        """
        if validate is True:
            pass

        segment = segment.get_discretized(shape=sd.shape)
        depths = np.rollaxis(segment.centers, 2)[-1]

        rise_times = calculator.get_rise_times(depths, sd.values)
        average_rt = calculator.get_average_rise_time(
            segment.dip, mw_to_m0(magnitude))

        rise_times = rise_times * (average_rt / rise_times.mean())
        slip_rates = np.zeros(segment.shape + (np.round(rise_times / calculator.time_delta).astype(np.int).max()+1,))

        for i in np.ndindex(segment.shape):
            t = rise_times[i]
            if t != 0:
                sr = calculator.nsrc(float(t)) * sd.values[i]
                slip_rates[i][:len(sr)] = sr

        self._slip_rates = slip_rates

    def __call__(self):
        """
        """
        return self._slip_rates
