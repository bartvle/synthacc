"""
The 'source.rupture.slip' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import scipy.stats

from ...apy import (Object, is_pos_integer, is_number, is_pos_number,
    is_2d_numeric_array)
from ... import space2
from ... import stats
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

        super().__init__(w, l, *slip.shape, validate=False)

        self._values = slip


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

    def _calculate_shape(self, w, l, dw, dl, validate=True):
        """
        Both nw and nl must be odd.
        """
        if validate is True:
            pass
        
        nw = round(w / dw)
        nl = round(l / dl)

        if not ((nw % 2) == 1):
            nw = round(w / (2*dw)) * 2 + 1
        if not ((nl % 2) == 1):
            nl = round(l / (2*dl)) * 2 + 1

        return nw, nl


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
        sdg = RandomFieldSDG(self, segment.width, segment.length, magnitude,
            rigidity, validate=validate)

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

        return stats.VonKarmanACF(h=h)

    def get_aw(self, w):
        """
        """
        mean, sd = 0.55 + 0.35 * w / 1000, 0.78

        return float(scipy.stats.truncnorm.rvs(-2, +2, mean, sd) * 1000)

    def get_al(self, l):
        """
        """
        mean, sd = 1.54 + 0.33 * l / 1000, 2.3

        return float(scipy.stats.truncnorm.rvs(-2, +2, mean, sd) * 1000)


class RandomFieldSDG(SlipDistributionGenerator):
    """
    Random field slip distribution generator.
    """

    def __init__(self, calculator, w, l, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(calculator, RandomFieldSDC))
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_number(magnitude) and is_pos_number(rigidity))

        nw, nl = self._calculate_shape(w, l, calculator.dw, calculator.dl)

        self._calculator = calculator
        self._magnitude = magnitude
        self._rigidity = rigidity

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, nw, nl, validate=False)

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

        srfg = stats.SpatialRandomFieldGenerator(
            self._surface.w, self._surface.l,
            self._surface.nw, self._surface.nl, acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        mean_slip = (mw_to_m0(self._magnitude) /
            (self._surface.area * self._rigidity))
        slip = mean_slip * (1 + field * self._calculator.sd)
        slip[slip < 0] = 0

        indices = np.ones_like(slip)
        n = 20
        for i in range(1, n):
             indices[:,i-1] = i/n
             indices[:,0-i] = i/n
        for i in range(1, n):
             indices[-i,i-1:-i] = i/n

        slip *= indices
        slip *= (mean_slip / slip.mean())

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
    sources[:,0] = randoms[1] * (w-1*radii)
    sources[:,1] = randoms[2] * (l-2*radii) + radii
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
        sdg = CompositeSourceSDG(self, segment.width, segment.length,
            magnitude, rigidity, validate=validate)

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

        p = (surface.area/np.pi)**(3/2) * (3 - self.dimension) / (rmax - rmin)

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

    def __init__(self, calculator, w, l, magnitude, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(calculator, CompositeSourceSDC))
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))

        nw, nl = self._calculate_shape(w, l, calculator.d, calculator.d)

        surface = space2.DiscretizedRectangularSurface(
            w, l, nw, nl, validate=validate)

        rmin = calculator.rminf * calculator.d
        rmax = calculator.rmaxf * w

        p = calculator.get_p(rmin, rmax, surface)
        n = calculator.get_n(rmin, rmax, p)

        c = ((1.5 / np.pi) * (mw_to_m0(magnitude) /
            (np.sqrt(surface.area/np.pi)**3 * rigidity)))

        self._calculator = calculator
        self._surface = surface
        self._rmin = rmin
        self._rmax = rmax
        self._p = p
        self._n = n
        self._c = c

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
        
        res = (sources[:,2]**2 - space2.distance(
            self._surface.xgrid, self._surface.ygrid,
            sources[:,0], sources[:,1],)**2)
        con = np.zeros_like(res)
        indices = res > 0
        con[indices] = np.sqrt(res[indices])

        slip = self._c * np.sum(con, axis=2)

        sd = CompositeSourceSD(self._surface.w, self._surface.l, slip, sources)

        return sd


class NormalizedSlipRateCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self, rise_time, validate=True):
        """
        """
        pass


class LiuEtAl2006NSRC(NormalizedSlipRateCalculator):
    """
    Normalized slip rate calculator of Liu et al. (2006).
    """

    def __call__(self, time_delta, rise_time, validate=True):
        """
        See Liu et al. (2006) p. 2121 eq. 7a and 7b.
        """
        if validate is True:
            assert(is_pos_number(rise_time))

        if rise_time <= time_delta:
            """
            """
            return np.array([0, 1 / time_delta, 0])

        times = time_delta * np.arange(
            np.round(rise_time / time_delta) + 1)

        t1 = 0.13*rise_time
        t2 = rise_time - t1
        cn = np.pi / (1.4*np.pi*t1 + 1.2*t1 + 0.3*np.pi*t2)

        i1 = times < t1
        i3 = times >= 2*t1
        i2 = ~(i1 | i3)

        f = np.zeros_like(times)
        f[i1] = (0.7 - 0.7*np.cos(np.pi*times[i1]/t1) +
                    0.6*np.sin(np.pi*times[i1]/(2.*t1)))
        f[i2] = (1.0 - 0.7*np.cos(np.pi*times[i2]/t1) +
                    0.3*np.cos(np.pi*(times[i2]-t1)/t2))
        f[i3] = (0.3 + 0.3*np.cos(np.pi*(times[i3]-t1)/t2))

        nsrs = cn * f

        try:
            nsrs /= np.sum(nsrs*time_delta)
        except Exception as e:
            print('nsrs', time_delta, rise_time, np.sum(nsrs*time_delta))
            raise e

        return nsrs


class LiuArchuleta2004NSRC(NormalizedSlipRateCalculator):
    """
    Normalized slip rate calculator of Liu & Archuleta (2004).
    """

    def __call__(self, time_delta, rise_time, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(rise_time))

        if rise_time <= time_delta:
            """
            """
            return np.array([0, 1 / time_delta, 0])

        times = time_delta * np.arange(
            np.round(rise_time / time_delta) +1)

        t = times/rise_time
        nsrs = t * (1-t)**4

        try:
            nsrs /= np.sum(nsrs*time_delta)
        except Exception as e:
            print('nsrs', time_delta, rise_time, np.sum(nsrs*time_delta))
            raise e

        return nsrs


class SchmedesEtAl2010NSRC(NormalizedSlipRateCalculator):
    """
    Normalized slip rate calculator of Schmedes et al. (2010).
    """

    def __call__(self, time_delta, rise_time, validate=True):
        """
        See Schmedes et al. (2010) p. 4 eq. 2.
        """
        if validate is True:
            assert(is_pos_number(rise_time))

        peak_time = rise_time / 5
        tail_time = rise_time - peak_time

        if peak_time < (2*time_delta):
            return LiuArchuleta2004NSRC()(time_delta, rise_time)

        times1 = time_delta * np.arange(np.round(tail_time / time_delta) +1)
        times2 = time_delta * np.arange(np.round(peak_time / time_delta) +1)

        p1 = np.zeros_like(times1)
        p2 = np.zeros_like(times2)

        p1[1:-1] = np.sqrt(tail_time - times1[1:-1]) / np.sqrt(times1[1:-1])
        p2 = np.sin(np.pi*times2 / peak_time)

        nsrs = np.convolve(p1, p2, mode='full')

        try:
            nsrs /= np.sum(nsrs*time_delta)
        except Exception as e:
            print('nsrs', time_delta, rise_time, np.sum(nsrs*time_delta))
            raise e

        return nsrs


class RiseTimeCalculator(Object):
    """
    Rise time calculator based on Graves & Pitarka (2010, 2015).
    """

    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        depths = segment.get_depths(sd.shape)

        rise_times = self.get_rise_times(depths, sd.values)
        average_rt = self.get_average_rise_time(
            segment.dip, mw_to_m0(magnitude))

        rise_times = rise_times * (average_rt / rise_times.mean())

        return rise_times

    def get_rise_times(self, depths, slip):
        """
        See Graves & Pitarka (2010) p. 2098 eq. 7. and Graves & Pitarka (2015).
        """
        rts = np.interp(depths, [5000, 8000, 17000, 20000], [2, 1, 1, 2])
        rts *= (slip/100)**(1/2)

        return rts

    def get_average_rise_time(self, dip, moment):
        """
        See Graves & Pitarka (2010) p. 2099 eq. 8 and 9. Adjusted for moment in
        Nm instead of dyn-cm.
        """
        art = np.interp(dip, [45, 60], [0.82, 1]) * 1.6 * 10**-9 * (10**7*moment)**(1/3)

        return art


class SlipRateCalculator(Object):
    """
    """

    def __init__(self, d, rtc, nsrc):
        """
        d: time_delta
        rtc: 'RiseTimeCalculator' instance
        nsrc: 'NormalizedSlipRateCalculator' instance
        """
        self._d, self._rtc, self._nsrc = d, rtc, nsrc

    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        rise_times = self._rtc(segment, magnitude, sd)
        slip_rates = np.zeros(
            sd.shape + (np.round(rise_times/self._d).astype(np.int).max()+1,))

        for i in np.ndindex(sd.shape):
            t = float(rise_times[i])
            if t != 0:
                sr = self._nsrc(self._d, t) * sd.values[i]
                
                try:
                    assert(np.all(sr >= 0))
                    slip_rates[i][:len(sr)] = sr
                except Exception as e:
                    print('nsrs', self._nsrc)
                    raise e

        return slip_rates
