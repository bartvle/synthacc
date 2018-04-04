"""
The 'source.ruptures' module.
"""


from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit
import numpy as np
import numpy.ma as ma
import scipy.special

from ..apy import (Object, is_number, is_pos_number, is_pos_integer,
    is_2d_numeric_array, is_3d_numeric_array)
from .. import space
from ..earth.flat import RectangularSurface, DiscretizedRectangularSurface
from .moment import (NormalizedMomentRateFunction, MomentRateFunction,
    NormalizedSlipRateFunction, calculate as calculate_moment, m0_to_mw,
    mw_to_m0)
from .mechanism import FocalMechanism, is_rake
from .faults import RIGIDITY


@jit(nopython=True)
def _distance(x1, y1, x2, y2):
    """
    """
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


class Surface(Object):
    """
    A two-dimensional rupture surface.
    """

    def __init__(self, w, l, dw, dl, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw))
            assert(is_pos_number(dl))

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


class PointRupture(Object):
    """
    """

    def __init__(self, point, focal_mechanism, moment, nmrf=None, validate=True):
        """
        """
        point = space.Point(*point, validate=validate)

        if validate is True:
            assert(type(focal_mechanism) is FocalMechanism)
            assert(is_pos_number(moment))
            if nmrf is not None:
                assert(type(nmrf) is NormalizedMomentRateFunction)

        self._point = point
        self._focal_mechanism = focal_mechanism
        self._moment = moment
        self._nmrf = nmrf

    @property
    def point(self):
        """
        """
        return self._point

    @property
    def focal_mechanism(self):
        """
        """
        return self._focal_mechanism

    @property
    def moment(self):
        """
        """
        return self._moment

    @property
    def nmrf(self):
        """
        """
        return self._nmrf

    @property
    def moment_tensor(self):
        """
        """
        return self._focal_mechanism.get_moment_tensor(self.moment)


class SimpleRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, slip, nsrf=None, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is RectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            if nsrf is not None:
                assert(type(nmrf) is NormalizedSlipRateFunction)
            assert(is_pos_number(rigidity))

        self._surface = surface
        self._hypo = hypo
        self._rake = rake
        self._slip = slip
        self._nsrf = nsrf
        self._rigidity = rigidity

        ## cached properties
        self._discretized, self._spacing = None, None
        
    @property
    def surface(self):
        """
        """
        return self._surface

    @property
    def hypo(self):
        """
        """
        return self._hypo

    @property
    def rake(self):
        """
        """
        return self._rake

    @property
    def slip(self):
        """
        """
        return self._slip

    @property
    def nsrf(self):
        """
        """
        return self._nsrf

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    @property
    def area(self):
        """
        """
        return self.surface.area

    @property
    def epi(self):
        """
        """
        return space.Point(self.hypo.x, self.hypo.y, 0)

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        return FocalMechanism(self.surface.strike, self.surface.dip, self.rake)

    @property
    def moment(self):
        """
        """
        moment = calculate_moment(
            self.area, self.slip, self.rigidity, validate=False)

        return moment

    @property
    def magnitude(self):
        """
        """
        return m0_to_mw(self.moment)

    def get_hypo_distance(self, point, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space.distance(*self.hypo, *p)
        return d

    def get_epi_distance(self, point, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space.distance(*self.epi, *p)
        return d

    def get_rup_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
            assert(is_pos_number(spacing))

        if self._discretized is None or self._spacing != spacing:
            w, l = self._surface.width, self._surface.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self.surface.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)

        x, y, z = space.nearest(*p, xs, ys, zs)
        d = space.distance(x, y, z, *p)

        return d

    def get_jb_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
            assert(is_pos_number(spacing))

        if self._discretized is None or self._spacing != spacing:
            w, l = self._surface.width, self._surface.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self.surface.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)
        zs = np.zeros_like(zs)

        x, y, z = space.nearest(*p, xs, ys, zs)
        d = space.distance(x, y, z, *p)

        return d

    def plot(self):
        """
        """
        fig, ax = plt.subplots()

        ulc, urc, llc, lrc = self.surface.corners

        ax.plot([ulc.y, urc.y], [ulc.x, urc.x], c='r', lw=2)

        ax.fill(
            [ulc.y, urc.y, lrc.y, llc.y],
            [ulc.x, urc.x, lrc.x, llc.x],
            color='coral', alpha=0.5,
            )

        ax.scatter([self.hypo.y], [self.hypo.x], marker='*', s=50)

        ax.axis('equal')

        x_label, y_label = 'East (m)', 'North (m)'
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        plt.show()


class KinematicRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, time_delta, slip_rates, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is DiscretizedRectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(slip_rates.shape[:2] == surface.shape)
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(is_pos_number(rigidity))

        self._surface = surface
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._rigidity = rigidity

    def __len__(self):
        """
        """
        return len(self.surface)

    def __iter__(self):
        """
        """
        slip = self.slip
        area = self.surface.cell_area
        centers = self.surface.centers
        fm = self.focal_mechanism

        for i in np.ndindex(self.surface.shape):
            moment = calculate_moment(float(slip[i], area, self.rigidity))

            if moment == 0:
                return None

            x, y, z = centers[i]
            x = float(x)
            y = float(y)
            z = float(z)
            point = space.Point(x, y, z)

            nmrf = NormalizedMomentRateFunction(
            self.time_delta, self._slip_rates[i] / slip[i])

            yield PointRupture(point, fm, moment, nmrf)

    @property
    def surface(self):
        """
        """
        return self._surface

    @property
    def hypo(self):
        """
        """
        return self._hypo

    @property
    def rake(self):
        """
        """
        return self._rake

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    @property
    def area(self):
        """
        """
        return self.surface.area

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        return FocalMechanism(self.surface.strike, self.surface.dip, self.rake)

    @property
    def onsets(self):
        """
        """
        onsets = np.zeros(self.surface.shape)
        for i in np.ndindex(onsets.shape):
            indices = np.where(self._slip_rates[i] != 0)[0]
            if len(indices) != 0:
                onsets[i] = indices[0] * self._time_delta
            else:
                onsets[i] = np.nan

        return onsets

    @property
    def slip(self):
        """
        """
        return np.sum(self._slip_rates, axis=2) * self._time_delta

    @property
    def mean_slip(self):
        """
        """
        return np.mean(self.slip)

    @property
    def mrf(self):
        """
        return: 'source.moment.MomentRateFunction' instance
        """
        moment_rates = (self._slip_rates * self.surface.cell_area *
            self.rigidity)
        moment_rates = np.sum(moment_rates, axis=(0,1))
        mrf = MomentRateFunction(self._time_delta, moment_rates)
        return mrf

    @property
    def moment(self):
        """
        """
        return self.mrf.moment

    @property
    def magnitude(self):
        """
        """
        return m0_to_mw(self.moment)

    def _plot(self, data, label, title=None, validate=True):
        """
        """
        if validate is True:
            assert(data.shape == self._surface.shape)

        f, ax = plt.subplots()
        xs, ys = self.surface.get_front_projected_corners()

        m = ma.masked_where(np.isnan(data), data)

        p = ax.pcolormesh(xs, ys, m)

        ax.axis('scaled')
        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim(ax.get_yaxis().get_data_interval())
        ax.invert_yaxis()

        x_label, y_label = 'Along strike (m)', 'Along dip (m)'
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=0.25)
        cbar = f.colorbar(p, cax=cax, norm=mpl.colors.Normalize(vmin=0))
        cbar.set_label(label)

        ax.set_title(title)

        plt.show()

    def plot_onsets(self):
        """
        """
        self._plot(self.onsets, 'Onset (s)', 'Rupture onset')

    def plot_slip(self):
        """
        """
        self._plot(self.slip, 'Slip (m)', 'Slip distribution')


class SlipDistribution(Surface):
    """
    A spatial slip distribution.
    """

    def __init__(self, w, l, slip, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(slip))

        dw = w / slip.shape[0]
        dl = l / slip.shape[1]

        super().__init__(w, l, dw, dl, validate=False)

        self._slip = slip

    @property
    def surface(self):
        """
        """
        return Surface(self.w, self.l, self.dw, self.dl, validate=False)

    @property
    def slip(self):
        """
        """
        return self._slip

    def plot(self, size=None, png_filespec=None, validate=True):
        """
        """
        f, ax = plt.subplots(figsize=size)

        extent = [0, self.l/1000, self.w/1000, 0]
        p = ax.imshow(self._slip, interpolation='bicubic', extent=extent)
        plt.axis('scaled')

        xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=0.25)
        cbar = f.colorbar(p, cax=cax)
        cbar.set_label('Slip (m)')

        plt.tight_layout()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


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
            assert(isinstance(acf, ACF))
            assert(is_pos_number(aw))
            assert(is_pos_number(al))

        self._surface = Surface(w, l, dw, dl, validate=False)

        self._acf = acf
        self._aw = aw
        self._al = al

    def __call__(self, magnitude, rigidity=RIGIDITY, std=1, seed=None, validate=True):
        """
        """
        if validate is True:
            assert(is_number(magnitude))
            assert(is_pos_number(rigidity))
            if seed is not None:
                assert(is_pos_integer(seed))

        rw = self.surface.nw // 2 ## index of middle row
        rl = self.surface.nl // 2 ## index of middle col

        ## Take upper half of psd (including middle row)
        psd = self.psd[:rw+1]

        ## Normalize PSD for iFFT
        amplitudes = np.sqrt(psd/psd.max())
        # amplitudes = np.sqrt(psd)

        ## Set DC-component to 0
        amplitudes[rw, rl] = 0

        ## Get random phases
        if seed is not None:
            np.random.seed(seed)
        phases = 2 * np.pi * np.random.random(size=amplitudes.shape)

        ## Construct DFT for upper half (positive frequency rows)
        Y = amplitudes * np.cos(phases) + 1j * amplitudes * np.sin(phases)

        ## Construct DFT for lower half (negative frequency rows)
        Y = np.concatenate((Y, np.conj(np.fliplr(np.flipud(Y[:-1])))))

        ## Construct DFT for for middle row (zero frequency row)
        Y[rw,:rl:-1] = np.conj(Y[rw,:rl])

        # ## Shift
        dft = np.fft.ifftshift(Y)

        ## inverse 2D complex FFT
        ## remaining imaginary part is due to machine precision
        field = np.real(np.fft.ifft2(dft))

        # Remove mean and scale to unit variance
        field = field / np.std(field, ddof=1)

        if np.mean(field) < 0:
            field *= -1 # (small) positive mean

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

    @property
    def kw(self):
        """
        nw wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nw this is in practice not the sampling rate but
        (nw//2 / nw) * (1/dw)) * np.linspace(-2*np.pi, 2*np.pi, self.nw)

        returns kw from - over 0 to +
        """
        return 2*np.pi * np.fft.fftshift(
            np.fft.fftfreq(self.surface.nw, self.surface.dw))

    @property
    def kl(self):
        """
        nl wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nl this is in practice not the sampling rate but
        (nl//2 / nl) * (1/dl) * np.linspace(-2*np.pi, 2*np.pi, self.nl)

        returns kl from - over 0 to +
        """
        return 2*np.pi * np.fft.fftshift(
            np.fft.fftfreq(self.surface.nl, self.surface.dl))

    @property
    def kr(self):
        """
        Radial wavenumbers
        """
        kr = np.sqrt(np.add.outer(
            self.kw**2 * self.aw**2,
            self.kl**2 * self.al**2,
        ))
        return kr

    @property
    def psd(self):
        """
        """
        return self.acf.get_psd(self.kr, a=self.aw*self.al)


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

        self._surface = Surface(w, l, d, d, validate=validate)

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

        distances = _distance(x1, y1, sources[:,0], sources[:,1])

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


class KinematicRuptureGenerator(Object):
    """
    """

    def __init__(self, time_delta, sdg, rfg, ttc, validate=True):
        """
        """

    def __call__(self):
        """
        """


class GP2016KinematicRuptureGenerator(Object):
    """
    Graves & Pitarka (2016) kinematic rupture generator (GP15.4).
    """

    def __init__(self, time_delta, velocity, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_pos_number(velocity))
            assert(is_pos_number(rigidity))

        self._time_delta = time_delta
        self._velocity = velocity
        self._rigidity = rigidity

    def __call__(self, surface, rake, magnitude, validate=True):
        """
        """
        if validate is True:
            assert(type(surface) is RectangularSurface)
            assert(is_rake(rake))
            assert(is_number(magnitude))

        moment = mw_to_m0(magnitude)

        hypo = surface.get_random()

        w, l = surface.width, surface.length

        nw = int(w / 100 // 2 * 2 + 1)
        nl = int(l / 100 // 2 * 2 + 1)

        surface = surface.get_discretized(shape=(nw, nl))

        dw = surface.spacing[0]
        dl = surface.spacing[1]

        acf = VonKarmanACF(h=0.75)

        aw = 10**(1/3*magnitude-1.6) * 1000
        al = 10**(1/2*magnitude-2.5) * 1000

        g = RFSlipDistributionGenerator(w, l, dw, dl, acf, aw, al)

        sd = g(magnitude, self.rigidity)

        _, _, depths = np.rollaxis(surface.centers, 2)
        rise_times = self._get_rise_times(depths, sd.slip)

        average = self._get_average_rise_time(surface.dip, moment)
        rise_times *= (average / rise_times.mean())

        onsets = (space.distance(*hypo, *np.rollaxis(surface.centers, 2)) /
            self.velocity)

        n_onsets = np.round(onsets / self.time_delta).astype(np.int)

        n_rise_times = np.round(rise_times / self.time_delta).astype(np.int)

        n = n_onsets + n_rise_times

        slip_rates = np.zeros(surface.shape + (n.max()+2,))

        nsrf_g = LiuEtAl2006NormalizedSlipRateGenerator(self.time_delta)

        for i in np.ndindex(surface.shape):
            t = rise_times[i]
            if t != 0:
                srf = nsrf_g(float(t))
                slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf * sd.slip[i]

        rupture = KinematicRupture(surface, hypo, rake, self._time_delta,
            slip_rates, self._rigidity)

        return rupture

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def velocity(self):
        """
        """
        return self._velocity

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    def _get_average_rise_time(self, dip, moment):
        """
        See Graves & Pitarka (2010) p. 2099 eq. 8 and 9. Adjusted for moment in
        Nm instead of dyn-cm.
        """
        factor = np.interp(dip, [45, 60], [0.82, 1])
        t = factor * 1.6 * 10**-9 * (10**7*moment)**(1/3)
        return t

    def _get_rise_times(self, depths, slip):
        """
        See Graves & Pitarka (2010) p. 2098 eq. 7.
        """
        return np.interp(depths, [5000, 8000], [2, 1]) * (slip/100)**(1/2)
