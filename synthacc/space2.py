"""
The 'space2' module.
"""


from abc import ABC, abstractmethod

from numba import jit
import numpy as np
import scipy.special

from .apy import PRECISION, Object, is_number, is_pos_number, is_pos_integer


class Point(Object):
    """
    A point.
    """

    def __init__(self, x, y, validate=True):
        """
        x: number, x coordinate
        y: number, y coordinate
        """
        if validate is True:
            assert(is_number(x))
            assert(is_number(y))

        if abs(x) < 10**-PRECISION:
            x = 0
        if abs(y) < 10**-PRECISION:
            y = 0

        self._x = x
        self._y = y

    def __repr__(self):
        """
        """
        s = '< space2.Point | '
        s += 'x={:{}.3f}'.format(self.x, '+' if self.x else '')
        s += ', '
        s += 'y={:{}.3f}'.format(self.y, '+' if self.y else '')
        s += ' >'

        return s

    def __getitem__(self, i):
        """
        """
        return (self._x, self._y)[i]

    def __eq__(self, other):
        """
        return: boolean
        """
        assert(type(other) is self.__class__)

        x, y = other
        x_eq = np.abs(self.x - x) < 10**-PRECISION
        y_eq = np.abs(self.y - y) < 10**-PRECISION

        return (x_eq and y_eq)

    @property
    def x(self):
        """
        return: number, x coordinate
        """
        return self._x

    @property
    def y(self):
        """
        return: number, y coordinate
        """
        return self._y


class RectangularSurface(Object):
    """
    The origin is in the upper left corner. X is along W and Y is along L.
    """

    def __init__(self, w, l, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))

        self._w = w
        self._l = l

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
    def area(self):
        """
        """
        return self.w * self.l

    def get_random(self, seed=None, validate=True):
        """
        Get a random point on the surface.

        return: 'space2.Point' instance
        """
        if validate is True:
            assert(seed is None or is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        x = np.random.uniform(0, 1) * self.w
        y = np.random.uniform(0, 1) * self.l

        return Point(x, y)


class DiscretizedRectangularSurface(RectangularSurface):
    """
    The origin is in the upper left corner. X is along W and Y is along L. The
    dw and dl parameters are recalculated based on nw and nl.
    """

    def __init__(self, w, l, dw, dl, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw) and dw <= w/2)
            assert(is_pos_number(dl) and dl <= l/2)

        super().__init__(w, l, validate=False)

        nw = round(w / dw)
        nl = round(l / dl)
        dw = w / nw
        dl = l / nl
        ws = np.linspace(0+dw/2, w-dw/2, nw)
        ls = np.linspace(0+dl/2, l-dl/2, nl)

        self._grid = np.dstack(np.meshgrid(ls, ws))
        self._dw = dw
        self._dl = dl

    def __len__(self):
        """
        """
        return np.prod(self.shape)

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
        return self._grid[:,0,1]

    @property
    def ys(self):
        """
        return: 1d numerical array
        """
        return self._grid[0,:,0]

    @property
    def xgrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,1]

    @property
    def ygrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,0]


class ACF(ABC, Object):
    """
    An autocorrelation function.
    """

    @abstractmethod
    def __call__(self, r, a):
        """
        """
        pass

    @abstractmethod
    def get_psd(self, k, a):
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

    def __call__(self, r, a=1):
        """
        """
        r = r + (r[1]-r[0])
        acf = r**self.h * scipy.special.kv(self.h, r/a)
        acf /= acf[0]

        return acf

    @property
    def h(self):
        """
        """
        return self._h

    def get_psd(self, k, a=1):
        """
        """
        return a/(1+k**2)**(self.h+1)


class SpatialRandomFieldCalculator(Object):
    """
    """

    def __init__(self, dw, dl, acf, aw, al, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(dw))
            assert(is_pos_number(dl))
            assert(isinstance(acf, ACF))
            assert(is_pos_number(aw))
            assert(is_pos_number(al))

        self._dw = dw
        self._dl = dl
        self._acf = acf
        self._aw = aw
        self._al = al

    def __call__(self, nw, nl, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(nw) and nw % 2 == 1)
            assert(is_pos_integer(nl) and nl % 2 == 1)

        srf = SpatialRandomFieldGenerator(nw, nl, self.dw, self.dl,
            self.acf, self.aw, self.al, validate=False)

        return srf

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


class SpatialRandomFieldGenerator(SpatialRandomFieldCalculator):
    """
    The origin is in the upper left corner. X is along W and Y is along L.
    """

    def __init__(self, nw, nl, dw, dl, acf, aw, al, validate=True):
        """
        """
        super().__init__(dw, dl, acf, aw, al, validate=validate)

        if validate is True:
            assert(is_pos_integer(nw) and nw % 2 == 1)
            assert(is_pos_integer(nl) and nl % 2 == 1)

        self._nw = nw
        self._nl = nl

        self._kw = self._calc_kw()
        self._kl = self._calc_kl()
        self._kr = self._calc_kr()

        self._psd = self._calc_psd()

        self._rw = self._nw // 2 ## index of middle row
        self._rl = self._nl // 2 ## index of middle col

        self._amplitudes = self._calc_amplitudes()

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            if seed is not None:
                assert(is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        ## Get random phases
        phases = 2 * np.pi * np.random.random(size=self._amplitudes.shape)

        ## 1. Construct DFT for upper half (positive frequency rows)
        real = self._amplitudes * np.cos(phases)
        imag = self._amplitudes * np.sin(phases)
        Y =  real + 1j * imag

        ## 2. Construct DFT for lower half (negative frequency rows)
        flip_1 = np.flipud(Y[:-1])
        flip_2 = np.fliplr(flip_1)
        Y = np.concatenate((Y, np.conj(flip_2)))

        ## 3. Construct DFT for for middle row (zero frequency row)
        Y[self._rw,:self._rl:-1] = np.conj(Y[self._rw,:self._rl])

        ## inverse FFT (remaining imaginary part is due to machine precision)
        field = np.real(np.fft.ifft2(np.fft.ifftshift(Y)))

        ## Remove mean and scale to unit variance
        field = field / np.std(field, ddof=1)

        ## Positive (small) mean
        if np.mean(field) < 0:
            field *= -1

        return field

    @property
    def nw(self):
        """
        """
        return self._nw

    @property
    def nl(self):
        """
        """
        return self._nl

    @property
    def shape(self):
        """
        """
        return (self.nw, self.nl)

    @property
    def kw(self):
        """
        """
        return self._kw

    @property
    def kl(self):
        """
        """
        return self._kl

    @property
    def kr(self):
        """
        """
        return self._kr

    def _calc_kw(self):
        """
        nw wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nw this is in practice not the sampling rate but
        (nw//2 / nw) * (1/dw)) * np.linspace(-2*np.pi, 2*np.pi, self.nw)

        returns kw from - over 0 to +
        """
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self._nw, self._dw))

    def _calc_kl(self):
        """
        nl wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nl this is in practice not the sampling rate but
        (nl//2 / nl) * (1/dl) * np.linspace(-2*np.pi, 2*np.pi, self.nl)

        returns kl from - over 0 to +
        """
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self._nl, self._dl))

    def _calc_kr(self):
        """
        Radial wavenumbers
        """
        kr = np.sqrt(np.add.outer(
            self._kw**2 * self._aw**2,
            self._kl**2 * self._al**2,
        ))
        return kr

    def _calc_psd(self):
        """
        """
        return self._acf.get_psd(self._kr, a=self._aw*self._al)

    def _calc_amplitudes(self):
        """
        """
        ## Take upper half of psd (including middle row)
        psd = self._psd[:self._rw+1]

        ## Normalize PSD for iFFT
        amplitudes = np.sqrt(psd/psd.max())

        ## Set DC-component to 0
        amplitudes[self._rw,self._rl] = 0

        return amplitudes


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


def cartesian_to_polar(x, y, validate=True):
    """
    """
    if validate is True:
        assert(is_number(x))
        assert(is_number(y))

    r = float(distance(0, 0, x, y))
    a = float(np.degrees(np.arctan2(y, x)))

    if abs(r) < 10**-PRECISION:
        r = 0
    if abs(a) < 10**-PRECISION:
        a = 0

    return r, a


def polar_to_cartesian(r, a, validate=True):
    """
    """
    if validate is True:
        assert(is_number(r))
        assert(is_number(a))

    x = float(r * np.cos(np.radians(a)))
    y = float(r * np.sin(np.radians(a)))

    if abs(x) < 10**-PRECISION:
        x = 0
    if abs(y) < 10**-PRECISION:
        y = 0

    return x, y
