"""
The 'space2' module.
"""


from abc import ABC, abstractmethod

from numba import jit
import numpy as np

from .apy import PRECISION, Object, is_number, is_pos_number, is_pos_integer


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


class SpatialRandomFieldGenerator(Object):
    """
    """

    def __init__(self, nw, nl, dw, dl, acf, aw, al, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(nw) and nw % 2 == 1)
            assert(is_pos_integer(nl) and nl % 2 == 1)
            assert(is_pos_number(dw))
            assert(is_pos_number(dl))
            assert(isinstance(acf, ACF))
            assert(is_pos_number(aw))
            assert(is_pos_number(al))

        self._nw = nw
        self._nl = nl
        self._dw = dw
        self._dl = dl
        self._acf = acf
        self._aw = aw
        self._al = al

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            if seed is not None:
                assert(is_pos_integer(seed))

        rw = self.nw // 2 ## index of middle row
        rl = self.nl // 2 ## index of middle col

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

        ## Shift
        dft = np.fft.ifftshift(Y)

        ## inverse 2D complex FFT
        ## remaining imaginary part is due to machine precision
        field = np.real(np.fft.ifft2(dft))

        # Remove mean and scale to unit variance
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

    @property
    def kw(self):
        """
        nw wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nw this is in practice not the sampling rate but
        (nw//2 / nw) * (1/dw)) * np.linspace(-2*np.pi, 2*np.pi, self.nw)

        returns kw from - over 0 to +
        """
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.nw, self.dw))

    @property
    def kl(self):
        """
        nl wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nl this is in practice not the sampling rate but
        (nl//2 / nl) * (1/dl) * np.linspace(-2*np.pi, 2*np.pi, self.nl)

        returns kl from - over 0 to +
        """
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.nl, self.dl))

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
