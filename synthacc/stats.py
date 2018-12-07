"""
The 'stats' module.
"""


from abc import ABC, abstractmethod

import numpy as np
import scipy.special

from .apy import Object, is_pos_integer, is_pos_number
from .space2 import DiscretizedRectangularSurface


class ACF(ABC, Object):
    """
    An autocorrelation function.
    """

    @abstractmethod
    def __call__(self, r, a, validate=True):
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


class SpatialRandomFieldGenerator(DiscretizedRectangularSurface):
    """
    """

    def __init__(self, w, l, nw, nl, acf, aw, al, validate=True):
        """
        """
        super().__init__(w, l, nw, nl, validate=validate)

        if validate is True:
            assert(nw % 2 == 1)
            assert(nl % 2 == 1)
            assert(isinstance(acf, ACF))
            assert(is_pos_number(aw))
            assert(is_pos_number(al))

        self._acf = acf
        self._aw = aw
        self._al = al

        self._kw = self._calc_kw()
        self._kl = self._calc_kl()
        self._kr = self._calc_kr()

        self._psd = self._calc_psd()

        self._amplitudes = self._calc_amplitudes()

    def __call__(self, seed=None, validate=True):
        """
        """
        if validate is True:
            assert(seed is None or is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        ## Get random phases
        phases = 2 * np.pi * np.random.random(size=self._amplitudes.shape)

        ## 1. Construct DFT for upper half (positive frequency rows)
        real = self._amplitudes * np.cos(phases)
        imag = self._amplitudes * np.sin(phases)
        y =  real + 1j * imag

        ## 2. Construct DFT for lower half (negative frequency rows)
        flip_1 = np.flipud(y[:-1])
        flip_2 = np.fliplr(flip_1)
        y = np.concatenate((y, np.conj(flip_2)))

        ## 3. Construct DFT for for middle row (zero frequency row)
        y[self.nw//2,:self.nl//2:-1] = np.conj(y[self.nw//2,:self.nl//2])

        ## inverse FFT (remaining imaginary part is due to machine precision)
        field = np.real(np.fft.ifft2(np.fft.ifftshift(y)))

        ## Remove mean, scale to unit variance and give (small) positive mean
        field = field / np.std(field, ddof=1)

        if np.mean(field) < 0:
            field *= -1

        return field

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
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.nw, self.dw))

    def _calc_kl(self):
        """
        nl wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd nl this is in practice not the sampling rate but
        (nl//2 / nl) * (1/dl) * np.linspace(-2*np.pi, 2*np.pi, self.nl)

        returns kl from - over 0 to +
        """
        return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.nl, self.dl))

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
        Take upper half of psd (including middle row), normalize PSD for iFFT,
        calculate amplitudes and set DC-component to 0.
        """
        psd = self._psd[:self.nw//2+1]
        psd /= psd.max()

        amplitudes = np.sqrt(psd)
        amplitudes[self.nw//2,self.nl//2] = 0

        return amplitudes
