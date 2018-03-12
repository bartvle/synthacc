"""
The 'math.random' module.
"""


from abc import ABC, abstractmethod

import scipy.special
import numpy as np

from ..apy import Object, is_pos_number, is_pos_integer


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

    def __init__(self, dimension, spacing, acf, acd, seed=None, validate=True):
        """
        dimension: number of samples (width, length), must be odd
        spacing: (d_w, d_l)
        acf: autocorrelation function
        acd: autocorrelation distance
        seed: integer
        """
        if validate is True:
            assert(len(dimension) == 2)
            assert(is_pos_integer(dimension[0]) and dimension[0] % 2 == 1)
            assert(is_pos_integer(dimension[1]) and dimension[1] % 2 == 1)
            assert(len(spacing) == 2)
            assert(is_pos_number(spacing[0]))
            assert(is_pos_number(spacing[1]))
            assert(isinstance(acf, ACF))
            assert(len(acd) == 2)
            assert(is_pos_number(acd[0]))
            assert(is_pos_number(acd[1]))
            if seed is not None:
                assert(is_pos_integer(seed))

        self._n_w = dimension[0]
        self._n_l = dimension[1]

        self._d_w = spacing[0]
        self._d_l = spacing[1]

        self._acf = acf
        self._a_w = acd[0]
        self._a_l = acd[1]

        self._seed = seed

    def __call__(self):
        """
        """
        r_w = self._n_w // 2 ## index of middle row
        r_l = self._n_l // 2 ## index of middle col

        ## Take upper half of psd (including middle row)
        psd = self.psd[:r_w+1]

        ## Normalize PSD for iFFT
        amplitudes = np.sqrt(psd/psd.max())
        # amplitudes = np.sqrt(psd)

        ## Set DC-component to 0
        amplitudes[r_w, r_l] = 0

        ## Get random phases
        if self.seed is not None:
            np.random.seed(self.seed)
        phases = 2 * np.pi * np.random.random(size=amplitudes.shape)

        ## Construct DFT for upper half (positive frequency rows)
        Y = amplitudes * np.cos(phases) + 1j * amplitudes * np.sin(phases)

        ## Construct DFT for lower half (negative frequency rows)
        Y = np.concatenate((Y, np.conj(np.fliplr(np.flipud(Y[:-1])))))

        ## Construct DFT for for middle row (zero frequency row)
        Y[r_w,:r_l:-1] = np.conj(Y[r_w,:r_l])

        ## Shift
        dft = np.fft.ifftshift(Y)

        ## inverse 2D complex FFT
        ## remaining imaginary part is due to machine precision
        field = np.real(np.fft.ifft2(dft))

        # Remove mean and scale to unit variance
        field = field / np.std(field, ddof=1)

        # if np.mean(field) < 0:
            # field *= -1 # (small) positive mean

        return field

    @property
    def n_w(self):
        """
        """
        return self._n_w

    @property
    def n_l(self):
        """
        """
        return self._n_l

    @property
    def d_w(self):
        """
        """
        return self._d_w

    @property
    def d_l(self):
        """
        """
        return self._d_l

    @property
    def acf(self):
        """
        """
        return self._acf

    @property
    def a_w(self):
        """
        """
        return self._a_w

    @property
    def a_l(self):
        """
        """
        return self._a_l

    @property
    def k_w(self):
        """
        n_w wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd n_w this is in practice not the sampling rate but
        (n_w//2 / n_w) * (1/d_w)) * np.linspace(-2*np.pi, 2*np.pi, self.n_w)

        returns k_w from - over 0 to +
        """
        return 2*np.pi * np.fft.fftshift(np.fft.fftfreq(self._n_w, self._d_w))

    @property
    def k_l(self):
        """
        n_l wavenumbers from -half the sampling rate to +half the sampling rate

        For an odd n_l this is in practice not the sampling rate but
        (n_l//2 / n_l) * (1/d_l) * np.linspace(-2*np.pi, 2*np.pi, self.n_l)

        returns k_l from - over 0 to +
        """
        return 2*np.pi * np.fft.fftshift(np.fft.fftfreq(self._n_l, self._d_l))

    @property
    def k_r(self):
        """
        Radial wavenumbers
        """
        k_r = np.sqrt(np.add.outer(
            self.k_w**2 * self.a_w**2,
            self.k_l**2 * self.a_l**2,
        ))
        return k_r

    @property
    def psd(self):
        """
        """
        return self.acf.get_psd(self.k_r, a=self.a_w*self.a_l)

    @property
    def seed(self):
        """
        """
        return self._seed
