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

    def __init__(self, dimension, spacing, acf, cd, seed=None, validate=True):
        """
        dimension: (width, length)
        spacing: (d_w, d_l)
        acf: autocorrelation function
        cd: correlation distance
        seed: integer
        """
        if validate is True:
            assert(len(dimension) == 2)
            assert(is_pos_number(dimension[0]))
            assert(is_pos_number(dimension[1]))
            assert(len(spacing) == 2)
            assert(is_pos_number(spacing[0]) and spacing[0] <= dimension[0])
            assert(is_pos_number(spacing[1]) and spacing[1] <= dimension[1])
            assert(isinstance(acf, ACF))
            assert(len(cd) == 2)
            assert(is_pos_number(cd[0]))
            assert(is_pos_number(cd[1]))
            if seed is not None:
                assert(is_pos_integer(seed))

        n_w = int(dimension[0]/spacing[0])
        n_l = int(dimension[1]/spacing[1])

        ## check if w / d_w and l / d_l give even int
        assert(dimension[0] % spacing[0] == 0 and n_w % 2 == 0)
        assert(dimension[1] % spacing[1] == 0 and n_l % 2 == 0)

        ## ifft needs odd number of points in each direction
        self._r_w = int(n_w / 2)
        self._r_l = int(n_l / 2)

        self._n_w = n_w + 1
        self._n_l = n_l + 1

        self._d_w = spacing[0]
        self._d_l = spacing[1]

        self._acf = acf
        self._a_w = cd[0]
        self._a_l = cd[1]

        self._seed = seed

    def __call__(self):
        """
        """
        r_w = self._r_w
        r_l = self._r_l

        psd = self.psd[:self._r_w+1]

        ## normalize psd for iFFT
        amplitudes = np.sqrt(psd/psd.max())

        ## Set DC-component to 0 (zero-mean field)
        amplitudes[r_w,r_l] = 0

        ## random phases
        if self.seed is not None:
            np.random.seed(self.seed)

        phases = 2 * np.pi * np.random.random(size=amplitudes.shape)

        ## Construct DFT (without negative frequency terms)
        Y = amplitudes * np.cos(phases) + 1j * amplitudes * np.sin(phases)

        ## Include negative frequency terms (complex conjugates of positive ones)
        U = np.zeros((self.n_w, self.n_l), dtype=complex)
        Y = np.concatenate((Y, np.conj(np.fliplr(np.flipud(Y[0:r_w,:])))))

        for i in range(0,r_l):
            Y[r_w,-i+self.n_l-1] = np.conj(Y[r_w,i])

        ## Set ULQ of U with LRQ of Y (including midpoint)
        for i in range(0,r_w+1):
            for j in range(0,r_l+1):
                U[i,j] = Y[i+r_w,j+r_l]

        ## Set LRQ of U with ULQ of Y
        for i in range(r_w+1, self.n_w):
            for j in range(r_l+1, self.n_l):
                U[i,j] = Y[i-r_w-1,j-r_l-1]

        ## Set URQ of U with LLQ of Y
        for i in range(0,r_w+1):
            for j in range(r_l+1, self.n_l):
                U[i,j] = Y[i+r_w,j-r_l-1]

        ## Set LLQ of U with URQ of Y
        for i in range(r_w+1, self.n_w):
            for j in range(0,r_l+1):
                U[i,j] = Y[i-1-r_w,j+r_l]

        ## 2D inverse FFT (remaining imaginary parts are due to machine precision)
        field = np.real(np.fft.ifft2(U))

        ## Remove mean and scale to unit variance
        field = field / np.std(field, ddof=1)

        if np.mean(field) < 0:
            field *= -1 # (small) positive mean

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
        """
        k_w = ((self._r_w/(self.n_w*self.d_w)) *
            np.linspace(-2*np.pi, 2*np.pi, self.n_w))
        return k_w

    @property
    def k_l(self):
        """
        """
        k_l = ((self._r_l/(self.n_l*self.d_l)) *
            np.linspace(-2*np.pi, 2*np.pi, self.n_l))
        return k_l

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
        return self.acf.get_psd(self.k_r, self.a_w*self.a_l)   

    @property
    def seed(self):
        """
        """
        return self._seed
