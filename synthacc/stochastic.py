"""
The 'stochastic' module.
"""


import matplotlib.pyplot as plt
import numpy as np

from .apy import Object, is_pos_number
from .units import MOTION_SI as SI_UNITS
from .spectral import FAS


class OmegaSquareSourceModel(Object):
    """
    """

    RD = 1000  ## reference distance (in m)

    def __init__(self, m0, sd, rp, pf, fsf, rho, vel, validate=True):
        """
        """
        self._m0 = m0
        self._sd = sd
        self._rp = rp
        self._pf = pf
        self._fsf = fsf
        self._rho = rho
        self._vel = vel

    def __call__(self, frequencies, gmt='dis', validate=True):
        """
        See Boore (2003) eq. 5.
        """
        if validate is True:
            assert(gmt in ('dis', 'vel', 'acc'))

        amplitudes = self._m0 * self._constant * self._get_shape(frequencies)

        if gmt == 'vel':
            amplitudes *= (2*np.pi*frequencies)**1
        if gmt == 'acc':
            amplitudes *= (2*np.pi*frequencies)**2

        return FAS(frequencies, amplitudes, unit=SI_UNITS[gmt])

    @property
    def m0(self):
        """
        """
        return self._m0

    @property
    def cf(self):
        """
        Corner frequency (in Hz). See Boore (2003) eq. 4 (modified to SI
        units).
        """
        return 4.906e-01 * (self._sd / self._m0)**(1/3) * self._vel

    @property
    def _constant(self):
        """
        See Boore (2003) eq. 7.
        """
        c = float((self._rp * self._pf * self._fsf) /
            (4 * np.pi * self._rho * self._vel**3 * self.RD))
        return c

    def _get_shape(self, frequencies):
        """
        See Boore (2003) eq. 6 and table 2.
        """
        return 1 / (1 + (frequencies / self.cf)**2)


class GeometricalSpreadingModel(Object):
    """
    """

    def __init__(self, segments, validate=True):
        """
        """
        if validate is True:
            assert(segments[0][0] == 1)

        self._segments = segments

    def __call__(self, distances, validate=True):
        """
        """
        distances = np.asarray(distances)

        if distances.ndim == 0:
            distances = distances[np.newaxis]
            is_number = 1
        else:
            is_number = 0

        gs = np.ones(distances.shape, dtype=float)

        for d, s in self._segments:
            i = np.where(distances >= d)
            if i[0].size != 0:
                gs[i] = gs[i][0] * (distances[i]/d)**s

        if is_number:
            gs = float(gs)

        return gs

    def plot(self, distances, max_distance=None, title=None, size=None, filespec=None):
        """
        """
        plot_geometrical_spreading_models([self], distances,
            max_distance=max_distance, title=title, size=size,
            filespec=filespec)


class QModel(Object):
    """
    """

    def __init__(self, fr1, qr1, sr1, ft1, ft2, fr2, qr2, sr2):
        """
        """
        self._fr1 = fr1
        self._qr1 = qr1
        self._sr1 = sr1
        self._ft1 = ft1
        self._ft2 = ft2
        self._qr2 = qr2
        self._fr2 = fr2
        self._sr2 = sr2

    def __call__(self, frequencies):
        """
        """
        qt1 = self._qr1*(self._ft1/self._fr1)**self._sr1
        qt2 = self._qr2*(self._ft2/self._fr2)**self._sr2
        st = np.log10(qt2/qt1) / np.log10(self._ft2/self._ft1)
        qs = qt1 * (frequencies / self._ft1)**st

        i1 = np.where(frequencies <= self._ft1)
        i2 = np.where(frequencies >= self._ft2)
        qs[i1] = self._qr1*(frequencies[i1]/self._fr1)**self._sr1
        qs[i2] = self._qr2*(frequencies[i2]/self._fr2)**self._sr2
        return qs

    def plot(self, frequencies, min_q=None, max_q=None, title=None, size=None, filespec=None):
        """
        """
        plot_q_models([self], frequencies, min_q=min_q, max_q=max_q,
            title=title, size=size, filespec=filespec)


class AttenuationModel(Object):
    """
    """

    def __init__(self, q_model, c_q, kappa=0):
        """
        """
        self._q_model = q_model
        self._c_q = c_q
        self._kappa = kappa

    def __call__(self, distance, frequencies, validate=True):
        """
        """
        if validate is True:
            pass

        q = self._q_model(frequencies)
        akappaq = distance/(self._c_q*q)
        kappa_f = self._kappa

        return np.exp(-np.pi*(kappa_f + akappaq)*frequencies)

    def plot(self, distances, frequencies, title=None, size=None, filespec=None):
        """
        """
        plot_attenuation_models([self], distances, frequencies=frequencies,
            title=title, size=size, filespec=filespec)


class SiteModel(Object):
    """
    """

    def __init__(self, factors):
        """
        """
        self._factors = factors

    def __call__(self, frequencies):
        """
        """
        factors = np.ones(frequencies.shape)
        for i, f in enumerate(frequencies):
            found = False
            if None:
                pass
            elif f <= self._factors[+0][0]:
                factors[i] = self._factors[+0][1]
            elif f >= self._factors[-1][0]:
                factors[i] = self._factors[-1][1]
            else:
                for j in range(len(self._factors)):
                    if (f >= self._factors[:,0][j] and
                        f < self._factors[:,0][j+1]):
                        slope = (np.log10(
                            (self._factors[:,1][j+1]/self._factors[:,1][j])) /
                            (self._factors[:,0][j+1]-self._factors[:,0][j]))
                        factors[i] = self._factors[:,1][j]*10**(
                            slope*(f-self._factors[:,0][j]))
                        found = True
                        break
                assert(found)
        return factors


def get_windowed_noise(time_delta, duration, eta=0.2, nano=0.05, ft=1.3, validate=True):
    """
    See Boore (2003).
    """
    if validate is True:
        assert(is_pos_number(time_delta))
        assert(is_pos_number(duration))

    size = int(duration / time_delta)

    times = time_delta * np.arange(size)

    te = ft * duration
    c1 = -(eta * np.log(nano)) / (1 + (eta * (np.log(eta) - 1)))
    c2 = c1 / eta
    c3 = (np.exp(1) / eta)**c1

    window = c3 * (times/te)**c1 * np.exp(-c2*(times/te))
    windowed_noise = np.random.normal(size=size) * window

    return times, windowed_noise


def plot_geometrical_spreading_models(gsms, distances, labels=None, max_distance=None, title=None, size=None, filespec=None):
    """
    """
    fig, ax = plt.subplots(figsize=size)

    for i, gsm in enumerate(gsms):

        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[i]

        ax.loglog(distances, gsm(distances), **kwargs)

    ax.set_xlim([1, max_distance])
    ax.set_ylim([None, 1])

    ax.grid()

    if labels is not None:
        ax.legend()

    x_label, y_label = 'Distance (km)', 'Geometrical spreading'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if title is not None:
        ax.set_title(title)

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
    plt.close(fig)


def plot_q_models(q_models, frequencies, labels=None, min_q=None, max_q=None, title=None, size=None, filespec=None):
    """
    """
    fig, ax = plt.subplots(figsize=size)

    for i, q_model in enumerate(q_models):

        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[i]

        ax.loglog(frequencies, q_model(frequencies), **kwargs)

    ax.set_ylim([min_q, max_q])

    ax.grid()

    if labels is not None:
        ax.legend()

    x_label, y_label = 'Frequencies (1/s)', 'Q'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if title is not None:
        ax.set_title(title)

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
    plt.close(fig)


def plot_attenuation_models(ams, distances, frequencies, labels=None, title=None, size=None, filespec=None, validate=True):
    """
    """
    if validate is True:
        pass

    fig, ax = plt.subplots(figsize=size)

    for f in frequencies:
        for i, am in enumerate(ams):

            kwargs = {}
            if labels is not None:
                kwargs['label'] = str(labels[i]) + ' (%s Hz)' % f

            ax.plot(distances, [am(float(d), np.array([f])) for d in distances], **kwargs)

    ax.grid()

    if labels is not None:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
    plt.close(fig)
