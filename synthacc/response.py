"""
The 'response' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
from scipy.integrate import quad

from .apy import Object, is_pos_number, is_fraction, is_1d_numeric_array
from .units import MOTION as UNITS, MOTION_SI as SI_UNITS
from .spectral import fft, ifft
from .plot import set_space


class ResponseSpectrum(Object):
    """
    """

    def __init__(self, periods, responses, unit, damping=0.05, validate=True):
        """
        """
        periods, responses = np.asarray(periods), np.asarray(responses)

        if validate is True:
            assert(is_1d_numeric_array(periods))
            assert(np.all(periods >= 0) and np.all(np.diff(periods) > 0))
            assert(is_1d_numeric_array(responses))
            assert(np.all(responses >= 0))
            assert(periods.shape == responses.shape)
            assert(unit in UNITS)
            assert(is_fraction(damping))

        self._periods, self._responses = periods, responses
        self._unit = unit
        self._damping = damping

    def __len__(self):
        """
        """
        return len(self.periods)

    @property
    def periods(self):
        """
        """
        return np.copy(self._periods)

    @property
    def responses(self):
        """
        """
        return np.copy(self._responses)

    @property
    def unit(self):
        """
        """
        return self._unit

    @property
    def damping(self):
        """
        """
        return self._damping

    @property
    def pgm(self):
        """
        Get the peak ground motion (PGM) (in unit of response spectrum).
        """
        return self.get_pgm()

    @property
    def gmt(self):
        """
        Ground motion type (displacement, velocity or acceleration).

        return: string
        """
        return UNITS[self.unit].quantity

    @property
    def max_response_period(self):
        """
        """
        return self._periods[self._responses.argmax()]

    def has_pgm(self):
        """
        """
        return 0 in self.periods

    def get_responses(self, unit=None, validate=True):
        """
        """
        if validate is True:
            if unit is not None:
                assert(UNITS[unit].quantity == self.gmt)

        if unit is None or unit == self._unit:
            return self.responses
        else:
            return self.responses * (UNITS[self.unit] / UNITS[unit])

    def get_pgm(self, unit=None):
        """
        Get the peak ground motion (PGM) (in given unit).

        return: pos number
        """
        if self.has_pgm():
            return float(self.get_responses(unit=unit)[0])
        else:
            return None

    def get_max_response(self, unit=None):
        """
        """
        return float(self.get_responses(unit=unit).max())

    def plot(self, color=None, style=None, width=None, unit=None, space='linlog', pgm_period=0.01, min_period=None, max_period=None, max_response=None, title=None, size=None, png_filespec=None):
        """
        """
        labels, colors, styles, widths = None, None, None, None
        if color is not None:
            colors = [color]
        if style is not None:
            styles = [style]
        if width is not None:
            widths = [width]

        plot_response_spectra([self], labels, colors, styles, widths, unit,
            space, pgm_period, min_period, max_period, max_response, title,
            size, png_filespec)


class ResponseCalculator(ABC, Object):
    """
    Calculate response for a single degree of freedom (SDOF) oscillator.
    """

    @abstractmethod
    def __call__(self, time_delta, accelerations, frequencies, damping, gmt, validate):
        """
        Returns relative displacement, relative velocity and/or absolute
        acceleration.
        """
        if validate is True:
            if gmt is not None:
                assert(gmt[:3] in ('dis', 'vel', 'acc'))

    def get_response_spectrum(self, acc, periods, damping=0.05, gmt='acc', pgm_frequency=100, validate=True):
        """
        """
        if validate is True:
            assert(is_1d_numeric_array(periods) and
                np.all(np.diff(periods) > 0))

        if periods[0] == 0:
            periods = np.copy(periods)
            periods[0] = 1 / pgm_frequency

        responses = self.__call__(acc.time_delta, acc.get_amplitudes('m/s2'), 1/periods, damping, gmt, validate=validate)
        responses = np.abs(responses).max(axis=0)
        unit = SI_UNITS[gmt[:3]]
        rs = ResponseSpectrum(periods, responses, unit, damping)

        return rs


class NewmarkBetaRC(ResponseCalculator):
    """
    Calculate response of SDOF in time domain with Newmark-beta method
    (Newmark, 1959).
    """

    def __init__(self, method='lin', validate=True):
        """
        method: 'lin' (linear) or 'avg' (average) acceleration method
        """
        if validate is True:
            assert(method in ('lin', 'avg'))
        
        self._a, self._b = {'lin': (1/2, 1/6), 'avg': (1/2, 1/4)}[method]

    def __call__(self, time_delta, accelerations, frequencies, damping=0.05, gmt=None, validate=True):
        """
        """
        super().__call__(time_delta, accelerations, frequencies, damping, gmt,
            validate)

        ca = self._a * time_delta
        cb = self._b * time_delta**2
        c1 = -ca + time_delta
        c2 = -cb + time_delta**2/2

        w = 2 * np.pi * frequencies
        c = 2 * w * damping
        k = w**2
        d = 1 + c * ca + k * cb

        rdis = np.zeros((len(accelerations), len(frequencies)))
        rvel = np.zeros((len(accelerations), len(frequencies)))
        racc = np.zeros((len(accelerations), len(frequencies)))
        aacc = np.zeros((len(accelerations), len(frequencies)))
        racc[0,:] = accelerations[0]
        aacc[0,:] = accelerations[0]

        for i in range(len(accelerations)-1):
            p1 = rvel[i] + racc[i] * time_delta / 2
            p2 = rdis[i] + rvel[i] * time_delta + c2 * racc[i]
            racc_iadd1 = (-accelerations[i+1] - c * p1 - k * p2) / d
            racc[i+1] = racc_iadd1
            rvel[i+1] = racc[i] * c1 + racc_iadd1 * ca + rvel[i]
            rdis[i+1] = racc[i] * c2 + racc_iadd1 * cb + rvel[i] * \
                time_delta + rdis[i]
            aacc[i+1] = racc_iadd1 + accelerations[i+1]

        if gmt is None:
            return rdis, rvel, aacc
        else:
            return {'dis': rdis, 'vel': rvel, 'acc': aacc}[gmt[:3]]


class NigamJenningsRC(ResponseCalculator):
    """
    Calculate response of SDOF in time domain with method of Nigam & Jennings
    (1969).
    """

    def __call__(self, time_delta, accelerations, frequencies, damping=0.05, gmt=None, validate=True):
        """
        """
        super().__call__(time_delta, accelerations, frequencies, damping, gmt,
            validate)

        omega = 2 * np.pi * frequencies
        omega2 = omega**2
        omega_d = omega * np.sqrt(1 - damping**2)
        f1 = (2 * damping) / (omega**3 * time_delta)
        f2 = 1 / omega2
        f3 = damping * omega
        f4 = 1 / omega_d
        f5 = f3 * f4
        f6 = 2 * f3
        e = np.exp(-f3 * time_delta)
        e_sin = e * np.sin(omega_d * time_delta)
        e_cos = e * np.cos(omega_d * time_delta)
        h_sub = (omega_d * e_cos) - (f3 * e_sin)
        h_add = (omega_d * e_sin) + (f3 * e_cos)

        shape = (len(accelerations)-1, len(frequencies))
        rdis = np.zeros(shape)
        rvel = np.zeros(shape)
        aacc = np.zeros(shape)
        
        for i in range(0, shape[0]):
            diff = accelerations[i+1] - accelerations[i]
            z_1 = f2 * diff
            z_2 = f2 * accelerations[i]
            z_3 = f1 * diff
            z_4 = z_1 / time_delta

            if i == 0:
                rdis_imin1 = 0
                rvel_imin1 = 0
            else:
                rdis_imin1 = rdis[i-1]
                rvel_imin1 = rvel[i-1]

            b_val = rdis_imin1 + z_2 - z_3
            a_val = (f4 * rvel_imin1) + (f5 * b_val) + (f4 * z_4)

            rdis[i] = (a_val * e_sin) + (b_val * e_cos) + z_3 - z_2 - z_1
            rvel[i] = (a_val * h_sub) - (b_val * h_add) - z_4
            aacc[i] = (-f6 * rvel[i]) - (omega2 * rdis[i, :]) ## eq 7

        if gmt is None:
            return rdis, rvel, aacc
        else:
            return {'dis': rdis, 'vel': rvel, 'acc': aacc}[gmt[:3]]


class SpectralRC(ResponseCalculator):
    """
    Calculate response of SDOF in spectral domain.
    """

    def _calc_gmt(self, time_delta, accelerations, frequencies, damping, gmt):
        """
        """
        dft = fft(time_delta, accelerations)

        responses = np.zeros((len(accelerations), len(frequencies)))
        for i, f in enumerate(frequencies):
            response_dft = dft[1] * frf(dft[0], f, damping, gmt, validate=False)
            responses[:,i] = ifft(dft[0], response_dft, time_delta)
            if gmt == 'acc':
                responses[:,i] += accelerations

        return responses

    def __call__(self, time_delta, accelerations, frequencies, damping=0.05, gmt=None, validate=True):
        """
        """
        super().__call__(time_delta, accelerations, frequencies, damping, gmt,
            validate)

        if gmt is None:
            rdis = self._calc_gmt(
                time_delta, accelerations, frequencies, damping, 'dis')
            rvel = self._calc_gmt(
                time_delta, accelerations, frequencies, damping, 'vel')
            aacc = self._calc_gmt(
                time_delta, accelerations, frequencies, damping, 'acc')

            return rdis, rvel, aacc

        else:
            return self._calc_gmt(
                time_delta, accelerations, frequencies, damping, gmt=gmt[:3])


class PeakCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self):
        """
        """


class CartwrightLonguetHiggins1956PC(PeakCalculator):
    """
    Cartwright & Longuet-Higgins (1956).
    """

    def __call__(self, m0, m2, m4, duration):
        """
        """
        fz = np.sqrt(m2/m0)/(2*np.pi)
        fe = np.sqrt(m4/m2)/(2*np.pi)
        nz = 2*fz*duration
        ne = 2*fe*duration
        eta = nz/ne
        pf = np.sqrt(2) * quad(
                lambda z: 1. - (1. - eta * np.exp(-z * z)) ** ne, 0, np.inf)[0]

        return pf


class RVTCalculator(Object):
    """
    Estimate response spectrum from Fourier amplitude spectrum (FAS) with
    Random Vibration Theory (RVT). RVT uses an estimate of the ratio of peak
    motion to rms motion. Parseval’s theorem is used to obtain the rms motion.
    See Boore (2003).
    """

    def __init__(self, pc=CartwrightLonguetHiggins1956PC(), validate=True):
        """
        """
        if validate is True:
            assert(isinstance(pc, PeakCalculator))

        self._pc = pc

    def _calc_spectral_moments(self, orders, frequencies, amplitudes):
        """
        Frequencies must be spaced close enough to get correct result!
        """
        spectral_moments = []
        a_squared = amplitudes**2
        two_pi_f = 2 * np.pi * frequencies
        for o in orders:
            sm = 2 * np.trapz(a_squared * np.power(two_pi_f, o), frequencies)
            spectral_moments.append(sm)
        return spectral_moments

    def __call__(self, fas, frequencies, damping, duration, rms_duration_fnc=None, gmt='dis'):
        """
        Responses are dis.
        """
        responses = []
        for f in frequencies:
            if rms_duration_fnc is None: ## no correction
                rms_duration = duration
            else:
                rms_duration = rms_duration_fnc(f)

            response_amplitudes = fas.amplitudes * np.abs(frf(fas.frequencies, float(f), damping, gmt=gmt))# * (2 * np.pi * f)**2
            m0, m1, m2, m4 = self._calc_spectral_moments([0, 1, 2, 4], fas.frequencies, response_amplitudes)
            y_rms = np.sqrt(m0/rms_duration)
            pf = self._pc(m0, m2, m4, duration)
            response = y_rms * pf
            responses.append(response)

        unit = SI_UNITS[gmt]

        return ResponseSpectrum(1/frequencies, np.array(responses), unit, damping)


def frf(dft_frequencies, sdofo_frequency, damping, gmt, validate=True):
    """
    Frequency response function (FRF) for a single degree of freedom (SDOF)
    oscillator. Acceleration to displacement, velocity or acceleration.
    """
    if validate is True:
        assert(is_pos_number(sdofo_frequency))
        assert(is_fraction(damping))

    dft_frequencies = 2 * np.pi * dft_frequencies
    sdofo_frequency = 2 * np.pi * sdofo_frequency

    dum = {'dis': 1, 'vel': 1.j * dft_frequencies,
        'acc': -1 * (dft_frequencies**2)}[gmt[:3]]

    frf = -1 * dum / (sdofo_frequency**2 - dft_frequencies**2 + 1.j *
        (2 * damping * dft_frequencies * sdofo_frequency))

    return frf


def plot_response_spectra(response_spectra, labels=None, colors=None, styles=None, widths=None, unit=None, space='linlog', pgm_period=0.01, min_period=None, max_period=None, max_response=None, title=None, size=None, png_filespec=None):
    """
    """
    if unit is None:
        unit = response_spectra[0].unit

    fig, ax = plt.subplots(figsize=size)

    max_response_ = 0

    for i, rs in enumerate(response_spectra):

        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[i]
        if colors is not None:
            kwargs['c'] = colors[i]
        if styles is not None:
            kwargs['ls'] = styles[i]
        if widths is not None:
            kwargs['lw'] = widths[i]

        periods, responses = rs.periods, rs.get_responses(unit)

        ax.plot(periods[1:-1], responses[1:-1], zorder=i*2+1, **kwargs)

        if rs.has_pgm():
            ax.scatter(pgm_period, responses[0], s=widths[i]*np.pi**3, c=colors[i], zorder=i*2+2)

        max_response_ = max([max_response_, responses.max()])

    ax.grid()

    if labels is not None:
        ax.legend()

    set_space(ax, space)

    if max_response is None:
        max_response = max_response_

    ax.set_xlim([min_period, max_period])
    ax.set_ylim([0., max_response * 1.1])

    def formatter(x, pos):
        dec_places = int(np.maximum(-np.log10(x), 0))
        formatstring = '{{:.{:1d}f}}'.format(dec_places)
        formatstring = formatstring.format(x)
        return formatstring

    ax.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(formatter))

    x_label = 'Period (s)'
    y_label = '%s (%s)' % (rs.gmt.capitalize(), unit)
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if title is not None:
        ax.set_title(title)

    if png_filespec is not None:
        plt.savefig(png_filespec)
    else:
        plt.show()
    plt.close(fig)
