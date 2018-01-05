"""
The 'spectral' module.
"""


import matplotlib.pyplot as plt
import numpy as np

from .apy import (Object, is_pos_number, is_1d_numeric_array,
    is_1d_complex_array)
from .units import MOTION as UNITS, MOTION_SI as SI_UNITS
from .response import frf
from .plot import set_space


class DFT(Object):
    """
    A discrete Fourier transform (DFT) that is already normalized (i.e.
    amplitudes must be multiplied by time delta).
    """

    def __init__(self, frequencies, amplitudes, unit, validate=True):
        """
        """
        frequencies = np.asarray(frequencies, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=complex)

        if validate is True:
            assert(is_1d_numeric_array(frequencies))
            assert(is_1d_complex_array(amplitudes))
            assert(frequencies.shape == amplitudes.shape)
            assert(unit in UNITS)

        self._frequencies = frequencies
        self._amplitudes = amplitudes
        self._unit = unit

    def __len__(self):
        """
        """
        return len(self._frequencies)

    @property
    def frequencies(self):
        """
        """
        return self._frequencies[:]

    @property
    def amplitudes(self):
        """
        """
        return self._amplitudes[:]

    @property
    def unit(self):
        """
        """
        return self._unit

    @property
    def real(self):
        """
        """
        return self._amplitudes.real[:]

    @property
    def imag(self):
        """
        """
        return self._amplitudes.imag[:]

    @property
    def magnitudes(self):
        """
        """
        magnitudes = np.sqrt(
            (self._amplitudes.real)**2 +
            (self._amplitudes.imag)**2
            )
        return magnitudes

    @property
    def angles(self):
        """
        """
        angles = np.arctan2(
            self._amplitudes.imag,
            self._amplitudes.real,
            )
        return angles

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS).
        """
        amplitudes = self.magnitudes
        fas = FAS(self.frequencies, amplitudes, self._unit, validate=False)
        return fas

    @property
    def fps(self):
        """
        Fourier phase spectrum (FPS).
        """
        amplitudes = self.angles
        fps = FPS(self.frequencies, amplitudes, validate=False)
        return fps

    @property
    def gmt(self):
        """
        Ground motion type (displacement, velocity or acceleration).

        return: string
        """
        return UNITS[self._unit].quantity

    def get_response(self, frequency, damping, gmt, validate=True):
        """
        Get response of SDOF oscillator with frequency response function (FRF).
        """
        amplitudes = self.get_amplitudes(unit='m/s2') * frf(
            self.frequencies, frequency, damping, gmt, validate=validate)
        unit = SI_UNITS[gmt]

        return DFT(self.frequencies, amplitudes, unit)

    def inverse(self, time_delta, validate=True):
        """
        """
        if validate is True:
            is_pos_number(time_delta)

        n = int(round(1 / (self._frequencies[1] * time_delta)))

        if validate is True:
            assert(len(self) == (n//2 + 1))

        amplitudes = np.fft.irfft(self._amplitudes / time_delta, n)

        return amplitudes


class AccDFT(DFT):
    """
    """

    def __init__(self, frequencies, amplitudes, unit, validate=True):
        """
        """
        if validate is True:
            assert(UNITS[unit].quantity == 'acceleration')

        super().__init__(frequencies, amplitudes, unit, validate)

    @classmethod
    def from_dft(cls, dft, validate=True):
        """
        """
        if validate is True:
            assert(type(dft) is DFT)

        acc_dft = cls(
            dft.frequencies,
            dft.amplitudes,
            dft.unit,
            validate=False,
            )

        return acc_dft


class FAS(Object):
    """
    Fourier amplitude spectrum (FAS) that is already normalized (i.e.
    amplitudes must be multiplied by time delta).
    """

    def __init__(self, frequencies, amplitudes, unit, validate=True):
        """
        """
        frequencies = np.asarray(frequencies, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(frequencies))
            assert(is_1d_numeric_array(amplitudes))
            assert(frequencies.shape == amplitudes.shape)
            assert(unit in UNITS)

        self._frequencies = frequencies
        self._amplitudes = amplitudes
        self._unit = unit

    def __len__(self):
        """
        """
        return len(self._frequencies)

    @property
    def frequencies(self):
        """
        """
        return self._frequencies[:]

    @property
    def amplitudes(self):
        """
        """
        return self._amplitudes[:]

    @property
    def unit(self):
        """
        """
        return self._unit

    @property
    def gmt(self):
        """
        Ground motion type (displacement, velocity or acceleration).

        return: string
        """
        return UNITS[self._unit].quantity

    def get_amplitudes(self, unit=None, validate=True):
        """
        """
        if validate is True:
            if unit is not None:
                assert(UNITS[unit].quantity == self.gmt)

        if unit is None or unit == self._unit:
            return self.amplitudes
        else:
            return self.amplitudes * (UNITS[self._unit] / UNITS[unit])

    def plot(self, color=None, style=None, width=None, unit=None, space='loglog', min_frequency=None, max_frequency=None, min_amplitude=None, max_amplitude=None, title=None, size=None, png_filespec=None):
        """
        """
        labels, colors, styles, widths = None, None, None, None
        if color is not None:
            colors = [color]
        if style is not None:
            styles = [style]
        if width is not None:
            widths = [width]

        plot_fass([self], labels, colors, styles, widths, unit, space,
            min_frequency, max_frequency, min_amplitude, max_amplitude, title,
            size, png_filespec)


class FPS(Object):
    """
    A Fourier phase spectrum (FPS).
    """

    def __init__(self, frequencies, amplitudes, validate=True):
        """
        """
        frequencies = np.asarray(frequencies, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(frequencies))
            assert(is_1d_numeric_array(amplitudes))
            assert(frequencies.shape == amplitudes.shape)

        self._frequencies = frequencies
        self._amplitudes = amplitudes

    def __len__(self):
        """
        """
        return len(self._frequencies)

    @property
    def frequencies(self):
        """
        """
        return self._frequencies[:]

    @property
    def amplitudes(self):
        """
        """
        return self._amplitudes[:]


def plot_fass(fass, labels=None, colors=None, styles=None, widths=None, unit=None, space='loglog', min_frequency=None, max_frequency=None, min_amplitude=None, max_amplitude=None, title=None, size=None, png_filespec=None):
    """
    """
    if unit is None:
        unit = fass[0].unit

    fig, ax = plt.subplots(figsize=size)

    for i, fas in enumerate(fass):

        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[i]
        if colors is not None:
            kwargs['c'] = colors[i]
        if styles is not None:
            kwargs['ls'] = styles[i]
        if widths is not None:
            kwargs['lw'] = widths[i]

        ax.plot(fas.frequencies, fas.get_amplitudes(unit), **kwargs)

    ax.grid()

    if labels is not None:
        ax.legend()

    set_space(ax, space)

    ax.set_xlim([min_frequency, max_frequency])
    ax.set_ylim([min_amplitude, max_amplitude])

    x_label = 'Frequency (1/s)'
    y_label = 'Amplitude (%s)' % unit

    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if title is not None:
        ax.set_title(title)

    if png_filespec is not None:
        plt.savefig(png_filespec)
    else:
        plt.show()
