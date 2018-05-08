"""
The 'response' module.
"""


import matplotlib.pyplot as plt
import numpy as np

from .apy import Object, is_pos_number, is_fraction, is_1d_numeric_array
from .units import MOTION as UNITS
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

    def plot(self, color=None, style=None, width=None, unit=None, space='linlog', pgm_period=0.01, min_period=None, max_period=None, title=None, size=None, png_filespec=None):
        """
        """
        labels, colors, styles, widths = None, None, None, None
        if color is not None:
            colors = [color]
        if style is not None:
            styles = [style]
        if width is not None:
            widths = [width]

        p = plot_response_spectra([self], labels, colors, styles, widths, unit,
            space, pgm_period, min_period, max_period, title, size,
            png_filespec)

        return p


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


def plot_response_spectra(response_spectra, labels=None, colors=None, styles=None, widths=None, unit=None, space='linlog', pgm_period=0.01, min_period=None, max_period=None, title=None, size=None, png_filespec=None):
    """
    """
    if unit is None:
        unit = response_spectra[0].unit

    fig, ax = plt.subplots(figsize=size)

    max_response = 0
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

        max_response = max([max_response, responses.max()])

    ax.grid()

    if labels is not None:
        ax.legend()

    set_space(ax, space)

    ax.set_xlim([min_period, max_period])
    ax.set_ylim([0., max_response * 1.1])

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
