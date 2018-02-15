"""
The 'recordings' module.

Component convention
--------------------
Z is positive up, N is positive north, E is positive east, R is positive from
source to receiver and T is positive right of R. X and Y are any orthogonal
horizontal components.
"""


import datetime

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime as _UTCDateTime, Trace as _Trace, read as _read

from .apy import (Object, is_boolean, is_number, is_non_neg_number,
    is_pos_number, is_pos_integer, is_1d_numeric_array, is_string)
from .time import Time, is_time
from .data import TimeSeries
from .units import MOTION as UNITS, MOTION_SI as SI_UNITS
from .earth import flat as earth
from .spectral import DFT, AccDFT
from .response import ResponseSpectrum


## Allowed components
_COMPONENTS = set(('Z', 'N', 'E', 'R', 'T', 'X', 'Y'))

_COMPONENTS_SETS = [
    ('X', 'Y', 'Z'),
    ('Z', 'N', 'E'),
    ('Z', 'R', 'T'),
]


class Pick(Object):
    """
    """

    def __init__(self, time, phase, validate=True):
        """
        """
        if not is_time(time):
            time = Time(time, validate=validate)

        if validate is True:
            assert(is_string(phase))

        self._time, self._phase = time, phase

    def __repr__(self):
        """
        """
        return '< Pick | %s | %s >' % (self.time, self.phase)

    @property
    def time(self):
        """
        """
        return self._time

    @property
    def phase(self):
        """
        """
        return self._phase


class Waveform(TimeSeries):
    """
    A waveform (has no unit).
    """

    def __init__(self, time_delta, amplitudes, start_time=0, validate=True):
        """
        """
        super().__init__(time_delta, start_time, validate=validate)

        amplitudes = np.asarray(amplitudes, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(amplitudes))

        header = {'delta': self.time_delta}
        if is_time(start_time):
            header['starttime'] = _UTCDateTime(start_time._time)

        self._trace = _Trace(amplitudes, header=header)

    def __len__(self):
        """
        """
        return len(self._amplitudes)

    @classmethod
    def from_trace(cls, trace, validate=True):
        """
        """
        if validate is True:
            assert(type(trace) is _Trace)

        td, a = float(trace.meta.delta), np.array(trace.data, dtype=float)

        return cls(td, a)

    @property
    def _amplitudes(self):
        """
        """
        return self._trace.data

    @property
    def sampling_rate(self):
        """
        """
        return 1 / self._time_delta

    @property
    def nyquist_frequency(self):
        """
        """
        return self.sampling_rate / 2

    def _slice(self, s_time, e_time, validate=True):
        """
        """
        if validate is True:
            if is_time(self._start_time):
                assert(is_time(s_time))
                assert(is_time(e_time))
            else:
                assert(is_non_neg_number(s_time))
                assert(is_non_neg_number(e_time))

        if is_time(self._start_time):
            s_time = _UTCDateTime(s_time._time)
            e_time = _UTCDateTime(e_time._time)
        else:
            s_time = (
                self._trace.stats.starttime + s_time - self._start_time)
            e_time = (
                self._trace.stats.starttime + e_time - self._start_time)

        return self._trace.copy().slice(s_time, e_time)

    def slice(self, s_time, e_time, validate=True):
        """
        """
        amplitudes = self._slice(s_time, t_time, validate=validate)

        wf = self.__class__(self.time_delta, amplitudes, start_time=s_time)

        return wf

    def _filter(self, f_type, frequency, corners, zero_phase, validate=True):
        """
        frequency: pos number or 2-tuple of pos numbers
        corners: pos integer
        zero_phase: bool
        """
        if validate is True:
            assert(f_type in ('lowpass', 'highpass', 'bandpass'))
            if f_type == 'bandpass':
                assert(type(frequency) is tuple)
                assert(len(frequency) == 2)
                assert(is_pos_number(frequency[0]))
                assert(is_pos_number(frequency[1]))
                assert(frequency[1] > frequency[0])
            else:
                assert(is_pos_number(frequency))
            assert(is_pos_integer(corners))
            assert(type(zero_phase) is bool)

        kwargs = {'corners': corners, 'zerophase': zero_phase}
        if f_type == 'bandpass':
            kwargs['freqmin'] = frequency[0]
            kwargs['freqmax'] = frequency[1]
        else:
            kwargs['freq'] = frequency

        return self._trace.copy().filter(f_type, **kwargs)

    def filter(self, f_type, frequency, corners=4, zero_phase=True, validate=True):
        """
        return: classs instance
        """
        amplitudes = self._filter(f_type, frequency, corners, zero_phase,
            validate=validate)

        wf = self.__class__(self.time_delta, amplitudes, self.start_time, 
            validate=False)

        return wf

    def _pad(self, before=0, after=0, validate=True):
        """
        Add zeros before and after.

        before: non neg number, time before (in s)
        after:  non neg number, time after  (in s)

        return: class instance
        """
        if validate is True:
            assert(is_non_neg_number(before))
            assert(is_non_neg_number(after))

        before = int(before / self.time_delta)
        after =  int(after  / self.time_delta)

        amplitudes = np.pad(self.amplitudes,
            (before, after), mode='constant', constant_values=(0, 0)
            )

        return amplitudes

    def pad(self, before=0, after=0, validate=True):
        """
        """
        amplitudes = self._pad(before, after, validate=validate)

        wf = self.__class__(self.time_delta, amplitudes)

        return wf


class Seismogram(Waveform):
    """
    A waveform with a unit. It is a component of a recording.
    """

    def __init__(self, time_delta, amplitudes, unit, start_time=0, validate=True):
        """
        """
        super().__init__(time_delta, amplitudes, start_time, validate=validate)

        if validate is True:
            assert(unit in UNITS)

        self._unit = unit

    def __neg__(self):
        """
        return: class instance
        """
        return -1 * self

    def __pos__(self):
        """
        return: class instance
        """
        return +1 * self

    def __add__(self, other):
        """
        Add another seismogram to this one. This can for example be used to
        stack synthetic seismograms from subruptures.

        return: new instance of self.__class__
        """
        ## validation of other
        assert(type(other) is self.__class__)
        assert(self.time_delta == other.time_delta)
        assert(len(self) == len(other))
        assert(self.unit == other.unit)
        if self.start_time is not None and other.start_time is not None:
            assert(self.start_time == other.start_time)

        amplitudes = self.amplitudes + other.amplitudes
        s = self.__class__(self._time_delta, amplitudes, self._unit,
            self.start_time or other.start_time)

        return s

    def __mul__(self, other):
        """
        return: class instance
        """
        ## validation of other
        assert(is_number(other))

        amplitudes = self._amplitudes * other
        s = self.__class__(self._time_delta, amplitudes, self._unit,
            self.start_time)

        return s

    @classmethod
    def from_trace(cls, trace, unit, validate=True):
        """
        """
        if validate is True:
            assert(type(trace) is _Trace and unit in UNITS)

        td, a = float(trace.meta.delta), np.array(trace.data, dtype=float)

        return cls(td, a, unit)

    @property
    def amplitudes(self):
        """
        return: 1d numeric array
        """
        return np.copy(self._amplitudes)

    @property
    def unit(self):
        """
        return: string
        """
        return self._unit

    @property
    def pgm(self):
        """
        Get the peak ground motion (PGM) (in unit of seismogram).
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
    def dft(self):
        """
        Discrete Fourier transform (DFT).
        """
        return self.get_dft()

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS).
        """
        return self.get_fas()

    def get_amplitudes(self, unit=None, validate=True):
        """
        Get amplitudes in given unit.

        return: 1d numeric array
        """
        if validate is True:
            if unit is not None:
                assert(UNITS[unit].quantity == self.gmt)

        if unit is None or unit == self._unit:
            return self.amplitudes
        else:
            return self.amplitudes * (UNITS[self.unit] / UNITS[unit])

    def get_abs_amplitudes(self, unit=None):
        """
        Get the abbsolute amplitudes (in given unit).

        return: 1d numeric array
        """
        return np.abs(self.get_amplitudes(unit=unit))

    def get_pgm(self, unit=None):
        """
        Get the peak ground motion (PGM) (in given unit).

        return: pos number
        """
        return float(np.max(self.get_abs_amplitudes(unit=unit)))

    def get_dft(self, unit=None, validate=True):
        """
        Discrete Fourier transform (DFT).
        """
        if validate is True:
            if unit is not None:
                assert(UNITS[unit].quantity == self.gmt)

        frequencies = np.fft.rfftfreq(len(self), self._time_delta)
        amplitudes = np.fft.rfft(self.get_amplitudes(unit))
        amplitudes *= self._time_delta

        return DFT(frequencies, amplitudes, unit or self._unit)

    def get_fas(self, unit=None, validate=True):
        """
        Fourier amplitude spectrum (FAS).
        """
        return self.get_dft(unit=unit, validate=validate).fas

    def slice(self, s_time, e_time, validate=True):
        """
        """
        amplitudes = self._slice(s_time, e_time, validate=validate)

        s = self.__class__(self.time_delta, amplitudes, self.unit,
            start_time=s_time, validate=False)

        return s

    def filter(self, f_type, frequency, corners=4, zero_phase=True, validate=True):
        """
        """
        amplitudes = self._filter(f_type, frequency, corners, zero_phase,
            validate=validate)

        s = self.__class__(self.time_delta, amplitudes, self.unit,
            self.start_time, validate=False)

        return s

    def pad(self, before=0, after=0, validate=True):
        """
        """
        amplitudes = self._pad(before, after, validate=validate)

        s = self.__class__(self.time_delta, amplitudes, unit=self.unit)

        return s

    def differentiate(self, validate=True):
        """
        """
        if self.unit == 'm':
            unit = 'm/s'
        elif self.unit == 'm/s':
            unit = 'm/s2'
        else:
            raise

        s = self.__class__(self.time_delta, self._trace.copy().differentiate(),
            unit, self.start_time, validate=False)

        return s

    def integrate(self, validate=True):
        """
        """
        if self.unit == 'm/s2':
            unit = 'm/s'
        elif self.unit == 'm/s':
            unit = 'm'
        else:
            raise

        s = self.__class__(self.time_delta, self._trace.copy().integrate(),
            unit, self.start_time, validate=False)

        return s

    def plot(self, color=None, style=None, width=None, unit=None, duration=None, picks=[], size=None, png_filespec=None, validate=True):
        """
        """
        colors, styles, widths = None, None, None
        if color is not None:
            colors = [[color]]
        if style is not None:
            styles = [[style]]
        if width is not None:
            widths = [[width]]

        p = plot_seismograms([[self]], colors=colors, styles=styles,
            widths=widths, unit=unit, duration=duration, picks=picks, 
            size=size, png_filespec=png_filespec, validate=validate)

        return p


class Accelerogram(Seismogram):
    """
    """

    def __init__(self, time_delta, amplitudes, unit, start_time=0, validate=True):
        """
        """
        if validate is True:
            assert(UNITS[unit].quantity == 'acceleration')

        super().__init__(
            time_delta, amplitudes, unit, start_time, validate)

    @classmethod
    def from_seismogram(cls, seismogram, validate=True):
        """
        """
        if validate is True:
            assert(type(seismogram) is Seismogram)

        accelerogram = cls(
            seismogram.time_delta,
            seismogram.amplitudes,
            seismogram.unit,
            seismogram.start_time,
            validate=False,
            )

        return accelerogram

    @property
    def pga(self):
        """
        Get the peak ground acceleration (PGA).
        """
        return self.get_pga()

    def get_pga(self, unit=None):
        """
        Get the peak ground acceleration (PGA).
        """
        return self.get_pgm(unit)

    def get_dft(self, unit=None, validate=True):
        """
        Discrete Fourier transform (DFT).
        """
        if validate is True:
            if unit is not None:
                assert(UNITS[unit].quantity == self.gmt)

        dft = super().get_dft(unit, validate)
        dft = AccDFT.from_dft(dft)

        return dft

    def get_responses(self, periods, damping=0.05, gmt='acc', pgm_frequency=100, validate=True):
        """
        """
        if validate is True:
            assert(is_1d_numeric_array(periods))
            assert(is_pos_number(pgm_frequency))

        if periods[0] == 0:
            periods = np.copy(periods)
            periods[0] = 1 / pgm_frequency

        dft = self.dft
        frequencies = 1 / periods
        responses = np.zeros((len(frequencies), len(self)))
        gmt = gmt[:3]
        for i, frequency in enumerate(frequencies):
            response_dft = dft.get_response(float(frequency), damping, gmt)
            responses[i] = response_dft.inverse(self._time_delta)
            if gmt == 'acc':
                responses[i] += self.get_amplitudes('m/s2')

        return responses

    def get_response_spectrum(self, periods, damping=0.05, gmt='acc', pgm_frequency=100, validate=True):
        """
        Response spectrum for SDOF oscillator. Responses are calculated with
        frequency response function (FRF). Calculates relative displacement,
        relative velocity or absolute acceleration. No pseudo response spectra!
        Units are m, m/s or m/s2.
        """
        responses = self.get_responses(periods, damping, gmt, pgm_frequency, validate)
        max_abs_responses = np.abs(responses).max(axis=1)
        unit = SI_UNITS[gmt[:3]]
        rs = ResponseSpectrum(periods, max_abs_responses, unit, damping)

        return rs


class Recording(Object):
    """
    It is checked if all components have same time delta, duration, unit and
        start time.
    """

    def __init__(self, components, validate=True):
        """
        """
        if validate is True:
            assert(type(components) is dict)

        self._set(components, validate=validate)

    @property
    def components(self):
        """
        Components of the recording.

        NOTE: A set is returned (and not a tuple) so one can easily test if two
            recordings have the same components.

        return: frozenset
        """
        return frozenset(self._components)

    @property
    def start_time(self):
        """
        """
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        """
        """
        for c in self.components:
            self._components[c].start_time = start_time
        self._start_time = start_time

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def duration(self):
        """
        """
        return self._duration

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

    @property
    def pgm(self):
        """
        Peak ground motion (PGM).
        """
        return max([self._components[c].pgm for c in self._components])

    def _set(self, components, validate=True):
        """
        """
        time_deltas = set()
        start_times = set()
        units = set()
        durations = set()

        for c, s in components.items():
            if validate is True:
                assert(c in _COMPONENTS)
                assert(type(s) is Seismogram)

            time_deltas.add(s.time_delta)
            start_times.add(s.start_time)
            units.add(s.unit)
            durations.add(s.duration)

        if validate is True:
            assert(len(time_deltas) == 1)
            assert(len(start_times) == 1)
            assert(len(units) == 1)
            assert(len(durations) == 1)

        self._components = components
        self._time_delta = time_deltas.pop()
        self._start_time = start_times.pop()
        self._unit = units.pop()
        self._duration = durations.pop()

    def get_component(self, component, validate=True):
        """
        Get a component of the recording.

        component: string

        return: 'recordings.Seismogram' instance
        """
        if validate is True:
            assert(component in self.components)

        return self._components[component]

    def rotate(self, back_azimuth, validate=True):
        """
        Rotate between ZRT and ZNE.
        """
        components = {}
        component_set = self.components
        zne = set(('Z', 'N', 'E'))
        zrt = set(('Z', 'R', 'T'))
        if component_set not in (zne, zrt):
            raise NotImplementedError

        elif component_set == zne:
            r, t = ne_to_rt(self.get_component('N').amplitudes,
                            self.get_component('E').amplitudes, back_azimuth,
                            validate=validate)
            components['R'] = Seismogram(
                self._time_delta, r, self._unit, self.start_time)
            components['T'] = Seismogram(
                self._time_delta, t, self._unit, self.start_time)

        elif component_set == zrt:
            n, e = rt_to_ne(self.get_component('R').amplitudes,
                            self.get_component('T').amplitudes, back_azimuth,
                            validate=validate)
            components['N'] = Seismogram(self._time_delta, n, self._unit,
                self.start_time)
            components['E'] = Seismogram(self._time_delta, e, self._unit,
                self.start_time)

        Z = self.get_component('Z').amplitudes
        components['Z'] = Seismogram(self._time_delta, Z, self._unit,
            self.start_time)

        return self.__class__(components)

    def slice(self, s_time, e_time, validate=True):
        """
        """
        components = {}
        for c, s in self._components.items():
            components[c] = s.slice(s_time, e_time, validate=validate)
        return self.__class__(components, validate=False)

    def filter(self, f_type, frequency, corners=4, zero_phase=True, validate=True):
        """
        """
        components = {}
        for c, s in self._components.items():
            components[c] = s.filter(f_type, frequency, corners, zero_phase,
            validate=validate)
        return self.__class__(components, validate=False)

    def pad(self, before=0, after=0, validate=True):
        """
        """
        components = {}
        for c, s in self._components.items():
            components[c] = s.pad(before, after, validate=validate)
        return self.__class__(components, validate=False)

    def differentiate(self, validate=True):
        """
        """
        components = {}
        for c, s in self._components.items():
            components[c] = s.differentiate(validate=validate)
        return self.__class__(components, validate=False)

    def integrate(self, validate=True):
        """
        """
        components = {}
        for c, s in self._components.items():
            components[c] = s.integrate(validate=validate)
        return self.__class__(components, validate=False)

    def convolve(self, a, validate=True):
        """
        """
        if validate is True:
            assert(is_1d_numeric_array(a))

        components = {}
        for c, s in self._components.items():
            components[c] = s.convolve(a, validate=False)
        return self.__class__(components, validate=False)

    def plot(self, duration=None, picks=[], size=None, png_filespec=None, validate=True):
        """
        """
        p = plot_recordings([self], duration=duration, picks=picks, size=size,
            png_filespec=png_filespec, validate=validate)

        return p


def ne_to_rt(n, e, back_azimuth, validate=True):
    """
    Rotate north and east components to radial and transversal components. See
    Havskov & Otttemöller (2010) p. 95.

    See the docstring of the module for the definition of the components.

    n: 1d numeric array
    e: 1d numeric array
    back_azimuth: azimuth from receiver to source

    return: r and t components
    """
    if validate is True:
        assert(is_1d_numeric_array(n))
        assert(is_1d_numeric_array(e))
        assert(len(n) == len(e))
        assert(earth.is_azimuth(back_azimuth))

    ba = np.radians(back_azimuth)
    r = - n * np.cos(ba) - e * np.sin(ba)
    t = + n * np.sin(ba) - e * np.cos(ba)

    return r, t


def rt_to_ne(r, t, back_azimuth, validate=True):
    """
    Rotate radial and transversal components to north and east components. See
    Havskov & Otttemöller (2010) p. 95.

    See the docstring of the module for the definition of the components.

    r: 1d numeric array
    t: 1d numeric array
    back_azimuth: azimuth from receiver to source

    return: n and e components
    """
    if validate is True:
        assert(is_1d_numeric_array(r))
        assert(is_1d_numeric_array(t))
        assert(len(r) == len(t))
        assert(earth.is_azimuth(back_azimuth))

    ba = np.radians(back_azimuth)
    n = - r * np.cos(ba) + t * np.sin(ba)
    e = - r * np.sin(ba) - t * np.cos(ba)

    return n, e


def plot_seismograms(seismograms, titles=None, labels=None, colors=None, styles=None, widths=None, unit=None, duration=None, scale=True, picks=[], title=None, size=None, png_filespec=None, validate=True):
    """
    seismograms: list of lists of 'recordings.Seismogram' instances
    titles: list of strings (default: None)
    labels: list of lists of strings (default: None)
    colors: list of lists of line colors (default: None)
    styles: list of lists of line styles (default: None)
    widths: list of lists of line widths (default: None)
    scale: boolean, all axes have same y scale (default: True)
    """
    if validate is True:
        assert(type(seismograms) is list)

    n = len(seismograms)

    if validate is True:
        if titles is not None:
            assert(type(titles) is list and len(titles) == n)
        if labels is not None:
            assert(type(labels) is list and len(labels) == n)
        if colors is not None:
            assert(type(colors) is list and len(colors) == n)
        if styles is not None:
            assert(type(styles) is list and len(styles) == n)
        if widths is not None:
            assert(type(widths) is list and len(widths) == n)
        assert(is_boolean(scale))

    fig = plt.figure(figsize=size)

    ## set background for y label
    bg = plt.subplot2grid((n, 1), (0, 0), rowspan=n)
    for s in ('bottom', 'top', 'left', 'right'):
        bg.spines[s].set_alpha(0)
    for tick in bg.get_xticklines():
        tick.set_alpha(0)
    for tick in bg.get_yticklines():
        tick.set_alpha(0)
    for label in bg.get_xticklabels():
        label.set_alpha(0)
    for label in bg.get_yticklabels():
        label.set_alpha(0)
    
    sharex = None
    sharey = None
    pgms = np.zeros(n)
    axes = []

    for i in range(n):

        if validate is True:
            assert(type(seismograms[i]) is list)
            if titles is not None:
                assert(is_string(titles[i]))
            if labels is not None:
                assert(type(labels[i]) is list and
                    len(labels[i]) == len(seismograms[i]))
            if colors is not None:
                assert(type(colors[i]) is list and
                    len(colors[i]) == len(seismograms[i]))
            if styles is not None:
                assert(type(styles[i]) is list and
                    len(styles[i]) == len(seismograms[i]))
            if widths is not None:
                assert(type(widths[i]) is list and
                    len(widths[i]) == len(seismograms[i]))

        ax = fig.add_subplot(n, 1, i+1, sharex=sharex, sharey=sharey)
        axes.append(ax)

        if sharex is None:
            sharex = ax
        if sharey is None:
            sharey = ax

        for j, s in enumerate(seismograms[i]):

            if validate is True:
                assert(type(s) is Seismogram)

            if unit is None:
                unit = s.unit
                gmt = UNITS[unit].quantity

            if validate is True:
                assert(s.gmt == gmt)

            kwargs = {}
            if labels is not None:
                kwargs['label'] = labels[i][j]
            if colors is not None:
                kwargs['c'] = colors[i][j]
            if styles is not None:
                kwargs['ls'] = styles[i][j]
            if widths is not None:
                kwargs['lw'] = widths[i][j]

            if duration is not None:
                s = s.slice(s.start_time, s.start_time+duration)

            if is_time(s.start_time):
                times = [s.start_time._time + datetime.timedelta(
                    microseconds=t*10**6) for t in s.rel_times]
            else:
                times = [s.start_time + t for t in s.rel_times]

            ax.plot(times, s.get_amplitudes(unit), **kwargs)

            pgms[i] = np.max([pgms[i], s.get_pgm(unit)])

            for p in picks:
                if type(p) is Pick:
                    time = p.time._time
                else:
                    time = p
                ax.axvline(x=time, color='r', ls='--')

        if titles is not None:
            ax.text(0.990, 0.925, s=titles[i], horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                color='dimgrey', fontsize=12, fontweight='bold')

        ax.grid()

        if labels is not None and titles is None:
            ax.legend()

        if scale is False:
            y_min = -pgms[i] * 1.1
            y_max = +pgms[i] * 1.1
            ax.set_ylim([y_min, y_max])

    if titles is not None and labels is not None:
        plt.legend(ax.get_legend_handles_labels())

    pgm = pgms.max()
    y_min = -pgm * 1.1
    y_max = +pgm * 1.1

    if scale is True:
        sharey.set_ylim([y_min, y_max])

    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].xaxis.set_label_text('Time (s)')

    bg.set_ylim([y_min, y_max])
    bg.set_ylabel('%s (%s)' % (gmt.capitalize(), unit))

    if title is not None:
        plt.suptitle(title)

    if png_filespec is not None:
        plt.savefig(png_filespec)
    else:
        plt.show()


def plot_recordings(recordings, labels=None, colors=None, styles=None, widths=None, unit=None, duration=None, picks=[], title=None, size=None, png_filespec=None, validate=True):
    """
    recordings: list of 'recordings.Recording' instances
    colors: list of line colors (default: None)
    styles: list of line styles (default: None)
    widths: list of line widths (default: None)
    """
    if validate is True:
        assert(type(recordings) is list)
        if labels is not None:
            assert(type(labels) is list and len(labels) == len(recordings))
        if colors is not None:
            assert(type(colors) is list and len(colors) == len(recordings))
        if styles is not None:
            assert(type(styles) is list and len(styles) == len(recordings))
        if widths is not None:
            assert(type(widths) is list and len(widths) == len(recordings))

    components_sets = set()
    for r in recordings:
        if validate is True:
            assert(type(r) is Recording)
        components_sets.add(r.components)
    
    if validate is True:
        assert(len(components_sets) == 1)

    components = components_sets.pop()

    for components_set in _COMPONENTS_SETS:
        if components == set(components_set):
            components = components_set
            break

    seismograms = []
    for c in components:
        seismograms.append([r.get_component(c) for r in recordings])

    titles = components

    if labels is not None:
        labels = [labels] * len(components)
    if colors is not None:
        colors = [colors] * len(components)
    if styles is not None:
        styles = [styles] * len(components)
    if widths is not None:
        widths = [widths] * len(components)

    p = plot_seismograms(seismograms, titles, labels, colors, styles, widths,
        unit=unit, duration=duration, picks=picks, title=title, size=size,
        png_filespec=png_filespec, validate=False)

    return p


def read(filespec, unit, validate=True):
    """
    """
    if validate is True:
        assert(is_string(filespec) and filespec.endswith('.sac'))

    [t] = _read(filespec)

    return Seismogram.from_trace(t, unit, validate)
