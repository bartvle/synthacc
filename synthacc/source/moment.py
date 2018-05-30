"""
The 'source.moment' module.
"""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from ..apy import (PRECISION, Object, is_number, is_pos_number,
    is_1d_numeric_array)
from ..math import SquareMatrix


class MomentTensor(Object):
    """
    A general moment tensor following the NED convection (x is north, y is east
    and z is down). Each of the nine Mij elements represents a pair of opposing
    forces, pointing in the i direction and separated in the j direction (force
    couple). The force at positive j points to positive i. To conserve angular
    moment Mij = Mji, leaving only six indepent elements.
    """

    COMPONENTS = ('xx', 'yy', 'zz', 'xy', 'yz', 'zx')

    def __init__(self, xx, yy, zz, xy, yz, zx, validate=True):
        """
        xx, yy, zz, xy, yz and zx: number
        """
        self._m = SquareMatrix(
            [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]], validate=validate)

    def __repr__(self):
        """
        """
        s = '< Moment Tensor |'
        for c in self.six:
            s += ' {}'.format(c)
        s += ' >'

        return s

    def __getitem__(self, item):
        """
        NOTE: Indices start at 1!
        """
        e = self._m[item]

        if e.is_integer():
            return int(e)
        elif abs(e) < 10**-PRECISION:
            return 0
        else:
            return e

    def __add__(self, other):
        """
        return: new instance
        """
        assert(type(other) is MomentTensor)

        xx = self.xx + other.xx
        yy = self.yy + other.yy
        zz = self.zz + other.zz
        xy = self.xy + other.xy
        yz = self.yz + other.yz
        zx = self.zx + other.zx

        mt = self.__class__(xx, yy, zz, xy, yz, zx)

        return mt

    def __mul__(self, other):
        """
        return: new instance
        """
        assert(is_number(other))

        xx = self.xx * other
        yy = self.yy * other
        zz = self.zz * other
        xy = self.xy * other
        yz = self.yz * other
        zx = self.zx * other

        mt = self.__class__(xx, yy, zz, xy, yz, zx)

        return mt

    @property
    def xx(self):
        """
        return: number
        """
        return self[1,1]

    @property
    def yy(self):
        """
        return: number
        """
        return self[2,2]

    @property
    def zz(self):
        """
        return: number
        """
        return self[3,3]

    @property
    def xy(self):
        """
        return: number
        """
        return self[1,2]

    @property
    def yz(self):
        """
        return: number
        """
        return self[2,3]

    @property
    def zx(self):
        """
        return: number
        """
        return self[3,1]

    @property
    def six(self):
        """
        """
        return self.get_six()

    @property
    def trace(self):
        """
        return: number
        """
        return self._m.trace

    @property
    def moment(self):
        """
        Seismic moment (in Nm). See Shearer (2009) p. 247.
        """
        m0 = float(np.sqrt(np.sum(e**2 for e in self._m)) / np.sqrt(2))

        if abs(m0-round(m0)) < 10**-PRECISION:
            m0 = int(round(m0))

        return m0

    @property
    def magnitude(self):
        """
        Moment magnitude from seismic moment.
        """
        return m0_to_mw(self.moment)

    def get_six(self, convention='NED', validate=True):
        """
        The six independent components of the moment tensor in NED or USE.

        NED: (xx, yy, zz, xy, yz and zx) or (nn, ee, dd, ne, ed and dn)
        USE: (rr, tt, pp, rt, tp and pr) or (uu, ss, ee, us, se and eu)

        See Aki and Richards (2002) p. 112. for the NED-USE conversion.
        """
        if validate is True:
            assert(convention in ('NED', 'USE'))

        if convention == 'NED':
            return (+self.xx, +self.yy, +self.zz, +self.xy, +self.yz, +self.zx)
        else:
            return (+self.zz, +self.xx, +self.yy, +self.zx, -self.xy, -self.yz)

    def has_isotropic_part(self):
        """
        return: boolean
        """
        return self.trace != 0


class _TimeSeries(ABC, Object):
    """
    """

    def __init__(self, time_delta, start_time=0, validate=True):
        """
        time_delta: pos number (in s)
        start_time: number (in s) (default: 0)
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_number(start_time))

        self._time_delta = time_delta
        self._start_time = start_time

    @abstractmethod
    def __len__(self):
        """
        """
        pass

    @property
    def time_delta(self):
        """
        return: pos number (in s)
        """
        return self._time_delta

    @property
    def start_time(self):
        """
        return: number (in s)
        """
        return self._start_time

    @property
    def times(self):
        """
        """
        return self.time_delta * np.arange(len(self))


class _Function(_TimeSeries):
    """
    """

    def __init__(self, time_delta, values, start_time=0, validate=True):
        """
        time_delta: (in s)
        start_time: (in s)
        """
        super().__init__(time_delta, start_time, validate=validate)

        values = np.asarray(values, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(values))
            assert(values[0] == 0)
            assert(np.all(np.diff(values) >= 0))

        self._values = values

    def __len__(self):
        """
        """
        return len(self._values)


class SlipFunction(_Function):
    """
    Total slip in function of time.
    """

    def __init__(self, time_delta, slips, start_time=0, validate=True):
        """
        time_delta: (in s)
        slips: (in m)
        start_time: (in s)
        """
        super().__init__(time_delta, slips, start_time, validate=validate)

    @property
    def slips(self):
        """
        """
        return np.copy(self._values)

    @property
    def slip(self):
        """
        """
        return self._values[-1]

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.slips, c='dimgrey', lw=2)
        ax.fill_between(self.times, self.slips, color='lightgrey')

        if model is not None:
            ax.plot(model.times, model.slips, c='red', lw=2)

        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Slip (m)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Slip function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class MomentFunction(_Function):
    """
    Released moment in function of time.
    """

    def __init__(self, time_delta, moments, start_time=0, validate=True):
        """
        time_delta: (in s)
        moments: (in Nm)
        start_time: (in s)
        """
        super().__init__(time_delta, moments, start_time, validate=validate)

    @property
    def moments(self):
        """
        """
        return np.copy(self._values)

    @property
    def moment(self):
        """
        """
        return self._values[-1]

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.moments, c='dimgrey', lw=2)
        ax.fill_between(self.times, self.moments, color='lightgrey')

        if model is not None:
            ax.plot(model.times, model.moments, c='red', lw=2)

        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Moment (Nm)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Moment function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class _NormalizedFunction(_Function):
    """
    """

    def __init__(self, time_delta, values, start_time=0, validate=True):
        """
        """
        super().__init__(time_delta, values, start_time, validate=validate)

        if validate is True:
            assert(np.abs(values[-1] - 1) <= 10**-PRECISION)


class NormalizedSlipFunction(_NormalizedFunction):
    """
    Normalized total slip in function of time.
    """

    def __mul__(self, slip):
        """
        return: 'moment.SlipFunction' instance
        """
        assert(is_pos_number(slip))

        sf = SlipFunction(
            self.time_delta,
            self._values * slip,
            self.start_time,
            )

        return sf

    @property
    def slips(self):
        """
        """
        return np.copy(self._values)

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.slips, c='dimgrey', lw=2)
        ax.fill_between(self.times, self.slips, color='lightgrey')

        if model is not None:
            ax.plot(model.times, model.slips, c='red', lw=2)

        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Normalized slip'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Normalized slip function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class NormalizedMomentFunction(_NormalizedFunction):
    """
    Normalized released moment in function of time.
    """

    def __mul__(self, moment):
        """
        return: 'moment.MomentFunction' instance
        """
        assert(is_pos_number(moment))

        mf = MomentFunction(
            self.time_delta,
            self._values * moment,
            self.start_time,
            )

        return mf

    @property
    def moments(self):
        """
        """
        return np.copy(self._values)

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.moments, c='dimgrey', lw=2)
        ax.fill_between(self.times, self.moments, color='lightgrey')

        if model is not None:
            ax.plot(model.times, model.moments, c='red', lw=2)

        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Normalized moment'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Normalized moment function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class _RateFunction(_TimeSeries):
    """
    """

    def __init__(self, time_delta, rates, start_time=0, validate=True):
        """
        """
        super().__init__(time_delta, start_time, validate=validate)

        rates = np.asarray(rates, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(rates))
            assert(np.all(rates >= 0))
            assert(rates[+0] == 0)
            assert(rates[-1] == 0)

        self._rates = rates

    def __len__(self):
        """
        """
        return len(self.rates)

    @property
    def rates(self):
        """
        return: 1d numerical array
        """
        return np.copy(self._rates)


class SlipRateFunction(_RateFunction):
    """
    A slip rate function (or slip velocity function (SVF)).
    """

    @property
    def slip(self):
        """
        return: pos number
        """
        return float(np.sum(self._rates) * self._time_delta)

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.rates, c='dimgrey', lw=2)
        ax.fill(self.times, self.rates, c='lightgrey')

        if model is not None:
            ax.plot(model.times, model.rates, c='red', lw=2)

        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Slip rate (m/s)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Slip rate function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class MomentRateFunction(_RateFunction):
    """
    A moment rate function (or source time function (STF)).
    """

    @property
    def moment(self):
        """
        return: pos number
        """
        return float(np.sum(self._rates) * self._time_delta)

    @property
    def magnitude(self):
        """
        return: number
        """
        return m0_to_mw(self.moment)

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.rates, c='dimgrey', lw=2)
        ax.fill(self.times, self.rates, c='lightgrey')

        if model is not None:
            ax.plot(model.times, model.rates, c='red', lw=2)

        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Moment rate (Nm/s)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Moment rate function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class _NormalizedRateFunction(_RateFunction):
    """
    """

    def __init__(self, time_delta, rates, start_time=0, validate=True):
        """
        time_delta: (in s)
        rates: (in 1/s)
        start_time: (in s)
        """
        super().__init__(time_delta, rates, start_time, validate=validate)

        if validate is True:
            assert(np.abs((np.sum(rates) * self.time_delta) - 1)
            <= 10**-PRECISION)


class NormalizedSlipRateFunction(_NormalizedRateFunction):
    """
    A normalized slip rate function (normalized slip velocity function (SVF)).
    """

    def __mul__(self, slip):
        """
        return: 'moment.SlipRateFunction' instance
        """
        assert(is_pos_number(slip))

        srf = SlipRateFunction(
            self.time_delta,
            self._rates * slip,
            self.start_time,
            )

        return srf

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.rates, c='dimgrey', lw=2)
        ax.fill(self.times, self.rates, c='lightgrey')

        if model is not None:
            ax.plot(model.times, model.rates, c='red', lw=2)

        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Normalized slip rate (1/s)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Normalized slip rate function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class NormalizedMomentRateFunction(_NormalizedRateFunction):
    """
    A normalized moment rate function (normalized source time function (STF)).
    """

    def __mul__(self, moment):
        """
        return: 'moment.MomentRateFunction' instance
        """
        assert(is_pos_number(moment))

        mrf = MomentRateFunction(
            self.time_delta,
            self._rates * moment,
            self.start_time,
            )

        return mrf

    def plot(self, model=None, title=None, size=None, png_filespec=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.rates, c='dimgrey', lw=2)
        ax.fill(self.times, self.rates, c='lightgrey')

        if model is not None:
            ax.plot(model.times, model.rates, c='red', lw=2)

        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Normalized moment rate (1/s)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Normalized moment rate function')

        plt.grid()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()


class InstantRateGenerator(Object):
    """
    """

    def get_nmrf(self, time_delta=10**-PRECISION, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(time_delta))

        rates = [0, 1 / time_delta, 0]

        return NormalizedMomentRateFunction(time_delta, rates)


class ConstantRateGenerator(Object):
    """
    """

    def get_nmrf(self, time_delta, rise_time, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_pos_number(rise_time))

        n = int(round(rise_time / time_delta))
        rates = np.zeros(n+3)
        rates[1:-1] = 1 / ((n+1)*time_delta)

        return NormalizedMomentRateFunction(time_delta, rates)


class TriangularRateGenerator(Object):
    """
    Gives a (normalized) moment rate function with the shape of an isosceles
    triangle. The height of the triangle (i.e. the maximum rate) is equal to
    the inverse of half its base (i.e. the half duration).
    """

    def get_nmrf(self, time_delta, half_duration, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_pos_number(half_duration))

        n = int(round(half_duration / time_delta))
        assert(abs((n * time_delta) - half_duration) <= 10**-PRECISION)

        rates = np.zeros(2*n+1)
        rates[:n+1] = np.linspace(0, 1 / half_duration, n+1)
        rates[n+1:] = rates[:n][::-1]

        nmrf = NormalizedMomentRateFunction(time_delta, rates)

        return nmrf


def calculate(area, slip, rigidity, validate=True):
    """
    area: (in m²)
    slip: (in m)
    rigidity: (in Pa)

    return: seismic moment (in Nm)
    """
    if validate is True:
        assert(is_pos_number(area))
        assert(is_pos_number(slip))
        assert(is_pos_number(rigidity))

    return area * slip * rigidity


def m0_to_mw(m0, precise=True, validate=True):
    """
    Convert seismic moment in J (Nm) to moment magnitude. 1 dyne-cm = 10**-7
    Nm. The factor 10.7 can also be written as (2/3) * 16.1, which is more
    precise. Both are used. The difference in moment magnitude can be 0.1. The
    Global CMT Catalog for example uses (2/3) * 16.1 from February 1, 2006. See
    the "Note on calculation of moment magnitude" section on
    http://www.globalcmt.org/CMTsearch.html (last accessed on 10/10/2017). The
    rounding occurs in Hanks and Kanamori (1979).

    precise: bool, use precise factor or not (default: True)

    return: number, mw
    """
    if validate is True:
        assert(is_pos_number(m0))

    if precise is True:
        factor = (2/3) * 16.1
    else:
        factor = 10.7

    return float((2/3) * np.log10(float(m0)*10**7) - factor)


def mw_to_m0(mw, precise=True, validate=True):
    """
    Convert moment magnitude to seismic moment in J (Nm). 1 dyne-cm = 10**-7
    Nm. The factor 10.7 can also be written as (2/3) * 16.1, which is more
    precise. Both are used. The difference in moment magnitude can be 0.1. The
    Global CMT Catalog for example uses (2/3) * 16.1 from February 1, 2006. See
    the "Note on calculation of moment magnitude" section on
    http://www.globalcmt.org/CMTsearch.html (last accessed on 10/10/2017). The
    rounding occurs in Hanks and Kanamori (1979).

    precise: bool, use precise factor or not (default: True)

    return: number, m0 in J (Nm)
    """
    if validate is True:
        assert(is_number(mw))

    if precise is True:
        factor = (2/3) * 16.1
    else:
        factor = 10.7

    return float(10**((3/2) * (mw + factor)) / 10**7)
