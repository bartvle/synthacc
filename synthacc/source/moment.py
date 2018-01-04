"""
The 'source.moment' module.
"""


import matplotlib.pyplot as plt
import numpy as np

from ..apy import (PRECISION, Object, is_number, is_pos_number,
    is_1d_numeric_array)
from ..math.matrices import SquareMatrix
from ..data import TimeSeries


class MomentTensor(Object):
    """
    A general moment tensor following the NED convection (x is north, y is east
    and z is down).
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
        elif abs(e) < PRECISION:
            return 0
        else:
            return e

    def __add__(self, other):
        """
        return: class instance
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
        return: class instance
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
        Seismic moment (in Nm). This is the Frobenius norm of the tensor
        divided by the square root of 2.
        """
        return float(np.sqrt(np.sum(e**2 for e in self._m)) / np.sqrt(2))

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


class NormalizedMomentFunction(TimeSeries):
    """
    Normalized released moment in function of time.
    """

    def __init__(self, time_delta, moments, start_time=0, validate=True):
        """
        time_delta: (in s)
        moments: (dimensionless)
        start_time: (in s)
        """
        super().__init__(time_delta, start_time, validate=validate)

        if validate is True:
            assert(is_1d_numeric_array(moments))
            assert(np.all(moments >= 0))
            assert(np.all(np.diff(moments) >= 0))
            assert(moments[0] == 0)
            assert(abs(moments[-1] - 1) <= PRECISION)
            assert(abs(moments[-2] - 1) <= PRECISION)

        self._moments = moments

    def __len__(self):
        """
        """
        return len(self._moments)

    def __mul__(self, moment):
        """
        return: 'moment.MomentFunction' instance
        """
        assert(is_pos_number(moment))

        mf = MomentFunction(
            self.time_delta,
            self.moments * moment,
            self.start_time,
            )

        return mf

    @property
    def moments(self):
        """
        """
        return self._moments[:]

    def plot(self, model=None, title=None, size=None):
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
        plt.show()


class MomentFunction(TimeSeries):
    """
    Released moment in function of time.
    """

    def __init__(self, time_delta, moments, start_time=0, validate=True):
        """
        time_delta: (in s)
        moments: (in Nm)
        start_time: (in s)
        """
        super().__init__(time_delta, start_time, validate=validate)

        if validate is True:
            assert(is_1d_numeric_array(moments))
            assert(np.all(moments >= 0))
            assert(np.all(np.diff(moments) >= 0))
            assert(moments[0] == 0)
            assert(moments[-1] == moments[-2])

        self._moments = moments

    def __len__(self):
        """
        """
        return len(self._moments)

    @property
    def moments(self):
        """
        """
        return self._moments[:]

    @property
    def moment(self):
        """
        """
        return self._moments[-1]

    def plot(self, model=None, title=None, size=None):
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
        plt.show()


class NormalizedMomentRateFunction(TimeSeries):
    """
    A normalized moment rate function (or normalized source time function).
    """

    def __init__(self, time_delta, rates, start_time=0, validate=True):
        """
        time_delta: (in s)
        rates: (in 1/s)
        start_time: (in s)
        """
        super().__init__(time_delta, start_time, validate=validate)

        rates = np.asarray(rates, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(rates))
            assert(np.all(rates >= 0))
            assert(rates[+0] == 0)
            assert(rates[-1] == 0)
            assert(np.abs((np.sum(rates) * self.time_delta) - 1) <= PRECISION)

        self._rates = rates

    def __len__(self):
        """
        """
        return len(self.rates)

    def __mul__(self, moment):
        """
        return: 'moment.MomentRateFunction' instance
        """
        assert(is_pos_number(moment))

        mrf = MomentRateFunction(
            self.time_delta,
            self.rates * moment,
            self.start_time,
            )

        return mrf

    @property
    def rates(self):
        """
        return: 1d numerical array, rates (in 1/s)
        """
        return self._rates[:]

    def get_normalized_moment_function(self):
        """
        return: 'moment.NormalizedMomentFunction' instance
        """
        moments = np.cumsum(self._rates * self._time_delta)
        nmf = NormalizedMomentFunction(self._time_delta, moments, self._start_time)

        return nmf

    def plot(self, model=None, title=None, size=None):
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
        plt.show()


NormalizedSourceTimeFunction = NormalizedMomentRateFunction


class MomentRateFunction(TimeSeries):
    """
    A moment rate function (or source time function).
    """

    def __init__(self, time_delta, moment_rates, start_time=0, validate=True):
        """
        time_delta: (in s)
        moment_rates: (in Nm/s)
        start_time: (in s)
        """
        super().__init__(time_delta, start_time, validate=validate)

        moment_rates = np.asarray(moment_rates, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(moment_rates))
            assert(np.all(moment_rates >= 0))
            assert(moment_rates[+0] == 0)
            assert(moment_rates[-1] == 0)

        self._moment_rates = moment_rates

    def __len__(self):
        """
        """
        return len(self._moment_rates)

    @property
    def moment_rates(self):
        """
        return: 1d numerical array, moment rates (in Nm/s)
        """
        return self._moment_rates[:]

    @property
    def moment(self):
        """
        return: pos number
        """
        return float(np.sum(self._moment_rates) * self._time_delta)

    @property
    def magnitude(self):
        """
        return: number
        """
        return m0_to_mw(self.moment)

    def get_normalized(self):
        """
        return: 'moment.NormalizedRateFunction' instance
        """
        nrf = NormalizedMomentRateFunction(self.time_delta,
            self.moment_rates / self.moment, self.start_time)
        return nrf

    def get_moment_function(self):
        """
        return: 'moment.MomentFunction' instance
        """
        moments = np.cumsum(self._moment_rates * self._time_delta)
        mf = MomentFunction(self._time_delta, moments, self._start_time)

        return mf

    def plot(self, model=None, title=None, size=None):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        ax.plot(self.times, self.moment_rates, c='dimgrey', lw=2)
        ax.fill(self.times, self.moment_rates, c='lightgrey')

        if model is not None:
            ax.plot(model.times, model.moment_rates, c='red', lw=2)

        ax.set_ylim((0, None))

        x_label, y_label = 'Time (s)', 'Moment rate (Nm/s)'

        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        if title is None:
            ax.set_title('Moment rate function')

        plt.grid()
        plt.show()


SourceTimeFunction = MomentRateFunction


class InstantMomentRateGenerator(Object):
    """
    """

    def get_nmrf(self, time_delta=PRECISION, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(time_delta))

        return NormalizedMomentRateFunction(time_delta, [0, 1 / time_delta, 0])


class ConstantMomentRateGenerator(Object):
    """
    """

    def get_nmrf(self, rise_time, time_delta, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(rise_time))
            assert(is_pos_number(time_delta))

        n = int(round(rise_time / time_delta))
        rates = np.zeros(n+3)
        rates[1:-1] = 1 / ((n+1)*time_delta)

        return NormalizedMomentRateFunction(time_delta, rates)


class TriangularMomentRateGenerator(Object):
    """
    Gives a (normalized) moment rate function with the shape of an isosceles
    triangle. The height of the triangle (i.e. the maximum rate) is equal to
    the inverse of half its base (i.e. the half duration).
    """

    def get_nmrf(self, half_duration, time_delta, validate=True):
        """
        return: 'moment.NormalizedMomentRateFunction' instance
        """
        if validate is True:
            assert(is_pos_number(half_duration) and is_pos_number(time_delta))
            n = int(round(half_duration / time_delta))
            assert(abs((n * time_delta) - half_duration) <= PRECISION)

        n = int(round(half_duration / time_delta))

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

    precise: bool, use precise factor or not (default: precise)

    return: number, mw

    Used by
        'source.moment.MomentTensor.magnitude'
        'source.ruptures.SimpleRupture.magnitude'
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

    precise: bool, use precise factor or not (default: precise)

    return: number, m0 in J (Nm)
    """
    if validate is True:
        assert(is_number(mw))

    if precise is True:
        factor = (2/3) * 16.1
    else:
        factor = 10.7

    return float(10**((3/2) * (mw + factor)) / 10**7)
