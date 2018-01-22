"""
The 'time' module.
"""


import datetime
import math


from .apy import Object, is_non_neg_number, is_string


DEFAULT_PRECISION = 6


def _get_timedelta_from_seconds(seconds):
    """
    """
    assert(is_non_neg_number(seconds))

    m, s = math.modf(seconds)
    m, s = int(round(m*10**6)), int(s)
    td = datetime.timedelta(seconds=s, microseconds=m)

    return td


class Time(Object):
    """
    Immutable time.
    """

    def __init__(self, time, precision=None, validate=True):
        """
        """
        if is_string(time):
            assert(len(time) == 19 or 21 <= len(time) <= 26)
            assert(precision is None)
            date, totd = time.split(' ')
            date = map(int, date.split('-'))
            h, m, s = totd.split(':')
            s = s.split('.')
            if len(s) == 2:
                ms = s[1]
                precision = len(ms)
                while len(ms) < 6:
                    ms += '0'
                ms = int(ms)
            else:
                ms = 0
                precision = 0
            s = int(s[0])
            self._time = datetime.datetime(*date, int(h), int(m), s, ms)
        else:
            if precision is None:
                precision = DEFAULT_PRECISION
            rest = '{:06}'.format(time.microsecond)[precision:]
            assert(rest == '' or int(rest) == 0)

            self._time = datetime.datetime(time.year, time.month, time.day,
                time.hour, time.minute, time.second, time.microsecond)

        assert(precision in (0, 1, 2, 3, 4, 5, 6))

        self._precision = precision


    def __str__(self):
        """
        """
        s = '%04i-%02i-%02i %02i:%02i:%02i' % (self.year, self.month, self.day, self.hour, self.minute, self.second)
        if self.microsecond is not None:
            s += ('.%06i' % self.microsecond)[:self._precision+1]
        return s

    def __repr__(self):
        """
        """
        return '< Time | %s >' % self

    def __eq__(self, other):
        """
        """
        assert(is_time(other))
        return str(self) == str(other)

    def __ne__(self, other):
        """
        """
        return not self.__eq__(other)

    def __gt__(self, other):
        """
        """
        assert(is_time(other))
        assert(other.precision == self.precision)
        return self._time > other._time

    def __lt__(self, other):
        """
        """
        assert(is_time(other))
        assert(other.precision == self.precision)
        return self._time < other._time

    def __ge__(self, other):
        """
        """
        assert(is_time(other))
        assert(other.precision == self.precision)
        return self._time >= other._time

    def __le__(self, other):
        """
        """
        assert(is_time(other))
        assert(other.precision == self.precision)
        return self._time <= other._time

    def __hash__(self):
        """
        For set behavior.
        """
        return str(self).__hash__()

    def __add__(self, value):
        """
        value: non neg number (in seconds)
        """
        dt = self._time + _get_timedelta_from_seconds(value)

        return self.__class__(dt, precision=self.precision)

    def __sub__(self, other):
        """
        other: non neg number (in seconds) or 'Time' instance
        """
        if is_time(other):
            assert(other < self)
            print(self._time - other._time)
            return (self._time - other._time).total_seconds()
        else:
            dt = self._time - _get_timedelta_from_seconds(other)
            return self.__class__(dt, precision=self.precision)

    @property
    def year(self):
        """
        """
        return self._time.year

    @property
    def month(self):
        """
        """
        return self._time.month

    @property
    def day(self):
        """
        """
        return self._time.day

    @property
    def hour(self):
        """
        """
        return self._time.hour

    @property
    def minute(self):
        """
        """
        return self._time.minute

    @property
    def second(self):
        """
        """
        return self._time.second

    @property
    def microsecond(self):
        """
        """
        if self.precision == 0:
            return None
        else:
            return self._time.microsecond

    @property
    def precision(self):
        """
        """
        return self._precision

    @property
    def date(self):
        """
        """
        return Date(self._time.date())


class Date(Object):
    """
    Immutable date.
    """

    def __init__(self, *args, validate=True):
        """
        """
        if validate is True:
            assert(len(args) == 1 or len(args) == 3)

        if len(args) == 1:
            [date] = args
            if is_string(date):
                assert(len(date) == 10)
                y, m, d = map(int, date.split('-'))
            else:
                y = date.year
                m = date.month
                d = date.day
        else:
            y, m, d = args

        self._date = datetime.date(y, m, d)

    def __str__(self):
        """
        """
        s = '%04i-%02i-%02i' % (self.year, self.month, self.day)
        return s

    def __repr__(self):
        """
        """
        return '< Date | %s >' % self

    def __eq__(self, other):
        """
        """
        return self._date == self.__class__(other)._date

    def __ne__(self, other):
        """
        """
        return not self.__eq__(other)

    def __gt__(self, other):
        """
        """
        return self._date > self.__class__(other)._date

    def __lt__(self, other):
        """
        """
        return self._date < self.__class__(other)._date

    def __ge__(self, other):
        """
        """
        return self._date >= self.__class__(other)._date

    def __le__(self, other):
        """
        """
        return self._date <= self.__class__(other)._date

    def __hash__(self):
        """
        For set behavior.
        """
        return str(self).__hash__()

    @property
    def year(self):
        """
        """
        return self._date.year

    @property
    def month(self):
        """
        """
        return self._date.month

    @property
    def day(self):
        """
        """
        return self._date.day


def is_time(obj):
    """
    Check if object is time.
    """
    return type(obj) == Time


def is_date(obj):
    """
    Check if object is date.
    """
    return type(obj) == Date
