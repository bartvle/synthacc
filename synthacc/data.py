"""
The 'data' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from .apy import Object, is_number, is_pos_number, is_pos_integer


class TimeSeries(ABC, Object):
    """
    """

    def __init__(self, time_delta, start_time=0, validate=True):
        """
        time_delta: pos number, (in s)
        start_time: any number, (in s) (default: 0)
        """
        if validate is True:
            assert(is_pos_number(time_delta) and is_number(start_time))

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
        return: any number (in s)
        """
        return self._start_time

    @property
    def times(self):
        """
        return: 1d numeric array (in s)
        """
        return self.start_time + self.time_delta * np.arange(len(self))

    @property
    def duration(self):
        """
        return: pos number (in s)
        """
        return self.time_delta * (len(self) - 1)

    @property
    def end_time(self):
        """
        return: any number (in s)
        """
        return self.start_time + self.duration


class DataBase(ABC, Object):
    """
    """

    def __init__(self, records, validate=True):
        """
        """
        if validate is True:
            pass

        self._records = records

    def __len__(self):
        """
        """
        return len(self._records)

    def __iter__(self):
        """
        """
        for r in self._records:
            yield r

    def get(self, key, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(key))

        for r in self:
            if r.key == key:
                return r

        raise LookupError('No record with key %i' % key)