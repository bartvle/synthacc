"""
The 'data' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from .apy import (Object, is_number, is_non_neg_number, is_pos_number,
    is_pos_integer)
from .time import is_time


class TimeSeries(ABC, Object):
    """
    """

    def __init__(self, time_delta, start_time=0, validate=True):
        """
        time_delta: pos number (in s)
        start_time: non neg number (in s) or
            'time.Time' instance (default: 0)
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_non_neg_number(start_time) or
                is_time(start_time))

        self._time_delta = time_delta
        self.start_time = start_time

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
        return: non neg number (in s)
        """
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        """
        """
        self._start_time = start_time

    @property
    def rel_times(self):
        """
        """
        return self.time_delta * np.arange(len(self))

    @property
    def abs_times(self):
        """
        return: list of non neg numbers or 'time.Time' instances
        """
        l = [self._start_time + float(rel_time) for rel_time in self.rel_times]

        return l

    @property
    def duration(self):
        """
        return: pos number (in s)
        """
        return self.time_delta * (len(self) - 1)

    @property
    def end_time(self):
        """
        return: non neg number (in s) or 'time.Time' instance
        """
        return self.start_time + self.duration


class DataRecord(ABC, Object):
    """
    A data record from a database.
    """

    def __init__(self, key, validate=True):
        """
        key: integer
        """
        if validate is True:
            assert(is_pos_integer(key))

        self._key = key

    @property
    def key(self):
        """
        """
        return self._key


class DataBase(ABC, Object):
    """
    A database of data records.
    """

    def __init__(self, records, rec_cls, validate=True):
        """
        """
        if validate is True:
            assert(type(records) is list and issubclass(rec_cls, DataRecord))

            ## Check if records are of type rec_cls and keys are unique
            keys = []
            for r in records:
                assert(type(r) is rec_cls)
                keys.append(r.key)
            assert(len(set(keys)) == len(records))

        self._records = records
        self._rec_cls = rec_cls

    def __len__(self):
        """
        """
        return len(self._records)

    def __iter__(self):
        """
        """
        for r in self._records:
            yield r

    @property
    def keys(self):
        """
        """
        return [r.key for r in self]

    def get(self, key, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(key))

        for r in self:
            if r.key == key:
                return r

        raise LookupError('No record with key %i' % key)
