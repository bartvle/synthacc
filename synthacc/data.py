"""
The 'data' module.
"""


from abc import ABC

import matplotlib.pyplot as plt

from .apy import Object, is_boolean, is_pos_integer, is_1d_numeric_array


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


class Histogram(Object):
    """
    """

    def __init__(self, values, positive=False, validate=True):
        """
        """
        if validate is True:
            assert(is_1d_numeric_array(values))
            assert(is_boolean(positive))

        self._values = values
        self._positive = positive

    @property
    def mean(self):
        """
        """
        return self._values.mean()

    @property
    def min(self):
        """
        """
        return self._values.min()

    @property
    def max(self):
        """
        """
        return self._values.max()

    def plot(self, bins=None, size=None, png_filespec=None, validate=True):
        """
        """
        f, ax = plt.subplots(figsize=size)

        if self._positive is True:
            values = self._values[self._values > 0]
        else:
            values = self._values

        ax.hist(values, bins=bins, density=True)

        ax.grid()

        if self._positive is True:
            ax.set_xlim(0, None)

        plt.tight_layout()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()
