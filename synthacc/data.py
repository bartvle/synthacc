"""
The 'data' module.
"""


from abc import ABC

import matplotlib.pyplot as plt

from .apy import Object, is_pos_integer, is_1d_numeric_array


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

    def __init__(self, values, validate=True):
        """
        """
        if validate is True:
            assert(is_1d_numeric_array(values))

        self._values = values

    def plot(self, bins=None, size=None, png_filespec=None, validate=True):
        """
        """
        f, ax = plt.subplots(figsize=size)

        plt.hist(self._values[self._values > 0], bins=bins, density=True)

        plt.tight_layout()

        if png_filespec is not None:
            plt.savefig(png_filespec)
        else:
            plt.show()
