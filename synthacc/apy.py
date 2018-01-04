"""
The 'apy' module.

Module for tweaking Python. Arrays are Numpy arrays. 0d arrays are not allowed
(they only complicate things). We allow only the native Python int and float,
not Numpy ones.
"""


import numpy as np


T, F = True, False

## Precision for float comparison
PRECISION = 10**-10


class Object(object):
    """
    An object that can have only private attributes. They can be made public
    with properties and setters.
    """

    def __setattr__(self, attr, val):
        """
        The intended behavior is purely achieved by this '__setattr__' method.
        """
        ## TODO: see issue 1
        if attr.startswith('_') or attr in type(self).__dict__:
            super().__setattr__(attr, val)
        else:
            raise AttributeError('This object has no attribute "%s"' % attr)


def is_boolean(obj):
    """
    Check if object is an boolean.
    """
    return obj in (T, F)


def is_integer(obj):
    """
    Check if object is an integer.
    """
    return type(obj) is int


def is_non_neg_integer(obj):
    """
    Check if object is a positive integer or zero.
    """
    return (is_integer(obj) and obj >= 0)


def is_non_pos_integer(obj):
    """
    Check if object is a negative integer or zero.
    """
    return (is_integer(obj) and obj <= 0)


def is_pos_integer(obj):
    """
    Check if object is a positive integer and not zero.
    """
    return (is_integer(obj) and obj > 0)


def is_neg_integer(obj):
    """
    Check if object is a negative integer and not zero.
    """
    return (is_integer(obj) and obj < 0)


def is_number(obj):
    """
    Check if object is a real number.
    """
    return type(obj) in (int, float)


def is_non_neg_number(obj):
    """
    Check if object is a positive real number or zero.
    """
    return (is_number(obj) and obj >= 0)


def is_non_pos_number(obj):
    """
    Check if object is a negative real number or zero.
    """
    return (is_number(obj) and obj <= 0)


def is_pos_number(obj):
    """
    Check if object is a positive real number and not zero.
    """
    return (is_number(obj) and obj > 0)


def is_neg_number(obj):
    """
    Check if object is a negative real number and not zero.
    """
    return (is_number(obj) and obj < 0)


def is_fraction(obj):
    """
    Check if object is a fraction (real number between zero and one).
    """
    return (is_number(obj) and 0 <= obj <= 1)


def _is_array(obj, n):
    """
    Check if object is an n-dimensional (not 0) array.
    """
    assert(n != 0)
    if (type(obj) == np.ndarray):
        if n is None:
            return obj.ndim >= 1
        else:
            return obj.ndim == n
    else:
        return F


def is_array(obj):
    """
    Check if object is an array (not 0d).
    """
    return _is_array(obj, n=None)


def _is_numeric_array(obj, n):
    """
    Check if object is a n-dimensional array of real numbers.
    """
    return (_is_array(obj, n=n) and obj.dtype in
        (np.int32, np.int64, np.float64))


def is_numeric_array(obj):
    """
    Check if object is an array of real numbers.
    """
    return _is_numeric_array(obj, n=None)


def is_1d_numeric_array(obj):
    """
    Check if object is a 1d array of real numbers.
    """
    return _is_numeric_array(obj, n=1)


def is_2d_numeric_array(obj):
    """
    Check if object is a 2d array of real numbers.
    """
    return _is_numeric_array(obj, n=2)


def is_3d_numeric_array(obj):
    """
    Check if object is a 3d array of real numbers.
    """
    return _is_numeric_array(obj, n=3)


def is_numeric(obj):
    """
    Check if object is numeric (a real number or n-dimensional array of real
    numbers)
    """
    return is_number(obj) or is_numeric_array(obj)


def is_in_range(obj, min_val, max_val):
    """
    Check if object is numeric and between 'min_val' and 'max_val' (included).
    """
    b = (is_numeric(obj) and (
        np.all(min_val <= obj) and np.all(obj <= max_val)))

    return b


def is_complex(obj):
    """
    Check if object is complex number.
    """
    return type(obj) is complex


def _is_complex_array(obj, n):
    """
    Check if object is a n-dimensional numpy array of complex numbers.
    """
    return (_is_array(obj, n=n) and obj.dtype is np.dtype(np.complex128))


def is_complex_array(obj):
    """
    Check if object is a numpy array of complex numbers.
    """
    return _is_complex_array(obj, n=None)


def is_1d_complex_array(obj):
    """
    Check if object is a 1d numpy array of complex numbers.
    """
    return _is_complex_array(obj, n=1)


def is_2d_complex_array(obj):
    """
    Check if object is a 2d numpy array of complex numbers.
    """
    return _is_complex_array(obj, n=2)


def is_3d_complex_array(obj):
    """
    Check if object is a 3d numpy array of complex numbers.
    """
    return _is_complex_array(obj, n=3)


def is_string(obj, length=None):
    """
    Check if object is string (with length).
    """
    if length is not None:
        return type(obj) == str and len(obj) == length
    else:
        return type(obj) == str
