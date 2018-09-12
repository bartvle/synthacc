"""
Tests for 'apy' module.
"""


import unittest

from synthacc.apy import (T, F, PRECISION, Object, is_boolean, is_integer,
    is_non_neg_integer, is_non_pos_integer, is_pos_integer, is_neg_integer,
    is_number, is_non_neg_number, is_non_pos_number, is_pos_number,
    is_neg_number, is_fraction, is_array, is_numeric_array,
    is_1d_numeric_array, is_2d_numeric_array, is_3d_numeric_array, is_numeric,
    is_in_range, is_complex, is_complex_array, is_1d_complex_array, 
    is_2d_complex_array, is_3d_complex_array, is_string)


class TestObject(unittest.TestCase):
    """
    """

    def test_1(self):
        """
        Private attribute can be set.
        """
        class X(Object):
            pass

        x = X()
        x._x = True

        self.assertTrue(x._x)

    def test_2(self):
        """
        Public attribute can't be set.
        """
        class X(Object):
            pass

        x = X()

        def f():
            x.x = True

        self.assertRaises(AttributeError, f)

    def test_3(self):
        """
        Public attribute can with private attribute and property getter. Also
        if property is set on superclass!
        """
        class X(Object):

            def __init__(self, x):
                self._x = x

            @property
            def x(self):
                return self._x

        x = X(x=True)
        self.assertTrue(x._x)
        self.assertTrue(x.x)

        def fx():
            x.x = True

        self.assertRaises(AttributeError, fx)

        class Y(X):
            pass

        y = Y(x=True)

        self.assertTrue(y._x)
        self.assertTrue(y.x)

        def fy():
            y.x = False

        self.assertRaises(AttributeError, fy)


    def test_4(self):
        """
        Public attribute can with private attribute and property getter/setter.
        Also if property is set on superclass!
        """
        class X(Object):

            def __init__(self, x):
                self._x = x

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = val

        x = X(x=True)
        self.assertTrue(x._x)
        self.assertTrue(x.x)

        x.x = False
        self.assertFalse(x._x)
        self.assertFalse(x.x)

        class Y(X):
            pass

        y = Y(x=True)

        self.assertTrue(y._x)
        self.assertTrue(y.x)

        y.x = False
        self.assertFalse(y._x)
        self.assertFalse(y.x)
