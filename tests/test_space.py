"""
Tests for 'space' module.
"""


import unittest

import numpy as np

from synthacc.apy import PRECISION, is_number
from synthacc.math.matrices import Matrix
from synthacc.space import (Point, Plane, Vector, RotationMatrix,
    are_coordinates, prepare_coordinates, distance, nearest)


class TestPoint(unittest.TestCase):
    """
    """

    p  = Point(1, 2, 3)

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.x, 1)
        self.assertEqual(self.p.y, 2)
        self.assertEqual(self.p.z, 3)

    def test___getitem__(self):
        """
        """
        x, y, z = self.p
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(z, 3)

    def test___eq__(self):
        """
        """
        self.assertTrue(self.p == Point(1, 2, 3))
        self.assertTrue(self.p == (1, 2, 3))
        self.assertTrue(self.p != Point(1, 2, 4))
        self.assertTrue(self.p == Point(1, 2, 3.00000000001))
        self.assertTrue(self.p != Point(1, 2, 3.00000000010))

    def test_vector(self):
        """
        """
        v = self.p.vector
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, self.p.x)
        self.assertEqual(v.y, self.p.y)
        self.assertEqual(v.z, self.p.z)

    def test_get_translated(self):
        """
        """
        v = (2, 4, 6)
        p = self.p.get_translated(v)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 6)
        self.assertEqual(p.z, 9)

        v = Vector(*v)
        p = self.p.get_translated(v)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 6)
        self.assertEqual(p.z, 9)


class TestPlane(unittest.TestCase):
    """
    """

    p = Plane(1, 2, 3, 4)

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.a, 1)
        self.assertEqual(self.p.b, 2)
        self.assertEqual(self.p.c, 3)
        self.assertEqual(self.p.d, 4)

    def test___getitem__(self):
        """
        """
        a, b, c, d = self.p
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)
        self.assertEqual(d, 4)

    def test_from_points(self):
        """
        """
        ulc = (693464.651747485, 718457.100946086, 0)
        urc = (743611.663104304, 696707.115590921, 0)
        llc = (698059.315075598, 729050.608497159, 20000)

        p = Plane.from_points(ulc, urc, llc)

        self.assertEqual(round(p.d), 1022226448407260)


class TestVector(unittest.TestCase):
    """
    """

    v = Vector(1, 2, 3)

    def test___getitem__(self):
        """
        """
        x, y, z = self.v
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(z, 3)

    def test___eq__(self):
        """
        """
        self.assertTrue(self.v == Vector(1, 2, 3))
        self.assertTrue(self.v == (1, 2, 3))
        self.assertTrue(self.v != Vector(1, 2, 4))
        self.assertTrue(self.v == Vector(1, 2, 3.00000000001))
        self.assertTrue(self.v != Vector(1, 2, 3.00000000010))

    def test___add__(self):
        """
        """
        v = self.v + Vector(1, 2, 3)
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, 2)
        self.assertEqual(v.y, 4)
        self.assertEqual(v.z, 6)

        v = self.v + (1, 2, 3)
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, 2)
        self.assertEqual(v.y, 4)
        self.assertEqual(v.z, 6)

    def test___sub__(self):
        """
        """
        v = self.v - Vector(3, 2, 1)
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, -2)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, 2)

        v = self.v - (3, 2, 1)
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, -2)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, 2)

    def test___mul__(self):
        """
        """
        v1 = Vector(1, 2, 3)
        v2 = v1 * 3
        self.assertEqual(v2.x, 3)
        self.assertEqual(v2.y, 6)
        self.assertEqual(v2.z, 9)
        self.assertEqual(v2.magnitude, v1.magnitude * 3)

    def test___truediv__(self):
        """
        """
        v1 = Vector(1.5, 3, 4.5)
        v2 = v1 / 3
        self.assertEqual(v2.x, 0.5)
        self.assertEqual(v2.y, 1.0)
        self.assertEqual(v2.z, 1.5)
        self.assertEqual(v2.magnitude, v1.magnitude / 3)

    def test___pos__(self):
        """
        """
        v = +self.v
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, +self.v.x)
        self.assertEqual(v.y, +self.v.y)
        self.assertEqual(v.z, +self.v.z)

    def test___neg__(self):
        """
        """
        v = -self.v
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, -self.v.x)
        self.assertEqual(v.y, -self.v.y)
        self.assertEqual(v.z, -self.v.z)

    def test_properties(self):
        """
        """
        self.assertEqual(self.v.x, 1)
        self.assertEqual(self.v.y, 2)
        self.assertEqual(self.v.z, 3)

    def test_row(self):
        """
        """
        m = self.v.row
        self.assertEqual(type(m), Matrix)
        self.assertEqual(m.order, (1, 3))

    def test_col(self):
        """
        """
        m = self.v.col
        self.assertEqual(type(m), Matrix)
        self.assertEqual(m.order, (3, 1))

    def test_magnitude(self):
        """
        """
        m = Vector(1, 1, 1).magnitude
        self.assertTrue(is_number(m))
        self.assertEqual(m, np.sqrt(3))

    def test_unit(self):
        """
        """
        u = self.v.unit
        self.assertEqual(u.get_angle(self.v), 0)
        self.assertEqual(u.magnitude, 1)
        self.assertEqual(Vector(0, 0, 0).unit, None)

    def test_angle(self):
        """
        """
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        self.assertTrue(abs(v1.get_angle(v2) - 90) < 10**-PRECISION)
        v1 = Vector(1, 0, 0)
        v2 = Vector(2, 0, 0)
        self.assertTrue(abs(v1.get_angle(v2) -  0) < 10**-PRECISION)
        v1 = Vector(1, 0, 0)
        v2 = Vector(1, 1, 0)
        self.assertTrue(abs(v1.get_angle(v2) - 45) < 10**-PRECISION)

        a = Vector(0, 0, 0).get_angle((1, 1, 1))
        self.assertEqual(a, None)
        a = Vector(1, 1, 1).get_angle((0, 0, 0))
        self.assertEqual(a, None)


class TestRotationMatrix(unittest.TestCase):
    """
    """

    def test_from_basic_rotations(self):
        """
        """
        r1 = RotationMatrix.from_basic_rotations(000., 000., 000.)
        r2 = RotationMatrix.from_basic_rotations(360., 360., 360.)
        r3 = RotationMatrix.from_basic_rotations(000., 000., 090.)
        r4 = RotationMatrix.from_basic_rotations(090., 000., 000.)
        r5 = RotationMatrix.from_basic_rotations(090., 000., 090.)
        self.assertTrue((r1._array == np.array(
            [[+1, +0, +0], [+0, +1, +0] ,[+0, +0, +1]])).all())
        self.assertTrue((r2._array == np.array(
            [[+1, +0, +0], [+0, +1, +0] ,[+0, +0, +1]])).all())
        self.assertTrue((r3._array == np.array(
            [[+0, -1, +0], [+1, +0, +0] ,[+0, +0, +1]])).all())
        self.assertTrue((r4._array == np.array(
            [[+1, +0, +0], [+0, +0, -1] ,[+0, +1, +0]])).all())
        self.assertTrue((r5._array == np.array(
            [[+0, +0, +1], [+1, +0, +0] ,[+0, +1, +0]])).all())


class Test(unittest.TestCase):
    """
    """

    def test_are_coordinates(self):
        """
        """
        self.assertTrue(are_coordinates((1, 2, 3)))
        self.assertFalse(are_coordinates((1, 2)))
        self.assertFalse(are_coordinates((None, 2, 3)))
        self.assertFalse(are_coordinates((1, None, 3)))
        self.assertFalse(are_coordinates((1, 2, None)))
        self.assertFalse(are_coordinates([1, 2, 3]))
