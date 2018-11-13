"""
Tests for 'space3' module.
"""


import unittest

import random

import numpy as np

from synthacc.apy import PRECISION, is_number
from synthacc.math import Matrix
from synthacc.space3 import (Point, Plane, Vector, RotationMatrix,
    are_coordinates, prepare_coordinates, distance, nearest)


class TestPoint(unittest.TestCase):
    """
    """

    x = random.uniform(-1000, 1000)
    y = random.uniform(-1000, 1000)
    z = random.uniform(-1000, 1000)
    p = Point(x, y, z)

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.x, self.x)
        self.assertEqual(self.p.y, self.y)
        self.assertEqual(self.p.z, self.z)

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.p[0], self.x)
        self.assertEqual(self.p[1], self.y)
        self.assertEqual(self.p[2], self.z)
        x, y, z = self.p
        self.assertEqual(x, self.x)
        self.assertEqual(y, self.y)
        self.assertEqual(z, self.z)

    def test___eq__(self):
        """
        """
        p = Point(0, 1, 2)
        self.assertTrue(p == Point(0, 1, 2))
        self.assertTrue(p == (0, 1, 2))
        self.assertTrue(p != Point(0, 1, 3))
        self.assertTrue(p == Point(0, 1, 2.00000000001))
        self.assertTrue(p != Point(0, 1, 2.00000000010))

    def test_vector(self):
        """
        """
        v = self.p.vector
        self.assertEqual(type(v), Vector)
        self.assertEqual(v.x, self.p.x)
        self.assertEqual(v.y, self.p.y)
        self.assertEqual(v.z, self.p.z)

    def test_translate1(self):
        """
        """
        p = Point(1, 2, 3)
        v = (2, 4, 6)
        p = p.translate(v)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 6)
        self.assertEqual(p.z, 9)

    def test_translate2(self):
        """
        """
        p = Point(1, 2, 3)
        v = Vector(*(2, 4, 6))
        p = p.translate(v)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 6)
        self.assertEqual(p.z, 9)

    def test_rotate1(self):
        """
        """
        r1 = RotationMatrix.from_basic_rotations(  0,   0,   0)
        r2 = RotationMatrix.from_basic_rotations(360, 360, 360)
        r3 = RotationMatrix.from_basic_rotations(  0,   0,  90)
        r4 = RotationMatrix.from_basic_rotations( 90,   0,   0)
        r5 = RotationMatrix.from_basic_rotations(  0,  90,   0)
        p1 = Point(1, 1, 0).rotate(r1)
        p2 = Point(1, 1, 0).rotate(r2)
        p3 = Point(1, 0, 0).rotate(r3)
        p4 = Point(0, 1, 0).rotate(r4)
        p5 = Point(1, 0, 0).rotate(r5)
        self.assertEqual(p1, (+1, +1, +0))
        self.assertEqual(p2, (+1, +1, +0))
        self.assertEqual(p3, (+0, +1, +0))
        self.assertEqual(p4, (+0, +0, +1))
        self.assertEqual(p5, (+0, +0, -1))

    def test_rotate2(self):
        """
        Rotate (1, 0, 0) 90 degrees around z in (1, 1, 0).
        """
        p = Point(1, 0, 0)
        o = Point(1, 1, 0)

        r = RotationMatrix.from_basic_rotations(0, 0, 90)
        p = p.rotate(r, origin=o)
        self.assertEqual(p, (2, 1, 0))


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

    def test_from_points(self):
        """
        """
        ulc = (693464.651747485, 718457.100946086, 0)
        urc = (743611.663104304, 696707.115590921, 0)
        llc = (698059.315075598, 729050.608497159, 20000)

        p = Plane.from_points(ulc, urc, llc)

        self.assertEqual(round(p.d), 1022226448407260)

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.p[0], 1)
        self.assertEqual(self.p[1], 2)
        self.assertEqual(self.p[2], 3)
        self.assertEqual(self.p[3], 4)
        a, b, c, d = self.p
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)
        self.assertEqual(d, 4)

    def test_normal(self):
        """
        """
        p = Plane.from_points((0, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertEqual(p.normal, (1, 0, 0))
        p = Plane.from_points((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.assertEqual(p.normal, (0, 0, 1))

    def test_get_distance(self):
        """
        """
        p = Plane.from_points((0, 0, 1), (0, 1, 1), (1, 0, 1))
        self.assertEqual(p.get_distance((0, 0, 0)), 1)
        p = Plane.from_points((0.5, 0.5, 0), (0, 1, 1), (1, 0, 1))
        self.assertTrue(abs(
            p.get_distance((0, 0, 0)) - float(np.sqrt(2)/2)) < 10**-PRECISION)
        p = Plane.from_points((0, 0, 1), (0, 2, 0), (2, 0, 0))
        self.assertEqual(p.get_distance((0, 0, 0)), np.sqrt(2) / np.sqrt(3))


class TestVector(unittest.TestCase):
    """
    """

    v = Vector(1, 2, 3)

    def test_properties(self):
        """
        """
        self.assertEqual(self.v.x, 1)
        self.assertEqual(self.v.y, 2)
        self.assertEqual(self.v.z, 3)

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

    def test___mul__1(self):
        """
        """
        v1 = Vector(1, 2, 3)
        v2 = v1 * 3
        self.assertEqual(v2.x, 3)
        self.assertEqual(v2.y, 6)
        self.assertEqual(v2.z, 9)
        self.assertEqual(v2.magnitude, v1.magnitude * 3)

    def test___mul__2(self):
        """
        """
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        self.assertEqual(v1 * v2, 1*4+2*5+3*6)
        self.assertEqual(v1 * v2, v1.magnitude * v2.magnitude *
            np.cos(np.radians(v1.get_angle(v2))))

    def test___truediv__(self):
        """
        """
        v1 = Vector(1.5, 3, 4.5)
        v2 = v1 / 3
        self.assertEqual(v2.x, 0.5)
        self.assertEqual(v2.y, 1.0)
        self.assertEqual(v2.z, 1.5)
        self.assertEqual(v2.magnitude, v1.magnitude / 3)

    def test___matmul__(self):
        """
        """
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        v =  v1 @ v2
        self.assertTrue(abs(v.magnitude - (v1.magnitude * v2.magnitude *
            np.sin(np.radians(v1.get_angle(v2))))) < 10**-PRECISION)

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

    def test_rotate(self):
        """
        """
        v = self.v.rotate(RotationMatrix.from_basic_rotations(z=180))
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, -2)
        self.assertEqual(v.z, +3)

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
        r1 = RotationMatrix.from_basic_rotations(  0,   0,   0)
        r2 = RotationMatrix.from_basic_rotations(360, 360, 360)
        r3 = RotationMatrix.from_basic_rotations(  0,   0,  90)
        r4 = RotationMatrix.from_basic_rotations( 90,   0,   0)
        r5 = RotationMatrix.from_basic_rotations( 90,   0,  90)
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

    def test_from_axis_and_angle(self):
        """
        """
        rx_cal = RotationMatrix.from_axis_and_angle((1, 0, 0), 30)
        ry_cal = RotationMatrix.from_axis_and_angle((0, 1, 0), 45)
        rz_cal = RotationMatrix.from_axis_and_angle((0, 0, 1), 60)

        rx_tgt = RotationMatrix.from_basic_rotations(x=30)
        ry_tgt = RotationMatrix.from_basic_rotations(y=45)
        rz_tgt = RotationMatrix.from_basic_rotations(z=60)

        self.assertTrue((rx_cal._array == rx_tgt._array).all())
        self.assertTrue((ry_cal._array == ry_tgt._array).all())
        self.assertTrue((rz_cal._array == rz_tgt._array).all())


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

    def test_prepare_coordinates(self):
        """
        """
        c1, c2, c3 = prepare_coordinates(1, 2, 3)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 2)
        self.assertEqual(c3, 3)

        c1, c2, c3 = prepare_coordinates(1., 2., np.array([3, 3]))
        self.assertListEqual(list(c1), [1., 1.])
        self.assertListEqual(list(c2), [2., 2.])
        self.assertListEqual(list(c3), [3., 3.])

    def test_distance(self):
        """
        """
        d = distance(0, 0, 0, 1, 1, np.array([0, 1]))
        self.assertListEqual(list(d), [np.sqrt(2), np.sqrt(3)])

    def test_nearest(self):
        """
        """
        shape = (100, 100, 100)
        xs = np.random.uniform(-1000, 1000, size=shape)
        ys = np.random.uniform(-1000, 1000, size=shape)
        zs = np.random.uniform(    0, 1000, size=shape)

        i = random.randint(0, 99)
        xs[i] = 1
        ys[i] = 1
        zs[i] = 1

        self.assertEqual(nearest(0.5, 0.5, 0.5, xs, ys, zs), (1, 1, 1))
