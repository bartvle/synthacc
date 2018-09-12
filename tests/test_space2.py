"""
Tests for 'space2' module.
"""


import unittest

import random

import numpy as np

from synthacc.apy import PRECISION
from synthacc.space2 import (Point, RectangularSurface,
    DiscretizedRectangularSurface, GaussianACF, ExponentialACF, VonKarmanACF,
    SpatialRandomFieldGenerator, prepare_coordinates, distance)


class TestPoint(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        x = random.uniform(-1000, 1000)
        y = random.uniform(-1000, 1000)
        p = Point(x, y)
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)

    def test___getitem__(self):
        """
        """
        p = Point(0, 1)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[1], 1)
        x, y = p
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test___eq__(self):
        """
        """
        p1 = Point(0, 1)
        p2 = Point(0, 1)
        self.assertTrue(p1 == p2)


class TestRectangularSurface(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        w = random.uniform(0.001, +1000)
        l = random.uniform(0.001, +1000)
        s = RectangularSurface(w, l)
        self.assertEqual(s.w, w)
        self.assertEqual(s.l, l)
        self.assertEqual(s.area, w * l)

    def test_get_random1(self):
        """
        """
        w = random.uniform(0.001, +1000)
        l = random.uniform(0.001, +1000)
        s = RectangularSurface(w, l)
        p = s.get_random()
        assert(0 <= p.x < s.w)
        assert(0 <= p.y < s.l)

    def test_get_random2(self):
        """
        """
        w = random.uniform(0.001, +1000)
        l = random.uniform(0.001, +1000)
        s = RectangularSurface(w, l)
        xmin = random.uniform(0, w)
        ymin = random.uniform(0, l)
        xmax = random.uniform(xmin, w)
        ymax = random.uniform(ymin, l)
        p = s.get_random(xmin, xmax, ymin, ymax)
        assert(0 <= p.x < s.w)
        assert(0 <= p.y < s.l)


class TestDiscretizedRectangularSurface(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        w = random.uniform(0.001, 1000)
        l = random.uniform(0.001, 1000)
        nw = random.randint(1, 1000)
        nl = random.randint(1, 1000)
        s = DiscretizedRectangularSurface(w, l, nw, nl)
        self.assertEqual(s.w, w)
        self.assertEqual(s.l, l)
        self.assertEqual(s.nw, nw)
        self.assertEqual(s.nl, nl)
        self.assertEqual(s.dw, w / nw)
        self.assertEqual(s.dl, l / nl)
        self.assertEqual(s.shape, (nw, nl))
        self.assertTrue(abs(s.cell_area - (s.dw * s.dl)) < 10**-PRECISION)

        w, l = 5, 10
        nw, nl = 5, 5
        s = DiscretizedRectangularSurface(w, l, nw, nl)
        xs, ys = [0.5, 1.5, 2.5, 3.5, 4.5], [1, 3, 5, 7, 9]
        self.assertTrue(np.all(s.xs == xs))
        self.assertTrue(np.all(s.ys == ys))
        self.assertEqual(s.cell_area, 2)

    def test_from_spacing(self):
        """
        """
        w, l, dw, dl, nw, nl = 10, 100, 0.9, 2, 11, 50
        s = DiscretizedRectangularSurface.from_spacing(w, l, dw, dl)
        self.assertEqual(s.w, w)
        self.assertEqual(s.l, l)
        self.assertEqual(s.nw, nw)
        self.assertEqual(s.nl, nl)
        self.assertEqual(s.dw, w / nw)
        self.assertEqual(s.dl, l / nl)
        self.assertEqual(s.shape, (nw, nl))


class TestSpatialRandomFieldGenerator(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        w, l, nw, nl, acf, aw, al = (
            101, 1001, 101, 1001, GaussianACF(), 50, 500)
        g = SpatialRandomFieldGenerator(w, l, nw, nl, acf, aw, al)
        self.assertEqual(g.aw, aw)
        self.assertEqual(g.al, al)
        self.assertEqual(g.shape, (nw, nl))

    def test___call__(self):
        """
        """
        w, l, nw, nl, acf, aw, al = (
            101, 1001, 101, 1001, GaussianACF(), 50, 500)
        g = SpatialRandomFieldGenerator(w, l, nw, nl, acf, aw, al)
        f = g()
        self.assertEqual(f.shape, (nw, nl))


class Test(unittest.TestCase):
    """
    """

    def test_prepare_coordinates(self):
        """
        """
        c1, c2 = prepare_coordinates(1, 2)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 2)

        c1, c2 = prepare_coordinates(1, np.array([2, 2]))
        self.assertListEqual(list(c1), [1, 1])
        self.assertListEqual(list(c2), [2, 2])

    def test_distance(self):
        """
        """
        d = distance(0, 0, 1, np.array([0, 1]))
        self.assertListEqual(list(d), [1, np.sqrt(2)])

#     def test_cartesian_to_polar(self):
#         """
#         """
#         c = cartesian_to_polar(0, 0)
#         self.assertEqual(c, (0., 0.))
#         c = cartesian_to_polar(1, 0)
#         self.assertEqual(c, (1.,  0.))
#         c = cartesian_to_polar(0, 1)
#         self.assertEqual(c, (1., 90.))
#         c = cartesian_to_polar(1, 1)
#         self.assertEqual(c, (np.sqrt(2), 45.))

#     def test_polar_to_cartesian(self):
#         """
#         """
#         c = polar_to_cartesian(0, 0)
#         self.assertEqual(c, (0, 0))
#         c = polar_to_cartesian(1, 0)
#         self.assertEqual(c, (1, 0))
#         c = polar_to_cartesian(1, 90)
#         self.assertEqual(c, (0, 1))
#         c = polar_to_cartesian(float(np.sqrt(2)), 45)
#         self.assertEqual(round(c[0], 10), 1)
#         self.assertEqual(round(c[1], 10), 1)
