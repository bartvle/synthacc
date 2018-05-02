"""
Tests for 'space2' module.
"""


import unittest

import numpy as np

from synthacc.space2 import (Point, RectangularSurface,
    DiscretizedRectangularSurface, GaussianACF, ExponentialACF, VonKarmanACF,
    SpatialRandomFieldGenerator, distance, cartesian_to_polar,
    polar_to_cartesian)


class TestDiscretizedRectangularSurface(unittest.TestCase):
    """
    """

    w = 6000
    l = 12000
    dw = 100
    dl = 150
    s = DiscretizedRectangularSurface(w, l, dw, dl)

    def test_properties(self):
        """
        """
        self.assertEqual(self.s.w, self.w)
        self.assertEqual(self.s.l, self.l)
        self.assertEqual(self.s.dw, self.dw)
        self.assertEqual(self.s.dl, self.dl)
        self.assertEqual(self.s.shape, (60, 80))
        self.assertEqual(self.s.area, 72000000)


class Test(unittest.TestCase):
    """
    """

    def test_cartesian_to_polar(self):
        """
        """
        c = cartesian_to_polar(0, 0)
        self.assertEqual(c, (0., 0.))
        c = cartesian_to_polar(1, 0)
        self.assertEqual(c, (1.,  0.))
        c = cartesian_to_polar(0, 1)
        self.assertEqual(c, (1., 90.))
        c = cartesian_to_polar(1, 1)
        self.assertEqual(c, (np.sqrt(2), 45.))

    def test_polar_to_cartesian(self):
        """
        """
        c = polar_to_cartesian(0, 0)
        self.assertEqual(c, (0, 0))
        c = polar_to_cartesian(1, 0)
        self.assertEqual(c, (1, 0))
        c = polar_to_cartesian(1, 90)
        self.assertEqual(c, (0, 1))
        c = polar_to_cartesian(float(np.sqrt(2)), 45)
        self.assertEqual(round(c[0], 10), 1)
        self.assertEqual(round(c[1], 10), 1)
