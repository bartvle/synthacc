"""
Tests for 'earth.flat' module.
"""


import unittest

from synthacc.apy import PRECISION
from synthacc.earth.flat import (Sites, Grid, Path, RectangularSurface,
    DiscretizedRectangularSurface, azimuth, is_azimuth, is_strike, is_dip)


class TestSites(unittest.TestCase):
    """
    """

    s = Sites([1, 2, 3], [4, 5, 6])

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.s), 3)


class TestGrid(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        drs = Grid((0, 50, -10, 10), 1)


class TestPath(unittest.TestCase):
    """
    """

    p = Path([0, 0, 1, 1], [0, 1, 1, 0])

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.length, 3)


class TestRectangularSurface(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        rs = RectangularSurface(0, 0, 100, 0, 10, 25, 60)

    def test_strike(self):
        """
        """
        rs = RectangularSurface(0, 0, +1, 0, 0, 1, 45)
        self.assertEqual(rs.strike,   0)

        rs = RectangularSurface(0, 0, +1, +1, 0, 1, 45)
        self.assertEqual(rs.strike,  45)

        rs = RectangularSurface(0, 0, 0, +1, 0, 1, 45)
        self.assertEqual(rs.strike,  90)

        rs = RectangularSurface(0, 0, -1, +1, 0, 1, 45)
        self.assertEqual(rs.strike, 135)

        rs = RectangularSurface(0, 0, -1, 0, 0, 1, 45)
        self.assertEqual(rs.strike, 180)

        rs = RectangularSurface(0, 0, -1, -1, 0, 1, 45)
        self.assertEqual(rs.strike, 225)

        rs = RectangularSurface(0, 0, 0, -1, 0, 1, 45)
        self.assertEqual(rs.strike, 270)

        rs = RectangularSurface(0, 0, +1, -1, 0, 1, 45)
        self.assertEqual(rs.strike, 315)

    def test_vectors(self):
        """
        """
        rs = RectangularSurface(0, 0, 100, 0, 5, 25, 45)

        asv = rs.as_vector
        self.assertEqual(asv.x, 100)
        self.assertEqual(asv.y, 0)
        self.assertEqual(asv.z, 0)

        adv = rs.ad_vector
        self.assertEqual(adv.x, 0)
        self.assertEqual(round(adv.y, PRECISION), 20)
        self.assertEqual(round(adv.z, PRECISION), 20)


class TestDiscretizedRectangularSurface(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        drs = RectangularSurface(0, 0, 100, 0, 10, 25, 60, 1)


class Test(unittest.TestCase):
    """
    """

    def test_azimuth(self):
        """
        """
        self.assertEqual(azimuth(0, 0, +1, 0) ,   0)
        self.assertEqual(azimuth(0, 0, +1, +1),  45)
        self.assertEqual(azimuth(0, 0, 0, +1) ,  90)
        self.assertEqual(azimuth(0, 0, -1, +1), 135)
        self.assertEqual(azimuth(0, 0, -1, 0) , 180)
        self.assertEqual(azimuth(0, 0, -1, -1), 225)
        self.assertEqual(azimuth(0, 0, 0, -1) , 270)
        self.assertEqual(azimuth(0, 0, +1, -1), 315)

    def test_is_dip(self):
        """
        """
        is_dip(60)

    def test_is_strike(self):
        """
        """
        is_strike(233)
