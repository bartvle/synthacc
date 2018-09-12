"""
Tests for 'earth.flat' module.
"""


import unittest

import numpy as np

from synthacc.apy import PRECISION
from synthacc import space2, space3
from synthacc.earth.flat import (Sites, Path, SimpleSurface,
    DiscretizedSimpleSurface, azimuth, is_azimuth, is_strike, is_dip)


class TestSites(unittest.TestCase):
    """
    """

    s = Sites([1, 2, 3], [4, 5, 6])

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.s), 3)
        self.assertListEqual(list(self.s.xs), [1, 2, 3])
        self.assertListEqual(list(self.s.ys), [4, 5, 6])

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.s[0], (1, 4))
        self.assertEqual(self.s[1], (2, 5))
        self.assertEqual(self.s[2], (3, 6))
        s1, s2, s3 = self.s
        self.assertEqual(s1, (1, 4))
        self.assertEqual(s2, (2, 5))
        self.assertEqual(s3, (3, 6))


class TestPath(unittest.TestCase):
    """
    """

    p = Path([0, 0, 1, 1], [0, 1, 1, 0])

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.length, 3)

    def test_get_simplified(self):
        """
        #TODO: This method does not yet work completely as it should.
        """
        p = Path([0, 0, 0, 0, 0], [0, 1, 2, 3, 4])
        p1 = p.get_simplified(n=1)
        self.assertEqual(len(p1), 2)


class TestSimpleSurface(unittest.TestCase):
    """
    """

    def test___contains__(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip = 0, 0, 10, 10, 10, 30, 45
        ss = SimpleSurface(x1, y1, x2, y2, ud, ld, dip)
        self.assertTrue(ss.lrc in ss)
        self.assertFalse((0, 0, 0) in ss)

    def test_properties(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip = 0, 0, 100, 0, 10, 30, 45

        ss = SimpleSurface(x1, y1, x2, y2, ud, ld, dip)
        self.assertEqual(ss.upper_depth, ud)
        self.assertEqual(ss.lower_depth, ld)
        self.assertEqual(ss.dip, dip)
        self.assertEqual(ss.ulc, (x1, y1, ud))
        self.assertEqual(ss.urc, (x2, y2, ud))
        self.assertEqual(ss.llc, (x1, ld-ud, ld))
        self.assertEqual(ss.lrc, (x2, ld-ud, ld))

        self.assertTrue(abs(ss.width - np.sqrt(2*(ld-ud)**2)) < 10**-PRECISION)
        self.assertEqual(ss.length, np.sqrt(x2**2+y2**2))
        self.assertAlmostEqual(ss.surface_width, ld-ud, PRECISION)
        self.assertEqual(ss.depth_range, ld-ud)

    def test_strike(self):
        """
        """
        ss = SimpleSurface(0, 0, +1, 0, 0, 1, 45)
        self.assertEqual(ss.strike,   0)

        ss = SimpleSurface(0, 0, +1, +1, 0, 1, 45)
        self.assertEqual(ss.strike,  45)

        ss = SimpleSurface(0, 0, 0, +1, 0, 1, 45)
        self.assertEqual(ss.strike,  90)

        ss = SimpleSurface(0, 0, -1, +1, 0, 1, 45)
        self.assertEqual(ss.strike, 135)

        ss = SimpleSurface(0, 0, -1, 0, 0, 1, 45)
        self.assertEqual(ss.strike, 180)

        ss = SimpleSurface(0, 0, -1, -1, 0, 1, 45)
        self.assertEqual(ss.strike, 225)

        ss = SimpleSurface(0, 0, 0, -1, 0, 1, 45)
        self.assertEqual(ss.strike, 270)

        ss = SimpleSurface(0, 0, +1, -1, 0, 1, 45)
        self.assertEqual(ss.strike, 315)

    def test_vectors(self):
        """
        """
        ss = SimpleSurface(0, 0, 100, 0, 5, 25, 45)

        asv = ss.as_vector
        self.assertEqual(asv.x, 100)
        self.assertEqual(asv.y, 0)
        self.assertEqual(asv.z, 0)

        adv = ss.ad_vector
        self.assertEqual(adv.x, 0)
        self.assertEqual(round(adv.y, PRECISION), 20)
        self.assertEqual(round(adv.z, PRECISION), 20)

    def test_plane(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip = 0, 0, 100, 0, 10, 30, 90
        ss = SimpleSurface(x1, y1, x2, y2, ud, ld, dip)
        p = ss.plane
        self.assertEqual(type(p), space3.Plane)
        self.assertEqual(p.get_distance(ss.lrc), 0)

    def test_surface(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip = 0, 0, 100, 0, 10, 30, 90
        ss = SimpleSurface(x1, y1, x2, y2, ud, ld, dip)
        s = ss.surface
        self.assertEqual(type(s), space2.RectangularSurface)
        self.assertEqual(s.w, ss.width)
        self.assertEqual(s.l, ss.length)

    def test_get_discretized(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip = 0, 0, 100, 0, 10, 30, 90
        ss = SimpleSurface(x1, y1, x2, y2, ud, ld, dip)

        dss = ss.get_discretized(shape=(20, 50))
        self.assertEqual(dss.shape, (20, 50))
        self.assertEqual(dss.spacing, (1, 2))


class TestDiscretizedSimpleSurface(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip, shape = 0, 0, 100, 0, 10, 30, 90, (20, 50)
        dss = DiscretizedSimpleSurface(x1, y1, x2, y2, ud, ld, dip, shape)
        self.assertEqual(dss.shape, shape)
        self.assertEqual(dss.spacing, (1, 2))
        self.assertEqual(dss.centers.shape, shape + (3,))
        self.assertEqual(dss.corners.shape, (shape[0] + 1, shape[1] + 1, 3))
        self.assertEqual(dss.cell_area, 2)

    def test_surface(self):
        """
        """
        x1, y1, x2, y2, ud, ld, dip, shape = 0, 0, 100, 0, 10, 30, 90, (20, 50)
        dss = DiscretizedSimpleSurface(x1, y1, x2, y2, ud, ld, dip, shape)
        ds = dss.surface
        self.assertEqual(type(ds), space2.DiscretizedRectangularSurface)
        self.assertEqual(ds.w, dss.width)
        self.assertEqual(ds.l, dss.length)
        self.assertEqual(ds.nw, shape[0])
        self.assertEqual(ds.nl, shape[1])


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

    def test_is_azimuth(self):
        """
        """
        self.assertTrue(is_azimuth(233))
        self.assertFalse(is_azimuth(-5))


    def test_is_dip(self):
        """
        """
        self.assertTrue(is_dip(60))
        self.assertFalse(is_dip(0))

    def test_is_strike(self):
        """
        """
        self.assertTrue(is_strike(233))
        self.assertFalse(is_strike(-5))
