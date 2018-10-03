"""
Tests for 'earth.flat' module.
"""


import os
import unittest

import numpy as np

from synthacc.apy import PRECISION
from synthacc import space2, space3
from synthacc.earth.flat import (Sites, Path, SimpleSurface,
    DiscretizedSimpleSurface, azimuth, is_azimuth, is_strike, is_dip,
    plot_rectangles)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


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

    def test___iter__(self):
        """
        """
        sites = [s for s in self.s]
        self.assertEqual(len(sites), 3)
        self.assertEqual(sites[0], (1, 4))
        self.assertEqual(sites[1], (2, 5))
        self.assertEqual(sites[2], (3, 6))

    def test_from_points(self):
        """
        """
        p1, p2 = (0, 1), (2, 3, 4)
        p3, p4 = (5, 6), (7, 8, 9)
        p3 = space2.Point(*p3)
        p4 = space3.Point(*p4)
        s = Sites.from_points([p1, p2, p3, p4])
        self.assertEqual(type(s), Sites)
        self.assertListEqual(list(s.xs), [0, 2, 5, 7])
        self.assertListEqual(list(s.ys), [1, 3, 6, 8])


class TestPath(unittest.TestCase):
    """
    """

    p = Path([0, 0, 1, 1], [0, 1, 1, 0])

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.length, 3)

    def test_from_points(self):
        """
        """
        p1, p2 = (0, 1), (2, 3, 4)
        p3, p4 = (5, 6), (7, 8, 9)
        p3 = space2.Point(*p3)
        p4 = space3.Point(*p4)
        p = Path.from_points([p1, p2, p3, p4])
        self.assertEqual(type(p), Path)
        self.assertListEqual(list(p.xs), [0, 2, 5, 7])
        self.assertListEqual(list(p.ys), [1, 3, 6, 8])

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

    def test_as_vector(self):
        """
        """
        ss = SimpleSurface(0, 0, 10, 0, 5, 25, 45)
        asv = ss.as_vector
        self.assertEqual(asv.x, 10)
        self.assertEqual(asv.y, 0)
        self.assertEqual(asv.z, 0)

        ss = SimpleSurface(0, 0, 0, -10, 5, 25, 45)
        asv = ss.as_vector
        self.assertEqual(asv.x, 0)
        self.assertEqual(asv.y, -10)
        self.assertEqual(asv.z, 0)

    def test_ad_vector(self):
        """
        """
        ss = SimpleSurface(0, 0, 10, 0, 5, 25, 45)
        adv = ss.ad_vector
        self.assertEqual(adv.x, 0)
        self.assertEqual(round(adv.y, PRECISION), 20)
        self.assertEqual(round(adv.z, PRECISION), 20)

        ss = SimpleSurface(0, 0, 0, -10, 5, 25, 45)
        adv = ss.ad_vector
        self.assertEqual(round(adv.x, PRECISION), 20)
        self.assertEqual(adv.y, 0)
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
        self.assertEqual(dss.cell_centers.shape, shape + (3,))
        self.assertEqual(dss.cell_corners.shape,
            (shape[0] + 1, shape[1] + 1, 3))
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

    def test_plot_rectangles(self):
        """
        """
        upper_depth, lower_depth, dip = 0, 10, 45

        p1 = (-5, +10)
        p2 = (+5, +10)
        p3 = (+10, +5)
        p4 = (+10, -5) 
        p5 = (+5, -10)
        p6 = (-5, -10)
        p7 = (-10, -5)
        p8 = (-10, +5)

        s1 = SimpleSurface(*p1, *p2, upper_depth, lower_depth, dip)
        s2 = SimpleSurface(*p2, *p3, upper_depth, lower_depth, dip)
        s3 = SimpleSurface(*p3, *p4, upper_depth, lower_depth, dip)
        s4 = SimpleSurface(*p4, *p5, upper_depth, lower_depth, dip)
        s5 = SimpleSurface(*p5, *p6, upper_depth, lower_depth, dip)
        s6 = SimpleSurface(*p6, *p7, upper_depth, lower_depth, dip)
        s7 = SimpleSurface(*p7, *p8, upper_depth, lower_depth, dip)
        s8 = SimpleSurface(*p8, *p1, upper_depth, lower_depth, dip)

        fs = os.path.join(OUTPUT_DIR, 'earth.flat.plot_rectangles.png')

        plot_rectangles([s1, s2, s3, s4, s5, s6, s7, s8], filespec=fs)
