"""
Tests for 'earth.geo' module.
"""


import unittest

import numpy as np

from synthacc.earth.geo import (Point, Path, is_lon, is_lat,
    are_coordinates, distance, project)


class TestPoint(unittest.TestCase):
    """
    """
    lon, lat, alt = 0, 1, 20.1
    p = Point(lon, lat, alt)

    def test_from_projection(self):
        """
        #TODO: implement test
        """
        pass

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.p[0], self.lon)
        self.assertEqual(self.p[1], self.lat)
        self.assertEqual(self.p[2], self.alt)
        lon, lat, alt = self.p
        self.assertEqual(lon, self.lon)
        self.assertEqual(lat, self.lat)
        self.assertEqual(alt, self.alt)

    def test___eq__(self):
        """
        """
        self.assertEqual(self.p, Point(0., 1., 20.1))
        self.assertNotEqual(self.p, Point(0., 1., 20.10001))

    def test_properties(self):
        """
        """
        self.assertEqual(self.p.lon, self.lon)
        self.assertEqual(self.p.lat, self.lat)
        self.assertEqual(self.p.alt, self.alt)
        self.assertEqual(self.p.depth, -self.alt)

    def test_get_geo_distance(self):
        """
        """
        self.assertEqual(self.p.get_geo_distance((0, 0)), distance(0, 0, 0, 1))

    def test_project(self):
        """
        #TODO: implement test
        """
        pass


class TestPath(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestSphericalEarth(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class Test(unittest.TestCase):
    """
    """

    def test_is_lon(self):
        """
        """
        self.assertTrue(is_lon(5.1))
        self.assertTrue(is_lon(-145.1))
        self.assertFalse(is_lon(180.1))
        self.assertTrue(is_lon(np.array([5.1, -145.1])))

    def test_is_lat(self):
        """
        """
        self.assertTrue(is_lat(1.2))
        self.assertTrue(is_lat(-89.3))
        self.assertFalse(is_lat(90.1))
        self.assertTrue(is_lat(np.array([1.2,  -89.3])))

    def test_are_coordinates(self):
        """
        """
        self.assertTrue(are_coordinates((0, 1)))
        self.assertTrue(are_coordinates((0, 1, 2)))
        self.assertFalse(are_coordinates((0, 1, '')))
        self.assertFalse(are_coordinates((0, 91, 2)))

    def test_distance(self):
        """
        Distances calculated with
        http://geographiclib.sourceforge.net/cgi-bin/GeodSolve
        """
        self.assertRaises(AssertionError, distance, '', 0, 0, 0)
        self.assertRaises(AssertionError, distance, 0, '', 0, 0)
        self.assertRaises(AssertionError, distance, 0, 0, '', 0)
        self.assertRaises(AssertionError, distance, 0, 0, 0, '')
        self.assertRaises(AssertionError, distance, 0, np.array([0]), 0, 0)
        self.assertRaises(AssertionError, distance, 0, 0, 0, np.array([0]))
        self.assertRaises(
            AssertionError, distance, np.array([0, 1]), np.array([0]), 0, 0)
        self.assertRaises(
            AssertionError, distance, 0, 0, np.array([0, 1]), np.array([0]))

        lon1, lat1 = -001.8494, +53.1472
        lon2, lat2 = +000.1406, +52.2044
        lon3, lat3 = +163.4539, +78.1692
        lon4, lat4 = -050.6923, -05.2369

        tgt_d12 = 00170648.997253939
        tgt_d13 = 05397860.320859759
        tgt_d14 = 07918708.570611600
        tgt_d23 = 05492562.133558812
        tgt_d24 = 07964495.675016765
        tgt_d34 = 11668218.434849779

        cal_d12 = round(distance(lon1, lat1, lon2, lat2), 8)
        cal_d34 = round(distance(lon3, lat3, lon4, lat4), 8)
        self.assertEqual(cal_d12, round(tgt_d12, 8))
        self.assertEqual(cal_d34, round(tgt_d34, 8))

        cal = distance(
            lon1,
            lat1,
            np.array([lon1, lon2]),
            np.array([lat1, lat2]),
            )
        tgt = np.array([0, tgt_d12])
        self.assertTrue(np.array_equal(np.round(cal, 8), np.round(tgt, 8)))

        cal = distance(
            np.array([lon1, lon2]),
            np.array([lat1, lat2]),
            np.array([lon1, lon3, lon4]),
            np.array([lat1, lat3, lat4]),
        )
        tgt = np.array([[0, tgt_d13, tgt_d14], [tgt_d12, tgt_d23, tgt_d24]])
        self.assertTrue(np.array_equal(np.round(cal, 8), np.round(tgt, 8)))

    def test_project(self):
        """
        #TODO: implement test
        """
        pass
