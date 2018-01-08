"""
Tests for 'earth.geo' module.
"""


import unittest

import numpy as np

from synthacc.earth.geo import Path, SphericalEarth, is_lon, is_lat


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
