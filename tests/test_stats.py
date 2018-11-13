"""
Tests for 'stats' module.
"""


import unittest

from synthacc.stats import (GaussianACF, ExponentialACF, VonKarmanACF,
    SpatialRandomFieldGenerator)


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
