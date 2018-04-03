"""
Tests for 'source.ruptures' module.
"""


import unittest

from synthacc.source.ruptures import (Surface, PointRupture, SimpleRupture,
    KinematicRupture, SlipDistribution, GaussianACF, ExponentialACF,
    VonKarmanACF, RFSlipDistribution, RFSlipDistributionGenerator,
    FCSlipDistribution, FCSlipDistributionGenerator,
    MASlipDistributionGenerator)

from synthacc.earth.flat import RectangularSurface
from synthacc.source.mechanism import FocalMechanism


class TestSurface(unittest.TestCase):
    """
    """

    w = 6000
    l = 12000
    dw = 100
    dl = 150
    s = Surface(w, l, dw, dl)

    def test_properties(self):
        """
        """
        self.assertEqual(self.s.w, self.w)
        self.assertEqual(self.s.l, self.l)
        self.assertEqual(self.s.w, self.w)
        self.assertEqual(self.s.shape, (60, 80))
        self.assertEqual(self.s.area, 72000000)


class TestPointRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        r = PointRupture((0, 0, 5000), FocalMechanism(0, 90, 0), 1)


class TestSimpleRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        surface = RectangularSurface(0, 0, 10000, 0, 0, 5000, 90)
        r = SimpleRupture(surface, (0, 0, 5000), 0, 1)
