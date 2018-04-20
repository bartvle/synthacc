"""
Tests for 'source.ruptures' module.
"""


import unittest

import numpy as np

from synthacc.source.rupture.models import (PointRupture, SimpleRupture,
    KinematicRupture, SlipDistribution, RFSlipDistribution,
    RFSlipDistributionGenerator, FCSlipDistribution,
    FCSlipDistributionGenerator, LiuEtAl2006NormalizedSlipRateGenerator,
    GP2016KinematicRuptureGenerator)

from synthacc.earth.flat import RectangularSurface
from synthacc.source.mechanism import FocalMechanism


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


class TestSlipDistribution(unittest.TestCase):
    """
    """

    w = 6000
    l = 12000
    dw = 100
    dl = 150
    s = SlipDistribution(w, l, np.random.random((60, 80)))

    def test_properties(self):
        """
        """
        self.assertEqual(self.s.w, self.w)
        self.assertEqual(self.s.l, self.l)
        self.assertEqual(self.s.dw, self.dw)
        self.assertEqual(self.s.dl, self.dl)
        self.assertEqual(self.s.shape, (60, 80))
        self.assertEqual(self.s.area, 72000000)

    def test_surface(self):
        """
        """
        s = self.s.surface
        self.assertEqual(s.w, self.w)
        self.assertEqual(s.l, self.l)
        self.assertEqual(s.dw, self.dw)
        self.assertEqual(s.dl, self.dl)
        self.assertEqual(s.shape, (60, 80))


class TestGP2016KinematicRuptureGenerator(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        surface = RectangularSurface(0, 0, 10000, 0, 0, 5000, 90)
        krg = GP2016KinematicRuptureGenerator(0.01, 2700)
        kr = krg(surface, 0, 7)
