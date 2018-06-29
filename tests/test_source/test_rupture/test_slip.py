"""
Tests for 'source.rupture.slip' module.
"""


import unittest

import numpy as np

from synthacc.source.rupture.slip import (SlipDistribution, RFSlipDistribution,
    RFSlipDistributionCalculator, MaiBeroza2002RFSDC,
    RFSlipDistributionGenerator, CSSlipDistribution,
    CSSlipDistributionCalculator, CSSlipDistributionGenerator,
    LiuEtAl2006NormalizedSlipRateCalculator, GP2010SlipRateCalculator,
    GP2010SlipRateGenerator)


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
