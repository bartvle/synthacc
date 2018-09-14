"""
Tests for 'source.rupture.slip' module.
"""


import unittest

import numpy as np

from synthacc.source.rupture.slip import (SlipDistribution, RandomFieldSD,
    RandomFieldSDC, MaiBeroza2002RFSDC, RandomFieldSDG, CompositeSourceSD,
    CompositeSourceSDC, CompositeSourceSDG, LiuEtAl2006NSRC,
    LiuArchuleta2004NSRC, RiseTimeCalculator, SlipRateCalculator)


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


class TestRandomFieldSD(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestRandomFieldSDC(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestMaiBeroza2002RFSDC(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestRandomFieldSDG(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestCompositeSourceSD(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestCompositeSourceSDC(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestCompositeSourceSDG(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestLiuEtAl2006NSRC(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TesTLiuArchuleta2004NSRC(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestRiseTimeCalculator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestSlipRateCalculator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass
