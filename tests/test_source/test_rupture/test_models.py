"""
Tests for 'source.rupture.models' module.
"""


import unittest

import numpy as np

from synthacc.earth.flat import DiscretizedSimpleSurface
from synthacc.source.mechanism import FocalMechanism
from synthacc.source.moment import MomentRateFunction

from synthacc.source.rupture.models import (PointRupture, FiniteRupture,
    KinematicRuptureCalculator, KinematicRuptureGenerator,
    KinematicRuptureCalculatorLogicTree)


class TestPointRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        r = PointRupture((0, 0, 5000), FocalMechanism(0, 90, 0), 1)
        self.assertEqual(r.nmrf, None)


class TestFiniteRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        surface = DiscretizedSimpleSurface(
            0, 0, 10000, 0, 0, 5000, 90, (10, 100))
        rake = np.zeros(surface.shape)
        slip_rates = np.zeros(surface.shape+(3,))
        slip_rates[:,:,1] = np.ones(surface.shape)
        r = FiniteRupture(surface, (0, 0, 5000), rake, 1, slip_rates)
        self.assertEqual(type(r.mrf), MomentRateFunction)


class TestKinematicRuptureCalculator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestKinematicRuptureGenerator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestKinematicRuptureCalculatorLogicTree(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass
