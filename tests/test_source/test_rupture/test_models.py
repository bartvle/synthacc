"""
Tests for 'source.rupture.models' module.
"""


import unittest

import numpy as np

from synthacc.earth.flat import DiscretizedSimpleSurface
from synthacc.source.mechanism import FocalMechanism
from synthacc.source.moment import MomentRateFunction

from synthacc.source.rupture.models import (PointRupture, SimpleRupture,
    SimpleFiniteRupture, ComposedFiniteRupture)


class TestPointRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        r = PointRupture((0, 0, 5000), FocalMechanism(0, 90, 0), 1)
        self.assertEqual(r.nmrf, None)


class TestSimpleRupture(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestSimpleFiniteRupture(unittest.TestCase):
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
        r = SimpleFiniteRupture(surface, (0, 0, 5000), rake, 1, slip_rates)
        self.assertEqual(type(r.mrf), MomentRateFunction)
