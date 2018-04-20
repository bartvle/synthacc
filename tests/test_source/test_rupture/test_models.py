"""
Tests for 'source.rupture.models' module.
"""


import unittest

from synthacc.source.rupture.models import (PointRupture, SimpleRupture,
    KinematicRupture, GP2016KinematicRuptureGenerator)

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


class TestGP2016KinematicRuptureGenerator(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        surface = RectangularSurface(0, 0, 10000, 0, 0, 5000, 90)
        krg = GP2016KinematicRuptureGenerator(0.01, 2700)
        kr = krg(surface, 0, 7)
