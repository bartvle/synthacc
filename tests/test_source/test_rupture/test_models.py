"""
Tests for 'source.rupture.models' module.
"""


import unittest

from synthacc.source.rupture.models import (PointRupture, SimpleRupture,
    KinematicRupture, KinematicRuptureCalculator, KinematicRuptureGenerator)

from synthacc.earth.flat import Rectangle
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
        surface = Rectangle(0, 0, 10000, 0, 0, 5000, 90)
        r = SimpleRupture(surface, (0, 0, 5000), 0, 1)


# class TestGP2016KinematicRuptureGenerator(unittest.TestCase):
#     """
#     """

#     def test_properties(self):
#         """
#         """
#         surface = Rectangle(100, 200, 10100, 200, 0, 5000, 90)
#         krg = GP2016KRG(0.01, 2700)
#         kr = krg(surface, 0, 7, hypo=(9100, 200, 3000))
#         kr = krg(surface, 0, 7, hypo=None)
