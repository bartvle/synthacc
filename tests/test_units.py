"""
Tests for 'unit' module.
"""


import unittest

from synthacc.units import (SI_PREFIXES, MOTION, Unit, make_motion_units,
    round_to_significant)


class TestUnit(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        dm = Unit('dm', 0.10, 'length')
        self.assertEqual(dm.symbol, 'dm')
        self.assertEqual(dm.si_scale, 0.10)
        self.assertEqual(dm.quantity, 'length')

    def test___truediv__(self):
        """
        """
        dm = Unit('dm', 0.10, 'length')
        cm = Unit('cm', 0.01, 'length',)
        self.assertEqual(dm / cm, 10.)


class Test(unittest.TestCase):
    """
    """

    def test_motion(self):
        """
        """
        self.assertIn('mm', MOTION)
        self.assertEqual(MOTION['mm'].symbol, 'mm')
        self.assertEqual(MOTION['mm'].si_scale, 0.001)
        self.assertEqual(MOTION['mm'].quantity, 'displacement')

    def test_round_to_significant(self):
        """
        """
        x = 12.345
        self.assertEqual(round_to_significant(x, 1), 10)
        self.assertEqual(round_to_significant(x, 2), 12)
        self.assertEqual(round_to_significant(x, 3), 12.3)
        self.assertEqual(round_to_significant(x, 4), 12.35)
        self.assertEqual(round_to_significant(x, 5), 12.345)
        self.assertEqual(type(round_to_significant(x, 1)), int)
        self.assertEqual(type(round_to_significant(x, 2)), int)
        self.assertEqual(type(round_to_significant(x, 3)), float)
        self.assertEqual(type(round_to_significant(x, 4)), float)
        self.assertEqual(type(round_to_significant(x, 5)), float)
