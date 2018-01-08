"""
Tests for 'unit' module.
"""


import unittest

from synthacc.units import SI_PREFIXES, MOTION, Unit, make_motion_units


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
