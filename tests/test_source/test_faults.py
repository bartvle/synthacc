"""
Tests for 'source.faults' module.
"""


import unittest

from synthacc.source.faults import RIGIDITY, SimpleFault


class TestSimpleFault(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        upper_depth = 10000
        lower_depth = 20000
        dip = 60
        rigidity = 3.1 * 10**10
        upper_sd = 12000
        lower_sd = 18000
        sf = SimpleFault(0, 0, 10000, 0, upper_depth, lower_depth, dip,
            rigidity, upper_sd, lower_sd)
        self.assertEqual(sf.upper_depth, upper_depth)
        self.assertEqual(sf.lower_depth, lower_depth)
        self.assertEqual(sf.dip, dip)
        self.assertEqual(sf.rigidity, rigidity)
        self.assertEqual(sf.upper_sd, upper_sd)
        self.assertEqual(sf.lower_sd, lower_sd)

        sf = SimpleFault(0, 0, 10000, 0, upper_depth, lower_depth, dip)
        self.assertEqual(sf.rigidity, RIGIDITY)
        self.assertEqual(sf.upper_sd, upper_depth)
        self.assertEqual(sf.lower_sd, lower_depth)
