"""
Tests for 'source.faults' module.
"""


import unittest

from synthacc.source.faults import (RIGIDITY, SimpleFault, ComposedFault,
    FaultGeometryCalculator)


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


class TestComposedFault(unittest.TestCase):
    """
    """

    p1 = (25000, 25000)
    p2 = (50000, 25000)
    p3 = (75000, 50000)
    p4 = (75000, 75000)

    cf = ComposedFault([p1, p2, p3, p4], 0, 25000, 45)

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.cf), 3)
        sf1, sf2, sf3 = self.cf
        self.assertEqual(sf1.ulc[:2], self.p1)
        self.assertEqual(sf1.urc[:2], self.p2)
        self.assertEqual(sf2.ulc[:2], self.p2)
        self.assertEqual(sf2.urc[:2], self.p3)
        self.assertEqual(sf3.ulc[:2], self.p3)
        self.assertEqual(sf3.urc[:2], self.p4)


class TestFaultGeometryCalculator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass
