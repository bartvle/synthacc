"""
Tests for 'source.mechanism' module.
"""


import unittest

from synthacc.source.mechanism import NodalPlane, FocalMechanism, is_rake


STRIKE, DIP, RAKE = 143., 68., -87.
MOMENT = 1.32*10**17 ## in Nm


class TestNodalPlane(unittest.TestCase):
    """
    """

    np = NodalPlane(STRIKE, DIP, RAKE)

    def test_properties(self):
        """
        """
        self.assertEqual(self.np.strike, STRIKE)
        self.assertEqual(self.np.dip, DIP)
        self.assertEqual(self.np.rake, RAKE)

    def test___getitem__(self):
        """
        """
        strike, dip, rake = self.np
        self.assertEqual(strike, STRIKE)
        self.assertEqual(dip, DIP)
        self.assertEqual(rake, RAKE)


class TestFocalMechanism(unittest.TestCase):
    """
    """

    fm = FocalMechanism(0, 90, -90)

    def test_normal_vector(self):
        """
        """
        nv = self.fm.normal_vector
        self.assertEqual(nv.x, 0)
        self.assertEqual(nv.y, 1)
        self.assertEqual(nv.z, 0)

    def test_slip_vector(self):
        """
        """
        sv = self.fm.slip_vector
        self.assertEqual(sv.x, 0)
        self.assertEqual(sv.y, 0)
        self.assertEqual(sv.z, 1)


class Test(unittest.TestCase):
    """
    """

    def test_is_rake(self):
        """
        """
        self.assertTrue(is_rake(90))
        self.assertFalse(is_rake(270))
