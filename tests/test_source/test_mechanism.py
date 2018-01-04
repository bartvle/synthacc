"""
Tests for 'source.mechanism' module.
"""


import unittest

from synthacc.source.mechanism import NodalPlane, FocalMechanism, is_rake


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
