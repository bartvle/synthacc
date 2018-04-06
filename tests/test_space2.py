"""
Tests for 'space2' module.
"""


import unittest


from synthacc.space2 import (DiscretizedRectangularSurface, GaussianACF,
    ExponentialACF, VonKarmanACF, SpatialRandomFieldGenerator, distance)


class TestDiscretizedRectangularSurface(unittest.TestCase):
    """
    """

    w = 6000
    l = 12000
    dw = 100
    dl = 150
    s = DiscretizedRectangularSurface(w, l, dw, dl)

    def test_properties(self):
        """
        """
        self.assertEqual(self.s.w, self.w)
        self.assertEqual(self.s.l, self.l)
        self.assertEqual(self.s.dw, self.dw)
        self.assertEqual(self.s.dl, self.dl)
        self.assertEqual(self.s.shape, (60, 80))
        self.assertEqual(self.s.area, 72000000)
