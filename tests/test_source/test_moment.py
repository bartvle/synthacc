"""
Tests for 'source.moment' module.
"""


import unittest

from synthacc.source.moment import (MomentTensor, SlipFunction, MomentFunction,
    NormalizedSlipFunction, NormalizedMomentFunction, SlipRateFunction,
    MomentRateFunction, NormalizedSlipRateFunction,
    NormalizedMomentRateFunction, InstantRateGenerator, ConstantRateGenerator,
    TriangularRateGenerator, calculate, m0_to_mw, mw_to_m0)


from synthacc.units import round_to_significant


class TestMomentTensor(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        mt = MomentTensor(1, 2, 3, 4, 5, 6)
        self.assertEqual(mt.xx, 1)
        self.assertEqual(mt.yy, 2)
        self.assertEqual(mt.zz, 3)
        self.assertEqual(mt.xy, 4)
        self.assertEqual(mt.yz, 5)
        self.assertEqual(mt.zx, 6)
        self.assertEqual(mt.six, (1, 2, 3, 4, 5, 6))
        self.assertEqual(mt.trace, 6)

    def test___add__(self):
        """
        """
        m1 = MomentTensor(+1.1, -2.2, +3.3, -4.4, +5.5, -6.6)
        m2 = MomentTensor(-9.9, -8.8, +7.7, +6.6, -5.5, -4.4)
        m = m1 + m2

        self.assertEqual(m.xx, +1.1 - 9.9)
        self.assertEqual(m.yy, -2.2 - 8.8)
        self.assertEqual(m.zz, +3.3 + 7.7)
        self.assertEqual(m.xy, -4.4 + 6.6)
        self.assertEqual(m.yz, +5.5 - 5.5)
        self.assertEqual(m.zx, -6.6 - 4.4)

    def test___mul__(self):
        """
        """
        for f in [-7.7, 7.7]:
            m = MomentTensor(+1.1, -2.2, +3.3, -4.4, +5.5, -6.6)
            m = m * f

            self.assertEqual(m.xx, +1.1 * f)
            self.assertEqual(m.yy, -2.2 * f)
            self.assertEqual(m.zz, +3.3 * f)
            self.assertEqual(m.xy, -4.4 * f)
            self.assertEqual(m.yz, +5.5 * f)
            self.assertEqual(m.zx, -6.6 * f)


class TestNormalizedSlipFunction(unittest.TestCase):
    """
    """

    nsf = NormalizedSlipFunction(0.1, [0, 0.2, 0.4, 0.6, 0.8, 1])

    def test___mul__(self):
        """
        """
        sf = self.nsf * 2
        self.assertEqual(type(sf), SlipFunction)
        self.assertEqual(sf.slip, 2)


class TestNormalizedMomentFunction(unittest.TestCase):
    """
    """

    nmf = NormalizedMomentFunction(0.1, [0, 0.2, 0.4, 0.6, 0.8, 1])

    def test___mul__(self):
        """
        """
        mf = self.nmf * 2
        self.assertEqual(type(mf), MomentFunction)
        self.assertEqual(mf.moment, 2)


class TestNormalizedSlipRateFunction(unittest.TestCase):
    """
    """

    nsrf = NormalizedSlipRateFunction(0.1, [0, 2, 2, 2, 2, 2, 0])

    def test___mul__(self):
        """
        """
        srf = self.nsrf * 2
        self.assertEqual(type(srf), SlipRateFunction)
        self.assertEqual(srf.slip, 2)


class TestNormalizedMomentRateFunction(unittest.TestCase):
    """
    """

    nmrf = NormalizedMomentRateFunction(0.1, [0, 2, 2, 2, 2, 2, 0])

    def test___mul__(self):
        """
        """
        mrf = self.nmrf * 2
        self.assertEqual(type(mrf), MomentRateFunction)
        self.assertEqual(mrf.moment, 2)


class Test(unittest.TestCase):
    """
    """

    m0, mw = 1.1*10**17, 5.3

    def test_m0_to_mw(self):
        """
        """
        self.assertEqual(round_to_significant(m0_to_mw(self.m0), 2),
            round_to_significant(self.mw, 2))

    def test_mw_to_m0(self):
        """
        """
        self.assertEqual(round_to_significant(mw_to_m0(self.mw), 2),
            round_to_significant(self.m0, 2))
