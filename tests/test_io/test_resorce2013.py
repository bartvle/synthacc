"""
Tests for 'io.resorce2013' module.
"""


import os
import unittest

import numpy as np

from synthacc.io.resorce2013 import read_acc, read_rs


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'resorce2013')


class Test(unittest.TestCase):
    """
    """

    def test_read_acc(self):
        """
        """
        acc = read_acc(os.path.join(DATA_DIR, '15279_V.cor.acc'))

        f_amplitudes = [
             1.1307000E-01,  3.2800000E-03, -9.9200000E-02, -8.7720000E-02,
             6.8530000E-02, -1.6225000E-01,  3.4440000E-01,  1.1594000E-01,
            -1.3037000E-01, -1.1941000E-01, -3.3900000E-02,  6.7750000E-02,
            -5.1100000E-03, -1.2273000E-01,  7.1140000E-02,  5.9760000E-02,
        ]
        l_amplitudes = [
            -4.2900000E-03, -2.1100000E-03, -3.0000000E-03, -2.8600000E-03,
            -3.8600000E-03, -3.2900000E-03, -3.6000000E-03, -3.3200000E-03,
            -3.6700000E-03, -4.2200000E-03, -4.3900000E-03, -4.2100000E-03,
            -4.2500000E-03, -4.2000000E-03, -4.2300000E-03, -4.2200000E-03,
        ]

        self.assertEqual(acc.time_delta, 0.02)
        self.assertListEqual(list(acc.amplitudes[:+16]), f_amplitudes)
        self.assertListEqual(list(acc.amplitudes[-16:]), l_amplitudes)
        self.assertEqual(acc.unit, 'm/s2')

    def test_read_rs(self):
        """
        """

        rs = read_rs(os.path.join(DATA_DIR, '15279_V.txt'))[1]

        periods = """
            0.000 0.010 0.020 0.030 0.040 0.050 0.075 0.100 0.110 0.120 0.130
            0.140 0.150 0.160 0.170 0.180 0.190 0.200 0.220 0.240 0.260 0.280
            0.300 0.320 0.340 0.360 0.380 0.400 0.420 0.440 0.460 0.480 0.500
            0.550 0.600 0.650 0.700 0.750 0.800 0.850 0.900 0.950 1.000 1.100
            1.200 1.300 1.400 1.500 1.600 1.700 1.800 1.900 2.000 2.200 2.400
            2.600 2.800 3.000 3.200 3.400 3.600 3.800 4.000 4.200 4.400 4.600
            4.800 5.000 5.500 6.000 6.500 7.000 7.500 8.000 8.500 9.000 9.500
            10.00
            """

        periods = list(np.array(periods.split(), dtype=float))

        f_responses = [
            3.9679000E-001, 4.0477622E-001, 4.0532070E-001, 5.5488199E-001,
            7.5385058E-001, 7.2152662E-001, 7.3033845E-001, 8.3194721E-001,
            7.8486401E-001, 9.5955211E-001, 1.0797322E+000, 1.2527586E+000,
            1.4042230E+000, 1.2237327E+000, 1.2319688E+000, 1.0519416E+000,
            ]

        l_responses = [
            4.2007200E-003, 3.7803600E-003, 3.5943200E-003, 3.4261000E-003,
            3.2630000E-003, 3.3139600E-003, 3.0978400E-003, 2.7881100E-003,
            2.1490700E-003, 1.9131700E-003, 1.7104300E-003, 1.5357800E-003,
            1.3851500E-003, 1.2545400E-003, 1.1407000E-003, 1.0410500E-003,
            ]

        self.assertListEqual(list(rs.periods), periods)
        self.assertListEqual(list(rs.responses[:+16]), f_responses)
        self.assertListEqual(list(rs.responses[-16:]), l_responses)
        self.assertEqual(rs.unit, 'm/s2')
        self.assertEqual(rs.damping, 0.05)
