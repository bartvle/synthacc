"""
Tests for 'io.esgmd2' module.
"""


import os
import unittest

import numpy as np

from synthacc.io.esgmd2 import read_cor, read_fas, read_spc


DATA_DIR = os.path.join(
    os.path.dirname(__file__), 'data', 'esgmd2')

UNITS = {'dis': 'm', 'vel': 'm/s', 'acc': 'm/s2'}


class Test(unittest.TestCase):
    """
    """

    def test_read_cor(self):
        """
        """
        dis, vel, acc = read_cor(os.path.join(DATA_DIR, '004676xa.cor'))

        time_delta = 0.005
        dis_f_amplitudes = [
            +0.00000E+00, -8.04332E-09, -1.60866E-08, -2.41300E-08,
            -3.21733E-08, -4.02166E-08, -4.82599E-08, -5.63033E-08,
            -6.43466E-08, -7.23899E-08, -8.04333E-08, -8.84766E-08,
            -9.65200E-08, -1.04563E-07, -1.12607E-07, -1.20650E-07,
        ]
        dis_l_amplitudes = [
            -9.72654E-08, -8.77383E-08, -7.86358E-08, -6.99553E-08,
            -6.16943E-08, -5.38502E-08, -4.64206E-08, -3.94030E-08,
            -3.27948E-08, -2.65934E-08, -2.07964E-08, -1.54012E-08,
            -1.04053E-08, -5.80606E-09, -1.60096E-09,  2.21252E-09,
        ]
        vel_f_amplitudes = [
            -1.60866E-06, -1.60866E-06, -1.60866E-06, -1.60866E-06,
            -1.60867E-06, -1.60867E-06, -1.60867E-06, -1.60867E-06,
            -1.60867E-06, -1.60867E-06, -1.60867E-06, -1.60867E-06,
            -1.60867E-06, -1.60867E-06, -1.60867E-06, -1.60867E-06,
        ]
        vel_l_amplitudes = [
            +1.94801E-06, +1.86284E-06, +1.77817E-06, +1.69402E-06,
            +1.61039E-06, +1.52724E-06, +1.44460E-06, +1.36246E-06,
            +1.28084E-06, +1.19972E-06, +1.11909E-06, +1.03898E-06,
            +9.59383E-07, +8.80308E-07, +8.01735E-07, +7.23655E-07,
        ]
        acc_f_amplitudes = [
            -9.90240E-11, -9.91940E-11, -9.93590E-11, -9.95200E-11,
            -9.96760E-11, -9.98280E-11, -9.99760E-11, -1.00120E-10,
            -1.00260E-10, -1.00390E-10, -1.00520E-10, -1.00640E-10,
            -1.00760E-10, -1.00880E-10, -1.00990E-10, -1.01090E-10,
        ]
        acc_l_amplitudes = [
            -1.70820E-05, -1.69850E-05, -1.68840E-05, -1.67770E-05,
            -1.66760E-05, -1.65810E-05, -1.64780E-05, -1.63760E-05,
            -1.62730E-05, -1.61760E-05, -1.60740E-05, -1.59710E-05,
            -1.58670E-05, -1.57630E-05, -1.56660E-05, -1.55660E-05,
        ]

        self.assertEqual(dis.time_delta, time_delta)
        self.assertEqual(vel.time_delta, time_delta)
        self.assertEqual(acc.time_delta, time_delta)
        self.assertListEqual(list(dis.amplitudes[:+16]), dis_f_amplitudes)
        self.assertListEqual(list(dis.amplitudes[-16:]), dis_l_amplitudes)
        self.assertListEqual(list(vel.amplitudes[:+16]), vel_f_amplitudes)
        self.assertListEqual(list(vel.amplitudes[-16:]), vel_l_amplitudes)
        self.assertListEqual(list(acc.amplitudes[:+16]), acc_f_amplitudes)
        self.assertListEqual(list(acc.amplitudes[-16:]), acc_l_amplitudes)
        self.assertEqual(dis.unit, UNITS['dis'])
        self.assertEqual(vel.unit, UNITS['vel'])
        self.assertEqual(acc.unit, UNITS['acc'])

    def test_read_fas(self):
        """
        """
        fas = read_fas(os.path.join(DATA_DIR, '004676xa.fas'))

        f_freqs = [1.22070E-02, 2.44141E-02, 3.66211E-02]
        l_freqs = [9.99756E+01, 9.99878E+01, 1.00000E+02]
        f_ampls = [2.29340E-06, 3.22889E-06, 2.14292E-05]
        l_ampls = [2.76323E-06, 2.65286E-06, 2.41304E-06]

        self.assertListEqual(list(fas.frequencies[:+3]), f_freqs)
        self.assertListEqual(list(fas.frequencies[-3:]), l_freqs)
        self.assertListEqual(list(fas.amplitudes[:+3]), f_ampls)
        self.assertListEqual(list(fas.amplitudes[-3:]), l_ampls)
        self.assertEqual(fas.unit, UNITS['acc'])

    def test_read_spc(self):
        """
        """
        rs = read_spc(os.path.join(DATA_DIR, '004676xa.spc'))

        f_ps = np.array([4.000E-02, 4.200E-02, 4.400E-02])
        l_ps = np.array([2.300E+00, 2.400E+00, 2.500E+00])
        f_rs = [
            [
                [1.210E-03, 8.940E-04, 7.670E-04],
                [1.730E-01, 1.330E-01, 1.150E-01],
                [2.470E+01, 2.000E+01, 1.890E+01],
                [1.728E-01, 1.337E-01, 1.205E-01],
            ],
            [
                [5.350E-04, 3.890E-04, 2.940E-04],
                [7.120E-02, 5.170E-02, 3.290E-02],
                [1.090E+01, 8.730E+00, 7.250E+00],
                [7.640E-02, 5.819E-02, 4.618E-02],
            ],
            [
                [3.580E-04, 3.000E-04, 2.640E-04],
                [4.690E-02, 3.800E-02, 2.780E-02],
                [7.280E+00, 6.720E+00, 6.500E+00],
                [5.112E-02, 4.488E-02, 4.147E-02],
            ],
            [
                [2.850E-04, 2.560E-04, 2.360E-04],
                [3.250E-02, 2.850E-02, 2.390E-02],
                [5.880E+00, 5.800E+00, 5.840E+00],
                [4.070E-02, 3.830E-02, 3.707E-02],
            ],
            [
                [2.520E-04, 2.290E-04, 2.070E-04],
                [2.080E-02, 1.960E-02, 1.780E-02],
                [5.280E+00, 5.250E+00, 5.250E+00],
                [3.599E-02, 3.426E-02, 3.252E-02],
            ],
        ]
        l_rs = [
            [
                [8.660E-02, 9.550E-02, 1.020E-01],
                [2.800E-01, 3.100E-01, 3.260E-01],
                [5.470E-01, 6.550E-01, 7.620E-01],
                [2.176E-01, 2.500E-01, 2.786E-01],
            ],
            [
                [7.770E-02, 8.530E-02, 9.080E-02],
                [2.660E-01, 2.880E-01, 2.990E-01],
                [4.910E-01, 5.850E-01, 6.780E-01],
                [1.953E-01, 2.233E-01, 2.480E-01],
            ],
            [
                [6.740E-02, 7.310E-02, 7.740E-02],
                [2.460E-01, 2.600E-01, 2.660E-01],
                [4.310E-01, 5.050E-01, 5.820E-01],
                [1.694E-01, 1.914E-01, 2.114E-01],
            ],
            [
                [5.750E-02, 5.880E-02, 6.160E-02],
                [2.170E-01, 2.240E-01, 2.250E-01],
                [3.830E-01, 4.160E-01, 4.720E-01],
                [1.445E-01, 1.539E-01, 1.683E-01],
            ],
            [
                [4.470E-02, 4.350E-02, 4.290E-02],
                [1.740E-01, 1.740E-01, 1.710E-01],
                [3.250E-01, 3.380E-01, 3.700E-01],
                [1.123E-01, 1.139E-01, 1.172E-01],
            ],
        ]
        gmts = ('dis', 'vel', 'acc', 'vel')
        dampings = [0., 0.02, 0.05, 0.1, 0.2]
        self.assertEqual(len(rs), 5)
        for i, rs_per_d in enumerate(rs):
            self.assertEqual(len(rs_per_d), 4)
            for j, r in enumerate(rs_per_d):
                self.assertListEqual(list(r.periods[:+3]), list(f_ps))
                self.assertListEqual(list(r.periods[-3:]), list(l_ps))
                self.assertListEqual(list(r.responses[:+3]), f_rs[i][j][::-1])
                self.assertListEqual(list(r.responses[-3:]), l_rs[i][j][::-1])
                self.assertEqual(r.unit, UNITS[gmts[j]])
                self.assertEqual(r.damping, dampings[i])
