"""
Tests for 'response' module.
"""


import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from synthacc.response import (ResponseSpectrum, NewmarkBetaCalculator,
    NigamJenningsCalculator, SpectralCalculator, frf, plot_response_spectra)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


## Test data from ESGMD2 004676xa

PERIODS = [
    4.000E-02, 4.200E-02, 4.400E-02, 4.600E-02, 4.800E-02, 5.000E-02,
    5.500E-02, 6.000E-02, 6.500E-02, 7.000E-02, 7.500E-02, 8.000E-02,
    8.500E-02, 9.000E-02, 9.500E-02, 1.000E-01, 1.100E-01, 1.200E-01,
    1.300E-01, 1.400E-01, 1.500E-01, 1.600E-01, 1.700E-01, 1.800E-01,
    1.900E-01, 2.000E-01, 2.200E-01, 2.400E-01, 2.600E-01, 2.800E-01,
    3.000E-01, 3.200E-01, 3.400E-01, 3.600E-01, 3.800E-01, 4.000E-01,
    4.200E-01, 4.400E-01, 4.600E-01, 4.800E-01, 5.000E-01, 5.500E-01,
    6.000E-01, 6.500E-01, 7.000E-01, 7.500E-01, 8.000E-01, 8.500E-01,
    9.000E-01, 9.500E-01, 1.000E+00, 1.100E+00, 1.200E+00, 1.300E+00,
    1.400E+00, 1.500E+00, 1.600E+00, 1.700E+00, 1.800E+00, 1.900E+00,
    2.000E+00, 2.100E+00, 2.200E+00, 2.300E+00, 2.400E+00, 2.500E+00,
    ]

RESPONSES = [
    6.500E+00, 6.720E+00, 7.280E+00, 6.760E+00, 7.090E+00, 8.490E+00,
    8.110E+00, 8.580E+00, 8.190E+00, 8.590E+00, 1.020E+01, 1.420E+01,
    1.590E+01, 1.510E+01, 1.610E+01, 1.460E+01, 1.370E+01, 1.190E+01,
    9.340E+00, 8.320E+00, 6.600E+00, 4.820E+00, 5.520E+00, 5.430E+00,
    5.240E+00, 4.310E+00, 3.810E+00, 3.990E+00, 3.600E+00, 3.110E+00,
    2.280E+00, 2.070E+00, 2.130E+00, 1.960E+00, 1.690E+00, 1.620E+00,
    1.510E+00, 1.680E+00, 1.870E+00, 2.010E+00, 1.930E+00, 1.720E+00,
    1.830E+00, 1.670E+00, 1.870E+00, 1.480E+00, 1.280E+00, 1.230E+00,
    9.550E-01, 7.310E-01, 7.000E-01, 7.530E-01, 7.760E-01, 9.540E-01,
    1.030E+00, 8.980E-01, 7.890E-01, 7.160E-01, 7.150E-01, 7.260E-01,
    7.200E-01, 6.940E-01, 6.470E-01, 5.820E-01, 5.050E-01, 4.310E-01,
]

DAMPING, UNIT = 0.05, 'm/s2'


class TestResponseSpectrum(unittest.TestCase):
    """
    """

    rs = ResponseSpectrum(PERIODS, RESPONSES, UNIT, DAMPING)

    def test_properties(self):
        """
        """
        self.assertListEqual(list(self.rs.periods), PERIODS)
        self.assertListEqual(list(self.rs.responses), RESPONSES)
        self.assertEqual(self.rs.unit, UNIT)
        self.assertEqual(self.rs.damping, DAMPING)

    def test_gmt(self):
        """
        """
        self.assertEqual(self.rs.gmt, 'acceleration')

    def test_has_pgm(self):
        """
        """
        self.assertFalse(self.rs.has_pgm())

    def test_get_max_response(self):
        """
        """
        self.assertEqual(self.rs.get_max_response(), max(RESPONSES))

    def test_plot(self):
        """
        """
        fs = os.path.join(OUTPUT_DIR, 'response.response_spectrum.plot.png')
        self.rs.plot(png_filespec=fs)


class Test(unittest.TestCase):
    """
    """

    def test_frf(self):
        """
        """
        frequencies = np.logspace(0, 2.4, 200)
        for i, gmt in enumerate(['dis', 'vel', 'acc']):
            responses = frf(frequencies, 100, 0.05, gmt)
            mgntds, angles = np.abs(responses), np.degrees(np.angle(responses))
            fig, axes = plt.subplots(2, sharex=True)
            axes[0].plot(frequencies, mgntds)
            axes[1].plot(frequencies, angles)
            axes[0].grid()
            axes[1].grid()
            axes[0].set_title('Amplitude')
            axes[1].set_title('Phase')

            plt.savefig(os.path.join(OUTPUT_DIR, 'response.frf.%s.png' % gmt))
