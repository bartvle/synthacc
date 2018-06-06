"""
Tests for 'io.smsim' module.
"""


import unittest

import os

from synthacc.io.smsim import read_fas, read_response_spectrum


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'smsim')


class Test(unittest.TestCase):
    """
    """

    def test_read_fas(self):
        """
        """
        fass = read_fas(
            os.path.join(DATA_DIR, 'ab06_bc.m7.50r0030.0_fs.col'))
        fas = fass[0]
        self.assertEqual(len(fass), 3)
        self.assertEqual(fas.frequencies[0], 1.000E-02)
        self.assertEqual(fas.frequencies[1], 1.207E-02)
        self.assertEqual(fas.frequencies[-2], 8.286E+01)
        self.assertEqual(fas.frequencies[-1], 1.000E+02)
        self.assertEqual(fas.amplitudes[0], 1.03981E+02)
        self.assertEqual(fas.amplitudes[1], 1.03546E+02)
        self.assertEqual(fas.amplitudes[-2], 9.67976E-07)
        self.assertEqual(fas.amplitudes[-1], 2.09392E-07)
        self.assertEqual(fas.unit, 'cm')

    def test_read_response_spectrum(self):
        """
        """
        rss = read_response_spectrum(
            os.path.join(DATA_DIR, 'ab06_bc.m7.50r030.0_rs.rv.col'))
        rs = rss[0]
        self.assertEqual(len(rss), 3)
        self.assertEqual(rs.periods[0], 0.0100)
        self.assertEqual(rs.periods[1], 0.0121)
        self.assertEqual(rs.periods[-2], 082.8643)
        self.assertEqual(rs.periods[-1], 100.0000)
        self.assertEqual(rs.responses[0], 6.275E-04)
        self.assertEqual(rs.responses[1], 9.273E-04)
        self.assertEqual(rs.responses[-2], 2.482E+01)
        self.assertEqual(rs.responses[-1], 2.406E+01)
        self.assertEqual(rs.unit, 'cm')
        self.assertEqual(rs.damping, 0.05)
