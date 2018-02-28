"""
Tests for 'io.scardec' module.
"""


import unittest

import os

from synthacc.io.scardec import read_moment_rate_function


DATA_DIR = os.path.join(
    os.path.dirname(__file__), 'data', 'scardec')


class Test(unittest.TestCase):
    """
    """

    def test_read_moment_rate_function(self):
        """
        """

        f_moment_rates = [
            0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 9.71781985E+14,
            3.10969430E+15, 6.48000238E+15, 1.00246920E+16, 1.33776087E+16,
            1.62696604E+16, 1.85934460E+16, 2.09084677E+16, 2.44649154E+16,
            3.45577364E+16, 5.07143425E+16, 8.50489166E+16, 1.27194649E+17,
        ]

        l_moment_rates = [
            2.09413715E+16, 1.92891641E+16 ,1.78497595E+16, 1.66070344E+16,
            1.55240476E+16, 1.45501616E+16, 1.36388168E+16, 1.27643357E+16,
            1.19365592E+16, 1.12000463E+16, 0.00000000E+00, 0.00000000E+00,
            0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00,
        ]

        mrf = read_moment_rate_function(os.path.join(DATA_DIR,
            'fctmoysource_20110312_014715_OFF_EAST_COAST_OF_HONSHU__JAPAN'))

        self.assertEqual(mrf.time_delta, 7.03125000*10**-2)
        self.assertEqual(len(mrf.rates), 245)
        self.assertListEqual(list(mrf.rates[:+16]), f_moment_rates)
        self.assertListEqual(list(mrf.rates[-16:]), l_moment_rates)
        self.assertEqual(mrf.start_time, -1.1250000)
