"""
Tests for 'spectral' module.
"""


import os
import unittest

import numpy as np

from synthacc.apy import PRECISION
from synthacc.units import round_to_significant
from synthacc.io.esgmd2 import read_cor

from synthacc.spectral import DFT, AccDFT, FAS, FPS, fft, ifft, plot_fass


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'spectral')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

with open(os.path.join(DATA_DIR, 'frequencies.txt'), 'r') as f:
    FREQUENCIES = list(np.array(f.read().split(), dtype=float))

with open(os.path.join(DATA_DIR, 'amplitudes.txt'), 'r') as f:
    AMPLITUDES = list(np.array(f.read().split(), dtype=float))

UNIT = 'm/s2'


class TestDFT(unittest.TestCase):
    """
    """

    def test_inverse(self):
        """
        """
        acc = read_cor(os.path.join(DATA_DIR, '004676xa.cor'))[-1]
        dft = acc.get_dft()
        cal_acc = dft.inverse(acc.time_delta)
        tgt_acc = acc.amplitudes
        for cal, tgt in zip(cal_acc[::20], tgt_acc[::20]):
            self.assertTrue(abs(cal - tgt) < 10**-PRECISION)


class TestAccDFT(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestFAS(unittest.TestCase):
    """
    """

    fas = FAS(FREQUENCIES, AMPLITUDES, UNIT) ## ESGMD2 004676xa

    def test_properties(self):
        """
        """
        self.assertListEqual(list(self.fas.frequencies), FREQUENCIES)
        self.assertListEqual(list(self.fas.amplitudes), AMPLITUDES)
        self.assertEqual(self.fas.unit, UNIT)

    def test_gmt(self):
        """
        """
        self.assertEqual(self.fas.gmt, 'acceleration')

    def test_plot(self):
        """
        """
        fs = os.path.join(OUTPUT_DIR, 'spectral.plot.png')
        self.fas.plot(filespec=fs)


class TestFPS(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class Test(unittest.TestCase):
    """
    """

    def test_fft(self):
        """
        #TODO: implement test
        """
        pass

    def test_ifft(self):
        """
        #TODO: implement test
        """
        pass

    def test_plot_fass(self):
        """
        #TODO: implement test
        """
        pass
