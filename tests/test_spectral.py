"""
Tests for 'spectral' module.
"""


import os
import unittest

import numpy as np

from synthacc.apy import PRECISION
from synthacc.spectral import DFT, AccDFT, FAS, FPS
from synthacc.io.esgmd2 import read_cor


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
        cal_acc = list(np.round(dft.inverse(acc.time_delta), PRECISION))
        tgt_acc = list(np.round(acc.amplitudes, PRECISION))
        self.assertListEqual(cal_acc[::20], tgt_acc[::20])


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
        self.fas.plot(png_filespec=fs)
