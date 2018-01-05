"""
Tests for 'recordings' module.
"""


import os
import unittest

import numpy as np

from synthacc.spectral import DFT, AccDFT, plot_fass
from synthacc.response import ResponseSpectrum, plot_response_spectra
from synthacc.io.esgmd2 import read_fas, read_spc
from synthacc.io.resorce2013 import read_acc, read_rs

from synthacc.recordings import (Waveform, Seismogram, Accelerogram,
    Recording, ne_to_rt, rt_to_ne, plot_seismograms, plot_recordings)


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'recordings')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

TIME_DELTA = 0.005

with open(os.path.join(DATA_DIR, 'amplitudes.txt'), 'r') as f:
    AMPLITUDES = list(np.array(f.read().split(), dtype=float))

UNIT = 'm/s2'

PGM = PGA = 3.8996000


class TestSeismogram(unittest.TestCase):
    """
    """

    s = Seismogram(TIME_DELTA, AMPLITUDES, UNIT) ## ESGMD2 004676xa

    def test_properties(self):
        """
        """
        self.assertEqual(self.s.time_delta, TIME_DELTA)
        self.assertListEqual(list(self.s.amplitudes), AMPLITUDES)
        self.assertEqual(self.s.unit, UNIT)

        self.assertEqual(self.s.pgm, PGM)
        self.assertEqual(self.s.gmt, 'acceleration')

    def test_get_pgm(self):
        """
        """
        self.assertEqual(self.s.get_pgm(UNIT), PGM)
        self.assertEqual(self.s.get_pgm(None), PGM)

    def test_get_dft(self):
        """
        """
        cal_dft = self.s.get_dft('m/s2')
        self.assertEqual(type(cal_dft), DFT)
        cal_fas = cal_dft.fas
        tgt_fas = read_fas(os.path.join(DATA_DIR, '004676xa.fas'))
        fs = os.path.join(OUTPUT_DIR, 'recordings.seismogram.dft.png')
        plot_fass([tgt_fas, cal_fas], labels=['tgt', 'cal'], colors=['r', 'b'],
            widths=[3, 1], title='dft calculation', png_filespec=fs)


class TestAccelerogram(unittest.TestCase):
    """
    """

    acc = Accelerogram(TIME_DELTA, AMPLITUDES, UNIT) ## ESGMD2 004676xa

    def test_properties(self):
        """
        """
        self.assertEqual(self.acc.time_delta, TIME_DELTA)
        self.assertListEqual(list(self.acc.amplitudes), AMPLITUDES)
        self.assertEqual(self.acc.unit, UNIT)

        self.assertEqual(self.acc.pga, PGA)
        self.assertEqual(self.acc.gmt, 'acceleration')

    def test_from_seismogram(self):
        """
        """
        s = Seismogram(TIME_DELTA, AMPLITUDES, UNIT)
        acc = Accelerogram.from_seismogram(s)
        self.assertEqual(type(acc), Accelerogram)
        self.assertEqual(acc.time_delta, TIME_DELTA)
        self.assertListEqual(list(acc.amplitudes), AMPLITUDES)
        self.assertEqual(self.acc.unit, UNIT)

    def test_get_pga(self):
        """
        """
        self.assertEqual(self.acc.get_pga(UNIT), PGA)
        self.assertEqual(self.acc.get_pga(None), PGA)

    def test_get_dft(self):
        """
        """
        cal_dft = self.acc.get_dft('m/s2')
        self.assertEqual(type(cal_dft), AccDFT)
        cal_fas = cal_dft.fas
        tgt_fas = read_fas(os.path.join(DATA_DIR, '004676xa.fas'))
        fs = os.path.join(OUTPUT_DIR, 'recordings.accelerogram.dft.png')
        plot_fass([tgt_fas, cal_fas], labels=['tgt', 'cal'], colors=['r', 'b'],
            widths=[2, 0.5], title='dft calculation', png_filespec=fs)

    def test_get_response_spectrum_1(self):
        """
        """
        tgt_rdis_rs, tgt_rvel_rs, tgt_aacc_rs, tgt_pvel_rs = read_spc(
            os.path.join(DATA_DIR, '004676xa.spc'))[2]
        periods = tgt_rdis_rs.periods
        cal_rdis_rs = self.acc.get_response_spectrum(
            periods, damping=0.05, gmt='dis')
        cal_rvel_rs = self.acc.get_response_spectrum(
            periods, damping=0.05, gmt='vel')
        cal_aacc_rs = self.acc.get_response_spectrum(
            periods, damping=0.05, gmt='acc')
        cal_pvel_rs = ResponseSpectrum(periods,
            cal_rdis_rs.responses * 2*np.pi/periods, unit='m/s', damping=0.05)

        labels, colors, widths = ['tgt', 'cal'], ['r', 'b'], [3, 1]

        fs1 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.1.rdis.png')
        fs2 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.1.rvel.png')
        fs3 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.1.aacc.png')
        fs4 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.1.pvel.png')
        plot_response_spectra([tgt_rdis_rs, cal_rdis_rs],
            labels=labels, colors=colors, widths=widths, png_filespec=fs1)
        plot_response_spectra([tgt_rvel_rs, cal_rvel_rs],
            labels=labels, colors=colors, widths=widths, png_filespec=fs2)
        plot_response_spectra([tgt_aacc_rs, cal_aacc_rs],
            labels=labels, colors=colors, widths=widths, png_filespec=fs3)
        plot_response_spectra([tgt_pvel_rs, cal_pvel_rs],
            labels=labels, colors=colors, widths=widths, png_filespec=fs4)

    def test_get_response_spectrum_2(self):
        """
        """
        rss = read_rs(os.path.join(DATA_DIR, '15279_V.txt'))
        tgt_02_rs, tgt_05_rs, tgt_07_rs, tgt_10_rs, tgt_20_rs, tgt_30_rs = rss
        periods = tgt_02_rs.periods
        acc = read_acc(os.path.join(DATA_DIR, '15279_V.cor.acc'))
        cal_02_rs = acc.get_response_spectrum(
            periods, damping=0.02, gmt='acc', pgm_frequency=150)
        cal_05_rs = acc.get_response_spectrum(
            periods, damping=0.05, gmt='acc', pgm_frequency=150)
        cal_30_rs = acc.get_response_spectrum(
            periods, damping=0.30, gmt='acc', pgm_frequency=150)

        labels, colors, widths = ['tgt', 'cal'], ['r', 'b'], [3, 1]

        fs1 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.2.02.png')
        fs2 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.2.05.png')
        fs3 = os.path.join(OUTPUT_DIR,
            'recordings.accelerogram.get_response_spectrum.2.30.png')
        plot_response_spectra([tgt_02_rs, cal_02_rs], labels=labels,
            colors=colors, widths=widths, unit='m/s2', png_filespec=fs1)
        plot_response_spectra([tgt_05_rs, cal_05_rs], labels=labels,
            colors=colors, widths=widths, unit='m/s2', png_filespec=fs2)
        plot_response_spectra([tgt_30_rs, cal_30_rs], labels=labels,
            colors=colors, widths=widths, unit='m/s2', png_filespec=fs3)
