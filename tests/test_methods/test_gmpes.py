"""
Tests for 'methods.gmpes' module.
"""


import os
import unittest

import numpy as np

from synthacc.methods.gmpes import (TECTONIC_REGIONS, DISTANCE_METRICS,
    AVAILABLE_GMPES, GMPE, find_gmpes, plot_gmpes_distance,
    plot_gmpes_magnitude, plot_gmpes_spectrum)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


class TestGMPE(unittest.TestCase):
    """
    """

    gmpe = GMPE('AkkarEtAlRjb2014')
    parameters = {'mag': 6.5, 'rake': 0, 'rjb': 40000, 'vs30': 800}

    def test_properties(self):
        """
        """
        self.assertEqual(self.gmpe.name, 'AkkarEtAlRjb2014')
        self.assertEqual(len(self.gmpe.periods), 62)
        self.assertEqual(self.gmpe.min_period, 0.01)
        self.assertEqual(self.gmpe.max_period, 4.00)
        self.assertEqual(self.gmpe.tectonic_region, 'active shallow crust')
        self.assertEqual(self.gmpe.distance_metric, 'rjb')
        parameters = self.gmpe.parameters
        self.assertEqual(len(parameters), 4)
        self.assertEqual(set(parameters), set(('rake', 'mag', 'rjb', 'vs30')))

    def test_has(self):
        """
        """
        self.assertTrue(self.gmpe.has_sa())
        self.assertTrue(self.gmpe.has_pga())
        self.assertTrue(self.gmpe.has_pgv())
        self.assertFalse(self.gmpe.has_pgd())

    def test_is_dependent(self):
        """
        """
        self.assertTrue(self.gmpe.is_dependent('vs30'))
        self.assertFalse(self.gmpe.is_dependent('rrup'))

    def test_get_pga(self):
        """
        """
        self.gmpe.get_pga(self.parameters)
        self.gmpe.get_pga(self.parameters, unit='g')

    def test_get_sa(self):
        """
        """
        self.gmpe.get_mean_sa(
            self.parameters, period=float(self.gmpe.periods[0]), unit='g')

    def test_get_mean_response_spectrum(self):
        """
        """
        rs = self.gmpe.get_response_spectrum(self.parameters, unit='g')

    def test_plot_distance(self):
        """
        Compare with fig 9 of Akkar et al 2013.
        """

        distances = np.linspace(1000, 200000, 100)

        parameters = {'mag': 7.5, 'rake': 0, 'vs30': 750}
        png_filespec=os.path.join(OUTPUT_DIR,
            'methods.gmpes.gmpe.plot_gmpes_distance1.png')
        plot_gmpes_distance([self.gmpe], 'pga', distances, parameters, None,
            unit='g', space='loglog', min_dis=1000, max_dis=200000,
            min_val=0.0001, max_val=2, png_filespec=png_filespec)

        parameters = {'mag': 4.5, 'rake': 0, 'vs30': 750}
        png_filespec=os.path.join(OUTPUT_DIR,
            'methods.gmpes.gmpe.plot_gmpes_distance2.png')
        plot_gmpes_distance([self.gmpe], 'pga', distances, parameters, None,
            unit='g', space='loglog', min_dis=1000, max_dis=200000,
            min_val=0.0001, max_val=2, png_filespec=png_filespec)

    def test_plot_magnitude(self):
        """
        """
        magnitudes = np.linspace(3, 8, 50)

        parameters = {'rake': 0, self.gmpe.distance_metric: 10000, 'vs30': 750}
        png_filespec=os.path.join(OUTPUT_DIR, 'methods.gmpes.gmpe.plot_gmpes_magnitude.png')
        plot_gmpes_magnitude([self.gmpe], 'pga', magnitudes, parameters, None,
            unit='g', png_filespec=png_filespec)


    def test_plot_spectrum(self):
        """
        """
        png_filespec=os.path.join(OUTPUT_DIR,
            'methods.gmpes.gmpe.plot_gmpes_spectrum.png')
        plot_gmpes_spectrum([self.gmpe], self.parameters, labels=None,
            unit='g', png_filespec=png_filespec)


class Test(unittest.TestCase):
    """
    """

    def test_TECTONIC_REGIONS(self):
        """
        """
        self.assertIn('active shallow crust', TECTONIC_REGIONS)
        self.assertIn('stable shallow crust', TECTONIC_REGIONS)
