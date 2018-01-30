"""
Tests for 'methods.gmpes' module.
"""


import os
import unittest

import numpy as np

from synthacc.methods.gmpes import TECTONIC_REGIONS, GMPES, GMPE, find_gmpes


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


class TestGMPE(unittest.TestCase):
    """
    """

    gmpe = GMPE('AkkarEtAlRjb2014')

    def test_properties(self):
        """
        """
        self.assertEqual(self.gmpe.name, 'AkkarEtAlRjb2014')
        self.assertEqual(self.gmpe.tectonic_region, 'active shallow crust')
        parameters = self.gmpe.parameters
        self.assertEqual(len(parameters), 4)
        self.assertEqual(set(parameters), set(('rake', 'mag', 'rjb', 'vs30')))
        self.assertEqual(self.gmpe.distance_metric, 'rjb')
        self.assertEqual(len(self.gmpe.periods), 62)
        self.assertEqual(self.gmpe.min_period, 0.01)
        self.assertEqual(self.gmpe.max_period, 4.00)

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
        parameters = {'mag': 6.5, 'rjb': 40000}
        self.gmpe.get_pga(parameters, unit='g')

    def test_get_sa(self):
        """
        """
        parameters = {'mag': 6.5, 'rjb': 40000}
        self.gmpe.get_sa(
            parameters, period=float(self.gmpe.periods[0]), unit='g')

    def test_get_response_spectrum(self):
        """
        """
        parameters = {'mag': 6.5, 'rjb': 40000}
        rs = self.gmpe.get_response_spectrum(parameters, unit='g')
        rs.plot(png_filespec=os.path.join(
            OUTPUT_DIR, 'methods.gmpes.gmpe.get_response_spectrum.png'))

    def test_plot(self):
        """
        Compare with fig 9 of Akkar et al 2013.
        """
        distances = np.linspace(1000, 200000, 100)
        magnitude = 7.5
        parameters = {'vs30': 750, 'rake': 0}
        png_filespec = os.path.join(OUTPUT_DIR, 'methods.gmpes.gmpe.plot.png')
        self.gmpe.plot('pga', distances, magnitude, parameters, unit='g',
            min_dis=1000, max_dis=200000, min_val=0.0001, max_val=1,
            png_filespec=png_filespec)
