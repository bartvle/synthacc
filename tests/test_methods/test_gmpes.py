"""
Tests for 'methods.gmpes' module.
"""


import os
import unittest

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

    def test_get_response_spectrum(self):
        """
        """
        parameters = {'mag': 6.5, 'rjb': 40}
        rs = self.gmpe.get_response_spectrum(parameters, unit='g')
        rs.plot(png_filespec=os.path.join(
            OUTPUT_DIR, 'methods.gmpes.gmpe.get_response_spectrum.png'))
