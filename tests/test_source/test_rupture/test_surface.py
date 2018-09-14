"""
Tests for 'source.rupture.surface' module.
"""


import unittest

from synthacc.source.rupture.surface import Distribution


class TestDistribution(unittest.TestCase):
    """
    """

    class DummyDistribution(Distribution):
        LABEL, _values = None, None

    i = DummyDistribution(1, 2, 2, 4)

    def test_properties(self):
        """
        """
        self.assertEqual(self.i.values, None)
