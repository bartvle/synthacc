"""
Tests for 'source.rupture.surface' module.
"""


import unittest

from synthacc.source.rupture.surface import Distribution


class TestDistribution(unittest.TestCase):
    """
    """

    class Test(Distribution):
        LABEL, _values = None, None

    i = Test(1, 2, 0.1, 0.1)

    def test_properties(self):
        """
        """
        pass
