"""
Tests for 'source.faults' module.
"""


import unittest

from synthacc.source.faults import RIGIDITY, SingularFault, ComposedFault


class TestSingularFault(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        sf = SingularFault(0, 0, 10000, 0, 1000, 20000, 60)
