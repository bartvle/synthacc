"""
Tests for 'io.axisem' module.
"""


import os
import unittest

from synthacc.ground_models import ContinuousModel
from synthacc.io.axisem import read_ground_model


DATA_DIR = os.path.join(
    os.path.dirname(__file__), 'data', 'axisem')


class Test(unittest.TestCase):
    """
    """

    def test_read_ground_model(self):
        """
        """
        gm = read_ground_model(os.path.join(DATA_DIR, 'ak135f.txt'))
        self.assertEqual(type(gm), ContinuousModel)
        self.assertEqual(len(gm), 716)
