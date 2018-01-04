"""
Tests for 'io.axisem' module.
"""


import unittest

import os

from synthacc.io.axisem import read_ground_model


DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'axisem')
