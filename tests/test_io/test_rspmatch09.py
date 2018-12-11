"""
Tests for 'io.rspmatch09' module.
"""


import os
import unittest

from synthacc.units import round_to_significant
from synthacc.io.rspmatch09 import (read_acc, read_rsp, read_tgt, write_acc,
    write_tgt, write_inp, write_input, run)


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'rspmatch09')


class Test(unittest.TestCase):
    """
    """

    def test_read_acc(self):
        """
        """
        acc = read_acc(os.path.join(DATA_DIR, 'Run2.acc'))
        self.assertEqual(acc.time_delta, 0.005)
        self.assertEqual(len(acc), 6110)
        self.assertEqual(acc.unit, 'g')

    def test_read_rsp(self):
        """
        """
        rs, max_mis = read_rsp(os.path.join(DATA_DIR, 'Run2.rsp'))
        self.assertEqual(round_to_significant(float(rs.periods[0]), 4), 0.02933)
        self.assertEqual(len(rs), 122)
        self.assertEqual(rs.unit, 'g')
        self.assertEqual(rs.damping, 0.05)
        self.assertEqual(max_mis, 8.29)

    def test_read_tgt(self):
        """
        """
        rs = read_tgt(os.path.join(DATA_DIR, 'cms_T0.2_horiz.tgt'))
        self.assertEqual(len(rs), 200)
        self.assertEqual(rs.unit, 'g')
        self.assertEqual(rs.damping, 0.05)
