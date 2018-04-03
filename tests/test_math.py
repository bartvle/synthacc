"""
Tests for 'math' module.
"""


import unittest

import numpy as np

from synthacc.math import Matrix, SquareMatrix, IdentityMatrix


class TestMatrix(unittest.TestCase):
    """
    """

    m = Matrix([[1, 2], [3, 4], [5, 6]])

    def test_properties(self):
        """
        """
        self.assertEqual(self.m.nrows, 3)
        self.assertEqual(self.m.ncols, 2)
        self.assertEqual(self.m.order, (3, 2))
        self.assertEqual(self.m.size, 6)

    def test_rows(self):
        """
        """
        rows = self.m.rows
        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0]), (1, 2))
        self.assertEqual(tuple(rows[1]), (3, 4))
        self.assertEqual(tuple(rows[2]), (5, 6))

    def test_cols(self):
        """
        """
        cols = self.m.cols
        self.assertEqual(len(cols), 2)
        self.assertEqual(tuple(cols[0]), (1, 3, 5))
        self.assertEqual(tuple(cols[1]), (2, 4, 6))

    def test__getitem__(self):
        """
        """
        self.assertEqual(self.m[1, 1], 1)
        self.assertEqual(self.m[3, 2], 6)

    def test___iter__(self):
        """
        """
        self.assertListEqual([e for e in self.m], [1, 2, 3, 4, 5, 6])
        x, y, z = Matrix([[1, 2, 3]])
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(z, 3)

    def test___mul__(self):
        """
        """
        m = self.m * 2
        self.assertEqual(type(m), Matrix)
        self.assertTrue(
            (m._array == np.array([[2, 4], [6, 8], [10, 12]])).all())

        m1 = self.m * Matrix([[1], [2]])
        m2 = Matrix([[1, 2, 3]]) * self.m
        self.assertEqual(type(m1), Matrix)
        self.assertEqual(type(m2), Matrix)
        self.assertEqual(m1.order, (3, 1))
        self.assertEqual(m2.order, (1, 2))

    def test_is_square(self):
        """
        """
        self.assertFalse(self.m.is_square())
        m = Matrix([[1, 2], [3, 4]])
        self.assertTrue(m.is_square())


class TestSquareMatrix(unittest.TestCase):
    """
    """

    m = SquareMatrix([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        ])

    def test_trace(self):
        """
        """
        self.assertEqual(self.m.trace, 15)


class TestIdentityMatrix(unittest.TestCase):
    """
    """

    m = IdentityMatrix(3)
