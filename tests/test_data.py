"""
Tests for 'data' module.
"""


import unittest

from synthacc.data import TimeSeries, DataRecord, DataBase


class DummyDataRecord(DataRecord):
    pass


class DummyDataBase(DataBase):

    def __init__(self, records):
        """
        """
        super().__init__(records, DummyDataRecord)


class TestDataRecord(unittest.TestCase):
    """
    """

    def test_valdiation(self):
        """
        Only accept pos integer as key.
        """
        self.assertRaises(AssertionError, DummyDataRecord, 5.)
        self.assertRaises(AssertionError, DummyDataRecord, '')
        self.assertRaises(AssertionError, DummyDataRecord, -1)
        self.assertRaises(AssertionError, DummyDataRecord, (5, 6))
        self.assertRaises(AssertionError, DummyDataRecord, 0)


class TestDataBase(unittest.TestCase):
    """
    """
    r1 = DummyDataRecord(key=1)
    r2 = DummyDataRecord(key=2)
    r3 = DummyDataRecord(key=3)
    records = [r1, r2, r3]
    db = DummyDataBase(records)

    def test_properties(self):
        """
        """
        self.assertListEqual(self.db.keys, [1, 2, 3])

    def test___len__(self):
        """
        """
        self.assertEqual(len(self.db), 3)

    def test___iter__(self):
        """
        """
        for cal, tgt in zip(self.db, self.records):
            self.assertEqual(cal, tgt)

    def test_get(self):
        """
        """
        self.assertEqual(self.db.get(2), self.r2)

        def f():
            self.db.get(4)

        self.assertRaises(LookupError, f)
