"""
Tests for 'time' module.
"""


import unittest

import datetime

from obspy import UTCDateTime

from synthacc.time import Time, Date, is_time, is_date


class TestTime(unittest.TestCase):
    """
    """
    t1 = Time('2016-09-16 14:51:17')
    t2 = Time('2016-09-16 14:51:17.12345')
    t3 = Time('2016-09-16 14:51:17.123450')
    t4 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17))
    t5 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17, 12345))
    t6 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17, 20000), precision=2)
    t7 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17, 20000), precision=3)
    t8 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17, 12340), precision=5)
    t9 = Time(datetime.datetime(2016, 9, 16, 14, 51, 17, 123456))
    times = (t1, t2, t3, t4, t5, t9)

    def test_properties(self):
        """
        """
        for t in self.times:
            self.assertEqual(t.year, 2016)
            self.assertEqual(t.month, 9)
            self.assertEqual(t.day, 16)
            self.assertEqual(t.hour, 14)
            self.assertEqual(t.minute, 51)
            self.assertEqual(t.second, 17)
            self.assertEqual(t.date, Date('2016-09-16'))

        self.assertEqual(self.t1.microsecond, None)
        self.assertEqual(self.t1.precision, 0)
        self.assertEqual(self.t2.microsecond, 123450)
        self.assertEqual(self.t2.precision, 5)
        self.assertEqual(self.t3.microsecond, 123450)
        self.assertEqual(self.t3.precision, 6)
        self.assertEqual(self.t4.microsecond, 0)
        self.assertEqual(self.t4.precision, 6)
        self.assertEqual(self.t5.microsecond, 12345)
        self.assertEqual(self.t5.precision, 6)
        self.assertEqual(self.t6.microsecond, 20000)
        self.assertEqual(self.t6.precision, 2)
        self.assertEqual(self.t7.microsecond, 20000)
        self.assertEqual(self.t7.precision, 3)
        self.assertEqual(self.t8.microsecond, 12340)
        self.assertEqual(self.t8.precision, 5)
        self.assertEqual(self.t9.microsecond, 123456)
        self.assertEqual(self.t9.precision, 6)

    def test___str__(self):
        """
        """
        self.assertEqual(str(self.t1), '2016-09-16 14:51:17')
        self.assertEqual(str(self.t2), '2016-09-16 14:51:17.12345')
        self.assertEqual(str(self.t3), '2016-09-16 14:51:17.123450')
        self.assertEqual(str(self.t4), '2016-09-16 14:51:17.000000')
        self.assertEqual(str(self.t5), '2016-09-16 14:51:17.012345')
        self.assertEqual(str(self.t6), '2016-09-16 14:51:17.02')
        self.assertEqual(str(self.t7), '2016-09-16 14:51:17.020')
        self.assertEqual(str(self.t8), '2016-09-16 14:51:17.01234')
        self.assertEqual(str(self.t9), '2016-09-16 14:51:17.123456')

    def test___eq__(self):
        """
        """
        self.assertTrue(self.t4 == Time('2016-09-16 14:51:17.000000'))
        self.assertTrue(self.t5 == Time('2016-09-16 14:51:17.012345'))
        self.assertTrue(self.t6 == Time('2016-09-16 14:51:17.02'))
        self.assertTrue(self.t7 == Time('2016-09-16 14:51:17.020'))
        self.assertFalse(self.t4 == self.t5)
        self.assertFalse(self.t6 == self.t7)

    def test___add__(self):
        """
        """
        self.assertEqual(self.t4 + 55, Time(datetime.datetime(2016, 9, 16, 14, 52, 12)))
        self.assertEqual(self.t4 + 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 22)))
        self.assertEqual(self.t5 + 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 22, 12345)))
        self.assertEqual(self.t6 + 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 22, 20000), precision=2))
        self.assertEqual(self.t7 + 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 22, 20000), precision=3))
        self.assertEqual(self.t5 + 5.51, Time(datetime.datetime(2016, 9, 16, 14, 51, 22, 522345)))

    def test___sub__(self):
        """
        """
        self.assertEqual(self.t4 - 55, Time(datetime.datetime(2016, 9, 16, 14, 50, 22)))
        self.assertEqual(self.t4 - 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 12)))
        self.assertEqual(self.t5 - 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 12, 12345)))
        self.assertEqual(self.t6 - 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 12, 20000), precision=2))
        self.assertEqual(self.t7 - 5, Time(datetime.datetime(2016, 9, 16, 14, 51, 12, 20000), precision=3))
        self.assertEqual(self.t5 - 5.51, Time(datetime.datetime(2016, 9, 16, 14, 51, 11, 502345)))

        self.assertEqual(self.t9 - self.t5, 0.111111)


class TestDate(unittest.TestCase):
    """
    """

    d1 = Date('1986-07-21')
    d2 = Date(datetime.date(1986, 7, 21))
    d3 = Date(UTCDateTime(1986, 7, 21))
    d4 = Date(1986, 7, 21)
    dates = (d1, d2, d3, d4)

    def test_properties(self):
        """
        """
        for d in self.dates:
            self.assertEqual(d.year, 1986)
            self.assertEqual(d.month, 7)
            self.assertEqual(d.day, 21)

    def test___str__(self):
        """
        """
        for d in self.dates:
            self.assertEqual(str(d), '1986-07-21')

    def test___eq__(self):
        """
        """
        self.assertEqual(self.d1, self.d2)
        self.assertFalse(self.d1 == Date('1986-07-22'))

    def test_comparison(self):
        """
        """
        self.assertTrue(self.d1 == self.d2)
        self.assertTrue(self.d1 <= self.d2)
        self.assertTrue(self.d1 >= self.d2)
        self.assertTrue(self.d1 < '1986-07-22')
        self.assertTrue(self.d1 > '1986-07-20')
        self.assertFalse(self.d1 == '1986-07-20')
        self.assertFalse(self.d1 > self.d2)
        self.assertFalse(self.d1 > '1986-07-22')
        self.assertFalse(self.d1 < '1986-07-20')
