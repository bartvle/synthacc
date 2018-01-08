"""
Tests for 'observation' module.
"""


import unittest

from synthacc.time import Time

from synthacc.observation import Magnitude, Event, Catalog, Station, Network


class TestMagnitude(unittest.TestCase):
    """
    """

    mw = 1.5
    ms = 2.1
    mb = 4.3
    ml = 3.2
    md = 5.4
    m = Magnitude(mw=mw, ms=ms, mb=mb, ml=ml, md=md)

    def test___getattr__(self):
        """
        """
        self.assertEqual(self.m.mw, self.mw)
        self.assertEqual(self.m.ms, self.ms)
        self.assertEqual(self.m.mb, self.mb)
        self.assertEqual(self.m.ml, self.ml)
        self.assertEqual(self.m.md, self.md)

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.m['mw'], self.mw)
        self.assertEqual(self.m['ms'], self.ms)
        self.assertEqual(self.m['mb'], self.mb)
        self.assertEqual(self.m['ml'], self.ml)
        self.assertEqual(self.m['md'], self.md)

    def test___contains__(self):
        """
        """
        self.assertTrue('ml' in self.m)
        self.assertFalse('m' in self.m)
        self.assertTrue('Ml' in self.m)
        self.assertFalse('M' in self.m)

    def test_mag(self):
        """
        """
        self.assertEqual(self.m.mag, self.mw)


class TestEvent(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        lon, lat, depth, time, magnitude = (
            4.02, 51.64, 15.6, '1992-12-05 01:33:24', 5.1)
        e = Event(lon, lat, depth, time, magnitude)
        self.assertEqual(e.lon, lon)
        self.assertEqual(e.lat, lat)
        self.assertEqual(e.depth, depth)
        self.assertEqual(e.time, Time(time))
        self.assertEqual(e.magnitude, magnitude)
        self.assertEqual(e.mag, magnitude)

        magnitude = Magnitude(mw=6.3)
        e = Event(lon, lat, depth, time, magnitude)
        self.assertEqual(e.magnitude, magnitude)
        self.assertEqual(e.mag, 6.3)


class TestCatalog(unittest.TestCase):
    """
    """

    e1 = Event(4.02, 51.64, 15.6, '1992-12-05 01:33:24', 5.1, 1)
    e2 = Event(4.03, 51.65, 15.7, '1992-12-05 01:33:25', 5.2, 2)
    e3 = Event(4.04, 51.66, 15.8, '1992-12-05 01:33:26', 5.3, 3)
    c = Catalog([e1, e2, e3])

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.c), 3)
        self.assertListEqual(list(self.c.lons), [4.02, 4.03, 4.04])
        self.assertListEqual(list(self.c.lats), [51.64, 51.65, 51.66])
        self.assertListEqual(list(self.c.mags), [5.1, 5.2, 5.3])

    def test___getitem__(self):
        """
        """
        self.assertEqual(self.c[1].key, 2)
