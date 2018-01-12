"""
The 'earth.geo' module.


Module for an Earth modelled as an oblate spheroid with the WGS84 reference
ellipsoid and datum. Longitudes and latitudes are geodetic (not geocentric). In
WGS84 the center of the reference ellipsoid coincide with the center of mass of
the Earth.
"""


import numpy as np
import pyproj

from ..apy import (Object, is_number, is_non_neg_number, is_1d_numeric_array,
    is_in_range)
from .. import space


class Point(Object):
    """
    """

    def __init__(self, lon, lat, alt=0., validate=True):
        """
        alt: number, alt of point (in m) (default: 0.)
        """
        if validate is True:
            assert(is_number(lon) and is_lon(lon))
            assert(is_number(lat) and is_lat(lat))
            assert(is_number(alt))

        self._lon = float(lon)
        self._lat = float(lat)
        self._alt = float(alt)

    def __getitem__(self, i):
        """
        """
        return (self.lon, self.lat, self.alt)[i]

    @property
    def lon(self):
        """
        """
        return self._lon

    @property
    def lat(self):
        """
        """
        return self._lat

    @property
    def alt(self):
        """
        """
        return self._alt

    @property
    def depth(self):
        """
        """
        return self._alt * -1

    def project(self, proj):
        """
        """
        return space.Point(*project(self.lon, self.lat, proj), self.depth)


class Path(Object):
    """
    Points that form a path on the Earth's surface.
    """

    def __init__(self, lons, lats, validate=True):
        """
        lons: 1d numeric array, lons of points on path (same shape as lats)
        lats: 1d numeric array, lats of points on path (same shape as lons)
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(lons) and is_lon(lons))
            assert(is_1d_numeric_array(lats) and is_lat(lats))
            assert(lons.shape == lats.shape)

        self._points = np.stack((lons, lats), axis=-1)

    def __len__(self):
        """
        """
        return len(self._points)

    @property
    def points(self):
        """
        """
        return np.copy(self._points)

    @property
    def lons(self):
        """
        """
        return np.copy(self._points[:,0])

    @property
    def lats(self):
        """
        """
        return np.copy(self._points[:,1])

    @property
    def begin(self):
        """
        """
        return float(self._points[:,0][+0]), float(self._points[:,1][+0])

    @property
    def end(self):
        """
        """
        return float(self._points[:,0][-1]), float(self._points[:,1][-1])


class SphericalEarth(Object):
    """
    """

    def __init__(self, radius=6371000., validate=True):
        """
        radius: in m
        """
        if validate is True:
            assert(is_non_neg_number(radius))

        self._radius = radius

    @property
    def radius(self):
        """
        """
        return self._radius

    @property
    def circumference(self):
        """
        """
        return 2 * np.pi * self.radius

    def m2d(self, distance, validate=True):
        """
        Convert distance on the Earth's surface from meter to degree.

        distance: distance in meter
        """
        return distance * (360 / self.circumference)

    def d2m(self, distance, validate=True):
        """
        Convert distance on the Earth's surface from degree to meter.

        distance: distance in degree
        """
        return distance * (self.circumference / 360)


def project(lons, lats, proj):
    """
    Project lons and lats on a map.

    proj: EPSG number
    """
    wgs84_proj_ = pyproj.Proj(init='epsg:%i' % 4326)
    other_proj  = pyproj.Proj(init='epsg:%i' % proj)
    ys, xs = pyproj.transform(wgs84_proj_, other_proj, lons, lats)

    return xs, ys


def is_lon(obj):
    """
    Check if object is lon (number or nd numeric array).
    """
    return is_in_range(obj, -180, 180)


def is_lat(obj):
    """
    Check if object is lat (number or nd numeric array).
    """
    return is_in_range(obj,  -90,  90)
