"""
The 'earth.geo' module.


Module for an oblate spheroid Earth (WGS84 reference ellipsoid). Longitudes and
latitudes are geodetic (not geocentric). For projections the WGS84 geographic
coordinate system is used. The center of the reference ellipsoid coincides with
the center of mass of the Earth. Z points to the north pole.
"""


import numpy as np
import pyproj

from ..apy import (PRECISION, Object, is_number, is_non_neg_number,
    is_1d_numeric_array, is_in_range)
from .. import space
from ..units import MOTION as UNITS
from . import flat


_G = pyproj.Geod(ellps='WGS84') ## The WGS84 reference ellipsoid


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

        if abs(lon) < 10**-PRECISION:
            lon = 0
        if abs(lat) < 10**-PRECISION:
            lat = 0
        if abs(alt) < 10**-PRECISION:
            alt = 0

        self._lon = float(lon)
        self._lat = float(lat)
        self._alt = float(alt)

    def __repr__(self):
        """
        """
        s = '< geo.Point | lon={:.3f}, lat={:.3f}, alt={:.3f} >'.format(
            self.lon, self.lat, self.alt)
        return s

    def __getitem__(self, i):
        """
        """
        return (self.lon, self.lat, self.alt)[i]

    def __eq__(self, other):
        """
        return: boolean
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        lon, lat, alt = self.__class__(*other)
        lon_eq = np.abs(self.lon - lon) < 10**-PRECISION
        lat_eq = np.abs(self.lat - lat) < 10**-PRECISION
        alt_eq = np.abs(self.alt - alt) < 10**-PRECISION

        return (lon_eq and lat_eq and alt_eq)

    @classmethod
    def from_projection(cls, point, proj, validate=True):
        """
        proj: EPSG number
        """
        p = space.Point(*point, validate=validate)

        other_proj = pyproj.Proj(init='epsg:%i' % proj)
        wgs84_proj = pyproj.Proj(init='epsg:%i' % 4326)
        lon, lat = pyproj.transform(other_proj, wgs84_proj, p.y, p.x)

        return cls(lon, lat, -p.z)

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

    def get_geo_distance(self, point, unit='m', validate=True):
        """
        point: 'earth.geo.Point' instance or (lon, lat, (alt)) tuple
        """
        p1, p2 = self, Point(*point, validate=validate)

        d = distance(p1.lon, p1.lat, p2.lon, p2.lat, unit, validate=validate)

        return d

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

    def project(self, proj):
        """
        """
        xs, ys = project(self._points[:,0], self._points[:,1], proj=proj)

        return flat.Path(xs, ys)


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


def are_coordinates(obj):
    """
    Check if object are geo coordinates, i.e. a (lon, lat) or (lon, lat, alt)
    tuple.
    """
    if type(obj) is not tuple:
        return False
    if len(obj) not in (2, 3):
        return False
    if not (is_number(obj[0]) and is_lon(obj[0])):
        return False
    if not (is_number(obj[1]) and is_lat(obj[1])):
        return False
    if len(obj) == 3 and not is_number(obj[2]):
        return False
    return True


def distance(lon1, lat1, lon2, lat2, unit='m', validate=True):
    """
    Geodesic distance, i.e. the shortest distance between two points on the
    Earth's surface.

    lon1 : numeric, if array then shape must equal that of lat1
    lat1 : numeric, if array then shape must equal that of lon1
    lon2 : numeric, if array then shape must equal that of lat2
    lat2 : numeric, if array then shape must equal that of lon2
    unit : string, unit of distance (default: 'm')

    return: numeric (shape is 1.shape + 2.shape), distances in unit.

    NOTE: Gives the same result as Geographiclib and Obspy (up to mm level).
    """
    if validate is True:
        assert(is_lon(lon1))
        assert(is_lat(lat1))
        assert(is_lon(lon2))
        assert(is_lat(lat2))

    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)

    if validate is True:
        assert(UNITS[unit].quantity == 'displacement')
        assert(lon1.shape == lat1.shape)
        assert(lon2.shape == lat2.shape)

    s1 = lon1.shape
    s2 = lon2.shape
    s = s1 + s2  ## output shape

    lon1 = np.tile(
        lon1[(Ellipsis,)+tuple([None]*len(s2))], tuple([1]*len(s1)) + s2)
    lat1 = np.tile(
        lat1[(Ellipsis,)+tuple([None]*len(s2))], tuple([1]*len(s1)) + s2)
    lon2 = np.tile(
        lon2[tuple([None]*len(s1))+(Ellipsis,)], s1 + tuple([1]*len(s2)))
    lat2 = np.tile(
        lat2[tuple([None]*len(s1))+(Ellipsis,)], s1 + tuple([1]*len(s2)))

    lon1 = lon1.flatten()
    lat1 = lat1.flatten()
    lon2 = lon2.flatten()
    lat2 = lat2.flatten()

    d = np.reshape(_G.inv(lon1, lat1, lon2, lat2)[-1], s)
    d *= (UNITS['m'] / UNITS[unit])

    if d.ndim == 0:
        d = float(d)

    return d


def project(lons, lats, proj):
    """
    Project lons and lats on a map. The projected coordinate system is defined
    by its EPSG number.

    proj: EPSG number
    """
    wgs84_proj = pyproj.Proj(init='epsg:%i' % 4326)
    other_proj = pyproj.Proj(init='epsg:%i' % proj)
    ys, xs = pyproj.transform(wgs84_proj, other_proj, lons, lats)

    return xs, ys
