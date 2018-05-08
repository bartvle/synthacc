"""
The 'observation' module.
"""


import numpy as np

from .apy import (Object, is_number, is_non_neg_number, is_pos_integer,
    is_string)
from .time import Time, Date, is_time, is_date
from .data import DataRecord, DataBase
from .earth.geo import is_lon, is_lat


class Magnitude(Object):
    """
    """

    def __init__(self, validate=True, **magnitudes):
        """
        """
        if validate is True:
            assert(magnitudes != {})
            for t, v in magnitudes.items():
                assert(is_string(t))
                assert(is_number(v))

        self._magnitudes = {t.lower(): v for t, v in magnitudes.items()}

    def __getattr__(self, attr):
        """
        """
        return self._magnitudes.get(attr.lower(), None)

    def __getitem__(self, item):
        """
        """
        return self._magnitudes.get(item.lower(), None)

    def __contains__(self, t):
        """
        """
        return t.lower() in self._magnitudes

    @property
    def mag(self):
        """
        """
        for t in ('mw', 'ms', 'mb', 'ml', 'md'):
            if t in self:
                return getattr(self, t)
        raise


class Event(Object):
    """
    """

    def __init__(self, lon, lat, depth, time, magnitude, name=None, validate=True):
        """
        """
        if validate is True:
            assert(is_number(lon) and is_lon(lon))
            assert(is_number(lat) and is_lat(lat))
            assert(is_non_neg_number(depth))
            assert(is_number(magnitude) or type(magnitude) is Magnitude)
            if name is not None:
                assert(is_string(name))

        self._lon = lon
        self._lat = lat
        self._depth = depth

        if not is_time(time):
            time = Time(time, validate=validate)
        self._time = time

        self._magnitude = magnitude
        self._name = name

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
    def depth(self):
        """
        """
        return self._depth

    @property
    def time(self):
        """
        """
        return self._time

    @property
    def magnitude(self):
        """
        """
        return self._magnitude

    @property
    def name(self):
        """
        """
        return self._name

    @property
    def key(self):
        """
        """
        return self._key

    @property
    def mag(self):
        """
        """
        if type(self._magnitude) is Magnitude:
            return self._magnitude.mag
        else:
            return self._magnitude


class EventRecord(DataRecord, Event):
    """
    """

    def __init__(self, key, lon, lat, depth, time, magnitude, name=None, validate=True):
        """
        """
        DataRecord.__init__(self, key, validate=validate)
        Event.__init__(
            self, lon, lat, depth, time, magnitude, name, validate=validate)


class Catalog(DataBase):
    """
    """

    def __init__(self, events=[], validate=True):
        """
        """
        super().__init__(events, EventRecord, validate=validate)

    def __getitem__(self, item):
        """
        """
        assert(is_non_neg_number(item))
        return self._records[item]

    @property
    def lons(self):
        """
        """
        return np.array([e.lon for e in self])

    @property
    def lats(self):
        """
        """
        return np.array([e.lat for e in self])

    @property
    def mags(self):
        """
        """
        return np.array([e.mag for e in self])

    def query(self, region=None, min_date=None, max_date=None, min_depth=None, max_depth=None, min_magnitude=None, max_magnitude=None, validate=True):
        """
        """
        if validate is True:
            if region != None:
                assert(len(region) == 4)
                if region[0] != None: assert(is_lon(region[0]))
                if region[1] != None: assert(is_lon(region[1]))
                if region[2] != None: assert(is_lat(region[2]))
                if region[3] != None: assert(is_lat(region[3]))
            if type(min_date) == int: ## min_date is year
                min_date = Date(min_date, 1, 1)
            elif min_date is not None:
                min_date = Date(min_date)
            if type(max_date) == int: ## max_date is year
                max_date = Date(max_date, 12, 31)
            elif max_date is not None:
                max_date = Date(max_date)
            if min_depth != None:
               assert(is_non_neg_number(min_depth))
            if max_depth != None:
               assert(is_non_neg_number(max_depth))
            if min_magnitude != None:
                assert(is_number(min_magnitude))
            if max_magnitude != None:
                assert(is_number(max_magnitude))
        events = []
        for e in self:
            if region is not None:
                if (region[0] is not None and e.lon < region[0]):
                    continue
                if (region[1] is not None and e.lon > region[1]):
                    continue
                if (region[2] is not None and e.lat < region[2]):
                    continue
                if (region[3] is not None and e.lat > region[3]):
                    continue
            if min_date is not None and e.time.date < min_date:
                continue
            if max_date is not None and e.time.date > max_date:
                continue
            if (min_depth is not None and e.depth < min_depth):
                continue
            if (max_depth is not None and e.depth > max_depth):
                continue
            if min_magnitude is not None and (e.magnitude is None or
                e.mag < min_magnitude):
                continue
            if max_magnitude is not None and (e.magnitude is None or
                e.mag > max_magnitude):
                continue
            events.append(e)

        return self.__class__(events, validate=False)


class Station(Object):
    """
    """

    def __init__(self, code, lon, lat, alt, s_date=None, e_date=None, validate=True):
        """
        """
        if validate is True:
            assert(is_string(code))
            assert(is_number(lon) and is_lon(lon))
            assert(is_number(lat) and is_lat(lat))
            assert(is_number(alt))
            if s_date is not None:
                assert(is_date(s_date))
            if e_date is not None:
                assert(is_date(e_date))
                assert(s_date <= e_date)

        self._code = code
        self._lon = lon
        self._lat = lat
        self._alt = alt
        self._s_date = s_date
        self._e_date = e_date

    @property
    def code(self):
        """
        """
        return self._code

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
    def s_date(self):
        """
        """
        return self._s_date

    @property
    def e_date(self):
        """
        """
        return self._e_date

    @property
    def key(self):
        """
        """
        return self._key


class StationRecord(DataRecord, Station):
    """
    """

    def __init__(self, key, code, lon, lat, alt, s_date=None, e_date=None, validate=True):
        """
        """
        DataRecord.__init__(self, key, validate=validate)
        Station.__init__(
            self, code, lon, lat, alt, s_date, e_date, validate=validate)


class Network(DataBase):
    """
    """

    def __init__(self, stations, validate=True):
        """
        """
        super().__init__(stations, StationRecord, validate=validate)

    @property
    def lons(self):
        """
        """
        return np.array([s.lon for s in self])

    @property
    def lats(self):
        """
        """
        return np.array([s.lat for s in self])

    @property
    def alts(self):
        """
        """
        return np.array([s.alt for s in self])

    def query(self, region=None, validate=True):
        """
        """
        if validate is True:
            if region != None:
                assert(len(region) == 4)
                if region[0] != None: assert(is_lon(region[0]))
                if region[1] != None: assert(is_lon(region[1]))
                if region[2] != None: assert(is_lat(region[2]))
                if region[3] != None: assert(is_lat(region[3]))

        stations = []
        for s in self:
            if region is not None:
                if (region[0] is not None and s.lon < region[0]):
                    continue
                if (region[1] is not None and s.lon > region[1]):
                    continue
                if (region[2] is not None and s.lat < region[2]):
                    continue
                if (region[3] is not None and s.lat > region[3]):
                    continue
            stations.append(s)

        return self.__class__(stations, validate=False)
