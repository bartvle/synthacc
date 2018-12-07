"""
The 'source.mechanism' module. Strike, dip and rake follow the convention of
strike being measured clockwise from the north, dip right to the strike and
rake giving motion of the hanging wall relative to the foot wall. The hanging
wall is therefore always to the right of the strike.
"""


import numpy as np

from ..apy import Object, is_integer, is_number, is_pos_number
from .. import space3
from ..earth.flat import is_strike, is_dip
from .moment import MomentTensor


class NodalPlane(Object):
    """
    """

    def __init__(self, strike, dip, rake, validate=True):
        """
        """
        if validate is True:
            assert(is_strike(strike) and is_dip(dip) and is_rake(rake))

        self._s, self._d, self._r = strike, dip, rake

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        return (self._s, self._d, self._r)[i]

    @property
    def strike(self):
        """
        """
        return self._s

    @property
    def dip(self):
        """
        """
        return self._d

    @property
    def rake(self):
        """
        """
        return self._r

    @property
    def _radians(self):
        """
        """
        return np.radians([self._s, self._d, self._r])


class FocalMechanism(NodalPlane):
    """
    """

    @property
    def normal_vector(self):
        """
        Normal vector in NED coordinates (x is north, y is east and z is down).

        See Stein and Wyession (2003) p. 218, where x also points north, but y
        west and z up. Jost and Herrmann (1989) p. 40. show the formulas with
        correct orientation.

        return: 'space3.Vector' instance
        """
        s, d, _ = self._radians

        n = float(-np.sin(d) * np.sin(s))
        w = float(-np.sin(d) * np.cos(s))
        u = float(+np.cos(d))

        return space3.Vector(n, -w, -u)

    @property
    def slip_vector(self):
        """
        Slip vector in NED coordinates (x is north, y is east and z is down).

        See Stein and Wyession (2003) p. 218, where x also points north, but y
        west and z up. Jost and Herrmann (1989) p. 40. show the formulas with
        correct orientation.

        return: 'space3.Vector' instance
        """
        s, d, r = self._radians

        n = float(+np.cos(r)*np.cos(s) + np.sin(r)*np.cos(d)*np.sin(s))
        w = float(-np.cos(r)*np.sin(s) + np.sin(r)*np.cos(d)*np.cos(s))
        u = float(+np.sin(r)*np.sin(d))

        return space3.Vector(n, -w, -u)

    def get_moment_tensor(self, moment, validate=True):
        """
        See Aki and Richards (2002) p. 112.

        moment: pos number

        return: 'source.moment.MomentTensor' instance
        """
        if validate is True:
            assert(is_pos_number(moment))

        s, d, r = self._radians
        xx = -moment * (np.sin(d)*np.cos(r)*np.sin(2*s) +
            np.sin(2*d)*np.sin(r)*np.sin(s)**2)
        yy = +moment * (np.sin(d)*np.cos(r)*np.sin(2*s) -
            np.sin(2*d)*np.sin(r)*np.cos(s)**2)
        zz = +moment * (np.sin(2*d)*np.sin(r))
        xy = +moment * (np.sin(d)*np.cos(r)*np.cos(2*s) +
            np.sin(2*d)*np.sin(r)*np.sin(2*s)/2)
        yz = -moment * (np.cos(d)*np.cos(r)*np.sin(1*s) -
            np.cos(2*d)*np.sin(r)*np.cos(s))
        zx = -moment * (np.cos(d)*np.cos(r)*np.cos(1*s) +
            np.cos(2*d)*np.sin(r)*np.sin(s))

        return MomentTensor(xx, yy, zz, xy, yz, zx)


def is_rake(obj):
    """
    Check if object is rake.
    """
    return is_number(obj) and (-180 < obj <= 180)
