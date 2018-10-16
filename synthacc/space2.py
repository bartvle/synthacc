"""
The 'space2' module. 2d Euclidean space in a right-handed Cartesian coordinate
system.
"""


from abc import ABC, abstractmethod
import random

from numba import jit
import numpy as np

from .apy import (PRECISION, Object, is_integer, is_pos_integer, is_number,
    is_pos_number, is_array, is_numeric, is_in_range)


class Point(Object):
    """
    """

    def __init__(self, x, y, validate=True):
        """
        x: number, x coordinate
        y: number, y coordinate
        """
        if validate is True:
            assert(is_number(x))
            assert(is_number(y))

        if abs(x) < 10**-PRECISION:
            x = 0
        if abs(y) < 10**-PRECISION:
            y = 0

        self._x = x
        self._y = y

    def __repr__(self):
        """
        """
        s = '< space2.Point | '
        s += 'x={:{}.3f}'.format(self.x, '+' if self.x else '')
        s += ', '
        s += 'y={:{}.3f}'.format(self.y, '+' if self.y else '')
        s += ' >'

        return s

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        return (self._x, self._y)[i]

    def __eq__(self, other):
        """
        return: boolean
        """
        assert(type(other) is self.__class__)

        x, y = other
        x_eq = abs(self._x - x) < 10**-PRECISION
        y_eq = abs(self._y - y) < 10**-PRECISION

        return (x_eq and y_eq)

    @property
    def x(self):
        """
        return: number, x coordinate
        """
        return self._x

    @property
    def y(self):
        """
        return: number, y coordinate
        """
        return self._y


class RectangularSurface(Object):
    """
    Rectangular surface with upper left corner (ULC), upper right corner (URC),
    lower left corner (LLC) and lower right corner (LRC). The origin is the URC
    which in a right-handed Cartesian coordinate system means:
        x-axis goes from the ULC to the LLC
        y-axis goes from the ULC to the URC
    We call the sizes of the rectangular surface width (W) along the x-axis and
    length (L) along the y-axis.
    """

    def __init__(self, w, l, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))

        self._w = w
        self._l = l

    @property
    def w(self):
        """
        return: pos number, w
        """
        return self._w

    @property
    def l(self):
        """
        return: pos number, l
        """
        return self._l

    @property
    def area(self):
        """
        return: pos number, area
        """
        return self._w * self._l

    def get_random(self, xmin=None, xmax=None, ymin=None, ymax=None, seed=None, validate=True):
        """
        Get a random point on the surface (between xmin, xmax, ymin and ymax).

        return: 'space2.Point' instance
        """
        if validate is True:
            assert(xmin is None or (is_number(xmin) and
                is_in_range(xmin, 0, self.w)))
            assert(ymin is None or (is_number(ymin) and
                is_in_range(ymin, 0, self.l)))
            assert(ymax is None or (is_number(ymax) and
                is_in_range(ymax, ymin, self.l)))
            assert(seed is None or is_pos_integer(seed))

        if seed is not None:
            np.random.seed(seed)

        if xmin is None:
            xmin = 0
        if ymin is None:
            ymin = 0
        if xmax is None:
            xmax = self.w
        if ymax is None:
            ymax = self.l

        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        return Point(x, y)


class DiscretizedRectangularSurface(RectangularSurface):
    """
    """

    def __init__(self, w, l, nw, nl, validate=True):
        """
        """
        super().__init__(w, l, validate=validate)

        if validate is True:
            assert(is_pos_integer(nw))
            assert(is_pos_integer(nl))

        dw = w / nw
        dl = l / nl
        ws = np.linspace(0+dw/2, w-dw/2, nw)
        ls = np.linspace(0+dl/2, l-dl/2, nl)
        self._grid = np.dstack(np.meshgrid(ls, ws))

    @classmethod
    def from_spacing(cls, w, l, dw, dl, validate=True):
        """
        The dw and dl parameters are recalculated depending on w/nw and l/nl.
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw) and dw <= w)
            assert(is_pos_number(dl) and dl <= l)

        nw = round(w / dw)
        nl = round(l / dl)

        return cls(w, l, nw, nl, validate=False)

    def __len__(self):
        """
        """
        return self.nw * self.nl

    @property
    def shape(self):
        """
        """
        return self._grid.shape[:2]

    @property
    def nw(self):
        """
        """
        return self.shape[0]

    @property
    def nl(self):
        """
        """
        return self.shape[1]

    @property
    def dw(self):
        """
        """
        return self._w / self.nw

    @property
    def dl(self):
        """
        """
        return self._l / self.nl

    @property
    def xs(self):
        """
        return: 1d numerical array
        """
        return self._grid[:,0,1]

    @property
    def ys(self):
        """
        return: 1d numerical array
        """
        return self._grid[0,:,0]

    @property
    def xgrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,1]

    @property
    def ygrid(self):
        """
        return: 2d numerical array
        """
        return self._grid[:,:,0]

    @property
    def cell_area(self):
        """
        return: pos number
        """
        return self.area / len(self)


def are_coordinates(obj):
    """
    Check if object are coordinates, i.e. a 2-number tuple.
    """
    if type(obj) is not tuple:
        return False
    if len(obj) != 2:
        return False
    if not is_number(obj[0]):
        return False
    if not is_number(obj[1]):
        return False
    return True


def prepare_coordinates(c1, c2, validate=True):
    """
    Convert coordinate arrays to same shape.

    For example, when c1 is a number and c2 is an nd array, c1 will be
    converted to the shape of c2.

    c1: numeric,
        if array then shape equals that of c2 and c3 if they are arrays
    c2: numeric,
        if array then shape equals that of c3 and c1 if they are arrays

    return: 2-number tuple or 2-array tuple, coordinate arrays with same shape
    """
    if validate is True:
        assert(is_numeric(c1))
        assert(is_numeric(c2))

    c1_is_array = is_array(c1)
    c2_is_array = is_array(c2)

    ## if one or both are arrays
    if c1_is_array or c2_is_array:
        ## both
        if c1_is_array and c2_is_array:
            if validate is True:
                assert(c1.shape == c2.shape)
        ## only c1
        elif c1_is_array:
            c2 = np.tile(c2, c1.shape)
        ## only c2
        elif c2_is_array:
            c1 = np.tile(c1, c2.shape)

    return c1, c2


@jit(nopython=True)
def _distance(x1, y1, x2, y2):
    """
    """
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


def distance(x1, y1, x2, y2, validate=True):
    """
    Calculate distance.

    x1: numeric, if array then shape must match y1
    y1: numeric, if array then shape must match x1
    x2: numeric, if array then shape must match y2
    y2: numeric, if array then shape must match x2

    return: number or array, distances with shape (1.shape + 2.shape)
    """
    x1, y1 = prepare_coordinates(x1, y1, validate=validate)
    x2, y2 = prepare_coordinates(x2, y2, validate=validate)

    if is_array(x1) and is_array(x2):
        x1 = np.tile(x1[(Ellipsis,)+tuple([None]*len(x2.shape))],
                     tuple([1]*len(x1.shape)) + x2.shape)
        y1 = np.tile(y1[(Ellipsis,)+tuple([None]*len(y2.shape))],
                     tuple([1]*len(y1.shape)) + y2.shape)

    distance = _distance(x1, y1, x2, y2)

    if not is_array(distance):
        distance = float(distance)

    return distance


def cartesian2polar(x, y, validate=True):
    """
    """
    if validate is True:
        assert(is_number(x))
        assert(is_number(y))

    r = float(distance(0, 0, x, y))
    a = float(np.degrees(np.arctan2(y, x)))

    if abs(r) < 10**-PRECISION:
        r = 0
    if abs(a) < 10**-PRECISION:
        a = 0

    return r, a


def polar2cartesian(r, a, validate=True):
    """
    """
    if validate is True:
        assert(is_number(r))
        assert(is_number(a))

    x = float(r * np.cos(np.radians(a)))
    y = float(r * np.sin(np.radians(a)))

    if abs(x) < 10**-PRECISION:
        x = 0
    if abs(y) < 10**-PRECISION:
        y = 0

    return x, y
