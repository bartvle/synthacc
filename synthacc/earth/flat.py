"""
The 'earth.flat' module. A flat Earth as 3-dimensional Euclidean space in a
right-handed Cartesian coordinate system where x is north, y is east and z is
down (or depth). Earth's surface has z=0. The azimuth is the angle from x
(north) to y (east).
"""


import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import (Point as _Point, LineString as _LineString,
    Polygon as _Polygon)

from ..apy import (T, F, Object, is_integer, is_pos_integer, is_number,
    is_non_neg_number, is_1d_numeric_array)
from .. import space2, space3


class Sites(Object):
    """
    Irregular spaced points on the Earth's surface.
    """

    def __init__(self, xs, ys, validate=True):
        """
        xs: 1d numeric array, x coordinates of points on path
            (same shape as ys)
        ys: 1d numeric array, y coordinates of points on path
            (same shape as xs)
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(xs))
            assert(is_1d_numeric_array(ys))
            assert(len(xs) == len(ys))

        self._points = np.stack([xs, ys], axis=1)

    def __len__(self):
        """
        """
        return len(self._points)

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        x, y = self._points[i]
        x = float(x)
        y = float(y)

        return x, y

    def __iter__(self):
        """
        """
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_points(cls, points, validate=True):
        """
        """
        if validate is True:
            assert(type(points) in (cls, list))
 
        xs = np.zeros(len(points))
        ys = np.zeros(len(points))
        for i, p in enumerate(points):
            xs[i] = p[0]
            ys[i] = p[1]

        return cls(xs, ys)

    @property
    def xs(self):
        """
        return: 1d numeric array, x coordinates of points on path
        """
        return self._points[:,0]

    @property
    def ys(self):
        """
        return: 1d numeric array, y coordinates of points on path
        """
        return self._points[:,1]


class Path(Sites):
    """
    Points that form a path on the Earth's surface.
    """

    @property
    def length(self):
        """
        return: pos number, length of path (in m)
        """
        length = 0
        for i in range(len(self)-1):
            x1, y1, = self[i+0]
            x2, y2, = self[i+1]
            length += space3.distance(x1, y1, 0, x2, y2, 0, validate=False)

        return length

    def get_simplified(self, n, validate=True):
        """
        Simplification of path by iterative increasing of tolerance with one.
        It is possible that the output of this algorithm gives a number of
        pieces that is lower than the desired number.

        n: positive integer, number of pieces of path (n+1 points)
        """
        if validate is True:
            assert(is_pos_integer(n))

        l = _LineString([p for p in zip(self.xs, self.ys)])

        p = len(l.coords)
        t = 1
        while p > (n+1):
            s = l.simplify(tolerance=t, preserve_topology=True)
            p = len(s.coords)
            t += 1

        xs, ys = s.coords.xy

        return self.__class__(list(xs), list(ys))

    def plot(self, points=False, size=None, filespec=None, validate=True):
        """
        """
        plot_paths([self], points, size, filespec, validate=validate)


class SimpleSurface(Object):
    """
    A rectangular surface with upper and lower side below and parallel to
    Earth's surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, validate=True):
        """
        """
        if validate is True:
            assert(is_number(x1))
            assert(is_number(y1))
            assert(is_number(x2))
            assert(is_number(y2))
            assert(is_non_neg_number(upper_depth))
            assert(is_non_neg_number(lower_depth))
            assert(lower_depth > upper_depth)
            assert(is_dip(dip))

        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._upper_depth = upper_depth
        self._lower_depth = lower_depth
        self._dip = dip

    def __contains__(self, point, validate=True):
        """
        """
        point = space3.Point(*point, validate=validate)

        ulc, urc, llc, lrc = self.corners
        _polygon = _Polygon((
            (ulc.x, ulc.y), (urc.x, urc.y),
            (lrc.x, lrc.y), (llc.x, llc.y),
             ))
        _point = _Point(point.x, point.y)

        if (self.plane.get_distance(point) == 0 and (
            _polygon.contains(_point) or _polygon.intersects(_point))):
            return T
        else:
            return F

    @property
    def upper_depth(self):
        """
        return: non neg number, upper depth (in m)
        """
        return self._upper_depth

    @property
    def lower_depth(self):
        """
        return: nog neg number, lower depth (in m)
        """
        return self._lower_depth

    @property
    def dip(self):
        """
        return: number, dip (angle)
        """
        return self._dip

    @property
    def ulc(self):
        """
        return: 'space3.Point' instance
        """
        return space3.Point(self._x1, self._y1, self.upper_depth)

    @property
    def urc(self):
        """
        return: 'space3.Point' instance
        """
        return space3.Point(self._x2, self._y2, self.upper_depth)

    @property
    def llc(self):
        """
        return: 'space3.Point' instance
        """
        return self.ulc.translate(self.ad_vector)

    @property
    def lrc(self):
        """
        return: 'space3.Point' instance
        """
        return self.urc.translate(self.ad_vector)

    @property
    def corners(self):
        """
        """
        return (self.ulc, self.urc, self.llc, self.lrc)

    @property
    def ad_vector(self):
        """
        Vector along dip.

        return: 'space3.Vector' instance
        """
        rm = space3.RotationMatrix.from_basic_rotations(z=90)
        x, y, _ = self.as_vector.unit.rotate(rm) * self.surface_width
        v = space3.Vector(x, y, self.depth_range)

        return v

    @property
    def as_vector(self):
        """
        Vector along strike.

        return: 'space3.Vector' instance
        """
        return self.urc.vector - self.ulc.vector

    @property
    def width(self):
        """
        return: pos number, width (in m)
        """
        return self.ad_vector.magnitude

    @property
    def length(self):
        """
        return: pos number, length (in m)
        """
        return self.as_vector.magnitude

    @property
    def strike(self):
        """
        return: non neg number
        """
        return azimuth(self._x1, self._y1, self._x2, self._y2, validate=False)

    @property
    def surface_width(self):
        """
        return: pos number, surface width (in m)
        """
        return float(self.depth_range / np.tan(np.radians(self.dip)))

    @property
    def depth_range(self):
        """
        return: pos number
        """
        return self.lower_depth - self.upper_depth

    @property
    def area(self):
        """
        return: pos number, area (in m²)
        """
        return self.length * self.width

    @property
    def plane(self):
        """
        return: 'space3.Plane' instance
        """
        return space3.Plane.from_points(self.ulc, self.urc, self.llc)

    @property
    def surface(self):
        """
        return 'space2.RectangularSurface' instance
        """
        return space2.RectangularSurface(self.width, self.length)

    def get_discretized(self, shape, validate=True):
        """
        Get a discretized simple surface.

        return: 'earth.flat.DiscretizedSimpleSurface' instance
        """
        if validate is True:
            assert(type(shape) is tuple and len(shape) == 2)
            assert(is_pos_integer(shape[0]))
            assert(is_pos_integer(shape[1]))

        drs = DiscretizedSimpleSurface(self._x1, self._y1, self._x2, self._y2,
            self.upper_depth, self.lower_depth, self.dip, shape,
            validate=False)

        return drs


class DiscretizedSimpleSurface(SimpleSurface):
    """
    A discretized rectangular surface with upper and lower side below and
    parallel to Earth's surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, shape, validate=True):
        """
        """
        super().__init__(x1, y1, x2, y2, upper_depth, lower_depth, dip,
            validate=validate)

        if validate is True:
            assert(type(shape) is tuple and len(shape) == 2)
            assert(is_pos_integer(shape[0]))
            assert(is_pos_integer(shape[1]))

        self._shape = shape
        self._spacing = (self.width / shape[0], self.length / shape[1])

        corners, centers = self._discretize()
        self._cell_corners = corners
        self._cell_centers = centers

    def __len__(self):
        """
        """
        return np.prod(self.shape)

    @property
    def shape(self):
        """
        """
        return self._shape

    @property
    def spacing(self):
        """
        """
        return self._spacing

    @property
    def cell_corners(self):
        """
        """
        return self._cell_corners

    @property
    def cell_centers(self):
        """
        """
        return self._cell_centers

    @property
    def cell_area(self):
        """
        return: pos number
        """
        return self.area / len(self)

    @property
    def surface(self):
        """
        """
        s = space2.DiscretizedRectangularSurface(self.width, self.length,
            self._shape[0], self._shape[1])

        return s

    def _discretize(self, validate=True):
        """
        """
        nwc = self._shape[0] + 1
        nlc = self._shape[1] + 1

        ad_vector = self.ad_vector
        as_vector = self.as_vector

        ad_vectors = [ad_vector * float(m) for m in np.linspace(0, 1, nwc)]
        as_vectors = [as_vector * float(m) for m in np.linspace(0, 1, nlc)]

        corners = np.zeros((nwc, nlc, 3))
        for iw, il in np.ndindex((nwc, nlc)):
            p = self.ulc.translate(ad_vectors[iw] + as_vectors[il])
            corners[iw,il] = tuple(p)

        v = ad_vector.unit * self.spacing[0] + as_vector.unit * self.spacing[1]
        centers = corners[0:-1,0:-1] + (v.x, v.y, v.z)

        return corners, centers


def azimuth(x1, y1, x2, y2, validate=True):
    """
    Azimuth between north and line between two points.
    """
    a = np.degrees(np.arctan2(y2-y1, x2-x1))

    if a < 0:
        a += 360

    return float(a)


def is_azimuth(obj):
    """
    Check if object is azimuth.
    """
    return is_number(obj) and (0 <= obj < 360)


def is_strike(obj):
    """
    Check if object is strike.
    """
    return is_azimuth(obj)


def is_dip(obj):
    """
    Check if object is dip.
    """
    return is_number(obj) and (0 < obj <= 90)


def plot_paths(paths, points=False, size=None, filespec=None, validate=True):
    """
    """
    if validate is True:
        assert(type(paths) is list)

    _, ax = plt.subplots(figsize=size)

    for p in paths:
        ax.plot(p.ys, p.xs, c='k', lw=2)

        if points is True:
            ax.scatter(p.ys, p.xs, c='r')

    plt.axis('equal')

    x_label, y_label = 'East (m)', 'North (m)'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()


def plot_simple_surfaces(simple_surfaces, colors=None, styles=None, widths=None, fill_colors=None, size=None, filespec=None, validate=True):
    """
    """
    if validate is True:
        assert(type(simple_surfaces) is list)

    _, ax = plt.subplots(figsize=size)

    for i, r in enumerate(simple_surfaces):
        ulc, urc, llc, lrc = r.corners

        kwargs = {}
        if colors is not None:
            kwargs['color'] = colors[i]
        if styles is not None:
            kwargs['ls'] = styles[i]
        if widths is not None:
            kwargs['lw'] = widths[i]

        ax.plot([ulc.y, urc.y], [ulc.x, urc.x], **kwargs)

        kwargs = {}
        if fill_colors is not None:
            kwargs['color'] = fill_colors[i]

        ax.fill(
            [ulc.y, urc.y, lrc.y, llc.y],
            [ulc.x, urc.x, lrc.x, llc.x],
            alpha=0.5, **kwargs,
            )

    ax.axis('scaled')

    x_label, y_label = 'East (m)', 'North (m)'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
