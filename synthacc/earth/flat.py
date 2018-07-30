"""
The 'earth.flat' module.

Module for a flat Earth (3d Euclidean space in a right-handed Cartesian
coordinate system) where x is north, y is east and z is down (or depth).
Earth's surface has z=0. The azimuth is the angle from x (north) to y (east).
"""


import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import (Point as _Point, LineString as _LineString,
    Polygon as _Polygon)

from ..apy import (T, F, Object, is_number, is_non_neg_number, is_pos_number,
    is_integer, is_pos_integer, is_1d_numeric_array)
from .. import space2
from .. import space3


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

    def __getitem__(self, i):
        """
        """
        assert(is_integer(i))

        x, y = self._points[i]
        x = float(x)
        y = float(y)

        return x, y

    def __iter__(self):
        """
        """
        for i in range(len(self)):
            yield(self[i])

    @property
    def xs(self):
        """
        return: 1d numeric array, x coordinates of points on path
        """
        return np.copy(self._points[:,0])

    @property
    def ys(self):
        """
        return: 1d numeric array, y coordinates of points on path
        """
        return np.copy(self._points[:,1])


class Grid(Object):
    """
    Regular spaced points on the Earth's surface.
    """

    def __init__(self, outline, spacing, validate=True):
        """
        outline: (number, number, number, number) tuple
        spacing: pos number, spacing (in m)
        """
        if validate is True:
            assert(type(outline) is tuple)
            assert(len(outline) == 4)
            assert(is_number(outline[0]))
            assert(is_number(outline[1]))
            assert(is_number(outline[2]))
            assert(is_number(outline[3]))
            assert(is_pos_number(spacing))

        xs = np.arange(outline[0], outline[1] + spacing, spacing)
        ys = np.arange(outline[2], outline[3] + spacing, spacing)

        self._outline = outline
        self._spacing = spacing
        self._grid = np.dstack(np.meshgrid(xs, ys))

    def __len__(self):
        """
        """
        return np.prod(self.shape)

    def __getitem__(self, i):
        """
        """
        assert(type(i) is tuple and len(i) == 2)

        x, y = self._grid[i]
        x = float(x)
        y = float(y)

        return x, y

    def __iter__(self):
        """
        """
        for i in np.ndindex(self.shape):
            yield self[i]

    @property
    def outline(self):
        """
        """
        return self._outline

    @property
    def spacing(self):
        """
        """
        return self._spacing

    @property
    def xs(self):
        """
        return: 2d numerical array, xs
        """
        return self._grid[:,:,0]

    @property
    def ys(self):
        """
        return: 2d numerical array, ys
        """
        return self._grid[:,:,1]

    @property
    def shape(self):
        """
        return: (pos int, pos int) tuple, shape of grid
        """
        return self._grid.shape[:2]


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

    def get_simplified(self, n):
        """
        """
        l = _LineString([p for p in zip(self.xs, self.ys)])
        
        p = len(l.coords)
        t = 1
        while p > (n+1):
            s = l.simplify(tolerance=t, preserve_topology=True)
            p = len(s.coords)
            t += 1

        xs, ys = s.coords.xy

        return self.__class__(list(xs), list(ys))

    def plot(self, points=False, selection=[], size=None, validate=True):
        """
        """
        plot_paths([self], points, selection, size, validate=validate)


class Rectangle(Object):
    """
    A rectangular surface below the Earth's surface.
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

    def __contains__(self, point):
        """
        """
        point = space3.Point(*point)

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
    def depth_range(self):
        """
        return: pos number
        """
        return self.lower_depth - self.upper_depth

    @property
    def strike(self):
        """
        return: non neg number
        """
        return azimuth(self._x1, self._y1, self._x2, self._y2, validate=False)

    @property
    def dip(self):
        """
        return: number, dip (angle)
        """
        return self._dip

    @property
    def dip_azimuth(self):
        """
        return: non neg number
        """
        a = self.strike + 90
        if a >= 360:
            return a - 360
        else:
            return a

    @property
    def as_vector(self):
        """
        return: 'space.Vector' instance
        """
        return self.urc.vector - self.ulc.vector

    @property
    def ad_vector(self):
        """
        return: 'space.Vector' instance
        """
        rm = space3.RotationMatrix.from_basic_rotations(z=self.dip_azimuth)
        v = self.as_vector.unit.rotate(rm) * self.surface_width
        return space3.Vector(v.x, v.y, self.depth_range)

    @property
    def length(self):
        """
        return: pos number, length (in m)
        """
        return self.as_vector.magnitude

    @property
    def width(self):
        """
        return: pos number, width (in m)
        """
        return self.ad_vector.magnitude

    @property
    def area(self):
        """
        return: pos number, area (in m²)
        """
        return self.length * self.width

    @property
    def surface_width(self):
        """
        return: pos number, surface width (in m)
        """
        return float(self.depth_range / np.tan(np.radians(self.dip)))

    @property
    def center(self):
        """
        return: 'space3.Point' instance
        """
        return self.ulc.translate((self.as_vector + self.ad_vector) / 2)

    @property
    def plane(self):
        """
        return: 'space3.Plane' instance
        """
        return space3.Plane.from_points(self.ulc, self.urc, self.llc)

    @property
    def surface(self):
        """
        """
        return space2.RectangularSurface(self.width, self.length)

    def get_discretized(self, shape, validate=True):
        """
        Get a discretized rectangular surface.

        return: 'earth.flat.DiscretizedRectangle' instance
        """
        drs = DiscretizedRectangle(
            self._x1, self._y1,
            self._x2, self._y2,
            self.upper_depth,
            self.lower_depth,
            self.dip, shape,
            validate=validate,
            )
        return drs

    def get_random(self):
        """
        Get a random point on the surface.

        return: 'space3.Point' instance
        """
        l_vector = self.as_vector * np.random.uniform(0, 1)
        w_vector = self.ad_vector * np.random.uniform(0, 1)
        x, y, z = self.ulc.translate(l_vector + w_vector)

        return space3.Point(x, y, z)

    def plot(self, fill=True, validate=True):
        """
        """
        plot_rectangles([self], fill=fill, validate=validate)


class DiscretizedRectangle(Rectangle):
    """
    A discretized rectangular surface below the Earth's surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, shape, validate=True):
        """
        """
        super().__init__(x1, y1, x2, y2, upper_depth, lower_depth, dip,
            validate=validate)

        if validate is True:
            assert(type(shape) is tuple)
            assert(is_pos_integer(shape[0]))
            assert(is_pos_integer(shape[1]))

        self._shape = shape
        self._spacing = (self.width / shape[0], self.length / shape[1])

        corners, centers = self._discretize()

        self._corners = corners
        self._centers = centers

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
    def corners(self):
        """
        """
        return self._corners

    @property
    def centers(self):
        """
        """
        return self._centers

    @property
    def cell_area(self):
        """
        return: pos number
        """
        return self.area / len(self)

    def _discretize(self):
        """
        """
        l_n = self.shape[1] + 1
        w_n = self.shape[0] + 1

        as_vector = self.as_vector
        ad_vector = self.ad_vector

        as_vectors = [as_vector * float(m) for m in np.linspace(0, 1, l_n)]
        ad_vectors = [ad_vector * float(m) for m in np.linspace(0, 1, w_n)]

        corners = np.zeros((w_n, l_n, 3))
        for w_i, l_i in np.ndindex((w_n, l_n)):
            p = self.ulc.translate(ad_vectors[w_i] + as_vectors[l_i])
            corners[w_i,l_i] = tuple(p)

        v = ad_vector.unit * self.spacing[0] + as_vector.unit * self.spacing[1]

        centers = corners[0:-1,0:-1] + (v.x, v.y, v.z)

        return corners, centers

    def get_front_projected_corners(self):
        """
        """
        l, w = self.length, self.width
        nw, nl = self.shape
        xs = np.linspace(0, l, nl+1)
        ys = np.linspace(0, w, nw+1)
        xs, ys = np.meshgrid(xs, ys)
        return xs, ys

    def get_front_projected_centers(self):
        """
        """
        l, w = self.length, self.width
        nw, nl = self.shape
        cl = (l / nl) / 2
        cw = (w / nw) / 2
        xs = np.linspace(cl, l-cl, nl)
        ys = np.linspace(cw, w-cw, nw)
        xs, ys = np.meshgrid(xs, ys)
        return xs, ys


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
    return is_number(obj) and (0 <= obj < 360)


def is_dip(obj):
    """
    Check if object is dip.
    """
    return is_number(obj) and (0 < obj <= 90)


def plot_paths(paths, points=False, selection=[], size=None, validate=True):
    """
    """
    if validate is True:
        pass

    _, ax = plt.subplots(figsize=size)

    for p in paths:
        ax.plot(p.ys, p.xs, c='k', lw=2)

        if points is True:
            ax.scatter(p.ys, p.xs, c='k')

    for p in selection:
        ax.scatter(p[1], p[0], c='r')

    plt.axis('equal')

    x_label, y_label = 'East (m)', 'North (m)'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    plt.show()


def plot_rectangles(rectangles, fill=True, size=None, validate=True):
    """
    """
    if validate is True:
        pass

    _, ax = plt.subplots(figsize=size)

    for r in rectangles:
        ulc, urc, llc, lrc = r.corners

        ax.plot([ulc.y, urc.y], [ulc.x, urc.x], c='r', lw=2)

        if fill is True:
            ax.fill(
                [ulc.y, urc.y, lrc.y, llc.y],
                [ulc.x, urc.x, lrc.x, llc.x],
                color='coral', alpha=0.5,
                )

    ax.axis('equal')

    x_label, y_label = 'East (m)', 'North (m)'
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)

    plt.show()
