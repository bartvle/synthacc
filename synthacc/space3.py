"""
The 'space3' module. 3d Euclidean space in a right-handed Cartesian coordinate
system.
"""


import math

from numba import jit
import numpy as np

from .apy import PRECISION, Object, is_integer, is_number, is_array, is_numeric
from .math import Matrix, SquareMatrix


class Point(Object):
    """
    """

    def __init__(self, x, y, z=0, validate=True):
        """
        x: number, x coordinate
        y: number, y coordinate
        z: number, z coordinate (default: 0)
        """
        if validate is True:
            assert(is_number(x))
            assert(is_number(y))
            assert(is_number(z))

        if abs(x) < 10**-PRECISION:
            x = 0
        if abs(y) < 10**-PRECISION:
            y = 0
        if abs(z) < 10**-PRECISION:
            z = 0

        self._x = x
        self._y = y
        self._z = z

    def __repr__(self):
        """
        """
        s = '< space3.Point | '
        s += 'x={:{}.3f}'.format(self.x, '+' if self.x else '')
        s += ', '
        s += 'y={:{}.3f}'.format(self.y, '+' if self.y else '')
        s += ', '
        s += 'z={:{}.3f}'.format(self.z, '+' if self.z else '')
        s += ' >'

        return s

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        return (self._x, self._y, self._z)[i]

    def __eq__(self, other):
        """
        return: boolean
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        x, y, z = other
        x_eq = abs(self.x - x) < 10**-PRECISION
        y_eq = abs(self.y - y) < 10**-PRECISION
        z_eq = abs(self.z - z) < 10**-PRECISION

        return (x_eq and y_eq and z_eq)

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

    @property
    def z(self):
        """
        return: number, z coordinate
        """
        return self._z

    @property
    def vector(self):
        """
        return: 'space3.Vector' instance, position vector
        """
        return Vector(*self)

    def translate(self, vector, validate=True):
        """
        Translate point by a vector.

        vector: 'space3.Vector' instance or coordinates

        return: class instance
        """
        if validate is True:
            assert(type(vector) is Vector or are_coordinates(vector))

        x, y, z = vector
        x = self.x + x
        y = self.y + y
        z = self.z + z

        return self.__class__(x, y, z, validate=False)

    def rotate(self, rotation_matrix, origin=(0, 0, 0), validate=True):
        """
        Rotate around a point.

        rotation_matrix: 'space3.RotationMatrix' instance
        origin: 'space3.Point' instance or coordinates, Point to rotate around
            (default: (0, 0, 0), rotate around origin)

        return: class instance
        """
        if validate is True:
            assert(type(origin) is Point or are_coordinates(origin))

        origin, point = Point(*origin).vector, self

        ## translate -> origin
        point = point.translate(-origin)

        ## rotate
        x, y, z = rotation_matrix * point.vector.col
        point = Point(x, y, z)

        ## translate <- origin
        point = point.translate(+origin)

        return point


class Plane(Object):
    """
    A plane. The general equation of a plane is ax + by + cz + d = 0.
    """

    def __init__(self, a, b, c, d, validate=True):
        """
        a: number, a parameter
        b: number, b parameter
        c: number, c parameter
        d: number, d parameter
        """
        if validate is True:
            assert(is_number(a))
            assert(is_number(b))
            assert(is_number(c))
            assert(is_number(d))

        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        return (self._a, self._b, self._c, self._d)[i]

    @classmethod
    def from_points(cls, p1, p2, p3, validate=True):
        """
        #TODO: add assert that points are not on one line
        """
        p1 = Point(*p1, validate=validate)
        p2 = Point(*p2, validate=validate)
        p3 = Point(*p3, validate=validate)

        v1 = p2.vector - p1.vector
        v2 = p3.vector - p1.vector

        a, b, c = v1 @ v2

        d = -(a*p1.x + b*p1.y + c*p1.z)

        return cls(a, b, c, d, validate=False)

    @property
    def a(self):
        """
        return: number, a parameter
        """
        return self._a

    @property
    def b(self):
        """
        return: number, b parameter
        """
        return self._b

    @property
    def c(self):
        """
        return: number, c parameter
        """
        return self._c

    @property
    def d(self):
        """
        return: number, d parameter
        """
        return self._d

    @property
    def normal(self):
        """
        return: 'space3.Vector' instance
        """
        return Vector(self.a, self.b, self.c, validate=False)

    def get_distance(self, p, validate=True):
        """
        Get distance of a point to the plane.

        p: 'space3.Point' instance or coordinates

        return: pos number
        """
        x, y, z = Point(*p, validate=validate)

        d = (abs(self.a*x + self.b*y + self.c*z + self.d)
            / math.sqrt(self.a**2+self.b**2+self.c**2))

        if d < 10**-9: #TODO: Works only with precision of 9, not 10.
            d = 0

        return d


class Vector(Object):
    """
    A (free) vector (i.e. between origin and a point).
    """

    def __init__(self, x, y, z=0, validate=True):
        """
        x: number, x coordinate
        y: number, y coordinate
        z: number, z coordinate (default: 0)
        """
        if validate is True:
            assert(is_number(x))
            assert(is_number(y))
            assert(is_number(z))

        if abs(x) < 10**-PRECISION:
            x = 0
        if abs(y) < 10**-PRECISION:
            y = 0
        if abs(z) < 10**-PRECISION:
            z = 0

        self._x = x
        self._y = y
        self._z = z

    def __repr__(self):
        """
        """
        s = '< space3.Vector | '
        s += 'x={:{}.3f}'.format(self.x, '+' if self.x else '')
        s += ', '
        s += 'y={:{}.3f}'.format(self.y, '+' if self.y else '')
        s += ', '
        s += 'z={:{}.3f}'.format(self.z, '+' if self.z else '')
        s += ' >'
        return s

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        return (self._x, self._y, self._z)[i]

    def __eq__(self, other):
        """
        return: boolean
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        x, y, z = other
        x_eq = abs(self.x - x) < 10**-PRECISION
        y_eq = abs(self.y - y) < 10**-PRECISION
        z_eq = abs(self.z - z) < 10**-PRECISION

        return (x_eq and y_eq and z_eq)

    def __add__(self, other):
        """
        return: class instance
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        x, y, z = other
        x = self._x + x
        y = self._y + y
        z = self._z + z

        return self.__class__(x, y, z)

    def __sub__(self, other):
        """
        return: class instance
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        x, y, z = other
        x = self._x - x
        y = self._y - y
        z = self._z - z

        return self.__class__(x, y, z)

    def __mul__(self, other):
        """
        Scalar product if other is number or dot product if other is vector.

        Dot product = (x1x2 + y1y2 + z1z2)
                    = (magnitude1 * magnitude2) * cos(angle)
        a . b = b . a

        return: class instance
        """
        assert(is_number(other) or type(other) is self.__class__ or
            are_coordinates(other))

        if is_number(other):
            x = self._x * other
            y = self._y * other
            z = self._z * other
            return self.__class__(x, y, z)
        else:
            return float(np.dot(tuple(self), tuple(other)))

    def __truediv__(self, other):
        """
        return: class instance
        """
        assert(is_number(other))

        x = self._x / other
        y = self._y / other
        z = self._z / other

        return self.__class__(x, y, z)

    def __matmul__(self, other):
        """
        Cross product = (y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2)

        magnitude = (magnitude1 * magnitude2) * sin(angle)
        a x b = -(b x a)

        return: class instance
        """
        assert(type(other) is self.__class__ or are_coordinates(other))

        other = Vector(*other)

        x = self._y * other._z - self._z * other._y
        y = self._z * other._x - self._x * other._z
        z = self._x * other._y - self._y * other._x

        return Vector(x, y, z)

    def __pos__(self):
        """
        return: class instance
        """
        return self.__class__(+self.x, +self.y, +self.z)

    def __neg__(self):
        """
        return: class instance
        """
        return self.__class__(-self.x, -self.y, -self.z)

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

    @property
    def z(self):
        """
        return: number, z coordinate
        """
        return self._z

    @property
    def row(self):
        """
        """
        a = np.array([
            [self._x, self._y, self._z],
            ])
        return Matrix(a)

    @property
    def col(self):
        """
        """
        a = np.array([
            [self._x],
            [self._y],
            [self._z],
            ])
        return Matrix(a)

    @property
    def magnitude(self):
        """
        return: pos number, magnitude
        """
        return distance(0, 0, 0, self.x, self.y, self.z, validate=False)

    @property
    def unit(self):
        """
        return: class instance or None, unit vector
        """
        if self == (0, 0, 0):
            return None
        else:
            return self / self.magnitude

    def rotate(self, rotation_matrix, validate=True):
        """
        Rotate around origin.

        rotation_matrix: 'space3.RotationMatrix' instance

        return: class instance
        """
        if validate is True:
            assert(type(rotation_matrix) is RotationMatrix)

        return Vector(*(rotation_matrix * self.col))

    def get_angle(self, other, validate=True):
        """
        Angle with other vector.

        return: angle (in degrees)
        """
        if validate is True:
            assert(type(other) is self.__class__ or are_coordinates(other))

        other = Vector(*other)

        if self == (0, 0, 0) or other == (0, 0, 0):
            return None

        angle = math.degrees(
            math.acos(self * other / (self.magnitude * other.magnitude)))

        return float(angle)


class RotationMatrix(SquareMatrix):
    """
    Rotate around origin. To rotate around a point first translate the origin
    to that point, then do the rotation and finally translate back to the
    point.
    """

    def __init__(self, array, validate=True):
        """
        """
        super().__init__(array, validate=validate)

        if validate is True:
            assert(self.order == (3, 3))

    @classmethod
    def from_basic_rotations(cls, x=0, y=0, z=0, validate=True):
        """
        Rotation in x-y-z order.

        x: angle (in degrees) around the x axis (default: 0)
        y: angle (in degrees) around the y axis (default: 0)
        z: angle (in degrees) around the z axis (default: 0)
        """
        if validate is True:
            assert(is_number(x))
            assert(is_number(y))
            assert(is_number(z))

        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

        r_x = np.array([
            [1, 0, 0],
            [0, +math.cos(x), -math.sin(x)],
            [0, +math.sin(x), +math.cos(x)],
            ])
        r_y = np.array([
            [+math.cos(y), 0, +math.sin(y)],
            [0, 1, 0],
            [-math.sin(y), 0, +math.cos(y)],
            ])
        r_z = np.array([
            [+math.cos(z), -math.sin(z), 0],
            [+math.sin(z), +math.cos(z), 0],
            [0, 0, 1],
            ])

        r_x[abs(r_x) < 10**-PRECISION] = 0
        r_y[abs(r_y) < 10**-PRECISION] = 0
        r_z[abs(r_z) < 10**-PRECISION] = 0

        r = r_z.dot(r_y).dot(r_x)

        return cls(r)

    @classmethod
    def from_axis_and_angle(cls, axis, angle, validate=True):
        """
        Rotation around one axis.
        """
        if validate is True:
            assert(type(axis) is Vector or are_coordinates(axis))
            assert(is_number(angle))

        x, y, z = Vector(*axis).unit

        angle = math.radians(angle)
        cos = math.cos(angle)
        sin = math.sin(angle)

        r = np.array([
                [cos + x**2 * (1-cos), x*y*(1-cos)-z*sin, x*z*(1-cos)+y*sin],
                [y*x*(1-cos)+z*sin, cos + y**2 * (1-cos), y*z*(1-cos)-x*sin],
                [z*x*(1-cos)-y*sin, z*y*(1-cos)+x*sin, cos + z**2 * (1-cos)],
            ])

        return cls(r)


def are_coordinates(obj):
    """
    Check if object are coordinates, i.e. a 3-number tuple.
    """
    if type(obj) is not tuple:
        return False
    if len(obj) != 3:
        return False
    if not is_number(obj[0]):
        return False
    if not is_number(obj[1]):
        return False
    if not is_number(obj[2]):
        return False
    return True


def prepare_coordinates(c1, c2, c3, validate=True):
    """
    Convert coordinate arrays to same shape.

    For example, when c1 is a number, and c2 and c3 are nd arrays with the same
    shape, c1 will be converted to that shape.

    c1: numeric,
        if array then shape equals that of c2 and c3 if they are arrays
    c2: numeric,
        if array then shape equals that of c3 and c1 if they are arrays
    c3: numeric,
        if array then shape equals that of c1 and c2 if they are arrays

    return: 3-number tuple or 3-array tuple, coordinate arrays with same shape
    """
    if validate is True:
        assert(is_numeric(c1))
        assert(is_numeric(c2))
        assert(is_numeric(c3))

    c1_is_array = is_array(c1)
    c2_is_array = is_array(c2)
    c3_is_array = is_array(c3)

    ## if one or more are arrays
    if c1_is_array or c2_is_array or c3_is_array:
        ## all three
        if (c1_is_array and c2_is_array and c3_is_array):
            if validate is True:
                assert(c1.shape == c2.shape == c3.shape)
        ## two (c1 and c2)
        elif c1_is_array and c2_is_array:
            if validate is True:
                assert(c1.shape == c2.shape)
            c3 = np.tile(c3, c2.shape)
        ## two (c2 and c3)
        elif c2_is_array and c3_is_array:
            if validate is True:
                assert(c1.shape == c3.shape)
            c1 = np.tile(c1, c3.shape)
        ## two (c3 and c1)
        elif c3_is_array and c1_is_array:
            if validate is True:
                assert(c3.shape == c1.shape)
            c2 = np.tile(c2, c1.shape)
        ## only c1
        elif c1_is_array:
            c2 = np.tile(c2, c1.shape)
            c3 = np.tile(c3, c1.shape)
        ## only c2
        elif c2_is_array:
            c3 = np.tile(c3, c2.shape)
            c1 = np.tile(c1, c2.shape)
        ## only c3
        elif c3_is_array:
            c1 = np.tile(c1, c3.shape)
            c2 = np.tile(c2, c3.shape)

    return c1, c2, c3


@jit(nopython=True)
def _distance(x1, y1, z1, x2, y2, z2):
    """
    """
    return np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)


def distance(x1, y1, z1, x2, y2, z2, validate=True):
    """
    Calculate distance.

    x1: numeric, if array then shape must match y1 and z1
    y1: numeric, if array then shape must match z1 and x1
    z1: numeric, if array then shape must match x1 and y1
    x2: numeric, if array then shape must match y2 and z2
    y2: numeric, if array then shape must match z2 and x2
    z2: numeric, if array then shape must match x2 and y2

    return: number or array, distances with shape (1.shape + 2.shape)
    """
    x1, y1, z1 = prepare_coordinates(x1, y1, z1, validate=validate)
    x2, y2, z2 = prepare_coordinates(x2, y2, z2, validate=validate)

    if is_array(x1) and is_array(x2):
        x1 = np.tile(x1[(Ellipsis,)+tuple([None]*len(x2.shape))],
                     tuple([1]*len(x1.shape)) + x2.shape)
        y1 = np.tile(y1[(Ellipsis,)+tuple([None]*len(y2.shape))],
                     tuple([1]*len(y1.shape)) + y2.shape)
        z1 = np.tile(z1[(Ellipsis,)+tuple([None]*len(z2.shape))],
                     tuple([1]*len(z1.shape)) + z2.shape)
    distance = _distance(x1, y1, z1, x2, y2, z2)

    if not is_array(distance):
        distance = float(distance)

    return distance


def nearest(x, y, z, xs, ys, zs, validate=True):
    """
    Find nearest point in cloud.

    x: number, x coordinate of point
    y: number, y coordinate of point
    z: number, z coordinate of point
    xs: nd array, x coordinates of cloud
    ys: nd array, y coordinates of cloud
    zs: nd array, z coordinates of cloud

    return: 3-number tuple, (x, y, z) coordinates of nearest point
    """
    distances = distance(x, y, z, xs, ys, zs)
    index = np.unravel_index(distances.argmin(), distances.shape)

    x = float(xs[index])
    y = float(ys[index])
    z = float(zs[index])

    return x, y, z
