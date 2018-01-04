"""
The 'math.matrices' module.
"""


import numpy as np

from ..apy import Object, is_number, is_pos_integer, is_2d_numeric_array


class Matrix(Object):
    """
    """

    def __init__(self, array, validate=True):
        """
        """
        array = np.asarray(array, dtype=float)

        if validate is True:
            assert(is_2d_numeric_array(array))
            assert(array.size > 0)

        self._array = array

    def __len__(self):
        """
        """
        return self.size

    def __getitem__(self, item):
        """
        NOTE: Indices start at 1!
        """
        assert(type(item) is tuple and len(item) == 2)
        r = item[0] - 1
        c = item[1] - 1
        return float(self._array[r,c])

    def __iter__(self):
        """
        """
        for e in np.nditer(self._array):
            yield float(e)

    def __mul__(self, other):
        """
        """
        assert(isinstance(other, Matrix) or is_number(other))

        if is_number(other):
            array = self._array * other
        else:
            assert(other.nrows == self.ncols)
            array = np.dot(self._array, other._array)

        return self.__class__(array)

    @property
    def nrows(self):
        """
        return: pos integer
        """
        return self._array.shape[0]

    @property
    def ncols(self):
        """
        return: pos integer
        """
        return self._array.shape[1]

    @property
    def order(self):
        """
        return: 2-tuple
        """
        return (self.nrows, self.ncols)

    @property
    def size(self):
        """
        return: pos integer
        """
        return self.nrows * self.ncols

    @property
    def rows(self):
        """
        return: nrows-tuple
        """
        rows = []
        for i in range(self.nrows):
            rows.append(self._array[i,:][:])
        return tuple(rows)

    @property
    def cols(self):
        """
        return: ncols-tuple
        """
        cols = []
        for i in range(self.ncols):
            cols.append(self._array[:,i][:])
        return tuple(cols)

    def is_square(self):
        """
        return: boolean
        """
        return self.nrows == self.ncols


class SquareMatrix(Matrix):
    """
    Square matrix.
    """

    def __init__(self, array, validate=True):
        """
        """
        array = np.asarray(array, dtype=float)

        if validate is True:
            assert(is_2d_numeric_array(array))
            nrows = array.shape[0]
            ncols = array.shape[1]
            assert(nrows > 1)
            assert(ncols > 1)
            assert(nrows == ncols)

        self._array = array

    @property
    def trace(self):
        """
        return: number, sum of the main diagonal
        """
        return float(np.trace(self._array))


class IdentityMatrix(SquareMatrix):
    """
    A square matrix with ones on the main diagonal and zeros elsewhere.
    """

    def __init__(self, n, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(n))

        self._array = np.identity(n)
