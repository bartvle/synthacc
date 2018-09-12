"""
The 'source.faults' module.
"""


from ..apy import is_non_neg_number, is_pos_number
from ..earth import flat as earth
# from .moment import calculate as calculate_moment, m0_to_mw


## Average rigidity (in Pa) in crust (from USGS website)
RIGIDITY = 3.2 * 10**10


class SimpleFault(earth.SimpleSurface):
    """
    A fault with a rectangular surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, rigidity=RIGIDITY, upper_sd=None, lower_sd=None, validate=True):
        """
        """
        super().__init__(
            x1, y1, x2, y2, upper_depth, lower_depth, dip, validate=validate)

        if upper_sd is None:
            upper_sd = upper_depth
        if lower_sd is None:
            lower_sd = lower_depth

        if validate is True:
            assert(is_pos_number(rigidity))
            assert(is_non_neg_number(upper_sd) and upper_sd >= upper_depth)
            assert(is_non_neg_number(lower_sd) and lower_sd <= lower_depth)

        self._rigidity = rigidity
        self._upper_sd = upper_sd
        self._lower_sd = lower_sd

    @property
    def rigidity(self):
        """
        return: pos number
        """
        return self._rigidity

    @property
    def upper_sd(self):
        """
        """
        return self._upper_sd

    @property
    def lower_sd(self):
        """
        """
        return self._lower_sd

#     @property
#     def rectangle(self):
#         """
#         """
#         r = Rectangle(
#             self._x1, self._y1,
#             self._x2, self._y2,
#             self._upper_depth,
#             self._lower_depth,
#             self._dip,
#             self._rigidity)

#         return r

#     def get_max_moment(self, slip, validate=True):
#         """
#         slip: pos number, slip (in m)

#         return: pos number, maximum seismic moment
#         """
#         return calculate_moment(self.area, slip, self.rigidity, validate)

#     def get_max_magnitude(self, slip, validate=True):
#         """
#         slip: pos number, slip (in m)

#         return: number, maximum moment magnitude
#         """
#         return m0_to_mw(self.get_max_moment(slip, validate), validate=False)


# class ComposedFault(Object):
#     """
#     A fault composed of multiple simple faults.
#     """

#     def __init__(self, faults, validate=True):
#         """
#         faults: list of 'faults.SimpleFault' instances
#         """
#         if validate is True:
#             for f in faults:
#                 assert(type(f) is SimpleFault)

#         self._faults = faults

#     def __len__(self):
#         """
#         """
#         return len(self._faults)

#     def __getitem__(self, i):
#         """
#         """
#         assert(type(i) is int)
#         return self._faults[i]

#     def __iter__(self):
#         """
#         """
#         for f in self._faults:
#             yield f

#     @property
#     def length(self):
#         """
#         return: pos number
#         """
#         l = 0
#         for f in self:
#             l += f.length
#         return l

#     @property
#     def area(self):
#         """
#         return: pos number
#         """
#         a = 0
#         for f in self:
#             a += f.area
#         return a

#     def get_max_moment(self, slip, validate=True):
#         """
#         slip: pos number, slip (in m)

#         return: pos number, maximum seismic moment
#         """
#         if validate is True:
#             assert(is_pos_number(slip))

#         return sum([f.get_max_moment(slip, validate=False) for f in self])

#     def get_max_magnitude(self, slip, validate=True):
#         """
#         slip: pos number, slip (in m)

#         return: number, maximum moment magnitude
#         """
#         return m0_to_mw(self.get_max_moment(slip, validate), validate=False)
