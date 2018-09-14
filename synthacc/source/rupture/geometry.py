"""
The 'source.rupture.geometry' module.
"""


import numpy as np

from ...apy import Object, is_number, is_pos_number
from ... import space3
from ...earth import flat as earth
from ..faults import SimpleFault
from ..scaling import ScalingRelationship


class FaultGeometryCalculator(Object):
    """
    """

    def __init__(self, n, mrd, validate=True):
        """
        """
        self._n = n
        self._mrd = mrd

    def __call__(self, fault_data, validate=True):
        """
        """
        trace = fault_data['trace'].get_simplified(n=1)
        if self._n != 1:
            trace_n = fault_data['trace'].get_simplified(n=self._n)
            p1 = space3.Point(*trace[0])
            p2 = space3.Point(*trace[1])
            v = p2.vector - p1.vector
            v = v.unit * trace_n.length
            p2 = p1.translate(v)
            trace = earth.Path([p1.x, p2.x], [p1.y, p2.y])

        fault = SimpleFault(
            *trace[0], *trace[1], 0, self._mrd, 60, upper_sd=5000, lower_sd=20000)

        return fault


class FaultSegmentCalculator(Object):
    """
    Calculates rupture surface from fault and magnitude with magnitude to area
    or length scaling relationship and aspect ratio. An aspect ratio (length /
    width) is followed if the rupture width is smaller than the fault width. It
    must be greater than or equal to 1 (width <= length).
    """

    def __init__(self, sr, ar, sd=2, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(sr, ScalingRelationship))
            assert(sr.OF == 'm')
            assert(sr.TO in ('sl', 'l', 'w', 'a'))
            if type(ar) is tuple:
                assert(len(ar) == 2)
                assert(is_number(ar[0]))
                assert(is_number(ar[1]))
                assert(ar[0] >= 1 and ar[1] > ar[0])
            else:
                assert(is_number(ar) and ar >= 1)
            assert(is_pos_number(sd))

        self._sr = sr
        self._ar = ar
        self._sd = sd

    def __call__(self, fault, magnitude, validate=True):
        """
        return: 'earth.flat.Rectangle' instance
        """
        if validate is True:
            pass

        if type(self._ar) is tuple:
            ar = self._ar[0] + (
                np.random.uniform(0, 1) * (self._ar[1] - self._ar[0]))
        else:
            ar = self._ar

        if self._sr.TO in ('sl', 'l'):
            l = self._sr.sample(magnitude, n=self._sd)

            if l >= fault.length:
                l = fault.length

            w = min([fault.width, l / ar])

        elif self._sr.TO in ('w'):
            w = self._sr.sample(magnitude, n=self._sd)

            if w >= fault.width:
                w = fault.width

            l = min([fault.length, w * ar])

        else:
            a = self._sr.sample(magnitude, n=self._sd)

            if a >= fault.area:
                return fault

            w = min(np.sqrt(a / ar), fault.width)
            l = a / w

        w = float(w)
        l = float(l)

        assert(l >= w)

        surface = fault.surface
        advu = fault.ad_vector.unit
        asvu = fault.as_vector.unit
        wtv = advu * float(np.random.uniform(0, 1) * (surface.w - w))
        ltv = asvu * float(np.random.uniform(0, 1) * (surface.l - l))
        ulc = fault.ulc.translate(wtv + ltv)
        llc = ulc.translate(advu * w)
        urc = ulc.translate(asvu * l)

        upper_depth = ulc.z
        lower_depth = llc.z
        upper_sd = max([upper_depth, fault.upper_sd])
        lower_sd = min([lower_depth, fault.lower_sd])

        s = SimpleFault(ulc.x, ulc.y, urc.x, urc.y, upper_depth, lower_depth,
            fault.dip, fault.rigidity, upper_sd, lower_sd)

        return s
