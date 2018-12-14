"""
The 'source.rupture.geometry' module.
"""


import math
import random

from ...apy import Object, is_number, is_pos_number
from ..faults import ComposedFault
from ..scaling import ScalingRelationship


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
        return: 'source.faults.ComposedFault' instance
        """
        if validate is True:
            assert(type(fault) is ComposedFault)

        w, l = self._get_dimensions(fault, magnitude)

        if w == fault.width and l == fault.length:
            return fault

        surface = fault.surface

        ## 2d ulc of rupture
        x = random.uniform(0, 1) * (surface.w - w)
        y = random.uniform(0, 1) * (surface.l - l)

        rsp, rep = y, y + l
        f_length, parts = 0, []
        for s in fault:
            ssp, sep = f_length, f_length + s.length

            advu = s.ad_vector.unit
            asvu = s.as_vector.unit
            adv = advu * x

            if not (sep <= rsp or ssp >= rep):

                if ssp <= rsp < sep:
                    ulc = s.ulc.translate(adv + asvu * (rsp - f_length))
                else:
                    ulc = s.ulc.translate(adv)
                if ssp < rep <= sep:
                    urc = s.ulc.translate(adv + asvu * (rep - f_length))
                else:
                    urc = s.urc.translate(adv)

                parts.append([(ulc.x, ulc.y), (urc.x, urc.y)])

            f_length += s.length

        llc = ulc.translate(advu * w)

        upper_depth = ulc.z
        lower_depth = llc.z

        upper_sd = max([upper_depth, fault.upper_sd])
        lower_sd = min([lower_depth, fault.lower_sd])

        f = ComposedFault(parts, upper_depth, lower_depth, fault.dip,
            fault.rigidity, upper_sd, lower_sd)

        return f

    def _get_dimensions(self, fault, magnitude):
        """
        Calculate width and length based on aspect ratio and magnitude scaling
        relationship.
        """
        if type(self._ar) is tuple:
            ar = self._ar[0] + (
                random.uniform(0, 1) * (self._ar[1] - self._ar[0]))
        else:
            ar = self._ar
        
        ## Magnitude to length scaling relationship
        if self._sr.TO in ('sl', 'l'):
            l = self._sr.sample(magnitude, n=self._sd)

            if l >= fault.length:
                l = fault.length

            w = min([fault.width, l / ar])

        ## Magnitude to width scaling relationship
        elif self._sr.TO == 'w':
            w = self._sr.sample(magnitude, n=self._sd)

            if w >= fault.width:
                w = fault.width

            l = min([fault.length, w * ar])

        ## Magnitude to area scaling relationship
        else:
            assert(self._sr.TO == 'a')
    
            a = self._sr.sample(magnitude, n=self._sd)

            if a >= fault.area:
                return fault.width, fault.length

            w = min(math.sqrt(a / ar), fault.width)
            l = a / w

        assert(l >= w) #TODO: remove

        return w, l
