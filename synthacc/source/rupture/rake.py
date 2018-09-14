"""
The 'source.rupture.rake' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from ...apy import Object, is_pos_number, is_2d_numeric_array
from ... import space2
from .surface import Distribution
from .slip import MaiBeroza2002RFSDC


class RakeDistribution(Distribution):
    """
    """

    LABEL = 'Rake (°)'

    def __init__(self, w, l, rake, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(rake))
            # assert(np.all(slip >= 0))

        self._values = rake

        super().__init__(w, l, *rake.shape, validate=False)


class RakeDistributionCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        pass


class ConstantRDC(RakeDistributionCalculator):
    """
    """

    def __init__(self, validate=True):
        """
        """
        if validate is True:
            pass

    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        return RakeDistribution(sd.w, sd.l, np.ones_like(sd.values) * rake)


class RandomFieldRDC(RakeDistributionCalculator):
    """
    """

    def __init__(self, sd=1, validate=True):
        """
        """
        if validate is True:
            pass

        self._sd = sd

    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        rfc = MaiBeroza2002RFSDC(sd.dw, sd.dl, self._sd)
        acf = rfc.get_acf()
        aw = rfc.get_aw(sd.w)
        al = rfc.get_al(sd.l)

        srfg = space2.SpatialRandomFieldGenerator(sd.w, sd.l, sd.nw, sd.nl,
            acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        # c=1

        # sd_values = (sd.values - sd.avg) / np.std(sd.values, ddof=1)
        # sd_values = sd.values

        # print(field.mean(), field.std())
        # print(sd_values.mean(), sd_values.std())

        # dft = c*np.fft.rfft2(sd_values)+np.sqrt(1-c**2)*np.fft.rfft2(field)
        # dft[0,0] = 0
        # field = np.fft.irfft2(dft, s=sd.values.shape)

        # field = field / np.std(field, ddof=1)

        # print(field.mean(), field.std())

        rake = rake + field * self._sd

        return RakeDistribution(sd.w, sd.l, rake)
