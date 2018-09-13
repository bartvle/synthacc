"""
The 'source.scaling' module.
"""


from abc import ABC, abstractmethod

import numpy as np
import scipy.stats

from ..apy import Object
from .moment import mw_to_m0


PARAMS = ('m', 'a', 'sl', 'l', 'w', 'as', 'ms')


class ScalingRelationship(ABC, Object):
    """
    """

    OF = None
    TO = None
    SD = None
    FACTOR = 1

    def __init__(self):
        """
        """
        assert(self.OF in PARAMS)
        assert(self.TO in PARAMS)

    @abstractmethod
    def __call__(self, of):
        """
        """
        return None

    @abstractmethod
    def get_mean(self, to, n):
        """
        """
        return None

    @abstractmethod
    def get_sd(self, to, n):
        """
        """
        return None

    @abstractmethod
    def sample(self, of, n):
        """
        """
        return None

    @property
    def name(self):
        """
        """
        return self.__class__.__name__


class LinScalingRelationship(ScalingRelationship, ABC):
    """
    """

    @abstractmethod
    def __call__(self, of):
        """
        """
        return None

    def get_mean(self, of, validate=True):
        """
        """
        return self.FACTOR * self(of)

    def get_sd(self, of, n=1, validate=True):
        """
        """
        to = self(of)
        sd = self.SD * n
        sd_min = self.FACTOR * (to-sd)
        sd_max = self.FACTOR * (to+sd)

        return (sd_min, sd_max)

    def sample(self, of, n=2, validate=True):
        """
        """
        if self.SD is not None:
            v = scipy.stats.truncnorm.rvs(-n, +n, self(of), self.SD)
            return self.FACTOR * v
        else:
            return None


class LogScalingRelationship(ScalingRelationship, ABC):
    """
    Standard deviation does not change if scaling is power of 10.
    """

    @abstractmethod
    def __call__(self, of):
        """
        """
        return None

    def get_mean(self, of, validate=True):
        """
        """
        return self.FACTOR * 10**(self(of))

    def get_sd(self, of, n=1, validate=True):
        """
        """
        to = self(of)
        sd = self.SD * n
        sd_min = self.FACTOR * 10**(to-sd)
        sd_max = self.FACTOR * 10**(to+sd)

        return (sd_min, sd_max)

    def sample(self, of, n=2, validate=True):
        """
        """
        if self.SD is not None:
            v = scipy.stats.truncnorm.rvs(-n, +n, self(of), self.SD)
            return self.FACTOR * 10**(v)
        else:
            return None


class WC1994_m2a(LogScalingRelationship):
    """
    Relation between magnitude and area.

    See Wells & Coppersmith (1994) p. 990 (r=0.95, n=148).
    """

    OF, TO = 'm', 'a'
    SD = 0.24
    FACTOR = 10**6

    def __call__(self, of):
        """
        """
        return -3.49 + 0.91 * of


class WC1994_m2a_n(LogScalingRelationship):
    """
    Relation between magnitude and area.

    See Wells & Coppersmith (1994) p. 990 (r=0.92, n=22).
    """

    OF, TO = 'm', 'a'
    SD = 0.22
    FACTOR = 10**6

    def __call__(self, of):
        """
        """
        return -2.87 + 0.82 * of


class WC1994_a2m(LinScalingRelationship):
    """
    Relation between area and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.95).
    """

    OF, TO = 'a', 'm'
    SD = 0.24

    def __call__(self, of):
        """
        """
        return 4.07 + 0.98 * np.log10(of/10**6)


class WC1994_m2sl(LogScalingRelationship):
    """
    Relation between magnitude and surface length.

    See Wells & Coppersmith (1994) p. 990 (r=0.89, n=77).
    """

    OF, TO = 'm', 'sl'
    SD = 0.22
    FACTOR = 10**3

    def __call__(self, of):
        """
        """
        return -3.22 + 0.69 * of


class WC1994_m2l(LogScalingRelationship):
    """
    Relation between magnitude and surface length.

    See Wells & Coppersmith (1994) p. 990 (r=0.94, n=167).
    """

    OF, TO = 'm', 'l'
    SD = 0.22
    FACTOR = 10**3

    def __call__(self, of):
        """
        """
        return -2.44 + 0.59 * of


class WC1994_m2l_n(LogScalingRelationship):
    """
    Relation between magnitude and surface length.

    See Wells & Coppersmith (1994) p. 990 (r=0.88, n=24).
    """

    OF, TO = 'm', 'l'
    SD = 0.17
    FACTOR = 10**3

    def __call__(self, of):
        """
        """
        return -1.88 + 0.50 * of


class WC1994_sl2m(LinScalingRelationship):
    """
    Relation between surface length and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.89).
    """

    OF, TO = 'sl', 'm'
    SD = 0.28

    def __call__(self, of):
        """
        """
        return 5.08 + 1.16 * np.log10(of/1000)


class WC1994_m2as(LogScalingRelationship):
    """
    Relation between magnitude and average slip.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """

    OF, TO = 'm', 'as'
    SD = 0.36

    def __call__(self, of):
        """
        """
        return -4.80 + 0.69 * of


class WC1994_as2m(LinScalingRelationship):
    """
    Relation between average slip and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """

    OF, TO = 'as', 'm'
    SD = 0.39

    def __call__(self, of):
        """
        """
        return 6.93 + 0.82 * np.log10(of)


class WC1994_m2ms(LogScalingRelationship):
    """
    Relation between magnitude and maximum slip.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """

    OF, TO = 'm', 'ms'
    SD = 0.42

    def __call__(self, of):
        """
        """
        return -5.46 + 0.82 * of


class WC1994_ms2m(LinScalingRelationship):
    """
    Relation between maximum slip and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """

    OF, TO = 'ms', 'm'
    SD = 0.40

    def __call__(self, of):
        """
        """
        return 6.69 + 0.74 * np.log10(of)


class SommervilleEtAl1999_m2a(LinScalingRelationship):
    """
    Relation between magnitude and area.

    See Sommerville et al. (1999) p. 70.
    """

    OF, TO = 'm', 'a'
    SD = None
    FACTOR = 10**6

    def __call__(self, of):
        """
        """
        return 2.23 * 10**-15 * (10**7 * mw_to_m0(of))**(2/3)


class SommervilleEtAl1999_m2as(LinScalingRelationship):
    """
    Relation between magnitude and average slip.

    See Sommerville et al. (1999) p. 70.
    """

    OF, TO = 'm', 'as'
    SD = None
    FACTOR = 10**-2

    def __call__(self, of):
        """
        """
        return 1.56 * 10**-7 * (10**7 * mw_to_m0(of))**(1/3)


class Wesnousky2008_sl2m(LinScalingRelationship):
    """
    Relation between surface length and magnitude.

    See Wesnousky (2008) p. 1620 (r=0.82).
    """

    OF, TO = 'sl', 'm'
    SD = 0.28

    def __call__(self, of):
        """
        """
        return 5.30 + 1.02 * np.log10(of/1000)


class Leonard2014_m2a_scr_ds(LogScalingRelationship):
    """
    """

    OF, TO = 'm', 'l'
    SD = None
    FACTOR = 10**6

    def __call__(self, of):
        """
        """
        return (of - 4.19) / 1.00


class Leonard2014_m2l_scr_ds(LogScalingRelationship):
    """
    """

    OF, TO = 'm', 'l'
    SD = None
    FACTOR = 10**3

    def __call__(self, of):
        """
        """
        return (of - 4.32) / 1.667


class ThingbaijamEA2017_m2a_n(LogScalingRelationship):
    """
    Relation between magnitude and area.

    See Thingbaijam et al. (2017) p. 2231 (r=0.94).
    """

    OF, TO = 'm', 'a'
    SD = 0.181
    FACTOR = 10**6

    def __call__(self, of):
        """
        """
        return -2.551 + 0.808 * of


class ThingbaijamEA2017_m2l_n(LogScalingRelationship):
    """
    Relation between magnitude and area.

    See Thingbaijam et al. (2017) p. 2231 (r=0.94).
    """

    OF, TO = 'm', 'l'
    SD = 0.128
    FACTOR = 10**3

    def __call__(self, of):
        """
        """
        return -1.722 + 0.485 * of


class ThingbaijamEA2017_m2as_n(LogScalingRelationship):
    """
    Relation between magnitude and average slip.

    See Thingbaijam et al. (2017) p. 2234 (r=0.93).
    """

    OF, TO = 'm', 'as'
    SD = 0.195

    def __call__(self, of):
        """
        """
        return -4.967 + 0.693 * of
