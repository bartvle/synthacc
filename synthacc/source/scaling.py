"""
The 'source.scaling' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from .moment import mw_to_m0


PARAMS = ('m', 'a', 'sl', 'l', 'w', 'ad', 'md')


class ScalingRelationship(ABC):
    """
    """

    OF, TO = None, None

    STD = None

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
    def sample(self, of):
        """
        """
        return None


class LinScalingRelationship(ScalingRelationship, ABC):
    """
    """

    @abstractmethod
    def __call__(self, of):
        """
        """
        return None

    def sample(self, of):
        """
        """
        if self.STD is not None:
            return np.random.normal(self(of), self.STD)
        else:
            return None


class LogScalingRelationship(ScalingRelationship, ABC):
    """
    """

    @abstractmethod
    def __call__(self, of):
        """
        """
        return None

    def sample(self, of):
        """
        """
        if self.STD is not None:
            return 10**(np.random.normal(np.log10(self(of)), self.STD))
        else:
            return None


class WC1994_sl2m(LinScalingRelationship):
    """
    Relation between surface rupture length and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.89).
    """

    OF, TO = 'sl', 'm'

    STD = 0.28

    def __call__(self, sl):
        """
        """
        return 5.08 + 1.16 * np.log10(sl/1000)


class WC1994_m2sl(LogScalingRelationship):
    """
    Relation between magnitude and surface rupture length.

    See Wells & Coppersmith (1994) p. 990 (r=0.89).
    """

    OF, TO = 'm', 'sl'

    STD = 0.22

    def __call__(self, m):
        """
        """
        return 1000*10**(-3.22 + 0.69 * m)


class WC1994_a2m(LinScalingRelationship):
    """
    Relation between area and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.95).
    """

    OF, TO = 'a', 'm'

    STD = 0.24

    def __call__(self, a):
        """
        """
        return 4.07 + 0.98 * np.log10(a/10**6)


class WC1994_m2a(LogScalingRelationship):
    """
    Relation between magnitude and area.

    See Wells & Coppersmith (1994) p. 990 (r=0.95).
    """

    OF, TO = 'm', 'a'

    STD = 0.24

    def __call__(self, m):
        """
        """
        return 10**6 * 10**(-3.49 + 0.91 * m)


class WC1994_ad2m(LinScalingRelationship):
    """
    Relation between average displacement and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """

    OF, TO = 'ad', 'm'

    STD = 0.39

    def __call__(self, ad):
        """
        """
        return 6.93 + 0.82 * np.log10(ad)


class WC1994_m2ad(LogScalingRelationship):
    """
    Relation between magnitude and average displacement.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """

    OF, TO = 'm', 'ad'

    STD = 0.36

    def __call__(self, m):
        """
        """
        return 10**(-4.80 + 0.69 * m)


class WC1994_md2m(LinScalingRelationship):
    """
    Relation between maximum displacement and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """

    OF, TO = 'md', 'm'

    STD = 0.40

    def __call__(self, md):
        """
        """
        return 6.69 + 0.74 * np.log10(md)


class WC1994_m2md(LogScalingRelationship):
    """
    Relation between magnitude and maximum displacement.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """

    OF, TO = 'm', 'md'

    STD = 0.42

    def __call__(self, m):
        """
        """
        return 10**(-5.46 + 0.82 * m)


class SommervilleEtAl1999_m2a(LinScalingRelationship):
    """
    Relation between magnitude and area.

    See Sommerville et al. (1999) p. 70.
    """

    OF, TO = 'm', 'a'

    STD = None

    def __call__(self, m):
        """
        """
        return 2.23 * 10**-15 * (10**7 * mw_to_m0(m))**(2/3) * 10**6


class SommervilleEtAl1999_m2ad(LinScalingRelationship):
    """
    Relation between magnitude and average displacement.

    See Sommerville et al. (1999) p. 70.
    """

    OF, TO = 'm', 'ad'

    STD = None

    def __call__(self, m):
        """
        """
        return 1.56 * 10**-7 * (10**7 * mw_to_m0(m))**(1/3) / 100


class Wesnousky2008_sl2m(LinScalingRelationship):
    """
    Relation between surface rupture length and magnitude.

    See Wesnousky (2008) p. 1620 (r=0.82).
    """

    OF, TO = 'sl', 'm'

    STD = 0.28

    def __call__(self, sl):
        """
        """
        return 5.30 + 1.02 * np.log10(sl/1000)
