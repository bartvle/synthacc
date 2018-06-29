"""
The 'source.scaling' module.
"""


import numpy as np

from .moment import mw_to_m0


def WC1994_srl_to_m(srl):
    """
    Relation between surface rupture length and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.89).
    """
    return (5.08 + 1.16 * np.log10(srl/1000), 0.28)


def WC1994_m_to_srl(m):
    """
    Relation between magnitude and surface rupture length.

    See Wells & Coppersmith (1994) p. 990 (r=0.89).
    """
    return (1000*10**(-3.22 + 0.69 * m), 0.22)


def WC1994_a_to_m(a):
    """
    Relation between area and magnitude.

    See Wells & Coppersmith (1994) p. 990 (r=0.95).
    """
    return (4.07 + 0.98 * np.log10(a/10**6), 0.24)


def WC1994_m_to_a(m):
    """
    Relation between magnitude and area.

    See Wells & Coppersmith (1994) p. 990 (r=0.95).
    """
    return 10**6 * 10**(-3.49 + 0.91 * m)


def WC1994_ad_to_m(ad):
    """
    Relation between average displacement and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """
    return (6.93 + 0.82 * np.log10(ad), 0.39)


def WC1994_m_to_ad(m):
    """
    Relation between magnitude and average displacement.

    See Wells & Coppersmith (1994) p. 991 (r=0.75).
    """
    return (10**(-4.80 + 0.69 * m), 0.36)


def WC1994_md_to_m(md):
    """
    Relation between maximum displacement and magnitude.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """
    return (6.69 + 0.74 * np.log10(md), 0.40)


def WC1994_m_to_md(m):
    """
    Relation between magnitude and maximum displacement.

    See Wells & Coppersmith (1994) p. 991 (r=0.78).
    """
    return (10**(-5.46 + 0.82 * m), 0.42)


def SommervilleEtAl1999_m_to_a(m):
    """
    Relation between magnitude and area.

    See Sommerville et al. (1999) p. 70.
    """
    return (2.23 * 10**-15 * (10**7 * mw_to_m0(m))**(2/3) * 10**6, None)


def SommervilleEtAl1999_m_to_ad(m):
    """
    Relation between magnitude and average displacement.

    See Sommerville et al. (1999) p. 70.
    """
    return (1.56 * 10**-7 * (10**7 * mw_to_m0(m))**(1/3) / 100, None)


def Wesnousky2008_srl_to_m(srl):
    """
    Relation between surface rupture length and magnitude.

    See Wesnousky (2008) p. 1620 (r=0.82).
    """
    return (5.30 + 1.02 * np.log10(srl/1000), 0.28)
