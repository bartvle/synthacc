"""
The 'source.scaling' module.
"""


import numpy as np


def WC1994N_rl_to_m(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (4.86 + 1.32 * np.log10(rl/1000), 0.28)


def WC1994A_rl_to_m(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (5.08 + 1.16 * np.log10(rl/1000), 0.28)


def WC1994N_m_to_ad(m):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-4.45 + 0.63 * m), 0.33)


def WC1994A_m_to_ad(m):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-4.80 + 0.69 * m), 0.36)


def WC1994N_m_to_md(m):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-5.90 + 0.89 * m), 0.38)


def WC1994A_m_to_md(m):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-5.46 + 0.82 * m), 0.42)


def WC1994N_rl_to_ad(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-1.99 + 1.24 * np.log10(rl/1000)), 0.37)


def WC1994A_rl_to_ad(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-1.43 + 0.88 * np.log10(rl/1000)), 0.36)


def WC1994N_rl_to_md(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-1.98 + 1.51 * np.log10(rl/1000)), 0.41)


def WC1994A_rl_to_md(rl):
    """
    See Wells & Coppersmith (1994).
    """
    return (10**(-1.38 + 1.02 * np.log10(rl/1000)), 0.41)
