"""
The 'io.smsim' module.
"""


import numpy as np

from ..spectral import FAS
from ..response import ResponseSpectrum


UNITS = {'dis': 'cm', 'vel': 'cm/s', 'acc': 'cm/s2'}


def read_fas(col_filespec, validate=True):
    """
    """
    if validate is True:
        pass

    with open(col_filespec, 'r') as f:
        lines = list(f)
    frequencies = []
    dis_amplitudes = []
    vel_amplitudes = []
    acc_amplitudes = []
    for line in lines[2:]:
        s = line.split()
        frequencies.append(s[4])
        dis_amplitudes.append(s[6])
        vel_amplitudes.append(s[7])
        acc_amplitudes.append(s[8])
    frequencies = np.array(frequencies, dtype=float)
    dis_amplitudes = np.array(dis_amplitudes, dtype=float)
    vel_amplitudes = np.array(vel_amplitudes, dtype=float)
    acc_amplitudes = np.array(acc_amplitudes, dtype=float)
    dis = FAS(frequencies, dis_amplitudes, UNITS['dis'])
    vel = FAS(frequencies, vel_amplitudes, UNITS['vel'])
    acc = FAS(frequencies, acc_amplitudes, UNITS['acc'])
    return (dis, vel, acc)


def read_response_spectrum(col_filespec, validate=True):
    """
    """
    if validate is True:
        pass

    with open(col_filespec, 'r') as f:
        lines = list(f)
    periods = []
    adis = []
    pvel = []
    pacc = []
    for line in lines[2:]:
        s = line.split()
        periods.append(s[1])
        adis.append(s[9])
        pvel.append(s[10])
        pacc.append(s[11])
    periods = np.array(periods, dtype=float)
    adis = np.array(adis, dtype=float)
    pvel = np.array(pvel, dtype=float)
    pacc = np.array(pacc, dtype=float)
    adis = ResponseSpectrum(periods, adis, UNITS['dis'], 0.05)
    pvel = ResponseSpectrum(periods, pvel, UNITS['vel'], 0.05)
    pacc = ResponseSpectrum(periods, pacc, UNITS['acc'], 0.05)

    return adis, pvel, pacc
