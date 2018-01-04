"""
The 'io.esgmd2' module.

Module to read the European Strong Ground Motion Database 2 (ESGMD2). See
Ambraseys et al. (2004).
"""


import numpy as np

from ..ground.recordings import Seismogram
from ..spectral import FAS
from ..response import ResponseSpectrum


UNITS = {'dis': 'm', 'vel': 'm/s', 'acc': 'm/s2'}


def read_cor(cor_filespec):
    """
    Read a ESGMD2 cor file.
    """
    with open(cor_filespec, 'r') as f:
        lines = list(f)
    time_delta = float(lines[16].split()[2][0:-1])
    amplitudes = None
    dis_amplitudes = []
    vel_amplitudes = []
    acc_amplitudes = []
    for line in lines:
        if line.startswith('STOP'):
            break
        elif line.startswith('-> corrected dis'):
            amplitudes = dis_amplitudes
            continue
        elif line.startswith('-> corrected vel'):
            amplitudes = vel_amplitudes
            continue
        elif line.startswith('-> corrected acc'):
            amplitudes = acc_amplitudes
            continue
        if amplitudes is not None:
            amplitudes.extend(line.split())
    dis = Seismogram(
        time_delta, np.array(dis_amplitudes, dtype=float), UNITS['dis'])
    vel = Seismogram(
        time_delta, np.array(vel_amplitudes, dtype=float), UNITS['vel'])
    acc = Seismogram(
        time_delta, np.array(acc_amplitudes, dtype=float), UNITS['acc'])
    return (dis, vel, acc)


def read_fas(fas_filespec):
    """
    Read a ESGMD2 fas file.
    """
    with open(fas_filespec, 'r') as f:
        lines = list(f)
    freqs = []
    ampls = []
    data = 0
    for line in lines:
        if line.startswith('STOP'):
            break
        elif line.startswith('-> FREQ, FAS '):
            data = 1
            continue
        if data:
            f, p = line.split()
            freqs.append(f)
            ampls.append(p)
    freqs = np.array(freqs, dtype=float)
    ampls = np.array(ampls, dtype=float)
    return FAS(freqs, ampls, unit=UNITS['acc'])


def read_spc(spc_filespec):
    """
    Read a ESGMD2 spc file.
    """
    with open(spc_filespec, 'r') as f:
        lines = list(f)
    data = 0
    response_spectra = []
    periods, sas, svs, sds, pvs = [], [], [], [], []
    dampings = []
    for line in lines:
        if line.startswith('->') or line.startswith('STOP'):
            data = 1
            if line.startswith('->'):
                dampings.append(float(line.split()[-2][:-1]) / 100)
            if periods:
                periods = np.array(periods, dtype=float)
                damping = dampings.pop(0)
                sds = np.array(sds, dtype=float)
                svs = np.array(svs, dtype=float)
                sas = np.array(sas, dtype=float)
                pvs = np.array(pvs, dtype=float)
                rs_sd = ResponseSpectrum(
                    periods, sds, UNITS['dis'], damping)
                rs_sv = ResponseSpectrum(
                    periods, svs, UNITS['vel'], damping)
                rs_sa = ResponseSpectrum(
                    periods, sas, UNITS['acc'], damping)
                rs_pv = ResponseSpectrum(
                    periods, pvs, UNITS['vel'], damping)
                response_spectra.append((rs_sd, rs_sv, rs_sa, rs_pv))
            periods, sds, svs, sas, pvs = [], [], [], [], []
        elif data:
            period, sa, sv, sd, pv = line.split()
            periods.append(period)
            sds.append(sd)
            svs.append(sv)
            sas.append(sa)
            pvs.append(pv)
    return response_spectra
