"""
The 'io.resorce2013' module.

Module to read the Reference Database for Seismic Ground-motion Prediction in
Europe (RESORCE) 2013. See Akkar et al. (2014) and http://www.resorce-portal.eu
(last accessed on 05/01/2018).

Note that the use of RESORCE 2013 is granted for non-commercial purposes only.
The copy or the reproduction of parts of the database is not allowed.
"""


import numpy as np

from ..ground.recordings import Accelerogram
from ..response import ResponseSpectrum


def read_acc(acc_filespec):
    """
    """
    with open(acc_filespec, 'r') as f:
        lines = list(f)
    s = lines[5].split()
    time_delta = float(lines[6].split()[-1])
    amplitudes = []
    for line in lines[10:]:
        vals = [line[i:i+14] for i in range(0, len(line), 14)]
        if len(vals) == 6:
            del vals[-1]
        amplitudes.extend(vals)
    amplitudes = np.array(amplitudes, dtype=float)

    acc = Accelerogram(time_delta, amplitudes, unit='m/s2')

    return acc


def read_rs(rs_filespec):
    """
    """
    with open(rs_filespec, 'r') as f:
        lines = list(f)
    periods = []
    responses_02 = []
    responses_05 = []
    responses_07 = []
    responses_10 = []
    responses_20 = []
    responses_30 = []
    for line in lines[1:-1]:
        s = line.split()
        periods.append(s[0])
        responses_02.append(s[1])
        responses_05.append(s[2])
        responses_07.append(s[3])
        responses_10.append(s[4])
        responses_20.append(s[5])
        responses_30.append(s[6])
    periods = np.array(periods, dtype=float)
    responses_02 = np.array(responses_02, dtype=float)
    responses_05 = np.array(responses_05, dtype=float)
    responses_07 = np.array(responses_07, dtype=float)
    responses_10 = np.array(responses_10, dtype=float)
    responses_20 = np.array(responses_20, dtype=float)
    responses_30 = np.array(responses_30, dtype=float)
    rs_02 = ResponseSpectrum(periods, responses_02, 'm/s2', 0.02)
    rs_05 = ResponseSpectrum(periods, responses_05, 'm/s2', 0.05)
    rs_07 = ResponseSpectrum(periods, responses_07, 'm/s2', 0.07)
    rs_10 = ResponseSpectrum(periods, responses_10, 'm/s2', 0.10)
    rs_20 = ResponseSpectrum(periods, responses_20, 'm/s2', 0.20)
    rs_30 = ResponseSpectrum(periods, responses_30, 'm/s2', 0.30)
    response_spectra = [rs_02, rs_05, rs_07, rs_10, rs_20, rs_30]

    return response_spectra
