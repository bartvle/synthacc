"""
The 'io.scardec' module. Module for reading files from the SCARDEC database
(Vallée and Douet, 2016), a database of source time functions obtained by the
SCARDEC method (Vallée et al. 2011).

http://scardec.projects.sismo.ipgp.fr/ (last accessed on 10/10/2017)
"""


import numpy as np

from ..source.moment import MomentRateFunction


def read_moment_rate_function(filespec):
    """
    return: 'moment.MomentRateFunction' instance
    """
    with open(filespec, 'r') as f:
        lines = f.read().splitlines()

    times, rates = [], []
    for line in lines[2:]:
        s = line.split()
        times.append(s[0])
        rates.append(s[1])
    times = np.array(times, dtype=float)
    rates = np.array(rates, dtype=float)
    time_delta = float(times[1] - times[0])
    start_time = float(times[0])

    mrf = MomentRateFunction(time_delta, rates, start_time)

    return mrf
