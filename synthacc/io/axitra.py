"""
The 'io.axitra' module.

# I/O for the Axitra program (v20.08.2008). It is a Fortran code written by
# Olivier Courant after Cotton & Coutant (1997).

# - Moment is given in Nm
# - Recordings are given in XYZ with X north, Y east and Z up
"""


import os
from subprocess import Popen, PIPE

import numpy as np
from ..apy import (PRECISION, Object, is_pos_integer, is_number,
    is_non_neg_number, is_pos_number, is_string)
from .. import space3
from ..units import MOTION_SI as SI_UNITS
from ..source.mechanism import FocalMechanism
from ..ground_models import LayerModel
from ..greens_functions import GenericGreensFunction
from ..recordings import Seismogram, Recording


## Limits set in 'parameter.inc'. Axitra must be compiled with these values.
NSP = 6
NRP = 6
NCP = 150

## Fixed filenames
_DATA_FILENAME = 'axi.data'
_HIST_FILENAME = 'axi.hist'
_HEAD_FILENAME = 'axi.head'

## Free filenames
SRC_FILENAME = 'srcs'
RCV_FILENAME = 'rcvs'

DEFAULT_PARAMS = {'aw': 2, 'xl': 1000000, 'ikmax': 100000}

AXITRA_GF_FILENAME = 'Axitra.exe'
AXITRA_SS_FILENAME = 'Convm.exe'


class Wrapper(Object):
    """
    """

    def __init__(self, folder, nfreqs, duration, ground_model, params={}, validate=True):
        """
        folder: string, folder with axitra gf exe and axitra ss exe
        """
        if validate is True:
            assert(is_pos_number(nfreqs) and is_pos_number(duration))
            assert(type(ground_model) is LayerModel)
            assert(type(params) is dict)
            for p in params:
                assert(p not in ('nfreq', 'tl',))

        self._folder = folder

        self._ground_model = ground_model

        self._params = dict(DEFAULT_PARAMS)
        self._params['nfreq'] = nfreqs
        self._params['tl'] = duration
        self._params.update(params)

    @property
    def folder(self):
        """
        """
        return self._folder

    # @property
    # def time_delta(self):
        # """
        # """
        # return self._params['tl'] / ( 2 * self._params['nfreq'])

    @property
    def ground_model(self):
        """
        """
        return self._ground_model

    def _run(self, exe, com=''):
        """
        com: string, communication
        """
        p = Popen('cmd', stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=self.folder)
        c = p.communicate('{}\n{}'.format(exe, com).encode())

        if c[1] == b'':
            return True
        else:
            raise

    def get_greens_function(self, src_depth, distance, rcv_depth, validate=True):
        """
        """
        src = (0, 0, src_depth)
        rcv = (distance, 0, rcv_depth)
        write_input(self.folder, self._params, self._ground_model, src, rcv,
            validate=validate)

        self._run(AXITRA_GF_FILENAME)

        gf = read_greens_function(self.folder)

        return gf

    def get_recording(self, src, rcv, moment, focal_mechanism=None, stf=None, gmt='dis', t0=0, validate=True):
        """
        stf: tuple (type, parameter) or None,
            1: Ricker, pseudo period
            2: Step, source duration
            4: triangle, triangle width
            5: ramp, rise time
            Dirac is used when None

        return: 'recordings.Recording' instance
        """
        src = space3.Point(*src)
        rcv = space3.Point(*rcv)

        if validate is True:
            assert(src.z != rcv.z) ## Limitation of Axitra methodology
            if stf is not None:
                assert(type(stf) is tuple)
                assert(stf[0] in (1, 2, 4, 5))

        write_input(self.folder, self._params, self._ground_model, src, rcv,
            moment, focal_mechanism, t0, validate=validate)

        self._run(AXITRA_GF_FILENAME)

        if stf is not None:
            com = '{}\n{}\n'.format(stf[0], stf[1])
        else:
            com = '0\n'

        com += '%i\n' % {'dis': 1, 'vel': 2, 'acc': 3}[gmt]

        self._run(AXITRA_SS_FILENAME, com)

        r = read_recording(self.folder, 1, gmt)

        return r


def write_data_file(filespec, params, ground_model, n_srcs, n_rcvs, src_filename, rcv_filename, validate=True):
    """
    Write Axitra .data file.
    """
    if validate is True:
        assert(is_string(filespec))
        assert(type(params) is dict)
        assert(type(ground_model) is LayerModel)
        assert(len(ground_model) <= NCP)
        assert(is_pos_integer(n_srcs) and n_srcs <= NSP)
        assert(is_pos_integer(n_rcvs) and n_rcvs <= NRP)
        assert(is_string(src_filename))
        assert(is_string(rcv_filename))

    with open(filespec, 'w') as f:
        f.write('&input')
        f.write('\n')
        f.write('nc={},'.format(len(ground_model)))
        f.write('nfreq={},'.format(params['nfreq']))
        f.write('tl={},'.format(params['tl']))
        f.write('aw={},'.format(params['aw']))
        f.write('nr={},'.format(n_rcvs))
        f.write('ns={},'.format(n_srcs))
        f.write('xl={},'.format(params['xl']))
        f.write('ikmax={},'.format(params['ikmax']))
        f.write('\n')
        f.write('latlon=.false.,sourcefile="{}",statfile="{}"'.format(
            src_filename, rcv_filename))
        f.write('\n')
        f.write('//')
        f.write('\n')

        for l in ground_model:
            f.write('{} {} {} {} {} {}'.format(
                l.thickness, l.vp, l.vs, l.density, l.qp, l.qs))
            f.write('\n')


def _write_point_file(filespec, points, validate=True):
    """
    points: list of (x, y, z) tuples (in m)
    """
    if validate is True:
        assert(is_string(filespec))

    with open(filespec, 'w') as f:
        for i, point in enumerate(points):
            x, y, z = point
            if validate is True:
                assert(is_number(x))
                assert(is_number(y))
                assert(is_number(z))
            f.write('{} {} {}'.format(i+1, x, y))
            f.write('\n')
            f.write('{}'.format(z))
            f.write('\n')


def write_src_file(filespec, srcs, validate=True):
    """
    Write Axitra srcs file.

    srcs: list of (x, y, z) tuples (in m)
    """
    _write_point_file(filespec, srcs, validate=validate)


def write_rcv_file(filespec, rcvs, validate=True):
    """
    Write Axitra rcvs file.

    rcvs: list of (x, y, z) tuples (in m)
    """
    _write_point_file(filespec, rcvs, validate=validate)


def _get_source_parameters(source, validate=True):
    """
    If focal mechanism is None, the parameters for an explosion are given.

    source: (pos number, 'source.mechanism.FocalMechanism' instance, non neg
        number) tuple, (moment, focal mechanism, start time)
    """
    if validate is True:
        assert(type(source) is tuple and len(source) == 3)
        assert(is_pos_number(source[0]))
        if source[1] is not None:
            assert(type(source[1]) is FocalMechanism)
        assert(is_non_neg_number(source[2]))

    m, focal_mechanism, t = source

    s, d, r, w, l = 0, 0, 0, 0, 0
    if focal_mechanism is not None:
         s, d, r = focal_mechanism
    else:
         w, l = 1, -1 ## Parameters for explosion

    return m, s, d, r, w, l, t


def write_hist_file(filespec, sources, validate=True):
    """
    Write Axitra .hist file.

    sources: list of (pos number, 'source.mechanism.FocalMechanism' instance,
        non neg number) tuples
    """
    if validate is True:
        assert(is_string(filespec) and type(sources) is list)

    for i, source in enumerate(sources):
        m, s, d, r, w, l, t = _get_source_parameters(source, validate=validate)

        with open(filespec, 'w') as f:
            f.write('{} {} {} {} {} {} {} {}'.format(i+1, m, s, d, r, w, l, t))
            f.write('\n')


def write_input(folder, params, ground_model, src, rcv, moment=None, focal_mechanism=None, t0=0, validate=True):
    """
    Helper function to write the Axitra input for one src-rcv combination.

    params: dict
    ground_model: 'ground_models.LayerModel' instance
    src: (x, y, z) tuple (in m)
    rcv: (x, y, z) tuple (in m)
    moment: pos number or None
    focal_mechanism: 'source.mechanism.FocalMechanism' instance or None
    t0: non neg number, onset (in s)
    """
    if validate is True:
        assert(is_string(folder))

    n_srcs = 1
    n_rcvs = 1

    write_data_file(os.path.join(folder, _DATA_FILENAME), params,
        ground_model, n_srcs, n_rcvs, SRC_FILENAME, RCV_FILENAME, validate)

    ## If moment is None, only Green's functions are calculated 
    if moment is not None:
        write_hist_file(os.path.join(folder, _HIST_FILENAME),
            [(moment, focal_mechanism, t0)], validate=validate)

    write_src_file(
        os.path.join(folder, SRC_FILENAME), [src], validate=validate)
    write_rcv_file(
        os.path.join(folder, RCV_FILENAME), [rcv], validate=validate)


def read_head(filespec, validate=True):
    """
    Read Axitra .head file. This is a file with the frequencies for which the
    Green's functions are calculated.

    return: 1d numeric array
    """
    if validate is True:
        assert(is_string(filespec))

    with open(filespec, 'r') as f:
        lines = f.read().splitlines()

    frequencies = np.array([line.split()[2] for line in lines], dtype=float)

    return frequencies


def read_res(filespec, n_srcs, n_rcvs, n_frequencies, validate=True):
    """
    Read Axitra .res file. This is a binary file that contains the Green's
    functions for the three components of six elementary moment tensors in
    the frequency domain.

    return: 5d numeric array, (n_frequencies, n_srcs, n_rcvs, 3, 6)
    """
    if validate is True:
        assert(is_string(filespec))
        assert(is_pos_integer(n_srcs))
        assert(is_pos_integer(n_rcvs))
        assert(is_pos_integer(n_frequencies))

    dtype = np.dtype([('pad1', 'i4'), ('data', '6c16'), ('pad2', 'i4')])
    shape = (n_frequencies, n_srcs, n_rcvs, 3, 6)
    data = np.reshape(np.fromfile(filespec, dtype=dtype)['data'], shape)

    return data


def _get_time_delta(frequencies, validate=True):
    """
    Get time delta from frequencies. The first frequency must be zero.

    return: pos number
    """
    if validate is True:
        assert(frequencies[0] == 0)

    time_delta = float(1 / (frequencies[1] * len(frequencies) * 2))

    return time_delta


def read_greens_functions(folder, n_srcs, n_rcvs, validate=True):
    """
    return: list of 'greens_functions.GenericGreensFunction' instances
    """
    frequencies = read_head(os.path.join(folder, _HEAD_FILENAME))
    freq = read_res(
        os.path.join(folder, 'axi.res'), n_srcs, n_rcvs, len(frequencies))

    time_delta = _get_time_delta(frequencies, validate=validate)
    stack = np.fft.irfft(freq / time_delta, axis=0)

    n1 = 1 / (frequencies[1] * time_delta) #
    assert(abs(n1 - int(n1)) < 10**-PRECISION)  #
    n = int(n1) - 1                        #
    N = n//2 + 1                           #
    assert(N == len(frequencies))          #

    ggfs = []
    for i in range(n_srcs):
        for j in range(n_rcvs):
            components = []
            components.append(stack[:,i,j,2,4] - stack[:,i,j,2,3])  ## ZSS
            components.append(stack[:,i,j,0,4] - stack[:,i,j,0,3])  ## RSS
            components.append(stack[:,i,j,1,0] * -1)                ## TSS
            components.append(stack[:,i,j,2,1])                     ## ZDS
            components.append(stack[:,i,j,0,1])                     ## RDS
            components.append(stack[:,i,j,1,2])                     ## TDS
            components.append(stack[:,i,j,2,4] + stack[:,i,j,2,3])  ## ZDD
            components.append(stack[:,i,j,0,4] + stack[:,i,j,0,3])  ## RDD
            components.append(stack[:,i,j,2,5])                     ## ZEP
            components.append(stack[:,i,j,0,5])                     ## REP
            ggf = GenericGreensFunction(time_delta, components, gmt='dis')
            ggfs.append(ggf)

    return ggfs


def read_greens_function(folder, validate=True):
    """
    return: 'greens_functions.GenericGreensFunction' instance
    """
    [gf] = read_greens_functions(folder, n_srcs=1, n_rcvs=1, validate=validate)
    return gf


def read_seismogram(filespec, gmt='dis', validate=True):
    """
    """
    if validate is True:
        assert(is_string(filespec))

    with open(filespec, 'r') as f:
        lines = f.read().splitlines()

    time_delta = float(lines[0].split()[0])

    data = []
    for line in lines[1:]:
        data.extend(line.split())
    data = np.array(data, dtype=float)

    ## According to Convm.exe the output is in SI units.
    s = Seismogram(time_delta, data, unit=SI_UNITS[gmt])

    return s


def read_recording(folder, number=1, gmt='dis', validate=True):
    """
    """
    if validate is True:
        assert(is_string(folder))
        assert(is_pos_integer(number))

    fn = 'axi%03i' % number

    x = read_seismogram(
        os.path.join(folder, fn + '.X.dat'), gmt, False)
    y = read_seismogram(
        os.path.join(folder, fn + '.Y.dat'), gmt, False)
    z = read_seismogram(
        os.path.join(folder, fn + '.Z.dat'), gmt, False)

    r = Recording(components={'N': x, 'E': y, 'Z': z})

    return r
