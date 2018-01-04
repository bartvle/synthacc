"""
The 'io.qseis' module.

I/O for the Qseis program. It is a Fortran code written by Rongjiang Wang
after Wang (1999).

- Moment tensors are given in NED coordinates and Nm
- Green's functions and recordings are given in ZRT with Z down, R radial and T
  transverse
"""


import os
from subprocess import Popen, PIPE

import numpy as np

from ..apy import (PRECISION, T, F, Object, is_non_neg_number, is_pos_number, 
    is_string)
from .. import space
from ..earth import flat as earth
from ..ground.models import ContinuousModel
from ..source.moment import MomentTensor
from ..greens_functions import GenericGreensFunction
from ..ground.recordings import Seismogram, Recording


QSEIS_FILENAME = 'Qseis.exe'
INPUT_FILENAME = 'qseis.dat'

DEFAULT_PARAMS = {
    'rtb': 0, ##(reduced) time begin
    'time_reduction': 0, ## in m/s
    'stf_type': 1,
    'stf_nsamples': 5,
    'flat_earth': 1,
    'vp_res': 0.25,
    'vs_res': 0.25,
    'ro_res': 0.50,
}


class Wrapper(Object):
    """
    """

    def __init__(self, folder, nsamps, duration, ground_model, params={}, validate=True):
        """
        """
        if validate is True:
            assert(os.path.isdir(folder))
            assert(is_pos_number(nsamps) and is_pos_number(duration))
            assert(type(params) is dict)
            assert(type(ground_model) is ContinuousModel)
            for p in ('duration', 'nsamples'):
                assert(p not in params)

        self._folder = folder

        # self._time_delta = time_delta

        self._ground_model = ground_model

        # nsamples = duration / time_delta + 1
        # assert(abs(nsamples - int(nsamples)) < 10**-PRECISION)
        # nsamples = int(nsamples)

        self._params = dict(DEFAULT_PARAMS)
        self._params['nsamples'] = nsamps
        self._params['duration'] = duration
        self._params.update(params)

    @property
    def folder(self):
        """
        """
        return self._folder

    @property
    def ground_model(self):
        """
        """
        return self._ground_model

    def _run(self):
        """
        """
        p = Popen('cmd', stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=self.folder)
        c = p.communicate(
            '{}\n{}\n'.format(QSEIS_FILENAME, INPUT_FILENAME).encode()
            )

        if c[1] == b'':
            return True
        else:
            raise

    def get_greens_function(self, src_depth, distance, rcv_depth, validate=True):
        """
        src_depth: in m
        distance: in m
        rcv_depth: in m
        """
        if validate is True:
            assert(src_depth != rcv_depth) ## Limitation of Qseis methodology

        filespec = os.path.join(self.folder, INPUT_FILENAME)

        write_input(filespec, self._params, self._ground_model, src_depth,
            [distance], rcv_depth, moment_tensor=None, azimuths=None,
            validate=validate)

        self._run()

        [gf] = read_greens_functions(self.folder, src_depth, distance, rcv_depth)

        return gf

    def get_recording(self, src, rcv, mt, validate=True):
        """
        return: 'ground.recordings.Recording' instance
        """
        src = space.Point(*src)
        rcv = space.Point(*rcv)

        if validate is True:
            assert(src.z != rcv.z) ## Limitation of Qseis methodology

        filespec = os.path.join(self.folder, INPUT_FILENAME)
        d = space.distance(src.x, src.y, 0, rcv.x, rcv.y, 0)
        azimuth = earth.azimuth(src.x, src.y, rcv.x, rcv.y)

        write_input(filespec, self._params, self._ground_model, src.z, [d], rcv.z,
            mt, [azimuth], validate=validate)

        self._run()

        [r] = read_recordings(self.folder)

        return r


def write_input(filespec, params, ground_model, src_depth, distances, rcv_depth, moment_tensor, azimuths, validate=True):
    """
    params: dict
    src_depth: non neg number, src depth (in m)
    distances: list of pos numbers, distances (in m)
    rcv_depth: non neg number, rcv depth (in m)
    moment_tensor: None or 'source.moment.MomentTensor' instance (default:
        None), moment tensor in NED coordinates
    """
    if validate is True:
        assert(is_string(filespec))
        assert(type(params) is dict)
        assert(type(ground_model) is ContinuousModel)
        assert(is_non_neg_number(src_depth))
        assert(type(distances) is list)
        # for distance in distances:
            # assert(is_pos_number(distance))
        assert(is_non_neg_number(rcv_depth))
        if moment_tensor is not None:
            assert(type(moment_tensor) is MomentTensor)
            assert(type(azimuths) is list)
            for azimuth in azimuths:
                assert(earth.is_azimuth(azimuth))
        # else:
            # assert(azimuths == None)

        ## params
        assert(params['stf_type'] in (1, 2))

    with open(filespec, 'w') as f:
        f.write('{}'.format(src_depth / 1000)) ## in km
        f.write('\n')
        f.write('{}'.format(rcv_depth / 1000)) ## in km
        f.write('\n')
        f.write('0 1') ## use list of distances in km
        f.write('\n')
        f.write('{}'.format(len(distances)))
        f.write('\n')
        f.write(' '.join(map(str, [d / 1000 for d in distances]))) ## in km
        f.write('\n')
        f.write('{} {} {}'.format(params['rtb'], params['duration'], params['nsamples']))
        f.write('\n')
        f.write('1 {}'.format(params['time_reduction'] / 1000)) ## use km/s
        f.write('\n')

        ## Wavenumber integration
        f.write('0') ## slowness integration algorithm
        f.write('\n')
        f.write('0.0 0.0 0.0 0.0') ## low and high slowness cut-offs
        f.write('\n')
        f.write('1.0') ## sample wavenumber integration with Nyquist frequency
        f.write('\n')
        f.write('0.01') ## factor for suppressing time domain aliasing
        f.write('\n')

        ## Partial solutions
        f.write('0') ## no filtering of free surface effects
        f.write('\n')
        f.write('0 560.0') ## no filtering of waves with shallow penetration depth
        f.write('\n')
        f.write('0') ## no filtering
        f.write('\n')

        ## Source time function
        f.write('{} {}'.format(params['stf_nsamples'], params['stf_type'])) ## duration (in time samples) and form of wavelet
        f.write('\n')

        ## Receiver filtering
        f.write('(1.0,0.0)')
        f.write('\n')
        f.write('0')
        f.write('\n')
        f.write('# (0.0, 0.0), (0.0, 0.0)')
        f.write('\n')
        f.write('0')
        f.write('\n')
        f.write('# (-4.35425, 4.44222), (-4.35425,-4.44222)')
        f.write('\n')

        ## Calculate Green's function for explosion, strike-slip, dip-slip and
        ## compensated linear vector dipole.
        f.write('1 1 1 1 0 0')
        f.write('\n')
        f.write("'ex' 'ss' 'ds' 'cl' 'fz' 'fh'")
        f.write('\n')

        ## Moment tensor
        if moment_tensor is not None:
            f.write(
                '1 {} {} {} {} {} {} seis'.format(*moment_tensor.six))
        else:
            f.write('0')
        f.write('\n')

        ## Back-azimuth or azimuth of receiver (it is not clear if this is used for Green's
        ## function calculation, therefore we assert if back-azimuth is 0 when
        ## no moment tensor is given)
        if azimuths is None: ## use 0 or 180 for Green's function?
            f.write('0')
            f.write('\n')
            f.write('0')
            f.write('\n')
        else:
            f.write('1') ## use multiple azimuths
            f.write('\n')
            f.write(' '.join(map(str, azimuths)))
            f.write('\n')

        f.write('{}'.format(params['flat_earth'])) ## use flat earth transformation
        f.write('\n')

        ## Gradient resolution [%] of vp, vs, and ro (density), if <= 0, then
        ## default values (depending on wave length at cut-off frequency) will
        ## be used
        f.write('{} {} {}'.format(
            params['vp_res'],
            params['vs_res'],
            params['ro_res'],
            ))
        f.write('\n')

        ## Ground model
        f.write('{}'.format(len(ground_model)))
        f.write('\n')
        for i, (depth, m) in enumerate(ground_model):
            f.write('{} {} {} {} {} {} {}'.format(i+1, depth / 1000, ## in km
                m.vp / 1000, ## in km/s
                m.vs / 1000, ## in km/s
                m.density / 1000, ## in g/cm3
                m.qp,
                m.qs,
                ))
            f.write('\n')

        f.write('0') ## no different ground model at receiver


def read_t(filespec, validate=True):
    """
    return: (pos number, 2d numerical array) tuple, (time delta, (n recordings,
        n time samples))
    """
    if validate is True:
        assert(os.path.exists(filespec))

    data = np.loadtxt(filespec, skiprows=1)
    t0 = data[:,0][0]
    t1 = data[:,0][1]
    time_delta = float(t1 - t0)

    data = np.rollaxis(data[:,1:], axis=1)

    return time_delta, data


def read_gf(folder, validate=True):
    """
    """
    components = [
        read_t(os.path.join(folder, 'ss.tz')),
        read_t(os.path.join(folder, 'ss.tr')),
        read_t(os.path.join(folder, 'ss.tt')),
        read_t(os.path.join(folder, 'ds.tz')),
        read_t(os.path.join(folder, 'ds.tr')),
        read_t(os.path.join(folder, 'ds.tt')),
        read_t(os.path.join(folder, 'cl.tz')),
        read_t(os.path.join(folder, 'cl.tr')),
        read_t(os.path.join(folder, 'ex.tz')),
        read_t(os.path.join(folder, 'ex.tr')),
        ]

    time_deltas = set([c[0] for c in components])
    stack = np.dstack([c[1] for c in components])
    assert(len(time_deltas) == 1)
    time_delta = time_deltas.pop()

    return time_delta, stack


def read_greens_functions(folder, src_depth=None, distance=None, rcv_depth=None, validate=True):
    """
    Read a file with multiple Green's functions.

    Z is converted from down to up.
    """
    time_delta, stack = read_gf(folder, validate=validate)

    gfs = []
    for i in range(stack.shape[0]):
        components = []
        components.append(stack[i,:,0] * -1) ## ZSS
        components.append(stack[i,:,1])      ## RSS
        components.append(stack[i,:,2] * -1) ## TSS
        components.append(stack[i,:,3] * -1) ## ZDS
        components.append(stack[i,:,4])      ## RDS
        components.append(stack[i,:,5])      ## TDS (azimuth factor!)
        components.append(stack[i,:,6] * -2) ## ZDD
        components.append(stack[i,:,7] * +2) ## RDD
        components.append(stack[i,:,8] * -1) ## ZEP
        components.append(stack[i,:,9])      ## REP

        gf = GenericGreensFunction(time_delta, components, gmt='dis',
            src_depth=src_depth, distance=distance, rcv_depth=rcv_depth)
        gfs.append(gf)

    return gfs


def read_zrt(folder, filename='seis', validate=True):
    """
    return: (pos number, 3d numerical array) tuple, (time delta, (n recordings,
        n time samples, n components))
    """
    z = read_t(os.path.join(folder, filename + '.tz'), validate=validate)
    r = read_t(os.path.join(folder, filename + '.tr'), validate=validate)
    t = read_t(os.path.join(folder, filename + '.tt'), validate=validate)

    time_deltas = set([z[0], r[0], t[0]])
    stack = np.dstack([z[1], r[1], t[1]])
    assert(len(time_deltas) == 1)
    time_delta = time_deltas.pop()

    return time_delta, stack


def read_recordings(folder, filename='seis', validate=True):
    """
    Z is converted from down to up.

    return: list of 'ground.recordings.Recording' instances
    """
    time_delta, stack = read_zrt(folder, filename='seis', validate=validate)

    recordings = []
    for i in range(stack.shape[0]):
        components = {
            'Z': Seismogram(time_delta, stack[i,:,0], 'm') * -1,
            'R': Seismogram(time_delta, stack[i,:,1], 'm'),
            'T': Seismogram(time_delta, stack[i,:,2], 'm'),
        }
        recordings.append(Recording(components))

    return recordings
