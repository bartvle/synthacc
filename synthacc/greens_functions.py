"""
The 'greens_functions' module.
"""


import os

import h5py
import numpy as np

from .apy import (Object, is_non_neg_number, is_pos_number, is_non_neg_integer,
    is_pos_integer, is_2d_numeric_array, is_string)
from .data import TimeSeries
from .units import MOTION_SI as SI_UNITS
from .earth import flat as earth
from .source.moment import MomentTensor
from .recordings import (Seismogram, Recording, rt_to_ne,
    plot_seismograms)


class GenericGreensFunction(TimeSeries):
    """
    A Green's function for a certain src depth, distance and rcv depth. The ten
    components are ZSS, RSS, TSS, ZDS, RDS, TDS, ZDD, RDD, ZEP and REP. SS is a
    90° strike-slip, DS is a 90° dip-slip, DD is a 45° dip-slip and EP is an
    explosion. Z is up, R is radial and T is transverse. The src-rcv azimuth is
    0.

    ZSS (+1, -1,  0,  0,  0,  0) = +ZM2
    RSS (+1, -1,  0,  0,  0,  0) = +RM2
    TSS ( 0,  0,  0, -1,  0,  0) = -TM1
    ZDS ( 0,  0,  0,  0,  0, +1) = +ZM4
    RDS ( 0,  0,  0,  0,  0, +1) = +RM4
    TDS ( 0,  0,  0,  0, -1,  0) = -TM3
    ZDD (-1, -1, +2,  0,  0,  0) = +ZCL = ZM5 + ZM5 * 90°
    RDD (-1, -1, +2,  0,  0,  0) = +RCL = RM5 + RM5 * 90°
    ZEP (+1, +1, +1,  0,  0,  0) = +ZM6
    REP (+1, +1, +1,  0,  0,  0) = +RM6

    The M1-M6 correspond to the elementary moment tensors of Kikuchi and
    Kanamori (1991). CL is a compensated linear vector dipole.
    """

    COMPONENTS = (
        'ZSS', 'RSS', 'TSS', 'ZDS', 'RDS', 'TDS', 'ZDD', 'RDD', 'ZEP', 'REP')

    def __init__(self, time_delta, components, gmt, src_depth=None, distance=None, rcv_depth=None, model=None, validate=True):
        """
        """
        super().__init__(time_delta, validate=validate)

        components = np.asarray(components, dtype=float)

        if validate is True:
            assert(is_2d_numeric_array(components) and len(components) == 10)
            assert(gmt[:3] in SI_UNITS)
            if src_depth is not None:
                assert(is_non_neg_number(src_depth))
            if distance is not None:
                assert(is_pos_number(distance))
            if rcv_depth is not None:
                assert(is_non_neg_number(rcv_depth))
            if model is not None:
                assert(is_string(model))

        self._components = components
        self._gmt = gmt
        self._src_depth = src_depth
        self._distance = distance
        self._rcv_depth = rcv_depth
        self._model = model

    def __len__(self):
        """
        """
        return self._components.shape[1]

    @property
    def src_depth(self):
        """
        """
        return self._src_depth

    @property
    def distance(self):
        """
        """
        return self._distance

    @property
    def rcv_depth(self):
        """
        """
        return self._rcv_depth

    @property
    def gmt(self):
        """
        """
        return self._gmt

    @property
    def unit(self):
        """
        """
        return SI_UNITS[self.gmt[:3]]

    @property
    def model(self):
        """
        """
        return self._model

    def _get_component(self, component, validate=True):
        """
        """
        if validate is True:
            assert(component in self.COMPONENTS)

        return np.copy(self._components[self.COMPONENTS.index(component)])

    def get_component(self, component):
        """
        """
        s = Seismogram(
            self.time_delta, self._get_component(component), unit=self.unit)
        return s

    def get_recording(self, azimuth, moment_tensor, components='ZRT', validate=True):
        """
        See Minson & Dreger (2008).

        moment_tensor: 'source.moment.MomentTensor' instance

        return: 'recordings.Recording' instance
        """
        if validate is True:
            assert(earth.is_azimuth(azimuth))
            assert(type(moment_tensor) is MomentTensor)
            assert(components in ('ZRT', 'ZNE'))

        azimuth = np.radians(azimuth)

        (ZSS, RSS, TSS, ZDS, RDS, TDS, ZDD, RDD, ZEP, REP) = self._components

        xx, yy, zz, xy, yz, zx = moment_tensor.six

        z = np.zeros(len(self))
        r = np.zeros(len(self))
        t = np.zeros(len(self))

        z += xx * (+1*(ZSS/2)*np.cos(2*azimuth)-ZDD/6+ZEP/3)
        r += xx * (+1*(RSS/2)*np.cos(2*azimuth)-RDD/6+REP/3)
        t += xx * (+1*(TSS/2)*np.sin(2*azimuth))

        z += yy * (-1*(ZSS/2)*np.cos(2*azimuth)-ZDD/6+ZEP/3)
        r += yy * (-1*(RSS/2)*np.cos(2*azimuth)-RDD/6+REP/3)
        t += yy * (-1*(TSS/2)*np.sin(2*azimuth))

        z += zz * (ZDD/3 + ZEP/3)
        r += zz * (RDD/3 + REP/3)

        z += xy * +(ZSS * np.sin(2*azimuth))
        r += xy * +(RSS * np.sin(2*azimuth))
        t += xy * -(TSS * np.cos(2*azimuth))

        z += yz * +(ZDS * np.sin(azimuth))
        r += yz * +(RDS * np.sin(azimuth))
        t += yz * -(TDS * np.cos(azimuth))

        z += zx * +(ZDS * np.cos(azimuth))
        r += zx * +(RDS * np.cos(azimuth))
        t += zx * +(TDS * np.sin(azimuth))

        z = Seismogram(self._time_delta, z, unit=self.unit)

        if components == 'ZRT':
            r = Seismogram(self._time_delta, r, unit=self.unit)
            t = Seismogram(self._time_delta, t, unit=self.unit)
            components = {'Z': z, 'R': r, 'T': t}
        else:
            n, e = rt_to_ne(r, t, back_azimuth)
            n = Seismogram(self._time_delta, n, unit=self.unit)
            e = Seismogram(self._time_delta, e, unit=self.unit)
            components = {'Z': z, 'N': n, 'E': e}

        return Recording(components)

    def plot(self, duration=None, size=None, validate=True):
        """
        """
        p = plot_generic_greens_functions([self], duration=duration, size=size, validate=validate)

        return p


class GGFD(Object):
    """
    A database of generic Green's functions, stored in an HDF5 file.

    The keys are integer for src depth, distance and rcv depth (all in m).
    """

    def __init__(self, filespec, gmt, time_delta, duration, validate=True):
        """
        """
        if validate is True:
            assert(gmt[:3] in SI_UNITS)

        exists = os.path.exists(filespec)
        self._f = h5py.File(filespec)

        if not exists:
            self._f.attrs['gmt'] = gmt
            self._f.attrs['time_delta'] = time_delta
            self._f.attrs['duration'] = duration
        else:
            assert(self._f.attrs['gmt'] == gmt)
            assert(self._f.attrs['time_delta'] == time_delta)
            assert(self._f.attrs['duration'] == duration)

    def __len__(self):
        """
        """
        return len(self.keys)

    def __contains__(self, key):
        """
        """
        return self._get_key(*key) in self.keys

    @property
    def keys(self):
        """
        """
        return list(self._f.keys())

    @property
    def gmt(self):
        """
        """
        return self._f.attrs['gmt']

    @property
    def time_delta(self):
        """
        """
        return float(self._f.attrs['time_delta'])

    @property
    def duration(self):
        """
        """
        return float(self._f.attrs['duration'])

    @property
    def unit(self):
        """
        """
        return SI_UNITS[self.gmt[:3]]

    def _get_key(self, src_depth, distance, rcv_depth, validate=True):
        """
        """
        if validate is True:
            assert(is_non_neg_integer(src_depth))
            assert(is_pos_integer(distance))
            assert(is_non_neg_integer(rcv_depth))

        return '%i_%i_%i' % (src_depth, distance, rcv_depth)

    def add(self, gf, validate=True):
        """
        """
        if validate is True:
            assert(type(gf) is GenericGreensFunction)
            assert(gf.src_depth is not None)
            assert(gf.distance is not None)
            assert(gf.rcv_depth is not None)
            assert(gf.gmt == self.gmt)
            # print(gf.time_delta, self.time_delta)
            # assert(gf.time_delta == self.time_delta)
            # print(gf.duration, self.duration)
            # assert(gf.duration == self.duration)

        key = self._get_key(
            gf.src_depth, gf.distance, gf.rcv_depth, validate=False)

        assert(key not in self._f)

        self._f[key] = gf._components

    def get(self, src_depth, distance, rcv_depth, validate=True):
        """
        """
        key = self._get_key(src_depth, distance, rcv_depth, validate=validate)

        ggf = GenericGreensFunction(self.time_delta, np.copy(self._f[key]),
            gmt=self.gmt, src_depth=src_depth, distance=distance,
            rcv_depth=rcv_depth)

        return ggf


def plot_generic_greens_functions(ggfs, duration=None, scale=True, size=None, validate=True):
    """
    """
    if validate is True:
        assert(type(ggfs) is list)
        for ggf in ggfs:
            assert(type(ggf) is GenericGreensFunction)

    components = list(GenericGreensFunction.COMPONENTS)

    seismograms = []
    for c in components:
        seismograms.append([ggf.get_component(c) for ggf in ggfs])

    p = plot_seismograms(seismograms, titles=components, duration=duration,
        scale=scale, size=size)

    return p
