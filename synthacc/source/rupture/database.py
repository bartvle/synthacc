"""
The 'source.rupture.database' module.
"""


from abc import ABC
import os
import random

import h5py
import numpy as np
import pandas as pd

from ...apy import Object, is_string
from ...data import LogicTreeLeaf
from ...earth import flat as earth
from ..faults import ComposedFault
from .models import SimpleFiniteRupture, ComposedFiniteRupture


class FiniteRuptureDatabase(ABC, Object):
    """
    """

    def __init__(self, filespec, validate=True):
        """
        """
        if validate is True:
            pass

        self._f = h5py.File(filespec, 'a')

    def __len__(self):
        """
        """
        return len(self._f)

    def __contains__(self, key, validate=True):
        """
        """
        if validate is True:
            assert(is_string(key))

        return key in self._f

    def _generate_key(self):
        """
        """
        kg = lambda: '%06.i' % (random.randint(1, 999999),)

        key = kg()
        while key in self._f:
            key = kg()
    
        return key

    @property
    def keys(self):
        """
        """
        return tuple(self._f.keys())


class SimpleFiniteRuptureDatabase(FiniteRuptureDatabase):
    """
    """

    def add(self, r, l, validate=True):
        """
        """
        if validate is True:
            assert(type(r) is SimpleFiniteRupture)
            assert(type(l) is LogicTreeLeaf)

        key = self._generate_key()

        g = self._f.create_group(key)
        flat = r.slip_rates.flatten()
        inds = np.nonzero(flat)
        vals = flat[inds]

        g.attrs['x1'] = r.segment.ulc.x
        g.attrs['y1'] = r.segment.ulc.y
        g.attrs['x2'] = r.segment.urc.x
        g.attrs['y2'] = r.segment.urc.y
        g.attrs['ud'] = r.segment.upper_depth
        g.attrs['ld'] = r.segment.lower_depth
        g.attrs['dip'] = r.segment.dip
        g.attrs['shape'] = r.slip_rates.shape
        g.attrs['hypo_x'] = r.hypo.x
        g.attrs['hypo_y'] = r.hypo.y
        g.attrs['hypo_z'] = r.hypo.z
        g.attrs['td'] = r.time_delta
        g.attrs['rigidity'] = r.rigidity

        g.create_dataset('rake', data=r.rake.values)
        g.create_dataset('vals', data=vals)
        g.create_dataset('inds', data=inds)

        g.attrs['path'] = '_'.join([b[1] for b in l.path])
        g.attrs['prob'] = l.prob

    def get(self, key, validate=True):
        """
        """
        if key in self:
            g = self._f[key]
        else:
            raise LookupError

        shape = g.attrs['shape']
        x1 = float(g.attrs['x1'])
        y1 = float(g.attrs['y1'])
        x2 = float(g.attrs['x2'])
        y2 = float(g.attrs['y2'])
        ud = float(g.attrs['ud'])
        ld = float(g.attrs['ld'])
        dip = float(g.attrs['dip'])
        segment = earth.DiscretizedSimpleSurface(
            x1, y1, x2, y2, ud, ld, dip, (int(shape[0]), int(shape[1])))
        hypo = (
            float(g.attrs['hypo_x']),
            float(g.attrs['hypo_y']),
            float(g.attrs['hypo_z']),
            )
        rake = np.asarray(g['rake'])
        time_delta = float(g.attrs['td'])
        vals = g['vals']
        inds = g['inds']
        slip_rates = np.zeros(shape)
        slip_rates[np.unravel_index(inds, shape)] = vals
        rigidity = float(g.attrs['rigidity'])

        r = SimpleFiniteRupture(
            segment, hypo, rake, time_delta, slip_rates, rigidity)

        path = g.attrs['path']
        prob = g.attrs['prob']

        return r, path, prob


class ComposedFiniteRuptureDatabase(FiniteRuptureDatabase):
    """
    """

    def add(self, r, l, validate=True):
        """
        """
        if validate is True:
            assert(type(r) is ComposedFiniteRupture)
            assert(type(l) is LogicTreeLeaf)

        key = self._generate_key()

        g = self._f.create_group(key)
        flat = r.slip_rates.flatten()
        inds = np.nonzero(flat)
        vals = flat[inds]

        x1s, y1s = [], []
        x2s, y2s = [], []
        for p in r.segment:
            x1s.append(p.ulc.x)
            y1s.append(p.ulc.y)
            x2s.append(p.urc.x)
            y2s.append(p.urc.y)
        g.create_dataset('x1s', data=x1s)
        g.create_dataset('y1s', data=y1s)
        g.create_dataset('x2s', data=x2s)
        g.create_dataset('y2s', data=y2s)

        g.attrs['ud'] = r.segment.upper_depth
        g.attrs['ld'] = r.segment.lower_depth
        g.attrs['dip'] = r.segment.dip
        g.attrs['shape'] = r.slip_rates.shape
        g.attrs['hypo_x'] = r.hypo.x
        g.attrs['hypo_y'] = r.hypo.y
        g.attrs['hypo_z'] = r.hypo.z
        g.attrs['td'] = r.time_delta
        g.attrs['rigidity'] = r.rigidity

        g.create_dataset('rake', data=r.rake.values)
        g.create_dataset('vals', data=vals)
        g.create_dataset('inds', data=inds)

        g.attrs['path'] = '_'.join([b[1] for b in l.path])
        g.attrs['prob'] = l.prob

    def get(self, key, validate=True):
        """
        """
        if key in self:
            g = self._f[key]
        else:
            raise LookupError

        shape = g.attrs['shape']
        x1s = g['x1s']
        y1s = g['y1s']
        x2s = g['x2s']
        y2s = g['y2s']
        parts = []
        for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
            parts.append((
                (float(x1), float(y1)),
                (float(x2), float(y2)),
                ))

        ud = float(g.attrs['ud'])
        ld = float(g.attrs['ld'])
        dip = float(g.attrs['dip'])
        rigidity = float(g.attrs['rigidity'])

        segment = ComposedFault(parts, ud, ld, dip, rigidity,
            upper_sd=None, lower_sd=None)

        hypo = (
            float(g.attrs['hypo_x']),
            float(g.attrs['hypo_y']),
            float(g.attrs['hypo_z']),
            )
        rake = np.asarray(g['rake'])
        time_delta = float(g.attrs['td'])
        vals = g['vals']
        inds = g['inds']
        slip_rates = np.zeros(shape)
        slip_rates[np.unravel_index(inds, shape)] = vals
    
        r = ComposedFiniteRupture(segment, hypo, rake, time_delta, slip_rates)

        path = g.attrs['path']
        prob = g.attrs['prob']

        return r, path, prob


class FlatFileExporter(Object):
    """
    Flat file database of parameters of finiture rupture database.
    """

    COLUMNS = ['w', 'l', 'a', 'ud', 'ld', 'sd_avg', 'sd_max', 'asp_l',
        'asp_vl', 'rt_avg', 'rt_max', 'sr_std', 'sr_avg', 'sr_max', 'rt_std',
        'ltl1', 'ltl2', 'ltl3', 'ltl4', 'ltl5', 'ltl6', 'ltl7', 'ltl8']

    def __init__(self, db, filespec, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(db, FiniteRuptureDatabase))
    
        if os.path.exists(filespec):
            df = pd.read_pickle(filespec)
        else:
            df = pd.DataFrame(columns=self.COLUMNS)

        self._db = db
        self._df = df
        self._filespec = filespec

    def __len__(self):
        """
        """
        return len(self._df)

    @property
    def filespec(self):
        """
        """
        return self._filespec

    def add(self, key, update=True, validate=False):
        """
        """
        if update is False and key in self._df.index:
            return None

        r, path, prob = self._db.get(key)
        s, sd = r.segment, r.slip
        shape = sd.shape
        slip_rates = r.slip_rates

        asp_l = len(
            np.where(sd.values >= (sd.max * 1 / 3))[0]) / (shape[0]*shape[1])
        asp_v = len(
            np.where(sd.values >= (sd.max * 2 / 3))[0]) / (shape[0]*shape[1])

        rise_times = np.zeros(shape)
        for i in np.ndindex(shape):
            [nz] = np.nonzero(slip_rates[i])
            if len(nz) != 0:
                rise_times[i] = (nz[-1]-nz[0]) * r.time_delta

        rise_times = rise_times[rise_times != 0]
        rt_avg = rise_times.mean()
        rt_max = rise_times.max()
        rt_std = rise_times.std()

        slip_rates = slip_rates[slip_rates != 0]
        sr_avg = slip_rates.mean()
        sr_max = slip_rates.max()
        sr_std = slip_rates.std()

        self._df.loc[key] = [s.width, s.length, s.area, s.upper_depth,
            s.lower_depth, sd.avg, sd.max, asp_l, asp_v, rt_avg, rt_max,
            rt_std, sr_avg, sr_max, sr_std, *path.split('_')]

    def export(self):
        """
        """
        self._df.to_pickle(self._filespec)
