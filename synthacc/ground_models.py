"""
The 'ground_models' module.
"""


import os

import numpy as np
from obspy.taup.taup_create import build_taup_model as _build_taup_model

from .apy import (Object, is_boolean, is_non_neg_number, is_pos_number,
    is_integer, is_1d_numeric_array, is_string)


class Material(Object):
    """
    An isotropic linear elastic material, which has a linear relation between
    stress and strain.
    """

    def __init__(self, vp, vs, density, qp, qs, validate=True):
        """
        vp: (in m/s)
        vs: (in m/s)
        density: (in kg/m3)
        qp: (dimensionless)
        qs: (dimensionless)
        """
        if validate is True:
            assert(is_pos_number(vp) and is_non_neg_number(vs))
            assert(is_pos_number(density))
            assert(is_pos_number(qp) and is_non_neg_number(qs))

            ## theoretical minimum (see Shearer, 2009, p. 32)
            assert((vp/vs) >= np.sqrt(2))

        self._vp = vp
        self._vs = vs
        self._density = density
        self._qp = qp
        self._qs = qs

    @property
    def vp(self):
        """
        """
        return self._vp

    @property
    def vs(self):
        """
        """
        return self._vs

    @property
    def density(self):
        """
        """
        return self._density

    @property
    def qp(self):
        """
        """
        return self._qp

    @property
    def qs(self):
        """
        """
        return self._qs

    def to_layer(self, thickness, validate=True):
        """
        return: instance of 'ground_models.Layer'
        """
        if validate is True:
            assert(is_pos_number(thickness))

        l = Layer(thickness, self.vp, self.vs, self.density, self.qp, self.qs,
            validate=False)

        return l


class Layer(Material):
    """
    A one-dimensional layer of an isotropic linear elastic material.
    """

    def __init__(self, thickness, vp, vs, density, qp, qs, validate=True):
        """
        thickness: pos number, thickness (in m)
        vp: (in m/s)
        vs: (in m/s)
        density: (in kg/m3)
        qp: (dimensionless)
        qs: (dimensionless)
        """
        super().__init__(vp, vs, density, qp, qs, validate=validate)

        if validate is True:
            assert(is_pos_number(thickness))

        self._thickness = thickness

    @property
    def thickness(self):
        """
        """
        return self._thickness

    @property
    def material(self):
        """
        return: 'earth.models.Material' instance
        """
        m = Material(
            self.vp, self.vs, self.density, self.qp, self.qs, validate=False)

        return m


class LayerModel(Object):
    """
    An earth model defined by a sequence of layers of isotropic linear elastic
    materials.
    """

    def __init__(self, thicknesses, vps, vss, densities, qps, qss, half_space=None, name=None, validate=True):
        """
        thicknesses: (in m)
        vps: (in m/s)
        vss: (in m/s)
        densities: (in kg/m3)
        qps: (dimensionless)
        qss: (dimensionless)
        """
        thicknesses = np.asarray(thicknesses, dtype=float)
        vps = np.asarray(vps, dtype=float)
        vss = np.asarray(vss, dtype=float)
        densities = np.asarray(densities, dtype=float)
        qps = np.asarray(qps, dtype=float)
        qss = np.asarray(qss, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(thicknesses))
            assert(is_1d_numeric_array(vps))
            assert(is_1d_numeric_array(qps))
            assert(is_1d_numeric_array(densities))
            assert(is_1d_numeric_array(vss))
            assert(is_1d_numeric_array(qss))

            assert(len(thicknesses) == len(vps) == len(vss))
            assert(len(thicknesses) == len(densities))
            assert(len(thicknesses) == len(qps) == len(qss))

            assert(np.all(thicknesses > 0))
            assert(np.all(vps > 0))
            assert(np.all(qps > 0))
            assert(np.all(densities > 0))
            assert(np.all(vss >= 0))
            assert(np.all(qss >= 0))

            ## theoretical minimum (see Shearer, 2009, p. 32)
            assert(np.all((vps[vss != 0]/vss[vss != 0]) >= np.sqrt(2)))

            if half_space is not None:
                assert(type(half_space) is Material)
            if name is not None:
                assert(is_string(name))

        self._thicknesses = thicknesses
        self._vps = vps
        self._vss = vss
        self._densities = densities
        self._qps = qps
        self._qss = qss
        self._half_space = half_space
        self._name = name

    def __len__(self):
        """
        """
        return len(self._thicknesses)

    def __getitem__(self, i):
        """
        """
        assert(is_integer(i))

        thickness = self._thicknesses[i]
        vp = self._vps[i]
        vs = self._vss[i]
        density = self._densities[i]
        qp = self._qps[i]
        qs = self._qss[i]

        l = Layer(thickness, vp, vs, density, qp, qs, validate=False)

        return l

    def __iter__(self):
        """
        return: iterator of 'earth.models.Layer' instances
        """
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_layers(cls, layers, validate=True):
        """
        """
        if validate is True:
            for l in layers:
                assert(type(l) is Layer)

        thicknesses, vps, vss, densities, qps, qss = [], [], [], [], [], []
        for l in layers:
            thicknesses.append(l.thickness)
            vps.append(l.vp)
            vss.append(l.vs)
            densities.append(l.density)
            qps.append(l.qp)
            qss.append(l.qs)

        lm = cls(thicknesses, vps, vss, densities, qps, qss)

        return lm

    @property
    def thicknesses(self):
        """
        """
        return np.copy(self._thicknesses)

    @property
    def vps(self):
        """
        """
        return np.copy(self._vps)

    @property
    def vss(self):
        """
        """
        return np.copy(self._vss)

    @property
    def densities(self):
        """
        """
        return np.copy(self._densities)

    @property
    def qps(self):
        """
        """
        return np.copy(self._qps)

    @property
    def qss(self):
        """
        """
        return np.copy(self._qss)

    @property
    def half_space(self):
        """
        """
        return self._half_space

    @property
    def name(self):
        """
        """
        return self._name

    @property
    def thickness(self):
        """
        return: pos number, total thickness of the model (in m)
        """
        return np.sum(self._thicknesses)

    def to_continuous_model(self):
        """
        return: 'earth.models.ContinuousModel' instance
        """
        n = 2 * len(self)
        if self._half_space is not None:
            n += 1

        depths = np.zeros((n,))
        vps = np.zeros_like(depths)
        vss = np.zeros_like(depths)
        densities = np.zeros_like(depths)
        qps = np.zeros_like(depths)
        qss = np.zeros_like(depths)

        depth = 0
        for i, layer in enumerate(self):
            j = 2 * i

            depths[j+0] = depth
            depth += layer.thickness
            depths[j+1] = depth

            vps[[j,j+1]] = layer.vp
            vss[[j,j+1]] = layer.vs
            densities[[j,j+1]] = layer.density
            qps[[j,j+1]] = layer.qp
            qss[[j,j+1]] = layer.qs

        if self._half_space is not None:
            depths[-1] = depth
            vps[-1] = self.half_space.vp
            vss[-1] = self.half_space.vs
            densities[-1] = self.half_space.density
            qps[-1] = self.half_space.qp
            qss[-1] = self.half_space.qs

        assert(depths[-1] == self.thickness)

        cm = ContinuousModel(
            depths, vps, vss, densities, qps, qss, validate=False)

        return cm


class ContinuousModel(Object):
    """
    A ground model defined by a sequence of depths and materials. Layers and
    discontinuities can be modelled by specifying a depth twice, but with a
    different material.
    """

    def __init__(self, depths, vps, vss, densities, qps, qss, name=None, validate=True):
        """
        depths: (in m)
        vps: (in m/s)
        vss: (in m/s)
        densities: (in kg/m3)
        qps: (dimensionless)
        qss: (dimensionless)
        """
        depths = np.asarray(depths, dtype=float)
        vps = np.asarray(vps, dtype=float)
        vss = np.asarray(vss, dtype=float)
        densities = np.asarray(densities, dtype=float)
        qps = np.asarray(qps, dtype=float)
        qss = np.asarray(qss, dtype=float)

        if validate is True:
            assert(is_1d_numeric_array(depths))
            assert(is_1d_numeric_array(vps))
            assert(is_1d_numeric_array(qps))
            assert(is_1d_numeric_array(densities))
            assert(is_1d_numeric_array(vss))
            assert(is_1d_numeric_array(qss))

            assert(len(depths) == len(vps) == len(vss))
            assert(len(depths) == len(densities))
            assert(len(depths) == len(qps) == len(qss))

            assert(np.all(depths >= 0))
            assert(np.all(vps > 0))
            assert(np.all(qps > 0))
            assert(np.all(densities > 0))
            assert(np.all(vss >= 0))
            assert(np.all(qss >= 0))

            # assert depths are not descending
            # assert(np.all(np.diff(depths) >= 0))
            # assert depths occur max 2 times
            # assert(Counter(depths).most_common(1)[0][1] <= 2)

            ## theoretical minimum (see Shearer, 2009, p. 32)
            assert(np.all((vps[vss != 0]/vss[vss != 0]) >= np.sqrt(2)))

            if name is not None:
                assert(is_string(name))

        self._depths = depths
        self._vps = vps
        self._vss = vss
        self._densities = densities
        self._qps = qps
        self._qss = qss
        self._name = name

    def __len__(self):
        """
        """
        return len(self._depths)

    def __getitem__(self, i):
        """
        """
        assert(is_integer(i))

        depth = self._depths[i]
        vp = self._vps[i]
        vs = self._vss[i]
        density = self._densities[i]
        qp = self._qps[i]
        qs = self._qss[i]

        return depth, Material(vp, vs, density, qp, qs, validate=False)

    def __iter__(self):
        """
        return: iterator of (pos number, 'earth.flat.Layer' instance) tuples
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def depths(self):
        """
        """
        return np.copy(self._depths)

    @property
    def vps(self):
        """
        """
        return np.copy(self._vps)

    @property
    def vss(self):
        """
        """
        return np.copy(self._vss)

    @property
    def densities(self):
        """
        """
        return np.copy(self._densities)

    @property
    def qps(self):
        """
        """
        return np.copy(self._qps)

    @property
    def qss(self):
        """
        """
        return np.copy(self._qss)

    @property
    def name(self):
        """
        """
        return self._name

    @property
    def depth(self):
        """
        """
        return self._depths[-1]

    def to_layer_model(self, half_space=False, validate=True):
        """
        half_space: boolean (default: False)

        return: 'earth.models.LayerModel' instance
        """
        if validate is True:
            assert(is_boolean(half_space))

        thicknesses, vps, vss, densities, qps, qss = [], [], [], [], [], []
        for i in range(len(self)-1):
            td, tm = self[i+0]
            bd, bm = self[i+1]
            if td == bd:
                continue
            thicknesses.append(bd-td)
            vps.append((tm.vp+bm.vp)/2)
            vss.append((tm.vs+bm.vs)/2)
            densities.append((tm.density+bm.density)/2)
            qps.append((tm.qp+bm.qp)/2)
            qss.append((tm.qs+bm.qs)/2)

        if half_space is True:
            half_space = self[-1][1]
        else:
            half_space = None

        lm = LayerModel(thicknesses, vps, vss, densities, qps, qss, half_space)

        return lm


def qk_qm_to_qp_qs(qk, qm, vp, vs, validate=True):
    """
    Calculate P-wave (qp) and S-wave (qs) attenuation from bulk (qk) and shear
    (qm) attenuation (and vp and vs).

    NOTE: S-wave attenuation (qs) equals shear attenuation (qm). When qm and vs
        are zero, P-wave attenuation (qp) equals bulk attenuation (qk).

    See Shearer (2009) p. 173 and Stein & Wysession (2003) p. 192.
    """
    if validate is True:
        assert(is_pos_number(qk))
        assert(is_pos_number(vp))
        assert(is_non_neg_number(qm))
        assert(is_non_neg_number(vs))

        if qm == 0:
            assert(vs == 0)
        if vs == 0:
            assert(qm == 0)

    l = (4/3) * (vs/vp)**2

    if l != 0:
        qp = 1 / (l/qm + (1-l)/qk)
    else:
        ## qm and vs are 0
        qp = qk

    qs = qm

    return qp, qs


def build_taup_model(folder, ground_model, name=None, validate=True):
    """
    """
    if validate is True:
        assert(os.path.exists(folder))
        assert(type(ground_model) in (LayerModel, ContinuousModel))
        if name is not None:
            assert(is_string(name))
        assert(ground_model.name is not None or name is not None)

    if type(ground_model) is LayerModel:
        ground_model = ground_model.to_continuous_model()

    name = name or ground_model.name

    tvel_filespec = os.path.join(folder, name + '.tvel')

    with open(tvel_filespec, 'w') as f:
        f.write('## Comment')
        f.write('\n')
        f.write('## Comment')
        f.write('\n')
        for d, m in ground_model:
            f.write('%s' % (d / 1000,))
            f.write(' %s' % (m.vp / 1000,))
            f.write(' %s' % (m.vs / 1000,))
            f.write(' %s' % (m.density / 1000,))
            f.write('\n')

    _build_taup_model(tvel_filespec, folder)
