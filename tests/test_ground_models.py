"""
Tests for 'ground_models' module.
"""


import unittest

from synthacc.ground_models import (Material, Layer, LayerModel,
    ContinuousModel, qk_qm_to_qp_qs)


class TestMaterial(unittest.TestCase):
    """
    """
    vp = 5000.
    vs = 2886.
    density = 2700.
    qp = 1000.
    qs = 500.

    m = Material(vp, vs, density, qp, qs)

    def test_properties(self):
        """
        """
        self.assertEqual(self.m.vp, 5000.)
        self.assertEqual(self.m.vs, 2886.)
        self.assertEqual(self.m.density, 2700.)
        self.assertEqual(self.m.qp, 1000.)
        self.assertEqual(self.m.qs, 500.)

    def test_to_layer(self):
        """
        """
        l = self.m.to_layer(1000)
        self.assertEqual(type(l), Layer)
        self.assertEqual(l.thickness, 1000.)
        self.assertEqual(l.vp, 5000.)
        self.assertEqual(l.vs, 2886.)
        self.assertEqual(l.density, 2700.)
        self.assertEqual(l.qp, 1000.)
        self.assertEqual(l.qs, 500.)


class TestLayer(unittest.TestCase):
    """
    """

    thickness = 1000.
    vp = 5000.
    vs = 2886.
    density = 2700.
    qp = 1000.
    qs = 500.

    l = Layer(thickness, vp, vs, density, qp, qs)

    def test_properties(self):
        """
        """
        self.assertEqual(self.l.thickness, 1000.)
        self.assertEqual(self.l.vp, 5000.)
        self.assertEqual(self.l.vs, 2886.)
        self.assertEqual(self.l.density, 2700.)
        self.assertEqual(self.l.qp, 1000.)
        self.assertEqual(self.l.qs, 500.)

    def test_material(self):
        """
        """
        m = self.l.material
        self.assertEqual(type(m), Material)
        self.assertEqual(m.vp, 5000.)
        self.assertEqual(m.vs, 2886.)
        self.assertEqual(m.density, 2700.)
        self.assertEqual(m.qp, 1000.)
        self.assertEqual(m.qs, 500.)


class TestLayerModel(unittest.TestCase):
    """
    """

    thicknesses = [10000., 8000., 5000.]
    vps = [5800., 6800., 8035.5]
    vss = [3200., 3900., 4483.9]
    densities = [2600., 2920., 3641.]
    qps = [927.35, 876.15, 599.61]
    qss = [600., 600., 394.6]

    lm = LayerModel(thicknesses, vps, vss, densities, qps, qss)

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.lm), 3)

        layers = [l for l in self.lm]
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].vp, 5800.)
        self.assertEqual(layers[1].density, 2920.)
        self.assertEqual(layers[2].qs, 394.6)

        self.assertEqual(self.lm.thickness, 23000.)

    def test_to_continuous_1d_model(self):
        """
        """
        cm = self.lm.to_continuous_model()
        self.assertEqual(type(cm), ContinuousModel)
        self.assertEqual(len(cm), 6)


class TestContinuousModel(unittest.TestCase):
    """
    """

    depths = [
        0., 20000., 20000., 35000., 35000., 77000., 120000., 165000., 210000.]
    vps = [5800., 5800., 6500., 6500., 8040., 8045., 8050., 8175., 8300.]
    vss = [3360., 3360., 3750., 3750., 4470., 4485., 4500., 4509., 4518.]
    densities = [2720., 2720., 2920., 2920., 3320., 3346., 3371., 3399., 3426.]
    qps = [1340., 1340., 1340., 1340., 1340., 1340., 1340., 250., 250.]
    qss = [ 600.,  600.,  600.,  600.,  600.,  600.,  600., 100., 100.]

    cm = ContinuousModel(depths, vps, vss, densities, qps, qss)

    def test_properties(self):
        """
        """
        self.assertEqual(len(self.cm), 9)

        levels = [l for l in self.cm]
        self.assertEqual(len(levels), 9)
        d, m = levels[4]
        self.assertEqual(d, 35000.)
        self.assertEqual(m.vp, 8040.)
        self.assertEqual(m.vs, 4470.)
        self.assertEqual(m.density, 3320.)
        self.assertEqual(m.qp, 1340.)
        self.assertEqual(m.qs, 600.)

        self.assertEqual(self.cm.depth, 210000.)
