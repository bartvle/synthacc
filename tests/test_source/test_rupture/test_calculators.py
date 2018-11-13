"""
Tests for 'source.rupture.models' module.
"""


from functools import partial
import unittest

import numpy as np

from synthacc.earth.flat import DiscretizedSimpleSurface
from synthacc.source.mechanism import FocalMechanism
from synthacc.source.moment import MomentRateFunction
from synthacc.source.scaling import WC1994_m2a
from synthacc.source.faults import FaultGeometryCalculator
from synthacc.source.rupture.geometry import FaultSegmentCalculator
from synthacc.source.rupture.slip import (RandomFieldSDC, RiseTimeCalculator,
    SlipRateCalculator, LiuEtAl2006NSRC)
from synthacc.source.rupture.hypo import RandomHCC
from synthacc.source.rupture.velocity import ConstantVDC
from synthacc.source.rupture.rake import ConstantRDC

from synthacc.source.rupture.models import (PointRupture, SimpleFiniteRupture,
    KinematicRuptureCalculator, KinematicRuptureGenerator,
    KinematicRuptureCalculatorLogicTree)


class TestPointRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        r = PointRupture((0, 0, 5000), FocalMechanism(0, 90, 0), 1)
        self.assertEqual(r.nmrf, None)


class TestSimpleFiniteRupture(unittest.TestCase):
    """
    """

    def test_properties(self):
        """
        """
        surface = DiscretizedSimpleSurface(
            0, 0, 10000, 0, 0, 5000, 90, (10, 100))
        rake = np.zeros(surface.shape)
        slip_rates = np.zeros(surface.shape+(3,))
        slip_rates[:,:,1] = np.ones(surface.shape)
        r = SimpleFiniteRupture(surface, (0, 0, 5000), rake, 1, slip_rates)
        self.assertEqual(type(r.mrf), MomentRateFunction)


class TestKinematicRuptureCalculator(unittest.TestCase):
    """
    #TODO: implement test
    """

    def test_properties(self):
        """
        """
        fgc = partial(FaultGeometryCalculator, n=1, mrd=20000)
        fsc = partial(FaultSegmentCalculator, sr=WC1994_m2a(), ar=(1, 2), sd=2)
        sdc = partial(RandomFieldSDC, 100, 100, sd=1)
        hcc = partial(RandomHCC)
        src = partial(SlipRateCalculator, 0.05, RiseTimeCalculator(), LiuEtAl2006NSRC())
        vdc = partial(ConstantVDC, min_vel=2700, max_vel=2700)
        rdc = partial(ConstantRDC, sd=1)

        krlt = KinematicRuptureCalculatorLogicTree(fgc, fsc, sdc, hcc, src, vdc, rdc)


class TestKinematicRuptureGenerator(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestKinematicRuptureCalculatorLogicTree(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass
