"""
Tests for 'source.rupture.calculators' module.
"""


from functools import partial
import unittest

from synthacc.source.scaling import WC1994_m2a
from synthacc.source.faults import FaultGeometryCalculator
from synthacc.source.rupture.geometry import FaultSegmentCalculator
from synthacc.source.rupture.slip import (RandomFieldSDC, RiseTimeCalculator,
    SlipRateCalculator, LiuEtAl2006NSRC)
from synthacc.source.rupture.hypo import RandomHCC
from synthacc.source.rupture.velocity import ConstantVDC
from synthacc.source.rupture.rake import ConstantRDC

from synthacc.source.rupture.calculators import (KinematicRuptureCalculator,
    KinematicRuptureGenerator, KinematicRuptureCalculatorLogicTree)


class TestKinematicRuptureCalculator(unittest.TestCase):
    """
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
