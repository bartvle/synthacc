"""
The 'source.rupture.calculators' module.
"""


import numpy as np

from ...apy import Object
from ... import space3
from ...data import LogicTree
from ..faults import FaultGeometryCalculator
from .geometry import FaultSegmentCalculator
from .slip import SlipDistributionCalculator, SlipRateCalculator
from .hypo import HypoCenterCalculator
from .velocity import VelocityDistributionCalculator
from .rake import RakeDistributionCalculator
from .propagation import TravelTimeCalculator
from .models import ComposedFiniteRupture


class KinematicRuptureCalculator(Object):
    """
    """

    def __init__(self, fgc, fsc, sdc, hcc, src, vdc, rdc, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(fgc, FaultGeometryCalculator))
            assert(isinstance(fsc, FaultSegmentCalculator))
            assert(isinstance(sdc, SlipDistributionCalculator))
            assert(isinstance(hcc, HypoCenterCalculator))
            assert(isinstance(src, SlipRateCalculator))
            assert(isinstance(vdc, VelocityDistributionCalculator))
            assert(isinstance(rdc, RakeDistributionCalculator))
    
        self._fgc = fgc
        self._fsc = fsc
        self._sdc = sdc
        self._hcc = hcc
        self._src = src
        self._vdc = vdc
        self._rdc = rdc

    def __call__(self, fault_data, magnitude, rake, validate=True):
        """
        """
        fault = self._fgc(fault_data)
        segment = self._fsc(fault, magnitude, rake)
        #TODO: check this implementation
        #TODO: are all these parameters needed?
        # rd = self._rdc(segment, magnitude, rake)()
        sd = self._sdc(segment, magnitude, fault.rigidity)()
    
        ## Calculate 2D and 3D hypocenter
        hypo2 = self._hcc(segment, sd)
        start, part = segment.get_start_and_part(hypo2.y)
        x, y = hypo2.x, (hypo2.y - start)
        w_vector = part.ad_vector.unit * x
        l_vector = part.as_vector.unit * y
        hypo3 = space3.Point(*part.ulc.translate(w_vector + l_vector))

        sr = self._src(segment, magnitude, sd)
        assert(np.all(sr >= 0))
        vd = self._vdc(segment, magnitude, sd)

        #TODO: which delta to take?
        ttc = TravelTimeCalculator(vd, d=100)
        tts = ttc(*hypo2, ds=sd)

        #TODO: dev test / remove
        assert(sd.shape == sr.shape[:2] == tts.shape)

        #TODO: check this implementation
        n_onsets = np.round(tts.times / self._src._d).astype(np.int)
        i = np.ones(sr.shape) * np.arange(sr.shape[-1])
        i[sr==0] = 0
        n_rise_times = np.argmax(i, axis=2)
        n = n_onsets + n_rise_times
        slip_rates = np.zeros(sd.shape + (n.max()+2,))

        for i in np.ndindex(sd.shape):
            srf = sr[i][:n_rise_times[i]]
            slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf

        rd = self._rdc(segment, sd, rake).values

        r = ComposedFiniteRupture(segment, hypo3, rd, self._src._d, slip_rates)

        return r

    @property
    def fgc(self):
        """
        """
        return self._fgc

    @property
    def fsc(self):
        """
        """
        return self._fsc

    @property
    def sdc(self):
        """
        """
        return self._sdc

    @property
    def hcc(self):
        """
        """
        return self._hcc

    @property
    def src(self):
        """
        """
        return self._src

    @property
    def vdc(self):
        """
        """
        return self._vdc

    @property
    def rdc(self):
        """
        """
        return self._rdc

    def get_generator(self, fault_data, magnitude, rake, validate=True):
        """
        """
        krg = KinematicRuptureGenerator(
            self, fault_data, magnitude, rake, validate=validate)

        return krg


class KinematicRuptureGenerator(Object):
    """
    """

    def __init__(self, krc, fault_data, magnitude, rake, validate=True):
        """
        """
        if validate is True:
            assert(type(krc) is KinematicRuptureCalculator)

        self._krc = krc
        self._fault_data = fault_data
        self._magnitude = magnitude
        self._rake = rake

    def __call__(self, validate=True):
        """
        """
        r = self._krc(self._fault_data, self._magnitude, self._rake, validate)

        return r


class KinematicRuptureCalculatorLogicTree(LogicTree):
    """
    """

    CALCULATORS = [
        ('fgc', FaultGeometryCalculator),
        ('fsc', FaultSegmentCalculator),
        ('sdc', SlipDistributionCalculator),
        ('hcc', HypoCenterCalculator),
        ('src', SlipRateCalculator),
        ('vdc', VelocityDistributionCalculator),
        ('rdc', RakeDistributionCalculator),
    ]

    def __init__(self, fgc=None, fsc=None, sdc=None, hcc=None, src=None, vdc=None, rdc=None, validate=True):
        """
        """
        super().__init__()

        if validate is True:
            assert(fgc is None or issubclass(fgc.func, self.CALCULATORS[0][1]))
            assert(fsc is None or issubclass(fsc.func, self.CALCULATORS[1][1]))
            assert(sdc is None or issubclass(sdc.func, self.CALCULATORS[2][1]))
            assert(hcc is None or issubclass(hcc.func, self.CALCULATORS[3][1]))
            assert(src is None or issubclass(src.func, self.CALCULATORS[4][1]))
            assert(vdc is None or issubclass(vdc.func, self.CALCULATORS[5][1]))
            assert(rdc is None or issubclass(rdc.func, self.CALCULATORS[6][1]))

        self._fgc = fgc
        self._fsc = fsc
        self._sdc = sdc
        self._hcc = hcc
        self._src = src
        self._vdc = vdc
        self._rdc = rdc

    def __call__(self, sub=None, validate=True):
        """
        """
        leaf = self.sample(sub, validate)

        fgc = (self._fgc or leaf.get('fgc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('fgc_')})
        fsc = (self._fsc or leaf.get('fsc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('fsc_')})
        sdc = (self._sdc or leaf.get('sdc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('sdc_')})
        hcc = (self._hcc or leaf.get('hcc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('hcc_')})
        src = (self._src or leaf.get('src'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('src_')})
        vdc = (self._vdc or leaf.get('vdc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('vdc_')})
        rdc = (self._rdc or leaf.get('rdc'))(**{b[0].split('_')[1]: b[2] for
            b in leaf.path if b[0].startswith('rdc_')})

        krc = KinematicRuptureCalculator(fgc, fsc, sdc, hcc, src, vdc, rdc)

        return krc, leaf

    def get_generator(self, fault_data, magnitude, rake, validate=True):
        """
        """
        krglt = KinematicRuptureGeneratorLogicTree(
            self, fault_data, magnitude, rake, validate=validate)

        return krglt


class KinematicRuptureGeneratorLogicTree(Object):
    """
    """

    def __init__(self, krclt, fault_data, magnitude, rake, validate=True):
        """
        """
        if validate is True:
            assert(type(krclt) is KinematicRuptureCalculatorLogicTree)

        self._krclt = krclt
        self._fault_data = fault_data
        self._magnitude = magnitude
        self._rake = rake

    def __call__(self, validate=True):
        """
        """
        krc, leaf = self._krclt()
        krg = krc.get_generator(self._fault_data, self._magnitude, self._rake)

        return krg, leaf
 