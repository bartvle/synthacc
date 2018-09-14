"""
The 'source.rupture.models' module.
"""


import numpy as np

from ...apy import (Object, is_pos_number, is_2d_numeric_array,
    is_3d_numeric_array)
from ... import space3
from ...data import LogicTree
from ...earth import flat as earth
from ..moment import MomentRateFunction, NormalizedMomentRateFunction
from ..mechanism import FocalMechanism
from ..faults import RIGIDITY
from .geometry import FaultGeometryCalculator, FaultSegmentCalculator
from .slip import (SlipDistribution, SlipDistributionCalculator,
    SlipRateCalculator)
from .hypo import HypoCenterCalculator
from .velocity import VelocityDistributionCalculator
from .rake import RakeDistributionCalculator
from .propagation import TravelTimeCalculator


class PointRupture(Object):
    """
    """

    def __init__(self, point, focal_mechanism, moment, nmrf=None, validate=True):
        """
        """
        point = space3.Point(*point, validate=validate)

        if validate is True:
            assert(type(focal_mechanism) is FocalMechanism)
            assert(is_pos_number(moment))
            if nmrf is not None:
                assert(type(nmrf) is NormalizedMomentRateFunction)

        self._point = point
        self._focal_mechanism = focal_mechanism
        self._moment = moment
        self._nmrf = nmrf

    @property
    def point(self):
        """
        """
        return self._point

    @property
    def focal_mechanism(self):
        """
        """
        return self._focal_mechanism

    @property
    def moment(self):
        """
        """
        return self._moment

    @property
    def nmrf(self):
        """
        """
        return self._nmrf

    @property
    def moment_tensor(self):
        """
        """
        return self._focal_mechanism.get_moment_tensor(self.moment)


# class SimpleRupture(Object):
#     """
#     """

#     def __init__(self, segment, hypo, rake, slip, nsrf=None, rigidity=RIGIDITY, validate=True):
#         """
#         """
#         hypo = space3.Point(*hypo, validate=validate)

#         if validate is True:
#             assert(type(segment) is earth.Rectangle)
#             assert(hypo in segment)
#             assert(is_rake(rake))
#             assert(is_pos_number(slip))
#             if nsrf is not None:
#                 assert(type(nsrf) is NormalizedSlipRateFunction)
#             assert(is_pos_number(rigidity))

#         self._segment = segment
#         self._hypo = hypo
#         self._rake = rake
#         self._slip = slip
#         self._nsrf = nsrf
#         self._rigidity = rigidity

#         ## cached properties
#         self._discretized, self._spacing = None, None
        
#     @property
#     def segment(self):
#         """
#         """
#         return self._segment

#     @property
#     def hypo(self):
#         """
#         """
#         return self._hypo

#     @property
#     def rake(self):
#         """
#         """
#         return self._rake

#     @property
#     def slip(self):
#         """
#         """
#         return self._slip

#     @property
#     def nsrf(self):
#         """
#         """
#         return self._nsrf

#     @property
#     def rigidity(self):
#         """
#         """
#         return self._rigidity

#     @property
#     def area(self):
#         """
#         """
#         return self._segment.area

#     @property
#     def epi(self):
#         """
#         return: 'space3.Point' instance
#         """
#         return space3.Point(self.hypo.x, self.hypo.y, 0)

#     @property
#     def focal_mechanism(self):
#         """
#         return: 'source.mechanism.FocalMechanism' instance
#         """
#         fm = FocalMechanism(
#             self._segment.strike, self._segment.dip, self.rake)

#         return fm

#     @property
#     def moment(self):
#         """
#         """
#         moment = calculate_moment(
#             self.area, self.slip, self.rigidity, validate=False)

#         return moment

#     @property
#     def magnitude(self):
#         """
#         """
#         return m0_to_mw(self.moment)

#     def get_hypo_distance(self, point, validate=True):
#         """
#         return: pos number
#         """
#         p = space3.Point(*point)
#         if validate is True:
#             assert(p.z == 0)
#         d = space3.distance(*self.hypo, *p)
#         return d

#     def get_epi_distance(self, point, validate=True):
#         """
#         return: pos number
#         """
#         p = space3.Point(*point)
#         if validate is True:
#             assert(p.z == 0)
#         d = space3.distance(*self.epi, *p)
#         return d

#     def get_rup_distance(self, point, spacing=1000, validate=True):
#         """
#         return: pos number
#         """
#         p = space3.Point(*point)
#         if validate is True:
#             assert(p.z == 0)
#             assert(is_pos_number(spacing))

#         if self._discretized is None or self._spacing != spacing:
#             w, l = self._segment.width, self._segment.length
#             nw = int(round(w / spacing))
#             nl = int(round(l / spacing))
#             self._discretized = self._segment.get_discretized((nw, nl))
#             self._spacing = spacing

#         xs, ys, zs = np.rollaxis(self._discretized.corners, 2)

#         x, y, z = space3.nearest(*p, xs, ys, zs)
#         d = space3.distance(x, y, z, *p)

#         return d

#     def get_jb_distance(self, point, spacing=1000, validate=True):
#         """
#         return: pos number
#         """
#         p = space3.Point(*point)
#         if validate is True:
#             assert(p.z == 0)
#             assert(is_pos_number(spacing))

#         if self._discretized is None or self._spacing != spacing:
#             w, l = self._segment.width, self._segment.length
#             nw = int(round(w / spacing))
#             nl = int(round(l / spacing))
#             self._discretized = self._segment.get_discretized((nw, nl))
#             self._spacing = spacing

#         xs, ys, zs = np.rollaxis(self._discretized.corners, 2)
#         zs = np.zeros_like(zs)

#         x, y, z = space3.nearest(*p, xs, ys, zs)
#         d = space3.distance(x, y, z, *p)

#         return d

#     def plot(self):
#         """
#         """
#         _, ax = plt.subplots()

#         ulc, urc, llc, lrc = self._segment.corners

#         ax.plot([ulc.y, urc.y], [ulc.x, urc.x], c='r', lw=2)

#         ax.fill(
#             [ulc.y, urc.y, lrc.y, llc.y],
#             [ulc.x, urc.x, lrc.x, llc.x],
#             color='coral', alpha=0.5,
#             )

#         ax.scatter([self.hypo.y], [self.hypo.x], marker='*', s=50)

#         ax.axis('equal')

#         x_label, y_label = 'East (m)', 'North (m)'
#         ax.xaxis.set_label_text(x_label)
#         ax.yaxis.set_label_text(y_label)

#         plt.show()


class FiniteRupture(Object):
    """
    """

    def __init__(self, segment, hypo, rake, time_delta, slip_rates, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(segment) is earth.DiscretizedSimpleSurface)
            assert(hypo in segment)
            assert(is_2d_numeric_array(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(np.all(slip_rates >= 0))
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(segment.shape == slip_rates.shape[:2] == rake.shape)
            assert(is_pos_number(rigidity))

        self._segment = segment
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._rigidity = rigidity

    def __len__(self):
        """
        """
        return len(self._segment)

#     def __iter__(self):
#         """
#         """
#         slip = self.slip.values
#         area = self._segment.cell_area
#         centers = self._segment.centers
#         fm = self.focal_mechanism

#         for i in np.ndindex(self._segment.shape):
#             moment = calculate_moment(area, float(slip[i]), self.rigidity)

#             if moment == 0:
#                 return None

#             x, y, z = centers[i]
#             x = float(x)
#             y = float(y)
#             z = float(z)
#             point = space3.Point(x, y, z)

#             nmrf = NormalizedMomentRateFunction(
#             self.time_delta, self._slip_rates[i] / slip[i])

#             yield PointRupture(point, fm, moment, nmrf)

    @property
    def segment(self):
        """
        """
        return self._segment

    @property
    def hypo(self):
        """
        """
        return self._hypo

    @property
    def rake(self):
        """
        """
        return self._rake

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    @property
    def area(self):
        """
        """
        return self._segment.area

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        fm = FocalMechanism(
            self._segment.strike, self._segment.dip, self.rake)

        return fm

    @property
    def slip(self):
        """
        """
        sd = SlipDistribution(self._segment.width, self._segment.length,
            np.sum(self._slip_rates, axis=2) * self._time_delta)

        return sd

    @property
    def mrf(self):
        """
        return: 'source.moment.MomentRateFunction' instance
        """
        mrf = MomentRateFunction(self._time_delta, np.sum((self._slip_rates *
            self._segment.cell_area * self._rigidity), axis=(0,1)))

        return mrf

#     @property
#     def moment(self):
#         """
#         """
#         return self.mrf.moment

#     @property
#     def magnitude(self):
#         """
#         """
#         return m0_to_mw(self.moment)

#     def play(self, size=None, validate=True):
#         """
#         """
#         fig, ax = plt.subplots(figsize=size)

#         extent = [0, self._segment.length/1000, self._segment.width/1000, 0]

#         ims = []
#         for i in range(self._slip_rates.shape[-1]):
#             im = ax.imshow(self._slip_rates[:,:,i], animated=False,
#                 interpolation='none', extent=extent, vmin=0,
#                 vmax=self._slip_rates.max())
#             ims.append([im])

#         fig.colorbar(ims[0][0])

#         xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)

#         ani = animation.ArtistAnimation(
#             fig, ims, interval=50, blit=True).to_html5_video()

#         plt.close()

#         return ani


# class MagnitudeCalculator(Object):
#     """
#     """

#     OF = {'sl': 'length', 'l': 'length', 'w': 'width', 'a': 'area'}

#     def __init__(self, sr, validate=True):
#         """
#         """
#         if validate is True:
#             assert(isinstance(sr, ScalingRelationship))
#             assert(sr.OF in ('sl', 'l', 'w', 'a'))
#             assert(sr.TO == 'm')

#         self._sr = sr

#     def __call__(self, segment, validate=True):
#         """
#         """
#         if validate is True:
#             assert(type(segment) is earth.Rectangle)

#         m = self._sr.sample(getattr(segment, self.OF[self._sr.OF]))

#         return float(m)


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
        #print('type segment', type(segment))
        #TODO: check this implementation
        #TODO: are all these parameters needed?
        # rd = self._rdc(segment, magnitude, rake)()
        sd = self._sdc(segment, magnitude, fault.rigidity)()
    
        ## Calculate 2D and 3D hypocenter
        hypo2 = self._hcc(segment, sd)
        w_vector = segment.ad_vector.unit * hypo2.x
        l_vector = segment.as_vector.unit * hypo2.y
        hypo3 = space3.Point(*segment.ulc.translate(w_vector + l_vector))

        sr = self._src(segment, magnitude, sd)
        assert(np.all(sr >= 0))
        vd = self._vdc(segment, magnitude, sd)

        #TODO: which delta to take?
        ttc = TravelTimeCalculator(vd, d=100)
        #print(hypo2.x, vd.w) ##TODO Sometimes hypo.x is > vd.w
        #print(hypo2.y, vd.l)
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

        r = FiniteRupture(segment.get_discretized(sd.shape), hypo3, rd,
            self._src._d, slip_rates, fault.rigidity)

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

    def __call__(self, validate=True):
        """
        """
        leaf = self.sample()

        fgc = self._fgc or leaf.get('fgc')
        fsc = self._fsc or leaf.get('fsc')
        sdc = self._sdc or leaf.get('sdc')
        hcc = self._hcc or leaf.get('hcc')
        src = self._src or leaf.get('src')
        vdc = self._vdc or leaf.get('vdc')
        rdc = self._rdc or leaf.get('rdc')

        fgc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('fgc_')}
        fsc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('fsc_')}
        sdc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('sdc_')}
        hcc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('hcc_')}
        src_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('src_')}
        vdc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('vdc_')}
        rdc_params = {b[0].split('_')[1]: b[1] for b in leaf.path if b[0].startswith('rdc_')}

        fgc = fgc(**fgc_params)
        fsc = fsc(**fsc_params)
        sdc = sdc(**sdc_params)
        hcc = hcc(**hcc_params)
        src = src(**src_params)
        vdc = vdc(**vdc_params)
        rdc = rdc(**rdc_params)

        krc = KinematicRuptureCalculator(fgc, fsc, sdc, hcc, src, vdc, rdc)

        return krc, leaf

#     def get_generator(self, fault_data, magnitude, rake, validate=True):
#         """
#         """
#         krglt = KinematicRuptureGeneratorLogicTree(
#             self, fault_data, magnitude, rake, validate=validate)

#         return krglt


# class KinematicRuptureGeneratorLogicTree(Object):
#     """
#     """

#     def __init__(self, krclt, fault_data, magnitude, rake, validate=True):
#         """
#         """
#         if validate is True:
#             assert(type(krclt) is KinematicRuptureCalculatorLogicTree)

#         self._krclt = krclt
#         self._fault_data = fault_data
#         self._magnitude = magnitude
#         self._rake = rake

    # def __call__(self, validate=True):
    #     """
    #     """
#         cl = self._krclt()
#         gl = LogicTreeLeaf(cl.path, KinematicRuptureGenerator(
#             cl.model, self._fault, self._magnitude, self._rake))

#         return gl
