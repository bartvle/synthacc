"""
The 'source.rupture.models' module.
"""


import random

import matplotlib.pyplot as plt
from matplotlib import rc, animation
import numpy as np

from ...apy import Object, is_pos_number, is_3d_numeric_array
from ... import space3
from ...earth import flat as earth
from ..moment import (MomentRateFunction, NormalizedMomentRateFunction,
    NormalizedSlipRateFunction, calculate as calculate_moment, m0_to_mw)
from ..mechanism import FocalMechanism, is_rake
from ..faults import RIGIDITY
from .slip import SlipDistribution
from .propagation import TravelTimeCalculator


rc('animation', html='html5')


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


class SimpleRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, slip, nsrf=None, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is earth.Rectangle)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            if nsrf is not None:
                assert(type(nsrf) is NormalizedSlipRateFunction)
            assert(is_pos_number(rigidity))

        self._surface = surface
        self._hypo = hypo
        self._rake = rake
        self._slip = slip
        self._nsrf = nsrf
        self._rigidity = rigidity

        ## cached properties
        self._discretized, self._spacing = None, None
        
    @property
    def surface(self):
        """
        """
        return self._surface

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
    def slip(self):
        """
        """
        return self._slip

    @property
    def nsrf(self):
        """
        """
        return self._nsrf

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    @property
    def area(self):
        """
        """
        return self._surface.area

    @property
    def epi(self):
        """
        return: 'space3.Point' instance
        """
        return space3.Point(self.hypo.x, self.hypo.y, 0)

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        return FocalMechanism(self.surface.strike, self.surface.dip, self.rake)

    @property
    def moment(self):
        """
        """
        moment = calculate_moment(
            self.area, self.slip, self.rigidity, validate=False)

        return moment

    @property
    def magnitude(self):
        """
        """
        return m0_to_mw(self.moment)

    def get_hypo_distance(self, point, validate=True):
        """
        return: pos number
        """
        p = space3.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space3.distance(*self.hypo, *p)
        return d

    def get_epi_distance(self, point, validate=True):
        """
        return: pos number
        """
        p = space3.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space3.distance(*self.epi, *p)
        return d

    def get_rup_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space3.Point(*point)
        if validate is True:
            assert(p.z == 0)
            assert(is_pos_number(spacing))

        if self._discretized is None or self._spacing != spacing:
            w, l = self._surface.width, self._surface.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self.surface.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)

        x, y, z = space3.nearest(*p, xs, ys, zs)
        d = space3.distance(x, y, z, *p)

        return d

    def get_jb_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space3.Point(*point)
        if validate is True:
            assert(p.z == 0)
            assert(is_pos_number(spacing))

        if self._discretized is None or self._spacing != spacing:
            w, l = self._surface.width, self._surface.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self.surface.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)
        zs = np.zeros_like(zs)

        x, y, z = space3.nearest(*p, xs, ys, zs)
        d = space3.distance(x, y, z, *p)

        return d

    def plot(self):
        """
        """
        fig, ax = plt.subplots()

        ulc, urc, llc, lrc = self.surface.corners

        ax.plot([ulc.y, urc.y], [ulc.x, urc.x], c='r', lw=2)

        ax.fill(
            [ulc.y, urc.y, lrc.y, llc.y],
            [ulc.x, urc.x, lrc.x, llc.x],
            color='coral', alpha=0.5,
            )

        ax.scatter([self.hypo.y], [self.hypo.x], marker='*', s=50)

        ax.axis('equal')

        x_label, y_label = 'East (m)', 'North (m)'
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        plt.show()


class KinematicRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, time_delta, slip_rates, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is earth.Rectangle)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(is_pos_number(rigidity))

        self._surface = surface.get_discretized(slip_rates.shape[:2])
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._rigidity = rigidity

    def __len__(self):
        """
        """
        return len(self._surface)

#     def __iter__(self):
#         """
#         """
#         slip = self.slip
#         area = self.surface.cell_area
#         centers = self.surface.centers
#         fm = self.focal_mechanism

#         for i in np.ndindex(self.surface.shape):
#             moment = calculate_moment(float(slip[i], area, self.rigidity))

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
    def surface(self):
        """
        """
        return self._surface

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
        return self.surface.area

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        return FocalMechanism(self.surface.strike, self.surface.dip, self.rake)

#     @property
#     def onsets(self):
#         """
#         """
#         onsets = np.zeros(self.surface.shape)
#         for i in np.ndindex(onsets.shape):
#             indices = np.where(self._slip_rates[i] != 0)[0]
#             if len(indices) != 0:
#                 onsets[i] = indices[0] * self._time_delta
#             else:
#                 onsets[i] = np.nan

#         return onsets

    @property
    def slip(self):
        """
        """
        w, l = self.surface.width, self.surface.length
        slip = np.sum(self._slip_rates, axis=2) * self._time_delta

        return SlipDistribution(w, l, slip)

    @property
    def mrf(self):
        """
        return: 'source.moment.MomentRateFunction' instance
        """
        mrs = (self._slip_rates * self._surface.cell_area * self._rigidity)
        mrs = np.sum(mrs, axis=(0,1))
        mrf = MomentRateFunction(self._time_delta, mrs)

        return mrf

    @property
    def moment(self):
        """
        """
        return self.mrf.moment

    @property
    def magnitude(self):
        """
        """
        return m0_to_mw(self.moment)

    def play(self, size=None, validate=True):
        """
        """
        fig, ax = plt.subplots(figsize=size)

        extent = [0, self.surface.length/1000, self.surface.width/1000, 0]

        im = ax.imshow(self._slip_rates[:,:,1], animated=True, interpolation='bicubic', extent=extent, vmax=self._slip_rates.max())

        cbar = fig.colorbar(im)

        xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        def updatefig(j):
            im.set_array(self._slip_rates[:,:,j])
            return im,

        ani = animation.FuncAnimation(fig, updatefig, frames=self._slip_rates.shape[-1], interval=50, blit=True)

        return ani


class MagnitudeCalculator(Object):
    """
    """

    def __init__(self, relation, property):
        """
        """
        self._relation = relation
        self._property = property

    def __call__(self, surface):
        """
        """
        m = random.gauss(*self._relation(getattr(surface, self._property)))

        return float(m)


class KinematicRuptureCalculator(Object):
    """
    """

    def __init__(self, sdcs, srcs, vdcs, rc=None, ac=None, mc=None, validate=True):
        """
        """
        if validate is True:
            assert(type(sdcs) is list)
            assert(type(srcs) is list)
            assert(type(vdcs) is list)

        self._sdcs = sdcs
        self._srcs = srcs
        self._vdcs = vdcs
        self._rc = rc
        self._ac = ac
        self._mc = mc

    def __call__(self, fault, rake=None, area=None, magnitude=None, validate=True):
        """
        return: 'synthacc.source.rupture.models.KinematicRuptureGenerator'
            instance
        """
        if validate is True:
            pass

        krg = KinematicRuptureGenerator(self, fault, rake, area, magnitude, validate=False)

        return krg


class KinematicRuptureGenerator(Object):
    """
    """

    def __init__(self, calculator, fault, rake, area, magnitude, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(calculator, KinematicRuptureCalculator))

            if rake is not None:
                pass
            if area is not None:
                pass
            if magnitude is not None:
                pass

        self._calculator = calculator
        self._fault = fault
        self._rake = rake
        self._area = area
        self._magnitude = magnitude

    def __call__(self, validate=True):
        """
        """
        rectangle, surface = self._fault.rectangle, self._fault.surface

        if self._magnitude is None:
            magnitude = self._calculator._mc(rectangle)
        else:
            magnitude = self._magnitude

        sdc = random.choice(self._calculator._sdcs)
        src = random.choice(self._calculator._srcs)
        vdc = random.choice(self._calculator._vdcs)

        sd = sdc(rectangle, magnitude, self._fault.rigidity)()
        sr = src(rectangle, magnitude, sd)()
        vd = vdc(rectangle, magnitude, sd)()

        hypo2 = surface.get_random()
        w_vector = rectangle.ad_vector.unit * hypo2.x
        l_vector = rectangle.as_vector.unit * hypo2.y
        hypo3 = space3.Point(*rectangle.ulc.translate(w_vector + l_vector))

        ttc = TravelTimeCalculator(vd, d=100)
        tts = ttc(*hypo2, ds=sd)

        assert(sd.shape == sr.shape[:2] == tts.shape) ## dev test

        n_onsets = np.round(tts.times / src._time_delta).astype(np.int)
        i = np.ones(sr.shape) * np.arange(sr.shape[-1])
        i[sr==0] = 0
        n_rise_times = np.argmax(i, axis=2)
        n = n_onsets + n_rise_times
        slip_rates = np.zeros(sd.shape + (n.max()+2,))

        for i in np.ndindex(sd.shape):
            srf = sr[i][:n_rise_times[i]]
            slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf

        kr = KinematicRupture(rectangle, hypo3, self._rake, src._time_delta,
            slip_rates, self._fault.rigidity)

        return kr


# class GP2016KRG(KinematicRuptureGenerator):
#     """
#     Graves & Pitarka (2016) kinematic rupture generator (GP15.4).
#     """

#     def __init__(self, time_delta, velocity, rigidity=RIGIDITY, validate=True):
#         """
#         """
#         if validate is True:
#             assert(is_pos_number(time_delta))
#             assert(is_pos_number(velocity))
#             assert(is_pos_number(rigidity))

#         self._time_delta = time_delta
#         self._velocity = velocity
#         self._rigidity = rigidity

#         sdcs = 
    
#             acf = space2.VonKarmanACF(h=0.75)

#         aw = 10**(1/3*magnitude-1.6) * 1000
#         al = 10**(1/2*magnitude-2.5) * 1000

#         super.__init__(self, sdcs)

#     def __call__(self, surface, rake, magnitude, hypo=None, validate=True):
#         """
#         """
#         if validate is True:
#             assert(type(surface) is Rectangle)
#             assert(is_rake(rake))
#             assert(is_number(magnitude))

#         moment = mw_to_m0(magnitude)

#         if hypo is None:
#             hypo = surface.get_random()
#         else:
#             hypo = space3.Point(*hypo, validate=validate)

#         w, l = surface.width, surface.length

#         nw = int(w / 100 // 2 * 2 + 1)
#         nl = int(l / 100 // 2 * 2 + 1)

#         surface = surface.get_discretized(shape=(nw, nl))

#         dw = surface.spacing[0]
#         dl = surface.spacing[1]

#         g = RFSlipDistributionGenerator(w, l, dw, dl, acf, aw, al)

#         sd = g(magnitude, self.rigidity)

#         _, _, depths = np.rollaxis(surface.centers, 2)
#         rise_times = self._get_rise_times(depths, sd.slip)

#         average = self._get_average_rise_time(surface.dip, moment)
#         rise_times *= (average / rise_times.mean())

#         ## Propagation
#         vd = VelocityDistribution(w, l, np.ones(sd.shape)*self.velocity)

#         hv = hypo.vector - surface.outline.ulc.vector
#         wv = surface.outline.llc.vector - surface.outline.ulc.vector
#         lv = surface.outline.urc.vector - surface.outline.ulc.vector
#         x = float(np.cos(np.radians(hv.get_angle(wv))) * hv.magnitude)
#         y = float(np.cos(np.radians(hv.get_angle(lv))) * hv.magnitude)

#         ttc = TravelTimeCalculator(vd, d=100)
#         tts = ttc(x, y)

#         n_onsets = np.round(tts.times / self.time_delta).astype(np.int)

#         n_rise_times = np.round(rise_times / self.time_delta).astype(np.int)

#         n = n_onsets + n_rise_times

#         slip_rates = np.zeros(surface.shape + (n.max()+2,))

#         nsrf_g = LiuEtAl2006NormalizedSlipRateGenerator(self.time_delta)

#         for i in np.ndindex(surface.shape):
#             t = rise_times[i]
#             if t != 0:
#                 srf = nsrf_g(float(t))
#                 slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf * sd.slip[i]

#         rupture = KinematicRupture(surface, hypo, rake, self._time_delta,
#             slip_rates, self._rigidity)

#         return rupture

#     def _get_average_rise_time(self, dip, moment):
#         """
#         See Graves & Pitarka (2010) p. 2099 eq. 8 and 9. Adjusted for moment in
#         Nm instead of dyn-cm.
#         """
#         factor = np.interp(dip, [45, 60], [0.82, 1])
#         t = factor * 1.6 * 10**-9 * (10**7*moment)**(1/3)
#         return t

#     def _get_rise_times(self, depths, slip):
#         """
#         See Graves & Pitarka (2010) p. 2098 eq. 7.
#         """
#         return np.interp(depths, [5000, 8000], [2, 1]) * (slip/100)**(1/2)
