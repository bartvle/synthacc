"""
The 'source.rupture.models' module.
"""


import matplotlib.pyplot as plt
import numpy as np

from ...apy import Object, is_number, is_pos_number, is_3d_numeric_array
from ... import space2
from ... import space3
from ...earth.flat import RectangularSurface, DiscretizedRectangularSurface
from ..moment import (NormalizedMomentRateFunction, MomentRateFunction,
    NormalizedSlipRateFunction, calculate as calculate_moment, m0_to_mw,
    mw_to_m0)
from ..mechanism import FocalMechanism, is_rake
from ..faults import RIGIDITY
from .slip import (SlipDistribution, RFSlipDistributionGenerator,
    LiuEtAl2006NormalizedSlipRateGenerator)
from .velocity import VelocityDistribution
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


class SimpleRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, slip, nsrf=None, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is RectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            if nsrf is not None:
                assert(type(nmrf) is NormalizedSlipRateFunction)
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
        return self.surface.area

    @property
    def epi(self):
        """
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
            assert(type(surface) is DiscretizedRectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(slip_rates.shape[:2] == surface.shape)
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(is_pos_number(rigidity))

        self._surface = surface
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._rigidity = rigidity

    def __len__(self):
        """
        """
        return len(self.surface)

    def __iter__(self):
        """
        """
        slip = self.slip
        area = self.surface.cell_area
        centers = self.surface.centers
        fm = self.focal_mechanism

        for i in np.ndindex(self.surface.shape):
            moment = calculate_moment(float(slip[i], area, self.rigidity))

            if moment == 0:
                return None

            x, y, z = centers[i]
            x = float(x)
            y = float(y)
            z = float(z)
            point = space3.Point(x, y, z)

            nmrf = NormalizedMomentRateFunction(
            self.time_delta, self._slip_rates[i] / slip[i])

            yield PointRupture(point, fm, moment, nmrf)

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

    @property
    def onsets(self):
        """
        """
        onsets = np.zeros(self.surface.shape)
        for i in np.ndindex(onsets.shape):
            indices = np.where(self._slip_rates[i] != 0)[0]
            if len(indices) != 0:
                onsets[i] = indices[0] * self._time_delta
            else:
                onsets[i] = np.nan

        return onsets

    @property
    def slip(self):
        """
        """
        w, l = self.surface.width, self.surface.length
        slip = np.sum(self._slip_rates, axis=2) * self._time_delta
        return SlipDistribution(w, l, slip)

    @property
    def mean_slip(self):
        """
        """
        return self.slip.mean

    @property
    def mrf(self):
        """
        return: 'source.moment.MomentRateFunction' instance
        """
        moment_rates = (self._slip_rates * self.surface.cell_area *
            self.rigidity)
        moment_rates = np.sum(moment_rates, axis=(0,1))
        mrf = MomentRateFunction(self._time_delta, moment_rates)
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

    def play(self):
        """
        """
        pass


class GP2016KinematicRuptureGenerator(Object):
    """
    Graves & Pitarka (2016) kinematic rupture generator (GP15.4).
    """

    def __init__(self, time_delta, velocity, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(time_delta))
            assert(is_pos_number(velocity))
            assert(is_pos_number(rigidity))

        self._time_delta = time_delta
        self._velocity = velocity
        self._rigidity = rigidity

    def __call__(self, surface, rake, magnitude, validate=True):
        """
        """
        if validate is True:
            assert(type(surface) is RectangularSurface)
            assert(is_rake(rake))
            assert(is_number(magnitude))

        moment = mw_to_m0(magnitude)

        hypo = surface.get_random()

        w, l = surface.width, surface.length

        nw = int(w / 100 // 2 * 2 + 1)
        nl = int(l / 100 // 2 * 2 + 1)

        surface = surface.get_discretized(shape=(nw, nl))

        dw = surface.spacing[0]
        dl = surface.spacing[1]

        acf = space2.VonKarmanACF(h=0.75)

        aw = 10**(1/3*magnitude-1.6) * 1000
        al = 10**(1/2*magnitude-2.5) * 1000

        g = RFSlipDistributionGenerator(w, l, dw, dl, acf, aw, al)

        sd = g(magnitude, self.rigidity)

        _, _, depths = np.rollaxis(surface.centers, 2)
        rise_times = self._get_rise_times(depths, sd.slip)

        average = self._get_average_rise_time(surface.dip, moment)
        rise_times *= (average / rise_times.mean())

        ## Propagation
        vd = VelocityDistribution(w, l, np.ones(sd.shape)*self.velocity)

        hv = hypo.vector - surface.outline.ulc.vector
        wv = surface.outline.llc.vector - surface.outline.ulc.vector
        lv = surface.outline.urc.vector - surface.outline.ulc.vector
        x = float(np.cos(np.radians(hv.get_angle(lv))) * hv.magnitude)
        y = float(np.cos(np.radians(hv.get_angle(wv))) * hv.magnitude)

        ttc = TravelTimeCalculator(vd, d=100)
        tts = ttc(x, y)

        n_onsets = np.round(tts.times / self.time_delta).astype(np.int)

        n_rise_times = np.round(rise_times / self.time_delta).astype(np.int)

        n = n_onsets + n_rise_times

        slip_rates = np.zeros(surface.shape + (n.max()+2,))

        nsrf_g = LiuEtAl2006NormalizedSlipRateGenerator(self.time_delta)

        for i in np.ndindex(surface.shape):
            t = rise_times[i]
            if t != 0:
                srf = nsrf_g(float(t))
                slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf * sd.slip[i]

        rupture = KinematicRupture(surface, hypo, rake, self._time_delta,
            slip_rates, self._rigidity)

        return rupture

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def velocity(self):
        """
        """
        return self._velocity

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

    def _get_average_rise_time(self, dip, moment):
        """
        See Graves & Pitarka (2010) p. 2099 eq. 8 and 9. Adjusted for moment in
        Nm instead of dyn-cm.
        """
        factor = np.interp(dip, [45, 60], [0.82, 1])
        t = factor * 1.6 * 10**-9 * (10**7*moment)**(1/3)
        return t

    def _get_rise_times(self, depths, slip):
        """
        See Graves & Pitarka (2010) p. 2098 eq. 7.
        """
        return np.interp(depths, [5000, 8000], [2, 1]) * (slip/100)**(1/2)
