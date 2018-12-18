"""
The 'source.rupture.models' module.
"""


import numpy as np

from ...apy import (Object, is_pos_number, is_2d_numeric_array,
    is_3d_numeric_array)
from ... import space3
from ...earth import flat as earth
from ..moment import (MomentRateFunction, NormalizedMomentRateFunction,
    calculate as calculate_moment, m0_to_mw)
from ..mechanism import FocalMechanism, is_rake
from ..faults import RIGIDITY, ComposedFault
from .slip import SlipDistribution
from .rake import RakeDistribution


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

    def __init__(self, segment, hypo, rake, slip, nmrf=None, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(segment) is earth.SimpleSurface)
            assert(hypo in segment)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            if nmrf is not None:
                assert(type(nmrf) is NormalizedMomentRateFunction)
            assert(is_pos_number(rigidity))

        self._segment = segment
        self._hypo = hypo
        self._rake = rake
        self._slip = slip
        self._nmrf = nmrf
        self._rigidity = rigidity

        ## cached properties
        self._discretized, self._spacing = None, None
        
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
    def slip(self):
        """
        """
        return self._slip

    @property
    def nmrf(self):
        """
        """
        return self._nmrf

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
        fm = FocalMechanism(
            self._segment.strike, self._segment.dip, self.rake)

        return fm

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
            w, l = self._segment.width, self._segment.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self._segment.get_discretized((nw, nl))
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
            w, l = self._segment.width, self._segment.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self._segment.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)
        zs = np.zeros_like(zs)

        x, y, z = space3.nearest(*p, xs, ys, zs)
        d = space3.distance(x, y, z, *p)

        return d

    # def plot(self):
    #     """
    #     """
    #     _, ax = plt.subplots()

    #     ulc, urc, llc, lrc = self._segment.corners

    #     ax.plot([ulc.y, urc.y], [ulc.x, urc.x], c='r', lw=2)

    #     ax.fill(
    #         [ulc.y, urc.y, lrc.y, llc.y],
    #         [ulc.x, urc.x, lrc.x, llc.x],
    #         color='coral', alpha=0.5,
    #         )

    #     ax.scatter([self.hypo.y], [self.hypo.x], marker='*', s=50)

    #     ax.axis('equal')

    #     x_label, y_label = 'East (m)', 'North (m)'
    #     ax.xaxis.set_label_text(x_label)
    #     ax.yaxis.set_label_text(y_label)

    #     plt.show()


class SimpleFiniteRupture(Object):
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
        return np.prod(self.shape)

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
        return RakeDistribution(self.width, self.length, self._rake)

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def slip_rates(self):
        """
        """
        return self._slip_rates

    @property
    def shape(self):
        """
        """
        return self._slip_rates.shape[:2]

    @property
    def width(self):
        """
        """
        return self._segment.width

    @property
    def length(self):
        """
        """
        return self._segment.length

    @property
    def area(self):
        """
        """
        return self._segment.area

    @property
    def upper_depth(self):
        """
        """
        return self._segment.upper_depth

    @property
    def lower_depth(self):
        """
        """
        return self._segment.lower_depth

    @property
    def rigidity(self):
        """
        """
        return self._rigidity

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
            (self.area / len(self)) * self._rigidity), axis=(0,1)))

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
        return self.mrf.magnitude


class ComposedFiniteRupture(Object):
    """
    """

    def __init__(self, segment, hypo, rake, time_delta, slip_rates, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(segment) is ComposedFault)
            assert(hypo in segment)
            assert(is_2d_numeric_array(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(np.all(slip_rates >= 0))
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(slip_rates.shape[:2] == rake.shape)

        self._segment = segment
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._nls = self._calc_nls()

    def __len__(self):
        """
        """
        return np.prod(self.shape)

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
        return RakeDistribution(self.width, self.length, self._rake)

    @property
    def time_delta(self):
        """
        """
        return self._time_delta

    @property
    def slip_rates(self):
        """
        """
        return self._slip_rates

    @property
    def shape(self):
        """
        """
        return self._slip_rates.shape[:2]

    @property
    def width(self):
        """
        """
        return self._segment.width

    @property
    def length(self):
        """
        """
        return self._segment.length

    @property
    def area(self):
        """
        """
        return self._segment.area

    @property
    def upper_depth(self):
        """
        """
        return self._segment.upper_depth

    @property
    def lower_depth(self):
        """
        """
        return self._segment.lower_depth

    @property
    def rigidity(self):
        """
        """
        return self._segment.rigidity

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
            (self.area / len(self)) * self.rigidity), axis=(0,1)))

        return mrf

    def _calc_nls(self):
        """
        """
        l, nl = self.length, self.shape[1]
        nls = [round(nl * f.length / l) for f in self._segment]

        return nls
