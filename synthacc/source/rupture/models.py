"""
The 'source.rupture.models' module.
"""


import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ...apy import Object, is_number, is_pos_number, is_3d_numeric_array
from ... import space3
from ...earth import flat as earth
from ..moment import (MomentRateFunction, NormalizedMomentRateFunction,
    NormalizedSlipRateFunction, calculate as calculate_moment, m0_to_mw)
from ..mechanism import FocalMechanism, is_rake
from ..faults import RIGIDITY
from ..scaling import ScalingRelationship
from .slip import SlipDistribution, SlipDistributionCalculator
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

    def __init__(self, rectangle, hypo, rake, slip, nsrf=None, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(rectangle) is earth.Rectangle)
            assert(hypo in rectangle)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            if nsrf is not None:
                assert(type(nsrf) is NormalizedSlipRateFunction)
            assert(is_pos_number(rigidity))

        self._rectangle = rectangle
        self._hypo = hypo
        self._rake = rake
        self._slip = slip
        self._nsrf = nsrf
        self._rigidity = rigidity

        ## cached properties
        self._discretized, self._spacing = None, None
        
    @property
    def rectangle(self):
        """
        """
        return self._rectangle

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
        return self._rectangle.area

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
            self.rectangle.strike, self.rectangle.dip, self.rake)

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
            w, l = self._rectangle.width, self._rectangle.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self._rectangle.get_discretized((nw, nl))
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
            w, l = self._rectangle.width, self._rectangle.length
            nw = int(round(w / spacing))
            nl = int(round(l / spacing))
            self._discretized = self._rectangle.get_discretized((nw, nl))
            self._spacing = spacing

        xs, ys, zs = np.rollaxis(self._discretized.corners, 2)
        zs = np.zeros_like(zs)

        x, y, z = space3.nearest(*p, xs, ys, zs)
        d = space3.distance(x, y, z, *p)

        return d

    def plot(self):
        """
        """
        _, ax = plt.subplots()

        ulc, urc, llc, lrc = self._rectangle.corners

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


class FiniteRupture(Object):
    """
    """

    def __init__(self, rectangle, hypo, rake, time_delta, slip_rates, rigidity=RIGIDITY, validate=True):
        """
        """
        hypo = space3.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(rectangle) is earth.Rectangle)
            assert(hypo in rectangle)
            assert(is_rake(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(is_pos_number(rigidity))

        self._rectangle = rectangle.get_discretized(slip_rates.shape[:2])
        self._hypo = hypo
        self._rake = rake
        self._time_delta = time_delta
        self._slip_rates = slip_rates
        self._rigidity = rigidity

    def __len__(self):
        """
        """
        return len(self._rectangle)

    def __iter__(self):
        """
        """
        slip = self.slip.slip
        area = self._rectangle.cell_area
        centers = self._rectangle.centers
        fm = self.focal_mechanism

        for i in np.ndindex(self._rectangle.shape):
            moment = calculate_moment(area, float(slip[i]), self.rigidity)

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
    def rectangle(self):
        """
        """
        return self._rectangle

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
        return self._rectangle.area

    @property
    def focal_mechanism(self):
        """
        return: 'source.mechanism.FocalMechanism' instance
        """
        fm = FocalMechanism(
            self._rectangle.strike, self._rectangle.dip, self.rake)

        return fm

    @property
    def slip(self):
        """
        """
        w, l = self.rectangle.width, self.rectangle.length
        slip = np.sum(self._slip_rates, axis=2) * self._time_delta

        return SlipDistribution(w, l, slip)

    @property
    def mrf(self):
        """
        return: 'source.moment.MomentRateFunction' instance
        """
        mrf = MomentRateFunction(self._time_delta, np.sum((self._slip_rates *
            self.rectangle.cell_area * self._rigidity), axis=(0,1)))

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

        extent = [0, self.rectangle.length/1000, self.rectangle.width/1000, 0]

        ims = []
        for i in range(self._slip_rates.shape[-1]):
            im = ax.imshow(self._slip_rates[:,:,i], animated=False,
                interpolation='none', extent=extent, vmin=0,
                vmax=self._slip_rates.max())
            ims.append([im])

        fig.colorbar(ims[0][0])

        xlabel, ylabel = 'Along strike (km)', 'Along dip (km)'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True).to_html5_video()

        plt.close()

        return ani


class SurfaceCalculator(Object):
    """
    Calculates rupture surface from fault and magnitude with magnitude to area
    or length scaling relationship and aspect ratio. An aspect ratio (length /
    width) is followed if the rupture width is smaller than the fault width. It
    must be greater than or equal to 1 (width <= length).
    """

    def __init__(self, sr, ar, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(sr, ScalingRelationship))
            assert(sr.OF == 'm')
            assert(sr.TO in ('sl', 'l', 'w', 'a'))
            if type(ar) is tuple:
                assert(len(ar) == 2)
                assert(is_number(ar[0]))
                assert(is_number(ar[1]))
                assert(ar[0] >= 1 and ar[1] > ar[0])
            else:
                assert(is_number(ar) and ar >= 1)

        self._sr = sr
        self._ar = ar

    def __call__(self, fault, magnitude, validate=True):
        """
        return: 'earth.flat.Rectangle' instance
        """
        if validate is True:
            pass

        if type(self._ar) is tuple:
            ar = np.random.uniform(0, 1) * (self._ar[1] - self._ar[0])
        else:
            ar = self._ar

        if self._sr.TO in ('sl', 'l'):
            l = self._sr.sample(magnitude)

            if l >= fault.length:
                l = fault.length

            w = min([fault.width, l / ar])

        elif self._sr.TO in ('w'):
            w = self._sr.sample(magnitude)

            if w >= fault.width:
                w = fault.width

            l = min([fault.length, w * ar])

        else:
            a = self._sr.sample(magnitude)

            if a >= fault.area:
                return fault.rectangle

            w = min(np.sqrt(a / ar), fault.width)
            l = a / w

        w = float(w)
        l = float(l)

        surface = fault.surface
        advu = fault.ad_vector.unit
        asvu = fault.as_vector.unit
        wtv = advu * float(np.random.uniform(0, 1) * (surface.w - w))
        ltv = asvu * float(np.random.uniform(0, 1) * (surface.l - l))
        ulc = fault.ulc.translate(wtv + ltv)
        llc = ulc.translate(advu * w)
        urc = ulc.translate(asvu * l)

        r = earth.Rectangle(
            ulc.x, ulc.y, urc.x, urc.y, ulc.z, llc.z, fault.dip)

        return r


class MagnitudeCalculator(Object):
    """
    """

    OF = {'sl': 'length', 'l': 'length', 'w': 'width', 'a': 'area'}

    def __init__(self, sr, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(sr, ScalingRelationship))
            assert(sr.OF in ('sl', 'l', 'w', 'a'))
            assert(sr.TO == 'm')

        self._sr = sr

    def __call__(self, rectangle, validate=True):
        """
        """
        if validate is True:
            assert(type(rectangle) is earth.Rectangle)

        m = self._sr.sample(getattr(rectangle, self.OF[self._sr.OF]))

        return float(m)


class RakeCalculator(Object):
    """
    """

    def __init__(self, validate=True):
        """
        """
        pass


class KinematicRuptureCalculator(Object):
    """
    """

    def __init__(self, sdcs, srcs, vdcs, scs=None, mcs=None, rcs=None, validate=True):
        """
        """
        if validate is True:
            assert(type(sdcs) is list)
            for sdc in sdcs:
                assert(isinstance(sdc, SlipDistributionCalculator))
            assert(type(srcs) is list)
            assert(type(vdcs) is list)
            assert(scs is not None or mcs is not None)
            if scs is not None:
                assert(type(scs) is list)
                for sc in scs:
                    assert(type(sc) is SurfaceCalculator)
            if mcs is not None:
                assert(type(mcs) is list)
                for mc in mcs:
                    assert(type(mc) is MagnitudeCalculator)
            if rcs is not None:
                assert(type(rcs) is list)
                for rc in rcs:
                    assert(type(rc) is RakeCalculator)

        self._sdcs = tuple(sdcs)
        self._srcs = tuple(srcs)
        self._vdcs = tuple(vdcs)
        self._scs = tuple(scs) if scs is not None else None
        self._mcs = tuple(mcs) if mcs is not None else None
        self._rcs = tuple(rcs) if rcs is not None else None

    def __call__(self, fault, magnitude=None, rake=None, validate=True):
        """
        Takes fault(segment). If magnitude is given surface will be calculated
        with scaling relationship and aspect ratio. If no magnitude is given
        whole fault(segment) will rupture and magnitude will be calculated with
        scaling relationship.

        return: 'source.rupture.models.KinematicRuptureGenerator' instance
        """
        if validate is True:
            pass

        krg = KinematicRuptureGenerator(
            self, fault, magnitude, rake, validate=False)

        return krg

    @property
    def sdcs(self):
        """
        """
        return self._sdcs

    @property
    def srcs(self):
        """
        """
        return self._srcs

    @property
    def vdcs(self):
        """
        """
        return self._vdcs

    @property
    def rcs(self):
        """
        """
        return self._rcs

    @property
    def scs(self):
        """
        """
        return self._scs

    @property
    def mcs(self):
        """
        """
        return self._mcs


class KinematicRuptureGenerator(Object):
    """
    """

    def __init__(self, calculator, fault, magnitude, rake, validate=True):
        """
        Generates realizations for a scenario earthquake on a fault.
        """
        if validate is True:
            assert(type(calculator) is KinematicRuptureCalculator)

        self._calculator = calculator
        self._fault = fault
        self._magnitude = magnitude
        self._rake = rake

    def __call__(self, validate=True):
        """
        """
        if self._calculator._scs is not None:
            sc = random.choice(self._calculator._scs)
        if self._calculator._mcs is not None:
            mc = random.choice(self._calculator._mcs)
        if self._calculator._rcs is not None:
            rc = random.choice(self._calculator._rcs)

        sdc = random.choice(self._calculator._sdcs)
        src = random.choice(self._calculator._srcs)
        vdc = random.choice(self._calculator._vdcs)

        if self._magnitude is not None:
            magnitude = self._magnitude
            rectangle = sc(self._fault, magnitude)
        else:
            rectangle = self._fault.rectangle
            magnitude = mc(rectangle)

        rake = rc() if self._rake is None else self._rake

        #TODO: check this implementation
        #TODO: are all these parameters needed?
        sd = sdc(rectangle, magnitude, self._fault.rigidity)()
        sr = src(rectangle, magnitude, sd)()
        vd = vdc(rectangle, magnitude, sd)()

        hypo2 = rectangle.surface.get_random()
        w_vector = rectangle.ad_vector.unit * hypo2.x
        l_vector = rectangle.as_vector.unit * hypo2.y
        hypo3 = space3.Point(*rectangle.ulc.translate(w_vector + l_vector))

        #TODO: which delta to take?
        ttc = TravelTimeCalculator(vd, d=100)
        tts = ttc(*hypo2, ds=sd)

        #TODO: dev test / remove
        assert(sd.shape == sr.shape[:2] == tts.shape)

        #TODO: check this implementation
        n_onsets = np.round(tts.times / src._time_delta).astype(np.int)
        i = np.ones(sr.shape) * np.arange(sr.shape[-1])
        i[sr==0] = 0
        n_rise_times = np.argmax(i, axis=2)
        n = n_onsets + n_rise_times
        slip_rates = np.zeros(sd.shape + (n.max()+2,))

        for i in np.ndindex(sd.shape):
            srf = sr[i][:n_rise_times[i]]
            slip_rates[i][n_onsets[i]:n_onsets[i]+len(srf)] = srf

        r = FiniteRupture(rectangle, hypo3, rake, src._time_delta, slip_rates,
            self._fault.rigidity)

        return r

    @property
    def calculator(self):
        """
        """
        return self._calculator

    @property
    def fault(self):
        """
        """
        return self._fault
