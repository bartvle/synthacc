"""
The 'ruptures' module. Rectangular ruptures from a
'source.faults.SingularFault' class.
"""


from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.interpolate

from ..apy import Object, is_pos_integer, is_pos_number, is_3d_numeric_array
from ..math.random import ACF, SpatialRandomFieldGenerator
from .. import space
from ..earth.flat import RectangularSurface, DiscretizedRectangularSurface
from .moment import (NormalizedMomentRateFunction, MomentRateFunction,
    calculate as calculate_moment, m0_to_mw)
from .mechanism import FocalMechanism, is_rake
from .faults import RIGIDITY, SingularFault


class PointRupture(Object):
    """
    """

    def __init__(self, point, focal_mechanism, moment, nmrf=None, rigidity=RIGIDITY, validate=True):
        """
        """

        point = space.Point(*point, validate=validate)

        if validate is True:
            assert(type(focal_mechanism) is FocalMechanism)
            assert(is_pos_number(moment))
            if nmrf is not None:
                assert(type(nmrf) is NormalizedMomentRateFunction)

        self._point = point
        self._focal_mechanism = focal_mechanism
        self._moment = moment
        self._nmrf = nmrf
        self._rigidity = rigidity

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
    def rigidity(self):
        """
        """
        return self._rigidity

    @property
    def moment_tensor(self):
        """
        """
        return self._focal_mechanism.get_moment_tensor(self.moment)


class SimpleRupture(Object):
    """
    """

    def __init__(self, surface, hypo, rake, slip, rigidity=RIGIDITY, validate=True):

        hypo = space.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is RectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            assert(is_pos_number(rigidity))

        self._surface = surface
        self._hypo = hypo
        self._rake = rake
        self._slip = slip
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
        return space.Point(self.hypo.x, self.hypo.y, 0)

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
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space.distance(*self.hypo, *p)
        return d

    def get_epi_distance(self, point, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
        if validate is True:
            assert(p.z == 0)
        d = space.distance(*self.epi, *p)
        return d

    def get_rup_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
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

        x, y, z = space.nearest(*p, xs, ys, zs)
        d = space.distance(x, y, z, *p)

        return d

    def get_jb_distance(self, point, spacing=1000, validate=True):
        """
        return: pos number
        """
        p = space.Point(*point)
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

        x, y, z = space.nearest(*p, xs, ys, zs)
        d = space.distance(x, y, z, *p)

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
        hypo = space.Point(*hypo, validate=validate)

        if validate is True:
            assert(type(surface) is DiscretizedRectangularSurface)
            assert(hypo in surface)
            assert(is_rake(rake))
            assert(is_pos_number(time_delta))
            assert(is_3d_numeric_array(slip_rates))
            assert(np.all(slip_rates[:,:,+0] == 0))
            assert(np.all(slip_rates[:,:,-1] == 0))
            assert(slip_rates.shape[:2] == surface.shape)
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
            moment = float(slip[i] * area * self.rigidity)

            if moment == 0:
                continue

            x, y, z = centers[i]
            x = float(x)
            y = float(y)
            z = float(z)
            point = space.Point(x, y, z)

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
        return FocalMechanism(
            self.surface.outline.strike, self.surface.outline.dip, self.rake)

    @property
    def duration(self):
        """
        """
        return (self._slip_rates.shape[-1]-1) * self._time_delta

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
        return np.sum(self._slip_rates, axis=2) * self._time_delta

    @property
    def mean_slip(self):
        """
        """
        return np.mean(self.slip)

    @property
    def mrf(self):
        """
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

    def _plot(self, data, label, title, validate=True):
        """
        """
        if validate is True:
            assert(data.shape == self._surface.shape)

        f, ax = plt.subplots()
        xs, ys = self.surface.get_front_projected_corners()
        p = ax.pcolormesh(xs, ys, data)

        ax.axis('scaled')
        ax.set_xlim(ax.get_xaxis().get_data_interval())
        ax.set_ylim(ax.get_yaxis().get_data_interval())
        ax.invert_yaxis()

        x_label, y_label = 'Along strike (m)', 'Along dip (m)'
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)

        cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=0.25)
        cbar = f.colorbar(p, cax=cax, norm=mpl.colors.Normalize(vmin=0))
        cbar.set_label(label)

        ax.set_title(title)

        plt.show()

    def plot_onsets(self):
        """
        """
        self._plot(self.onsets, 'Onset (s)', 'Rupture onset')

    def plot_slip(self):
        """
        """
        self._plot(self.slip, 'Slip (m)', 'Slip distribution')


class RuptureGenerator(ABC, Object):
    """
    """

    @abstractmethod
    def get_rupture(self):
        """
        """
        pass


class RandomKinematicRuptureGenerator(RuptureGenerator):
    """
    Generator for a random kinematic rupture.
    """

    def __init__(self, velocity, acf, nmrf, seed=None, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(velocity))
            assert(isinstance(acf, ACF))
            assert(type(nmrf is NormalizedMomentRateFunction))
            if seed is not None:
                assert(is_pos_integer(seed))

        self._velocity = velocity
        self._acf = acf
        self._nmrf = nmrf
        self._seed = seed

    @property
    def velocity(self):
        """
        """
        return self._velocity

    @property
    def acf(self):
        """
        """
        return self._acf

    @property
    def nmrf(self):
        """
        """
        return self._nmrf

    @property
    def seed(self):
        """
        """
        return self._seed

    def get_rupture(self, fault, rake, slip, shape, validate=True):
        """
        return: 'ruptures.KinematicRupture' instance
        """
        if validate is True:
            assert(type(fault) is SingularFault)
            assert(is_rake(rake))
            assert(is_pos_number(slip))
            assert(type(shape) is tuple)
            assert(is_pos_integer(shape[0]) and (shape[0] % 2) == 0)
            assert(is_pos_integer(shape[1]) and (shape[1] % 2) == 0)

        surface = fault.surface

        ds = DiscretizedRectangularSurface(
            surface.ulc.x, surface.ulc.y,
            surface.urc.x, surface.urc.y,
            surface.upper_depth,
            surface.lower_depth,
            surface.dip, shape, validate=validate)

        hypo = surface.get_random()

        dimension = (surface.width, surface.length)
        cd = dimension

        srfg = SpatialRandomFieldGenerator(
            dimension, ds.spacing, self.acf, cd, self.seed)

        srf = srfg()

        xs, ys = ds.get_front_projected_corners()
        interp = scipy.interpolate.interp2d(xs[0,:], ys[:,0], srf)
        xs, ys = ds.get_front_projected_centers()
        srf = np.reshape(interp(xs[0,:], ys[:,0]), ds.shape)

        srf[srf < 0] = 0
        srf *= (slip / srf.mean())

        onsets = (space.distance(*hypo, *np.rollaxis(ds.centers, 2)) /
            self.velocity)

        n = np.round(onsets / self.nmrf.time_delta).astype(np.int)
        slip_rates = np.zeros(ds.shape + (n.max()+len(self.nmrf),))

        for i in np.ndindex(ds.shape):            
            slip_rates[i][n[i]:n[i]+len(self.nmrf)] = self.nmrf.rates * srf[i]

        rupture = KinematicRupture(ds, hypo, rake, self.nmrf.time_delta,
            slip_rates, fault.rigidity)

        return rupture
