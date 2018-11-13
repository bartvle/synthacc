"""
The 'gmpes' module.

For using Ground Motion Prediction Equations (GMPEs). For our implementation we
use the OpenQuake Engine (Pagani et al., 2014).
"""


import matplotlib.pyplot as plt
import numpy as np

from openquake.hazardlib.const import TRT as _TRT
from openquake.hazardlib.gsim import (get_available_gsims as
    _get_available_gsims)
from openquake.hazardlib.gsim.base import (GMPE as _GMPE, CoeffsTable as _CT,
    RuptureContext as _RC, DistancesContext as _DC, SitesContext as _SC)
from openquake.hazardlib.imt import (PGD as _PGD, PGV as _PGV, PGA as _PGA,
                                     SA as _SA)
from openquake.hazardlib.const import StdDev as _SD_TYPES

from .apy import Object, is_boolean, is_pos_number, is_string
from .units import MOTION as UNITS
from .response import (ResponseSpectrum, plot_response_spectra as
    _plot_response_spectra)
from .plot import set_space


_AVAILABLE_GSIMS = _get_available_gsims()

## See https://docs.openquake.org/oq-hazardlib/stable/openquake.hazardlib.html#module-openquake.hazardlib.imt
_UNITS = {'acc': 'g', 'vel': 'cm/s', 'dis': 'cm'}

## List of the possible tectonic regions
TECTONIC_REGIONS = [
    v.lower() for k, v in _TRT.__dict__.items() if not k.startswith('__')]

DISTANCE_METRICS = ('rhypo', 'repi', 'rrup', 'rjb')


class GMPE(Object):
    """
    A Ground Motion Prediction Equation (GMPE).
    """

    def __init__(self, name, validate=True):
        """
        """
        if validate is True:
            assert(name in AVAILABLE_GMPES)

        self._gmpe = _AVAILABLE_GSIMS[name]()
        self._name = name

        if self.has_sa():
            table = self._gmpe.COEFFS
            self._periods = np.array(
                sorted([float(sa.period) for sa in table.sa_coeffs.keys()]))
        else:
            self._periods = None

    def __call__(self, parameters, imt, unit='m/s2', validate=True):
        """
        """
        rc, dc, sc = self._get_context(parameters)

        mean, stddevs = self._gmpe.get_mean_and_stddevs(
            sc, rc, dc, imt, [_SD_TYPES.TOTAL])

        unit = UNITS[unit]
        if unit.quantity[:3] == 'acc':
            factor = UNITS[_UNITS['acc']] / unit
        elif unit.quantity[:3] == 'vel':
            factor = UNITS[_UNITS['vel']] / unit
        elif unit.quantity[:3] == 'dis':
            factor = UNITS[_UNITS['dis']] / unit

        stddevs_m = np.exp(mean[0] - stddevs[0]) * factor
        stddevs_p = np.exp(mean[0] + stddevs[0]) * factor
        mean = np.exp(mean[0]) * factor

        return mean, (stddevs_m, stddevs_p)

    @property
    def name(self):
        """
        return: string, name of the GMPE
        """
        return self._name

    @property
    def periods(self):
        """
        return: 1d num array or None
        """
        if self._periods is not None:
            return np.copy(self._periods)

    @property
    def min_period(self):
        """
        return: number or None
        """
        if self._periods is not None:
            return float(self._periods.min())

    @property
    def max_period(self):
        """
        return: number or None
        """
        if self._periods is not None:
            return float(self._periods.max())

    @property
    def tectonic_region(self):
        """
        return: string, one of TECTONIC_REGIONS.
        """
        return self._gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE.lower()

    @property
    def distance_metric(self):
        """
        """
        return tuple(self._gmpe.REQUIRES_DISTANCES)[0]

    @property
    def parameters(self):
        """
        """
        parameters = set()
        for group in ('RUPTURE_PARAMETERS', 'DISTANCES', 'SITES_PARAMETERS'):
            parameters = parameters.union(
                getattr(self._gmpe, 'REQUIRES_' + group))
        return parameters

    def _get_context(self, parameters):
        """
        """
        ## check if all required parameters are given
        for p in self.parameters:
            assert(p in parameters)

        rc = _RC()
        dc = _DC()
        sc = _SC()

        for p in parameters:

            if p in _RC._slots_:
                setattr(rc, p, parameters[p])

            elif p in _DC._slots_:
                val = np.asarray(parameters[p] / 1000) ## m to km
                if val.ndim == 0:
                    val = val[np.newaxis]
                setattr(dc, p, val)

            elif p in _SC._slots_:
                val = np.asarray(parameters[p])
                if val.ndim == 0:
                    val = val[np.newaxis]
                setattr(sc, p, val)

            else:
                raise ## parameter is not required

        return rc, dc, sc

    def has_pgd(self):
        """
        return: boolean
        """
        return _PGD in self._gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES

    def has_pgv(self):
        """
        return: boolean
        """
        return _PGV in self._gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES

    def has_pga(self):
        """
        return: boolean
        """
        return _PGA in self._gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES

    def has_sa(self):
        """
        return: boolean
        """
        return _SA in self._gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES

    def is_dependent(self, parameter):
        """
        """
        return parameter in self.parameters

    def get_pga(self, parameters, unit='m/s2', validate=True):
        """
        """
        if validate is True:
            assert(self.has_pga())
            assert(UNITS[unit].quantity == 'acceleration')

        return self(parameters, _PGA(), unit)

    def get_mean_pga(self, parameters, unit='m/s2', validate=True):
        """
        """
        return self.get_pga(parameters, unit, validate)[0]

    def get_pgv(self, parameters, unit='m/s', validate=True):
        """
        """
        if validate is True:
            assert(self.has_pgv())
            assert(UNITS[unit].quantity == 'velocity')

        return self(parameters, _PGV(), unit)

    def get_mean_pgv(self, parameters, unit='m/s', validate=True):
        """
        """
        return self.get_pgv(parameters, unit, validate)[0]

    def get_pgd(self, parameters, unit='m', validate=True):
        """
        """
        if validate is True:
            assert(self.has_pgd())
            assert(UNITS[unit].quantity == 'displacement')

        return self(parameters, _PGD(), unit)

    def get_mean_pgd(self, parameters, unit='m', validate=True):
        """
        """
        return self.get_pgd(parameters, unit, validate)[0]

    def get_sa(self, parameters, period, damping=0.05, unit='m/s2', validate=True):
        """
        """
        if validate is True:
            assert(self.has_sa())
            assert(is_pos_number(period) and period in self.periods)
            assert(UNITS[unit].quantity == 'acceleration')

        return self(parameters, _SA(period, damping * 100), unit)

    def get_mean_sa(self, parameters, period, damping=0.05, unit='m/s2', validate=True):
        """
        """
        return self.get_sa(parameters, period, damping, unit, validate)[0]

    def get_response_spectrum(self, parameters, damping=0.05, unit='m/s2', validate=True):
        """
        """
        if validate is True:
            assert(self.has_sa())

        periods = self.periods
        if self.has_pga():
            periods = [0] + periods

        means = np.zeros(len(periods))
        stddevs_m = np.zeros(len(periods))
        stddevs_p = np.zeros(len(periods))
        for i, p in enumerate(periods):
            if p != 0:
                imt = _SA(p, damping * 100)
            else:
                imt = _PGA()
            mean, stddevs = self(parameters, imt, unit)
            means[i] = mean
            stddevs_m[i] = stddevs[0]
            stddevs_p[i] = stddevs[1]

        mean = ResponseSpectrum(periods, means, unit=unit, damping=damping)
        stddev_m = ResponseSpectrum(periods, stddevs_m, unit=unit, damping=damping)
        stddev_p = ResponseSpectrum(periods, stddevs_p, unit=unit, damping=damping)

        return (mean, (stddev_m, stddev_p))

    def get_mean_response_spectrum(self, parameters, damping=0.05, unit='m/s2', validate=True):
        """
        """
        return self.get_response_spectrum(parameters, damping, unit, validate)[0]


def find_gmpes(tectonic_region=None, sa=None, pga=None, pgv=None, pgd=None, distance_metric=None, validate=True):
    """
    Find GMPEs.

    tectonic_region: string
    sa: boolean
    pga: boolean
    pgv: boolean
    pgd: boolean

    return: list of strings, names of GMPEs
    """
    if validate is True:
        if tectonic_region is not None:
            assert(tectonic_region.lower() in TECTONIC_REGIONS)
        if sa is not None:
            assert(is_boolean(sa))
        if pga is not None:
            assert(is_boolean(pga))
        if pgv is not None:
            assert(is_boolean(pgv))
        if pgd is not None:
            assert(is_boolean(pgd))
        if distance_metric is not None:
            assert(distance_metric in DISTANCE_METRICS)

    gmpes = []
    for name, gsim in _AVAILABLE_GSIMS.items():

        ## Only GMPEs (remove IPEs and GMPETable)
        if (not issubclass(gsim, _GMPE) or
            gsim is _AVAILABLE_GSIMS['GMPETable']):
            continue

        ## For GMPEs with spectral acceleration, we support only the ones that
        ## have a coefficient table called 'COEFFS' (that is not derived from a
        ## superclass), because our GMPE class requires the spectral periods,
        ## which can only be retrieved from this table. See issue 7.
        if _SA in gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
            if not ('COEFFS' in gsim.__dict__ and
                isinstance(gsim.COEFFS, _CT)):
                continue

        ## Only GMPEs with one distance metric
        if len(gsim.REQUIRES_DISTANCES) != 1:
            continue

        ## Tectonic region
        if (tectonic_region is not None and
            (gsim.DEFINED_FOR_TECTONIC_REGION_TYPE.lower() !=
                tectonic_region.lower())):
            continue

        ## SA, PGA, PGV and PGD
        if (sa is not None and sa is not
            (_SA in gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)):
            continue
        if (pga is not None and pga is not
            (_PGA in gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)):
            continue
        if (pgv is not None and pgv is not
            (_PGV in gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)):
            continue
        if (pgd is not None and pgd is not
            (_PGD in gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)):
            continue

        if distance_metric is not None:
            if tuple(gsim.REQUIRES_DISTANCES)[0] != distance_metric:
                continue

        gmpes.append(name)

    return gmpes


## List with names of available GMPEs
AVAILABLE_GMPES = find_gmpes()


def plot_gmpes_distance(gmpes, gmp, distances, parameters, labels=True, unit=None, space='linlog', min_dis=None, max_dis=None, min_val=None, max_val=None, size=None, filespec=None, validate=True):
    """
    Plot GMPEs for set of parameters in function of distance.

    gmp: string ('pga', 'pgv' or 'pgd') or pos number (sa period)
    """
    if validate is True:
        for gmpe in gmpes:
            assert(type(gmpe) is GMPE)
            if is_string(gmp):
                assert(getattr(gmpe, 'has_%s' % gmp)())
            else:
                assert(gmp in gmpe.periods)

    fig, ax = plt.subplots()

    for i, gmpe in enumerate(gmpes):
        means, stddevs_m, stddevs_p = [], [], []
        if is_string(gmp):
            f = getattr(gmpe, 'get_%s' % gmp)
            for d in distances:
                mean, stddevs = f({**{gmpe.distance_metric: d}, **parameters},
                    unit, validate=False)
                means.append(mean)
                stddevs_m.append(stddevs[0])
                stddevs_p.append(stddevs[1])
        else:
            for d in distances:
                mean, stddevs = gmpe.get_sa(
                    {**{gmpe.distance_metric: d}, **parameters}, gmp,
                    unit=unit, validate=False)
                means.append(mean)
                stddevs_m.append(stddevs[0])
                stddevs_p.append(stddevs[1])

        kwargs = {}
        if labels is True:
            kwargs['label'] = gmpe.name
        elif labels:
            kwargs['label'] = labels[i]

        [p] = plt.plot(distances, means, **kwargs)
        plt.plot(distances, stddevs_m, c=p.get_color(), ls='--')
        plt.plot(distances, stddevs_p, c=p.get_color(), ls='--')

    if labels is True or labels:
        ax.legend()

    ax.grid(which='both')

    set_space(ax, space)

    ax.set_xlim([min_dis, max_dis])
    ax.set_ylim([min_val, max_val])

    if is_string(gmp):
        gmp = gmp.upper()
    else:
        gmp = 'SA %ss' % gmp

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('%s (%s)' % (gmp, unit))

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
    plt.close(fig)


def plot_gmpes_magnitude(gmpes, gmp, magnitudes, parameters, labels=True, unit=None, space='linlin', min_mag=None, max_mag=None, min_val=None, max_val=None, size=None, filespec=None, validate=True):
    """
    Plot GMPEs for set of parameters in function of magnitude.

    gmp: string ('pga', 'pgv' or 'pgd') or pos number (sa period)
    """
    if validate is True:
        for gmpe in gmpes:
            assert(type(gmpe) is GMPE)
            if is_string(gmp):
                assert(getattr(gmpe, 'has_%s' % gmp)())
            else:
                assert(gmp in gmpe.periods)

    fig, ax = plt.subplots()

    for i, gmpe in enumerate(gmpes):
        means, stddevs_m, stddevs_p = [], [], []
        if is_string(gmp):
            f = getattr(gmpe, 'get_%s' % gmp)
            for m in magnitudes:
                mean, stddevs = f({**{'mag': m}, **parameters}, unit,
                    validate=False)
                means.append(mean)
                stddevs_m.append(stddevs[0])
                stddevs_p.append(stddevs[1])
        else:
            for m in magnitudes:
                mean, stddevs = gmpe.get_sa({**{'mag': m}, **parameters}, gmp,
                    unit=unit, validate=False)
                means.append(mean)
                stddevs_m.append(stddevs[0])
                stddevs_p.append(stddevs[1])

        kwargs = {}
        if labels is True:
            kwargs['label'] = gmpe.name
        elif labels:
            kwargs['label'] = labels[i]

        [p] = plt.plot(magnitudes, means, **kwargs)
        plt.plot(magnitudes, stddevs_m, c=p.get_color(), ls='--')
        plt.plot(magnitudes, stddevs_p, c=p.get_color(), ls='--')

    if labels is True or labels:
        ax.legend()

    ax.grid(which='both')

    set_space(ax, space)

    ax.set_xlim([min_mag, max_mag])
    ax.set_ylim([min_val, max_val])

    if is_string(gmp):
        gmp = gmp.upper()
    else:
        gmp = 'SA %ss' % gmp

    ax.set_xlabel('Magnitude')
    ax.set_ylabel('%s (%s)' % (gmp, unit))

    if filespec is not None:
        plt.savefig(filespec)
    else:
        plt.show()
    plt.close(fig)


def plot_gmpes_spectrum(gmpes, parameters, damping=0.05, labels=True, unit=None, space='linlog', min_period=None, max_period=None, size=None, filespec=None, validate=True):
    """
    Plot GMPEs for set of parameters in function of spectral period.
    """
    if validate is True:
        for gmpe in gmpes:
            assert(type(gmpe) is GMPE)
            assert(gmpe.has_sa())

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if labels is True:
        set_labels, labels = True, []
    else:
        set_labels = False

    rss, colors, styles, widths = [], [], [], None
    for i, gmpe in enumerate(gmpes):
        mean, stddevs = gmpe.get_response_spectrum(parameters, damping, unit,
            validate=False)

        rss.extend([mean] + list(stddevs))
        if set_labels is True:
            labels.extend([gmpe.name, '__nolegend__', '__nolegend__'])
        colors.extend([default_colors[i]] * 3)
        styles.extend(['-', '--', '--'])

    _plot_response_spectra(rss, labels, colors, styles, widths, unit, space, min_period=min_period, max_period=max_period, size=size, filespec=filespec)
