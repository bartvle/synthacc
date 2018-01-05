"""
The 'methods.gmpes' module.

For using Ground Motion Prediction Equations (GMPEs). For our implementation we
use the OpenQuake Engine (Pagani et al., 2014).
"""


import numpy as np

from openquake.hazardlib.const import TRT as _TRT
from openquake.hazardlib.gsim import (get_available_gsims as
    _get_available_gsims)
from openquake.hazardlib.gsim.base import (CoeffsTable as _CT, GMPE as _GMPE,
    RuptureContext as _RC, DistancesContext as _DC, SitesContext as _SC)
from openquake.hazardlib.imt import (PGD as _PGD, PGV as _PGV, PGA as _PGA,
                                     SA as _SA)
from openquake.hazardlib.const import StdDev as _SD_TYPES

from ..apy import Object, is_boolean
from ..response import ResponseSpectrum


_GSIMS = _get_available_gsims()

## List of the possible tectonic regions
TECTONIC_REGIONS = [
    v.lower() for k, v in _TRT.__dict__.items() if not k.startswith('__')]

DEFAULT_PARAMETERS = {'rake': 0, 'vs30': 800}


class GMPE(Object):
    """
    A Ground Motion Prediction Equation (GMPE).
    """

    def __init__(self, name, validate=True):
        """
        """
        if validate is True:
            assert(name in GMPES)

        self._gmpe = _GSIMS[name]()
        self._name = name

        if self.has_sa():
            table = self._gmpe.COEFFS
            self._periods = np.array(
                sorted([float(sa.period) for sa in table.sa_coeffs.keys()]))
        else:
            self._periods = None

    def __call__(self, parameters):
        """
        """
        parameters = dict(DEFAULT_PARAMETERS, **parameters)

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
                val = np.asarray(parameters[p] / 1000)
                if val.ndim == 0:
                    val = val[np.newaxis]
                setattr(dc, p, val)

            elif p in _SC._slots_:
                val = np.asarray(parameters[p])
                if val.ndim == 0:
                    val = val[np.newaxis]
                setattr(sc, p, val)

            else:
                raise

        return rc, dc, sc

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
    def parameters(self):
        """
        """
        parameters = set()
        for group in ('RUPTURE_PARAMETERS', 'DISTANCES', 'SITES_PARAMETERS'):
            parameters = parameters.union(
                getattr(self._gmpe, 'REQUIRES_' + group))
        return parameters

    @property
    def distance_metric(self):
        """
        """
        return tuple(self._gmpe.REQUIRES_DISTANCES)[0]

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

        rc, dc, sc = self(parameters)

        imt = _PGA()

        pga = np.exp(self._gmpe.get_mean_and_stddevs(
            sc, rc, dc, imt, [_SD_TYPES.TOTAL])[0][0])

        if unit != 'g':
            raise

        return pga

    def get_pgv(self):
        """
        """
        raise NotImplementedError

    def get_pgd(self):
        """
        """
        raise NotImplementedError

    def get_response_spectrum(self, parameters, damping=0.05, unit='m/s2', validate=True):
        """
        """
        if validate is True:
            assert(self.has_sa())

        rc, dc, sc = self(parameters)

        periods = self.periods
        if self.has_pga():
            periods = [0] + periods

        responses = np.zeros((len(periods),))

        for i, p in enumerate(periods):

            if p != 0:
                imt = _SA(p, damping * 100)
            else:
                imt = _PGA()

            responses[i] = np.exp(self._gmpe.get_mean_and_stddevs(
                sc, rc, dc, imt, [_SD_TYPES.TOTAL])[0][0])

        if unit != 'g':
            raise

        rs = ResponseSpectrum(periods, responses, unit=unit, damping=damping)

        return rs


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

    gmpes = []
    for name, gsim in _GSIMS.items():

        ## Only GMPEs (remove IPEs and GMPETable)
        if not issubclass(gsim, _GMPE) or gsim is _GSIMS['GMPETable']:
            continue

        ## For GMPEs with spectral acceleration, we support only the ones that
        ## have a coefficient table called 'COEFFS' (that is not derived from a
        ## superclass), because our GMPE class requires the spectral periods,
        ## which can only be retrieved from this table. See issue 2.
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
GMPES = find_gmpes()
