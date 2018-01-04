"""
Tests for 'source.moment' module.
"""


import unittest

from synthacc.source.moment import (MomentTensor, NormalizedMomentFunction,
    MomentFunction, NormalizedMomentRateFunction, NormalizedSourceTimeFunction,
    MomentRateFunction, SourceTimeFunction, InstantMomentRateGenerator,
    ConstantMomentRateGenerator, TriangularMomentRateGenerator, calculate,
    m0_to_mw, mw_to_m0)
