"""
Tests for 'source.rupture.propagation' module.
"""


import unittest

from synthacc.source.rupture.propagation import (VelocityDistribution,
    RFVelocityDistributionGenerator, GP2010VelocityDistributionGenerator,
    GP2016VelocityDistributionGenerator, TravelTimes,
    ConstantVelocityTravelTimeCalculator, TravelTimeCalculator)
