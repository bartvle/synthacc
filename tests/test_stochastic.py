"""
Tests for 'stochastic' module.
"""


import unittest

# import numpy as np
# import os

from synthacc.stochastic import (OmegaSquareSourceModel,
    GeometricalSpreadingModel, QModel, AttenuationModel)

# from synthacc.quantities import round_to_significant
# from synthacc.source.moment import m0_to_mw, mw_to_m0
# from synthacc.io.smsim import read_fas_model


# DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'stochastic')
# FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'visual')


class TestOmegaSquareSourceModel(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass

    # """
    # From SMSIM example:
    # r, f_ff, rmod, amag, kappa =   3.000E+01  0.000E+00  3.000E+01  7.500E+00  2.000E-02
    # const=   4.364E-24
    # amag, stress, fa, fb=   7.500E+00  2.500E+02  9.083E-02  9.083E-02
    # am0, am0b_m0fa=   1.995E+27  0.000E+00
    # """

    # mag = m0_to_mw(mw_to_m0(7.5, precise=False), precise=True)
    # sd = 250 * 10**5  ## magnitude independent stress drop (Pa)
    # rp = 0.55  ## radiation patttern
    # pf = 1 / np.sqrt(2)  ## partition factor
    # fsf = 2  ## free surface factor
    # rho = 2800  ## density (kg/m3)
    # vel = 3700  ## shear wave velocity (m/s)

    # source_model = OmegaSquareSourceModel(mag, sd, rp, pf, fsf, rho, vel)

    # def test_properties(self):
        # """
        # """
        # self.assertEqual(round_to_significant(self.source_model.seismic_moment, 4), 1.995E+27 / 10**7)
        # self.assertEqual(round_to_significant(self.source_model.corner_frequency, 4), 9.083E-02)
        # # self.assertEqual(round_to_significant(self.source_model.constant, 4), 4.364 * 10**-24)

    # def test_plot(self):
        # """
        # """
        # fs = os.path.join(FIG_DIR, 'methods.stochastic.omega_square_source_model.plot.png')
        # frequencies = np.logspace(-2, 2, 100)
        # self.source_model.plot(frequencies, png_filespec=fs)

class TestGeometricalSpreadingModel(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestQModel(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


class TestAttenuationModel(unittest.TestCase):
    """
    #TODO: implement test
    """
    pass


# class Test(unittest.TestCase):
    # """
    # """

    # def test(self):
        # """
        # """
        # tgt_fas = read_fas_model(os.path.join(DATA_DIR, 'ab06_bc.m7.50r0030.0_fs.col'))[0]

        # frequencies = tgt_fas.frequencies

        # mag = m0_to_mw(mw_to_m0(7.5, precise=False), precise=True)
        # sd = 250 * 10**5  ## magnitude independent stress drop (Pa)
        # rp = 0.55  ## radiation patttern
        # pf = 1 / np.sqrt(2)  ## partition factor
        # fsf = 2  ## free surface factor
        # rho = 2800  ## density (kg/m3)
        # vel = 3700  ## shear wave velocity (m/s)

        # source_model = OmegaSquareSourceModel(mag, sd, rp, pf, fsf, rho, vel)
        # source_model_fas = source_model.get_fas(frequencies)

        # gs_segments = [
            # (1, -1.3),
            # (70, 0.2),
            # (140, -0.5)
            # ]

        # gs_model = GeometricalSpreadingModel(gs_segments)

        # qr1 = 1000
        # ft1 = 0.2
        # fr1 = 0.02
        # s1 = 0.0
        # qr2 = 1272
        # ft2 = 1.4242
        # fr2 = 3.02
        # s2 = 0.32

        # q_model = QModel(fr1, qr1, s1, ft1, ft2, fr2, qr2, s2)

        # c_q = 3.7
        # kappa = 0.02  ## fm is zero and kappa is magnitude independent

        # att_model = AttenuationModel(q_model, c_q, kappa)

        # distance = 30

        # se = np.array([
            # (0.0001, 1.000),
            # (0.1014, 1.073),
            # (0.2402, 1.145),
            # (0.4468, 1.237),
            # (0.7865, 1.394),
            # (1.3840, 1.672),
            # (1.9260, 1.884),
            # (2.8530, 2.079),
            # (4.0260, 2.202),
            # (6.3410, 2.313),
            # (12.540, 2.411),
            # (21.230, 2.452),
            # (33.390, 2.474),
            # (82.000, 2.497),
            # ])
 
        # site = Site(se)

        # cal_fas = source_model_fas * gs_model(distance) * att_model(distance, frequencies) * site(frequencies)

        # unit = 'm'
        # self.assertListEqual(
            # [round_to_significant(float(x), 2) for x in tgt_fas.get_amplitudes(unit=unit)],
            # [round_to_significant(float(x), 2) for x in cal_fas.get_amplitudes(unit=unit)])
