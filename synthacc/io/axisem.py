"""
The 'io.axisem' module.

I/O for the AxiSEM program. See Nissen-Meyer et al. (2014).
"""


import numpy as np

from ..apy import is_string, is_pos_number
from ..ground.models import ContinuousModel, qk_qm_to_qp_qs


def read_ground_model(filespec, max_depth=None, validate=True):
    """
    filespec: raw string
    max_depth: pos number, exclude larger depths (in m) (default: None)
    """
    if validate is True:
        assert(is_string(filespec))
        if max_depth is not None:
            assert(is_pos_number(max_depth))

    with open(filespec, 'r') as f:
        lines = list(f)

    radius = float(lines[6].split()[0])
    depths, vps, vss, densities, qps, qss = [], [], [], [], [], []
    for line in lines[6:]:

        if line.startswith('#'):
            continue

        s = line.split()

        depth = radius - float(s[0])

        if max_depth is not None and depth > max_depth:
            break

        vp, vs = float(s[2]), float(s[3])
        density = s[1]
        qk, qm = float(s[4]), float(s[5])

        qp, qs = qk_qm_to_qp_qs(qk, qm, vp, vs)

        depths.append(depth)
        vps.append(vp)
        vss.append(vs)
        densities.append(density)
        qps.append(qp)
        qss.append(qs)

    name = 'ak135f'

    gm = ContinuousModel(depths, vps, vss, densities, qps, qss, name)

    return gm
