"""
The 'io.rspmatch09' module.
"""


import os
import shutil
from subprocess import Popen, PIPE

import numpy as np

from synthacc.recordings import Accelerogram
from synthacc.response import ResponseSpectrum


def read_acc(acc_filespec, remove_padding=False):
    """
    """
    with open(acc_filespec, 'r') as f:
        lines = f.read().splitlines()
    s1 = lines[1].split()
    time_delta = float(s1[1])
    amplitudes = []
    for line in lines[2:]:
        amplitudes.extend(line.split())
    amplitudes = np.array(amplitudes, dtype=float)
    if remove_padding is True:
        padding = int(s1[2])
        amplitudes = amplitudes[padding:]
    acc = Accelerogram(time_delta, amplitudes, unit='g')

    return acc


def read_rsp(rsp_filespec):
    """
    """
    with open(rsp_filespec, 'r') as f:
        lines = f.read().splitlines()
    freqs, resps = [], []
    for i, line in enumerate(lines):
        split = line.split()
        if len(split) > 1 and split[0] == 'Maximum' and split[1] == 'misfit':
            max_mis = float(split[3])
        if split == ['Matched', 'spectrum:']:
            break
    damping = float(lines[i+3].split()[0])
    j = i + 7
    for line in lines[j:]:
        split = line.split()
        freqs.append(split[0])
        resps.append(split[3])
    freqs = np.array(freqs, dtype=float)[::-1]
    resps = np.array(resps, dtype=float)[::-1]
    rs = ResponseSpectrum(1 / freqs, resps, 'g', damping)

    return rs, max_mis


def read_tgt(tgt_filespec):
    """
    """
    with open(tgt_filespec, 'r') as f:
        lines = f.read().splitlines()
    assert int(lines[1].split()[1]) == 1
    damping = float(lines[2].split()[0])
    freqs, resps = [], []
    for line in lines[3:]:
        f, _, _, r = line.split()[:4]
        freqs.append(f)
        resps.append(r)
    freqs = np.array(freqs, dtype=float)[::-1]
    resps = np.array(resps, dtype=float)[::-1]
    rs = ResponseSpectrum(1 / freqs, resps, 'g', damping)

    return rs


def write_acc(acc_filespec, accelerogram):
   """
   """
   with open(acc_filespec, 'w') as f:
       f.write('')
       f.write('\n')
       f.write('%i %s 0' % (len(accelerogram), accelerogram.time_delta))
       for amplitude in accelerogram.get_amplitudes('g'):
           f.write('\n')
           f.write('%.6E' % amplitude)
       f.close()


def write_tgt(tgt_filespec, response_spectrum, time_window):
    """
    """
    with open(tgt_filespec, 'w') as f:
        f.write('')
        f.write('\n')
        f.write('%i 1' % len(response_spectrum))
        f.write('\n')
        f.write('%s' % response_spectrum.damping)
        frequencies = 1 / response_spectrum.periods[::-1]
        responses = response_spectrum.get_responses(unit='g')[::-1]
        for frequency, response in zip(frequencies, responses):
            f.write('\n')
            f.write('{:>10.4f}{:>3d}{:>7d}{:>12.6f}'.format(
                frequency,
                int(time_window[0]),
                int(time_window[1]),
                response))


def write_inp(inp_filespec, max_iterations, tolerance, scaling, max_freq, min_freq, i_tgt_filespec, i_acc_filespec, o_acc_filespec, o_rsp_filespec, o_unm_filespec):
    """
    """
    with open(inp_filespec, 'w') as f:
        f.write('%i' % max_iterations)
        f.write('\n')
        f.write('%s' % tolerance)
        f.write('\n')
        f.write('1.0')
        f.write('\n')
        f.write('7')
        f.write('\n')
        f.write('1.25 0.25 1.0 4.0')
        f.write('\n')
        sp = '%f' % scaling[1]
        f.write('%i  %s' % (scaling[0], sp.rstrip('0')))
        f.write('\n')
        f.write('1') ## dt flag
        f.write('\n')
        f.write('1.0e-04')
        f.write('\n')
        f.write('30')
        f.write('\n')
        max_freq = '%f' % max_freq
        f.write('%s' % max_freq.rstrip('0'))
        f.write('\n')
        f.write('0.0 0.0 4')
        f.write('\n')
        f.write('0')
        f.write('\n')
        f.write('0  0.0')
        f.write('\n')
        min_freq = '%f' % min_freq
        f.write('%s %s' % (min_freq.rstrip('0'), max_freq.rstrip('0')))
        f.write('\n')
        f.write('0')
        f.write('\n')
        f.write('1.0')
        f.write('\n')
        f.write(i_tgt_filespec)
        f.write('\n')
        f.write(i_acc_filespec)
        f.write('\n')
        f.write(o_acc_filespec)
        f.write('\n')
        f.write(o_rsp_filespec)
        f.write('\n')
        f.write(o_unm_filespec)
        f.close()


def write_input(folder, seed_accelerogram, target_uhs, params):
    """
    """
    with open(os.path.join(folder, 'run.inp'), 'w') as f:
        f.write('%i' % params['passes'])
        for i in range(params['passes']):
            f.write('\n')
            rel_inp_filespec = 'i\\run%i.inp' % (i+1)
            f.write(rel_inp_filespec)
            o_acc_filespec = 'o\\run%i.acc' % (i+1)
            o_rsp_filespec = 'o\\run%i.rsp' % (i+1)
            o_unm_filespec = 'o\\run%i.unm' % (i+1)
            if i == 0:
                scaling = (2, params['scale_period'])
                i_acc_filespec = 'i\\seed_accelerogram.acc'
            else:
                scaling = (0, params['scale_period'])
                i_acc_filespec = 'o\\run%i.acc' % i
            i_tgt_filespec = 'i\\target_uhs.tgt'
            write_inp(
                os.path.join(folder, 'run%i.inp' % (i+1)),
                params['max_iterations'][i],
                params['tolerance'],
                scaling, params['max_freq'],
                params['min_freqs'][i],
                i_tgt_filespec,
                i_acc_filespec,
                o_acc_filespec,
                o_rsp_filespec,
                o_unm_filespec,
                )
    acc_filespec = os.path.join(folder, 'seed_accelerogram.acc')
    tgt_filespec = os.path.join(folder, 'target_uhs.tgt')
    write_acc(acc_filespec, seed_accelerogram)
    write_tgt(tgt_filespec, target_uhs, params['time_window'])


def run(folder, seed_accelerogram, target_uhs, params, remove_padding=True):
    """
    """
    i_dir = os.path.join(folder, 'i')
    o_dir = os.path.join(folder, 'o')
    if os.path.exists(i_dir):
        shutil.rmtree(i_dir)
    if os.path.exists(o_dir):
        shutil.rmtree(o_dir)
    os.mkdir(i_dir)
    os.mkdir(o_dir)
    write_input(os.path.join(folder, 'i'), seed_accelerogram, target_uhs, params)
    p = Popen('cmd', stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=folder)
    c = p.communicate(b'rspm09\ni\\run.inp\n')
    acc_filespec = os.path.join(o_dir, 'run%i.acc' % params['passes'])
    rsp_filespec = os.path.join(o_dir, 'run%i.rsp' % params['passes'])
    mod_acc = read_acc(acc_filespec, remove_padding=remove_padding)
    mod_rsp, max_mis = read_rsp(rsp_filespec)

    return mod_acc, mod_rsp, max_mis
