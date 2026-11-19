#!/usr/bin/env python3
"""Reproduce the systematic-error examples in Sec. V of the Zeno schedule paper.

Run with Python >=3.9 and NumPy:
    python paper_analysis/scripts/error_budget_checks.py
    python paper_analysis/scripts/error_budget_checks.py --json results.json

Model and conventions
---------------------
* Basis order is |Q,S>; the initial state is |+>_Q |0>_S.
* Omega*T = 3 and Omega*delta = pi/2. There are N recordings.
* Q=1 selects an offset Rx before the train; Q=0 selects it after.
* Each step is Rx(Omega*T/N), then exp(-i*phi*Z_Q*Z_S/2), then
  CNOT(S -> fresh memory), with phi=0.1 radians per recording step.
* The memories are unread. Tracing each fresh memory immediately after its
  CNOT is exactly a dephasing channel on S, including the Q-S correlations.
  This reduced simulation is valid for the observables here, not memory parity.
* Echo applies X_Q after floor(N/2) completed steps and again after step N,
  before the final controlled offset. Odd N uses the earlier central boundary.
* Q dephasing multiplies its off-diagonal blocks by eta before X readout.
* Visibility is the fringe amplitude 2*|Tr(rho[Q=0,Q=1])|. Conditional
  probabilities use the fixed +/- X basis, including the real fringe phase.

These are numerical illustrations for N=1,...,10, not a model of all device
errors or a proof of a bound for arbitrary N. The code was reconstructed from
these circuit conventions and checked independently against a full statevector
and the paper's ideal formulas; it is not a recovered historical script.
"""

import argparse
import json
from pathlib import Path

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
P0 = np.diag([1.0, 0.0])
P1 = np.diag([0.0, 1.0])
XQ = np.kron(X, I)
ZS = np.kron(I, Z)


def rx(angle):
    """The paper's drive rotation exp(-i*angle*X/2)."""
    return np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * X


def evolve(rho, gate):
    return gate @ rho @ gate.conj().T


def density_simulation(n, *, total_angle=3.0, offset_angle=np.pi / 2,
                       zz_phase=0.0, echo=False, coherence=1.0,
                       record=True):
    """Return Q-S density matrix after the circuit and before X readout."""
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not 0 <= coherence <= 1:
        raise ValueError('coherence must be in [0, 1]')
    if echo and n < 2:
        raise ValueError('the echo comparison requires at least two steps')
    psi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho = evolve(rho, np.kron(P0, I) + np.kron(P1, rx(offset_angle)))
    drive = np.kron(I, rx(total_angle / n))
    zz = np.diag(np.exp(-0.5j * zz_phase * np.array([1, -1, -1, 1])))
    for step in range(1, n + 1):
        rho = evolve(rho, drive)
        rho = evolve(rho, zz)
        if record:
            rho = 0.5 * (rho + ZS @ rho @ ZS)
        if echo and step == n // 2:
            rho = evolve(rho, XQ)
    if echo:
        rho = evolve(rho, XQ)
    rho = evolve(rho, np.kron(P0, rx(offset_angle)) + np.kron(P1, I))
    rho[:2, 2:] *= coherence
    rho[2:, :2] *= coherence
    return rho


def statevector_reference(n, *, zz_phase=0.0, echo=False):
    """Independent Q,S,M1,...,MN simulation with every memory retained."""
    psi = np.zeros((2, 2, 1), dtype=complex)
    psi[:, 0, 0] = 1 / np.sqrt(2)
    psi[1] = rx(np.pi / 2) @ psi[1]
    phases = np.exp(-0.5j * zz_phase * np.array([[1, -1], [-1, 1]]))
    for step in range(1, n + 1):
        psi = np.einsum('ij,qjm->qim', rx(3.0 / n), psi)
        psi *= phases[:, :, None]
        recorded = np.zeros((2, 2, 2 * psi.shape[2]), dtype=complex)
        recorded[:, 0, ::2] = psi[:, 0, :]
        recorded[:, 1, 1::2] = psi[:, 1, :]
        psi = recorded
        if echo and step == n // 2:
            psi = psi[::-1].copy()
    if echo:
        psi = psi[::-1].copy()
    psi[0] = rx(np.pi / 2) @ psi[0]
    amplitudes = psi.reshape(4, -1)
    return amplitudes @ amplitudes.conj().T


def observables(rho):
    """Fringe amplitude and fixed-basis conditional final-zero probabilities."""
    a, b, cross = rho[:2, :2], rho[2:, 2:], rho[:2, 2:]
    result = {
        'visibility': float(2 * abs(np.trace(cross))),
        'fringe_real': float(2 * np.trace(cross).real),
        'fringe_imag': float(2 * np.trace(cross).imag),
        'p0_unread': float((a[0, 0] + b[0, 0]).real),
    }
    for label, sign in [('plus', 1), ('minus', -1)]:
        selected = 0.5 * (a + b + sign * (cross + cross.conj().T))
        probability = float(np.trace(selected).real)
        result['p_' + label] = probability
        result['p0_' + label] = (
            float(selected[0, 0].real / probability) if probability > 1e-14 else None
        )
    return result


def validate():
    """Check channel equivalence, ideal formulas, normalization, and dephasing."""
    atol = 3e-13
    max_matrix_error = 0.0
    for n in range(1, 11):
        for phase, echo in [(0.0, False), (0.1, False), (0.0, True), (0.1, True)]:
            if echo and n == 1:
                continue
            rho = density_simulation(n, zz_phase=phase, echo=echo)
            full = statevector_reference(n, zz_phase=phase, echo=echo)
            error = float(np.max(np.abs(rho - full)))
            max_matrix_error = max(max_matrix_error, error)
            np.testing.assert_allclose(rho, full, atol=atol, rtol=0)
            np.testing.assert_allclose(np.trace(rho), 1, atol=atol, rtol=0)
            if np.min(np.linalg.eigvalsh(rho)) < -atol:
                raise AssertionError('density matrix is not positive')
            if phase == 0:
                values = observables(rho)
                expected_minus = 0.5 * (1 - np.cos(3.0 / n) ** n)
                np.testing.assert_allclose(values['visibility'], 0.5, atol=atol, rtol=0)
                np.testing.assert_allclose(values['p_minus'], 0.25, atol=atol, rtol=0)
                np.testing.assert_allclose(values['p0_minus'], expected_minus,
                                           atol=atol, rtol=0)
                if echo:
                    np.testing.assert_allclose(rho, density_simulation(n),
                                               atol=atol, rtol=0)
    ideal = observables(density_simulation(8))
    noisy = observables(density_simulation(8, coherence=0.9))
    # Dephasing is a mixture of the coherent state and its Q-dephased state.
    expected = (0.9 * ideal['p_minus'] * ideal['p0_minus']
                + 0.1 * ideal['p0_unread'] / 2) / (0.9 * ideal['p_minus'] + 0.1 / 2)
    np.testing.assert_allclose(noisy['p0_minus'], expected, atol=atol, rtol=0)
    baseline = observables(density_simulation(8, coherence=0.9, record=False))
    np.testing.assert_allclose(baseline['visibility'], 0.9, atol=atol, rtol=0)
    return {'passed': True, 'n_range': [1, 10],
            'max_density_statevector_difference': max_matrix_error,
            'tolerance': atol}


def calculate():
    rows = []
    for n in range(1, 11):
        ideal_minus = 0.5 * (1 - np.cos(3.0 / n) ** n)
        zz = observables(density_simulation(n, zz_phase=0.1))
        row = {'n': n, 'ideal_p0_minus': float(ideal_minus),
               'zz_visibility': zz['visibility'], 'zz_p0_minus': zz['p0_minus'],
               'zz_p0_minus_error': zz['p0_minus'] - ideal_minus}
        if n >= 2:
            echo = observables(density_simulation(n, zz_phase=0.1, echo=True))
            row.update(echo_visibility=echo['visibility'], echo_p0_minus=echo['p0_minus'],
                       echo_p0_minus_error=echo['p0_minus'] - ideal_minus)
        rows.append(row)
    ideal = observables(density_simulation(8))
    dephased = observables(density_simulation(8, coherence=0.9))
    summary = {
        'zz_visibility_n1': rows[0]['zz_visibility'],
        'zz_visibility_n10': rows[-1]['zz_visibility'],
        'zz_p0_minus_error_n10': rows[-1]['zz_p0_minus_error'],
        'echo_max_abs_p0_minus_error_n2_to_n10':
            max(abs(row['echo_p0_minus_error']) for row in rows[1:]),
        'echo_visibility_drift_n10': abs(rows[-1]['echo_visibility'] - 0.5),
        'dephasing_n8_ideal_p0_minus': ideal['p0_minus'],
        'dephasing_n8_eta_0_9_p0_minus': dephased['p0_minus'],
    }
    # Check the rounded numerical claims printed in Sec. V.
    for key, digits, expected in [
        ('zz_visibility_n1', 2, 0.50), ('zz_visibility_n10', 2, 0.38),
        ('zz_p0_minus_error_n10', 2, 0.17), ('echo_visibility_drift_n10', 3, 0.015),
        ('dephasing_n8_ideal_p0_minus', 3, 0.219),
        ('dephasing_n8_eta_0_9_p0_minus', 3, 0.260),
    ]:
        if round(summary[key], digits) != expected:
            raise AssertionError(f'{key} no longer reproduces the paper: {summary[key]}')
    if summary['echo_max_abs_p0_minus_error_n2_to_n10'] >= 5e-3:
        raise AssertionError('echo conditional-probability error exceeds the stated bound')
    return {'parameters': {'omega_t': 3.0, 'omega_delta': float(np.pi / 2),
                           'zz_phase_per_step': 0.1, 'zz_placement': 'after drive, before recording',
                           'echo_pulse_steps': 'floor(N/2), N', 'n_range': [1, 10]},
            'validation': validate(), 'rows': rows, 'paper_summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='also write the computed results to this path')
    args = parser.parse_args()
    result = calculate()
    print('N   ideal P0-    ZZ visibility   ZZ P0-       echo visibility   echo P0-')
    for row in result['rows']:
        echo = (f"{row['echo_visibility']:.9f}       {row['echo_p0_minus']:.9f}"
                if 'echo_visibility' in row else 'not applied')
        print(f"{row['n']:2d}  {row['ideal_p0_minus']:.9f}   {row['zz_visibility']:.9f}"
              f"     {row['zz_p0_minus']:.9f}   {echo}")
    print('\nSec. V checks:')
    for key, value in result['paper_summary'].items():
        print(f'  {key}: {value:.12f}')
    print('\nPASS: ideal formulas, full-statevector agreement, density-matrix validity,')
    print('      dephasing mixture, record-disabled baseline, and rounded paper values.')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Wrote {args.json}')


if __name__ == '__main__':
    main()
