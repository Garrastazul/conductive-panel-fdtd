"""
Tests for the conductive panel reflection/transmission project.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transfer_matrix import (
    panel_transfer_matrix,
    RT_from_transfer_matrix,
    reflection_transmission,
    reflection_transmission_stack,
    eta_0,
)
from compare import transfer_matrix_normalized
from fdtd_panel import run_fdtd_panel, run_fdtd_reference, compute_RT_fdtd


# ---------- Transfer Matrix Tests ----------

def test_lossless_dielectric_energy_conservation():
    """For a lossless panel, |R|^2 + |T|^2 = 1."""
    freq = np.linspace(1e6, 1e9, 500)
    R, T = reflection_transmission(freq, d=1e-3, eps_r=4.0, sigma=0.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.allclose(energy, 1.0, atol=1e-10)


def test_lossless_half_wave_slab():
    """A half-wave slab (n*d = lambda/2) should be transparent (R=0)."""
    from transfer_matrix import c_0
    eps_r = 4.0
    n = np.sqrt(eps_r)
    f0 = 1e9
    lam = c_0 / (f0 * n)
    d = lam / 2  # half-wave thickness

    R, T = reflection_transmission(np.array([f0]), d=d, eps_r=eps_r, sigma=0.0)
    assert np.abs(R[0]) < 1e-10


def test_lossy_panel_absorbs_energy():
    """For a lossy panel, |R|^2 + |T|^2 < 1."""
    freq = np.linspace(1e6, 1e9, 500)
    R, T = reflection_transmission(freq, d=1e-3, eps_r=4.0, sigma=100.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.all(energy < 1.0)
    assert np.all(energy > 0.0)


def test_vacuum_panel_no_reflection():
    """A panel with eps_r=1, sigma=0 (vacuum) should have R=0, T=1."""
    freq = np.linspace(1e6, 1e9, 100)
    R, T = reflection_transmission(freq, d=1e-3, eps_r=1.0, sigma=0.0)
    assert np.allclose(np.abs(R), 0.0, atol=1e-10)
    assert np.allclose(np.abs(T), 1.0, atol=1e-10)


def test_stack_single_layer_equals_panel():
    """A stack with one layer should equal a single panel."""
    freq = np.linspace(1e6, 1e9, 200)
    d, eps_r, sigma = 1e-3, 4.0, 50.0

    R1, T1 = reflection_transmission(freq, d=d, eps_r=eps_r, sigma=sigma)
    R2, T2 = reflection_transmission_stack(freq, [{'d': d, 'eps_r': eps_r, 'sigma': sigma}])

    assert np.allclose(R1, R2, atol=1e-12)
    assert np.allclose(T1, T2, atol=1e-12)


# ---------- Normalized Transfer Matrix Tests ----------

def test_normalized_vacuum_no_reflection():
    """Normalized: vacuum panel should be transparent."""
    freq = np.linspace(0.1, 10.0, 100)
    Phi = transfer_matrix_normalized(freq, d=0.1, eps_r=1.0, sigma=0.0)
    R, T = RT_from_transfer_matrix(Phi, eta0=1.0)
    assert np.allclose(np.abs(R), 0.0, atol=1e-10)
    assert np.allclose(np.abs(T), 1.0, atol=1e-10)


def test_normalized_lossless_energy_conservation():
    """Normalized: lossless panel should conserve energy."""
    freq = np.linspace(0.1, 10.0, 500)
    Phi = transfer_matrix_normalized(freq, d=0.1, eps_r=4.0, sigma=0.0)
    R, T = RT_from_transfer_matrix(Phi, eta0=1.0)
    energy = np.abs(R)**2 + np.abs(T)**2
    assert np.allclose(energy, 1.0, atol=1e-10)


# ---------- FDTD Tests ----------

def test_fdtd_free_space_no_reflection():
    """FDTD with no panel should give R~0, T~1."""
    N = 2001
    L = 4.0
    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=0.1, eps_r=1.0, sigma=0.0,
        pulse_sigma=0.06, t_final=2.0 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=0.1, pulse_sigma=0.06, t_final=2.0 * L,
    )
    freq, R, T = compute_RT_fdtd(panel_res, ref_res)

    # In the valid bandwidth, R should be ~0
    f_bw = 1.0 / (2.0 * np.pi * 0.06)
    mask = (freq > 0.1) & (freq < 2.0 * f_bw)
    assert np.max(np.abs(R[mask])) < 0.05


def test_fdtd_vs_analytical_dielectric():
    """FDTD R,T should match analytical for a lossless dielectric panel."""
    N = 4001
    L = 4.0
    d = 0.2
    eps_r = 4.0
    sigma = 0.0
    pulse_sigma = 0.06

    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=d, pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    freq, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    # Analytical
    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    mask = (freq > 0.1) & (freq < 1.5 * f_bw)
    f_valid = freq[mask]

    Phi = transfer_matrix_normalized(f_valid, d, eps_r, sigma)
    R_anal, T_anal = RT_from_transfer_matrix(Phi, eta0=1.0)

    # Check correlation in the valid band
    corr_R = np.corrcoef(np.abs(R_fdtd[mask]), np.abs(R_anal))[0, 1]
    corr_T = np.corrcoef(np.abs(T_fdtd[mask]), np.abs(T_anal))[0, 1]
    assert corr_R > 0.95, f"R correlation too low: {corr_R}"
    assert corr_T > 0.95, f"T correlation too low: {corr_T}"


def test_fdtd_vs_analytical_conductive():
    """FDTD R,T should match analytical for a slightly conductive panel."""
    N = 4001
    L = 4.0
    d = 0.2
    eps_r = 4.0
    sigma = 0.5
    pulse_sigma = 0.06

    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=d, pulse_sigma=pulse_sigma, t_final=2.5 * L,
    )
    freq, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    mask = (freq > 0.1) & (freq < 1.5 * f_bw)
    f_valid = freq[mask]

    Phi = transfer_matrix_normalized(f_valid, d, eps_r, sigma)
    R_anal, T_anal = RT_from_transfer_matrix(Phi, eta0=1.0)

    corr_R = np.corrcoef(np.abs(R_fdtd[mask]), np.abs(R_anal))[0, 1]
    corr_T = np.corrcoef(np.abs(T_fdtd[mask]), np.abs(T_anal))[0, 1]
    assert corr_R > 0.90, f"R correlation too low: {corr_R}"
    assert corr_T > 0.90, f"T correlation too low: {corr_T}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
