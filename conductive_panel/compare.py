"""
Compare the reflection and transmission coefficients of a slightly conductive
panel computed with:
    1. Transfer Matrix Method (analytical)
    2. FDTD simulation (numerical)

The FDTD simulation uses NORMALIZED units where c=1, eps_0=1, mu_0=1.
To compare with the transfer matrix (SI units), we need a frequency mapping.

In normalized units (c=1, eps_0=1, mu_0=1, eta_0=1):
    - The transfer matrix simplifies with mu_0=1, eps_0=1, eta_0=1.

We perform the comparison entirely in normalized units for consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fdtd1d import FDTD1D, gaussian
from transfer_matrix import (
    panel_transfer_matrix,
    RT_from_transfer_matrix,
    stack_transfer_matrix,
)
from fdtd_panel import run_fdtd_panel, run_fdtd_reference, compute_RT_fdtd


def transfer_matrix_normalized(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    """
    Transfer matrix in normalized units (c=1, mu_0=1, eps_0=1, eta_0=1).

    gamma = j*omega*sqrt(mu_r * eps_c)  with eps_c = eps_r - j*sigma/omega
    eta = sqrt(mu_r / eps_c)
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq

    eps_c = eps_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu_r * eps_c)
    eta = np.sqrt(mu_r / eps_c)

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch

    return Phi


def transfer_matrix_stack_normalized(freq, layers):
    """Transfer matrix for a stack of panels in normalized units."""
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0

    for layer in layers:
        Phi_i = transfer_matrix_normalized(
            freq,
            d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0] * Phi_i[:, 0, 0] + Phi_total[:, 0, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0] * Phi_i[:, 0, 1] + Phi_total[:, 0, 1] * Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0] * Phi_i[:, 0, 0] + Phi_total[:, 1, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0] * Phi_i[:, 0, 1] + Phi_total[:, 1, 1] * Phi_i[:, 1, 1]
        Phi_total = Phi_new

    return Phi_total


def compare_single_panel(
    d=0.1,
    eps_r=4.0,
    sigma=0.5,
    N=4001,
    L=4.0,
    pulse_sigma=0.06,
    save_fig=None,
):
    """
    Run FDTD and analytical comparison for a single conductive panel.
    """
    print(f"Panel: d={d}, eps_r={eps_r}, sigma={sigma}")
    print("Running FDTD with panel...")
    panel_res = run_fdtd_panel(
        N=N, L=L, panel_thickness=d, eps_r=eps_r, sigma=sigma,
        pulse_sigma=pulse_sigma,
    )
    print("Running FDTD reference (free space)...")
    ref_res = run_fdtd_reference(
        N=N, L=L, panel_thickness=d, pulse_sigma=pulse_sigma,
    )

    print("Computing FFT-based R and T...")
    freq_fdtd, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    # Analytical (normalized units, eta_0 = 1)
    f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
    Phi = transfer_matrix_normalized(f_anal, d, eps_r, sigma)
    R_anal, T_anal = RT_from_transfer_matrix(Phi, eta0=1.0)

    # Determine valid frequency range (where pulse has spectral content)
    # Gaussian pulse in freq domain has bandwidth ~ 1/(2*pi*sigma_pulse)
    f_bandwidth = 1.0 / (2.0 * np.pi * pulse_sigma)
    f_max_plot = min(3.0 * f_bandwidth, freq_fdtd.max())

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Conductive Panel: d={d}, $\\varepsilon_r$={eps_r}, $\\sigma$={sigma}',
        fontsize=14,
    )

    # |R|
    ax = axes[0, 0]
    mask = (freq_fdtd > 0.01) & (freq_fdtd < f_max_plot)
    ax.plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    mask_a = f_anal < f_max_plot
    ax.plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical')
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('|R|')
    ax.set_title('Reflection coefficient magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # |T|
    ax = axes[0, 1]
    mask = (freq_fdtd > 0.01) & (freq_fdtd < f_max_plot)
    ax.plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    mask_a = f_anal < f_max_plot
    ax.plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical')
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('|T|')
    ax.set_title('Transmission coefficient magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # |R|^2 + |T|^2 (energy conservation check)
    ax = axes[1, 0]
    # For lossy media, |R|^2 + |T|^2 < 1 (energy absorbed)
    ax.plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
            'b-', alpha=0.7, label='FDTD')
    ax.plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
            'r--', lw=2, label='Analytical')
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('$|R|^2 + |T|^2$')
    ax.set_title('Energy conservation (< 1 for lossy media)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Time-domain signals
    ax = axes[1, 1]
    ax.plot(panel_res['t_array'], panel_res['E_left'], 'b-', alpha=0.7, label='E left (inc+ref)')
    ax.plot(panel_res['t_array'], panel_res['E_right'], 'g-', alpha=0.7, label='E right (trans)')
    ax.plot(ref_res['t_array'], ref_res['E_left'], 'r--', alpha=0.5, label='E left (ref, no panel)')
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('E field')
    ax.set_title('Time-domain signals at observation points')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_fig}")
    plt.show()

    return freq_fdtd, R_fdtd, T_fdtd, f_anal, R_anal, T_anal


def compare_multilayer(
    layers=None,
    N=4001,
    L=6.0,
    pulse_sigma=0.04,
    save_fig=None,
):
    """
    Run FDTD and analytical comparison for a multi-layer panel.
    """
    if layers is None:
        layers = [
            {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
            {'d': 0.08, 'eps_r': 6.0, 'sigma': 1.0},
            {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
        ]

    total_d = sum(l['d'] for l in layers)
    panel_center = L / 2.0

    print(f"Multi-layer panel: {len(layers)} layers, total d={total_d:.3f}")

    # --- FDTD: build panel from layers ---
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    xH = (x[1:] + x[:-1]) / 2.0
    pulse_x0 = 0.8

    panel_left = panel_center - total_d / 2
    obs_left_x = panel_left - 0.4
    obs_right_x = panel_center + total_d / 2 + 0.4
    obs_left_idx = np.argmin(np.abs(x - obs_left_x))
    obs_right_idx = np.argmin(np.abs(x - obs_right_x))

    from fdtd1d import gaussian as gauss

    initial_e = gauss(x, pulse_x0, pulse_sigma)
    initial_h = gauss(xH, pulse_x0, pulse_sigma)

    # Panel simulation
    fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    # Set material properties layer by layer
    eps_r_profile = np.ones(N)
    sig_profile = np.zeros(N)
    z_start = panel_left
    for layer in layers:
        z_end = z_start + layer['d']
        mask = (x >= z_start) & (x <= z_end)
        eps_r_profile[mask] = layer.get('eps_r', 1.0)
        sig_profile[mask] = layer.get('sigma', 0.0)
        z_start = z_end

    fdtd.eps_r = eps_r_profile
    fdtd.sig = sig_profile

    t_final = 2.5 * L
    n_steps = round(t_final / fdtd.dt)
    dt = fdtd.dt

    t_array = np.zeros(n_steps)
    E_left_panel = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)

    for i in range(n_steps):
        fdtd._step()
        t_array[i] = fdtd.t
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # Reference simulation (free space)
    fdtd_ref = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = -gauss(xH, pulse_x0, pulse_sigma)

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)

    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # Compute R, T from FFT
    E_reflected = E_left_panel - E_left_ref
    E_transmitted = E_right_panel
    E_incident = E_right_ref

    E_ref_fft = np.fft.rfft(E_reflected)
    E_trans_fft = np.fft.rfft(E_transmitted)
    E_inc_fft = np.fft.rfft(E_incident)
    freq_fdtd = np.fft.rfftfreq(n_steps, d=dt)

    fft_mask = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))
    R_fdtd = np.zeros_like(freq_fdtd, dtype=complex)
    T_fdtd = np.zeros_like(freq_fdtd, dtype=complex)
    R_fdtd[fft_mask] = E_ref_fft[fft_mask] / E_inc_fft[fft_mask]
    T_fdtd[fft_mask] = E_trans_fft[fft_mask] / E_inc_fft[fft_mask]

    # Analytical
    f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
    Phi_stack = transfer_matrix_stack_normalized(f_anal, layers)
    R_anal, T_anal = RT_from_transfer_matrix(Phi_stack, eta0=1.0)

    f_bandwidth = 1.0 / (2.0 * np.pi * pulse_sigma)
    f_max_plot = min(3.0 * f_bandwidth, freq_fdtd.max())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_str = ', '.join([f"(d={l['d']}, er={l.get('eps_r',1)}, s={l.get('sigma',0)})" for l in layers])
    fig.suptitle(f'Multi-layer panel: {layer_str}', fontsize=11)

    mask = (freq_fdtd > 0.01) & (freq_fdtd < f_max_plot)
    mask_a = f_anal < f_max_plot

    axes[0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('|R|')
    axes[0].set_title('|R(f)|')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.7, label='FDTD')
    axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('|T|')
    axes[1].set_title('|T(f)|')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
                 'b-', alpha=0.7, label='FDTD')
    axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
                 'r--', lw=2, label='Analytical')
    axes[2].set_xlabel('Frequency')
    axes[2].set_ylabel('$|R|^2 + |T|^2$')
    axes[2].set_title('Energy check')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_fig}")
    plt.show()


if __name__ == '__main__':
    # Case 1: Slightly conductive panel (low sigma)
    print("=" * 60)
    print("Case 1: Slightly conductive panel")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=4.0, sigma=0.5)

    # Case 2: More conductive panel
    print("\n" + "=" * 60)
    print("Case 2: More conductive panel")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=1.0, sigma=2.0)

    # Case 3: Pure dielectric (sigma=0, verification)
    print("\n" + "=" * 60)
    print("Case 3: Pure dielectric (no loss)")
    print("=" * 60)
    compare_single_panel(d=0.1, eps_r=4.0, sigma=0.0)

    # Case 4: Multi-layer panel (bonus)
    print("\n" + "=" * 60)
    print("Case 4: Multi-layer panel (bonus)")
    print("=" * 60)
    compare_multilayer()
