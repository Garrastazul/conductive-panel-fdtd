import numpy as np
import matplotlib.pyplot as plt

from fdtd1d import FDTD1D
from slightly_conductive_panel import SlightlyConductivePanel

C = 1.0


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def modulated_gaussian(x, x0, sigma, k0):
    envelope = gaussian(x, x0, sigma)
    phase = np.cos(k0 * (x - x0))
    return envelope * phase


def transfer_matrix_coefficients(eps_r, sigma, d, k0, eps0=1.0, mu0=1.0):
    omega = C * k0
    eps_complex = eps0 * (eps_r + 1j * sigma / (omega * eps0))
    n2 = np.sqrt(eps_complex / eps0)
    if np.imag(n2) < 0:
        n2 = -n2
    n1 = 1.0
    n3 = 1.0
    k2 = k0 * n2

    r01 = (n1 - n2) / (n1 + n2)
    r12 = (n2 - n3) / (n2 + n3)

    t01 = 2 * n1 / (n1 + n2)
    t12 = 2 * n2 / (n2 + n3)

    exp_2ikd = np.exp(2j * k2 * d)
    denom = 1 + r01 * r12 * exp_2ikd

    r = (r01 + r12 * exp_2ikd) / denom
    t = (t01 * t12 * np.exp(1j * k2 * d)) / denom

    R = np.abs(r) ** 2
    T = np.abs(t) ** 2
    return r, t, R, T


def find_nearest_index(array, value):
    return np.argmin(np.abs(array - value))


def run_fdtd_probe(x, x0, sigma, k0, panel=None, boundaries=('mur', 'mur'),
                   x_ref=-1.9, x_tr=3.8, t_final=8.0, dt_per_frame=0.01):
    x = np.asarray(x)
    xH = (x[1:] + x[:-1]) / 2.0

    e0 = modulated_gaussian(x, x0, sigma, k0)
    h0 = modulated_gaussian(xH, x0, sigma, k0)

    fdtd = FDTD1D(x, boundaries=boundaries)
    if panel is not None:
        panel.apply_to(fdtd)
    fdtd.load_initial_field(e0)
    fdtd.h = h0.copy()

    n_frames = int(np.ceil(t_final / dt_per_frame)) + 1
    times = np.linspace(0.0, t_final, n_frames)
    left_probe_idx = find_nearest_index(x, x_ref)
    right_probe_idx = find_nearest_index(x, x_tr)
    left_probe_h_idx = find_nearest_index(xH, x_ref)
    right_probe_h_idx = find_nearest_index(xH, x_tr)

    probe_left_e = np.zeros(n_frames)
    probe_right_e = np.zeros(n_frames)
    probe_left_h = np.zeros(n_frames)
    probe_right_h = np.zeros(n_frames)

    probe_left_e[0] = fdtd.e[left_probe_idx]
    probe_right_e[0] = fdtd.e[right_probe_idx]
    probe_left_h[0] = fdtd.h[left_probe_h_idx]
    probe_right_h[0] = fdtd.h[right_probe_h_idx]

    for i in range(1, n_frames):
        fdtd.run_until(times[i])
        probe_left_e[i] = fdtd.e[left_probe_idx]
        probe_right_e[i] = fdtd.e[right_probe_idx]
        probe_left_h[i] = fdtd.h[left_probe_h_idx]
        probe_right_h[i] = fdtd.h[right_probe_h_idx]

    return times, probe_left_e, probe_left_h, probe_right_e, probe_right_h


def energy(e_signal, h_signal, times):
    return np.trapezoid(0.5 * (e_signal ** 2 + h_signal ** 2), times)


def energy_window(e_signal, h_signal, times, t_center, half_width):
    mask = np.abs(times - t_center) <= half_width
    return np.trapezoid(0.5 * (e_signal[mask] ** 2 + h_signal[mask] ** 2), times[mask])


def compare_panel_coefficients():
    x = np.linspace(-2.0, 4.0, 601)
    x0 = -1.2
    sigma = 0.08
    k0 = 12.0

    panel = SlightlyConductivePanel(
        x,
        xL_panel=0.0,
        d_panel=0.4,
        eps_panel=2.5,
        sigma_panel=0.08,
    )

    x_ref = -1.7
    x_tr = 3.3
    t_final = 7.0
    dt_per_frame = 0.01

    times_ref, _, _, probe_right_ref, probe_right_ref_h = run_fdtd_probe(
        x, x0, sigma, k0, panel=None,
        x_ref=x_ref, x_tr=x_tr, t_final=t_final, dt_per_frame=dt_per_frame
    )
    t_inc = (x_tr - x0) / C
    E_inc = energy_window(probe_right_ref, probe_right_ref_h, times_ref, t_inc, half_width=0.4)

    times_panel, probe_left_panel, probe_left_panel_h, probe_right_panel, probe_right_panel_h = run_fdtd_probe(
        x, x0, sigma, k0, panel=panel,
        x_ref=x_ref, x_tr=x_tr, t_final=t_final, dt_per_frame=dt_per_frame
    )
    t_refl = (0 - x0) / C + (0 - x_ref) / C
    n2 = np.sqrt(panel.eps_panel + 1j * panel.sigma_panel / k0)
    if np.imag(n2) < 0:
        n2 = -n2
    t_tr = (x_tr - x0) / C + (np.real(n2) - 1.0) * panel.d_panel / C

    E_refl = energy_window(probe_left_panel, probe_left_panel_h, times_panel, t_refl, half_width=0.4)
    E_tr = energy_window(probe_right_panel, probe_right_panel_h, times_panel, t_tr, half_width=0.4)

    R_fdtd = E_refl / E_inc
    T_fdtd = E_tr / E_inc

    r, t, R_analytic, T_analytic = transfer_matrix_coefficients(
        eps_r=panel.eps_panel,
        sigma=panel.sigma_panel,
        d=panel.d_panel,
        k0=k0
    )

    return {
        'R_fdtd': R_fdtd,
        'T_fdtd': T_fdtd,
        'R_analytic': R_analytic,
        'T_analytic': T_analytic,
        'r_analytic': r,
        't_analytic': t,
        'times': times_panel,
        'probe_left_panel': probe_left_panel,
        'probe_left_panel_h': probe_left_panel_h,
        'probe_right_panel': probe_right_panel,
        'probe_right_panel_h': probe_right_panel_h,
        'probe_right_ref': probe_right_ref,
        'panel': panel,
    }


def plot_results(results):
    panel = results['panel']
    times = results['times']

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, results['probe_left_panel'], label='Reflected E at probe (panel)')
    axes[0].plot(times, results['probe_left_panel_h'], label='Reflected H at probe (panel)', alpha=0.7)
    axes[0].set_ylabel('Field amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, results['probe_right_panel'], label='Transmitted E at probe (panel)')
    axes[1].plot(times, results['probe_right_panel_h'], label='Transmitted H at probe (panel)', alpha=0.7)
    axes[1].plot(times, results['probe_right_ref'], '--', label='Incident E at probe (vacuum ref)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Field amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('FDTD probe time signals for low-conductivity panel')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    panel.plot(ax=ax2)
    ax2.set_title('Panel material profile')
    plt.show()


def main():
    results = compare_panel_coefficients()

    print('FDTD reflection coefficient R_fdtd =', results['R_fdtd'])
    print('FDTD transmission coefficient T_fdtd =', results['T_fdtd'])
    print('Analytic reflection coefficient R_analytic =', results['R_analytic'])
    print('Analytic transmission coefficient T_analytic =', results['T_analytic'])
    print('Analytic complex r =', results['r_analytic'])
    print('Analytic complex t =', results['t_analytic'])
    print('R + T (analytic) =', results['R_analytic'] + results['T_analytic'])

    plot_results(results)

if __name__ == '__main__':
    main()
