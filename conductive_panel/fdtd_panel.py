"""
FDTD simulation of a pulse impinging on a conductive/dielectric panel.

The simulation domain is:
    [0, L] with Mur ABCs on both ends.

A narrow Gaussian pulse is launched as a right-traveling wave (E, H = -E)
from the left part of the domain.  A conductive/dielectric panel of
thickness d is placed in the middle of the domain.

Two observation points record the time history of E(t):
    - One to the LEFT of the panel  -> captures incident + reflected
    - One to the RIGHT of the panel -> captures transmitted

A reference simulation WITHOUT the panel gives the pure incident signal.
Then:
    R(f) = FFT(E_reflected) / FFT(E_incident)
    T(f) = FFT(E_transmitted) / FFT(E_incident)

This reuses the existing FDTD1D class from the repository.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fdtd1d import FDTD1D, gaussian


def run_fdtd_panel(
    N=4001,
    L=4.0,
    panel_center=2.0,
    panel_thickness=0.1,
    eps_r=1.0,
    sigma=0.0,
    mu_r=1.0,
    pulse_x0=0.8,
    pulse_sigma=0.06,
    t_final=None,
    obs_left_offset=0.4,
    obs_right_offset=0.4,
):
    """
    Run FDTD simulation of a pulse hitting a panel and return time-domain
    field histories at observation points.

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length (normalized units, c=1).
    panel_center : float
        Center position of the panel.
    panel_thickness : float
        Thickness of the panel.
    eps_r : float
        Relative permittivity of the panel.
    sigma : float
        Conductivity of the panel (in normalized units).
    mu_r : float
        Relative permeability of the panel.
    pulse_x0 : float
        Initial center of the Gaussian pulse.
    pulse_sigma : float
        Width of the Gaussian pulse.
    t_final : float or None
        Simulation end time. If None, auto-calculated.
    obs_left_offset : float
        Distance from panel left edge to left observation point.
    obs_right_offset : float
        Distance from panel right edge to right observation point.

    Returns
    -------
    dict with keys:
        't_array' : time samples
        'E_left'  : E-field at left observation point vs time
        'E_right' : E-field at right observation point vs time
        'x'       : spatial grid
        'dt'      : time step
        'obs_left_idx', 'obs_right_idx' : grid indices of obs points
    """
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    xH = (x[1:] + x[:-1]) / 2.0

    # Panel region
    panel_left = panel_center - panel_thickness / 2
    panel_right = panel_center + panel_thickness / 2

    # Observation points
    obs_left_x = panel_left - obs_left_offset
    obs_right_x = panel_right + obs_right_offset
    obs_left_idx = np.argmin(np.abs(x - obs_left_x))
    obs_right_idx = np.argmin(np.abs(x - obs_right_x))

    # Right-traveling Gaussian pulse: E = gaussian, H = +gaussian
    initial_e = gaussian(x, pulse_x0, pulse_sigma)
    initial_h = gaussian(xH, pulse_x0, pulse_sigma)

    # Create FDTD with Mur boundaries
    fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    # Set panel material properties
    fdtd.eps_r = np.where((x >= panel_left) & (x <= panel_right), eps_r, 1.0)
    fdtd.sig = np.where((x >= panel_left) & (x <= panel_right), sigma, 0.0)

    dt = fdtd.dt
    if t_final is None:
        t_final = 2.0 * L

    n_steps = round(t_final / dt)

    t_array = np.zeros(n_steps)
    E_left = np.zeros(n_steps)
    E_right = np.zeros(n_steps)

    for i in range(n_steps):
        fdtd._step()
        t_array[i] = fdtd.t
        E_left[i] = fdtd.e[obs_left_idx]
        E_right[i] = fdtd.e[obs_right_idx]

    return {
        't_array': t_array,
        'E_left': E_left,
        'E_right': E_right,
        'x': x,
        'dt': dt,
        'obs_left_idx': obs_left_idx,
        'obs_right_idx': obs_right_idx,
        'panel_left': panel_left,
        'panel_right': panel_right,
    }


def run_fdtd_reference(
    N=4001,
    L=4.0,
    pulse_x0=0.8,
    pulse_sigma=0.06,
    t_final=None,
    obs_left_x=None,
    obs_right_x=None,
    panel_center=2.0,
    panel_thickness=0.1,
    obs_left_offset=0.4,
    obs_right_offset=0.4,
):
    """
    Run a reference FDTD simulation WITHOUT a panel (free space)
    to obtain the incident pulse at the same observation points.
    """
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0

    panel_left = panel_center - panel_thickness / 2
    panel_right = panel_center + panel_thickness / 2

    if obs_left_x is None:
        obs_left_x = panel_left - obs_left_offset
    if obs_right_x is None:
        obs_right_x = panel_right + obs_right_offset

    obs_left_idx = np.argmin(np.abs(x - obs_left_x))
    obs_right_idx = np.argmin(np.abs(x - obs_right_x))

    initial_e = gaussian(x, pulse_x0, pulse_sigma)
    initial_h = gaussian(xH, pulse_x0, pulse_sigma)  # H = +E for right-traveling

    fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    dt = fdtd.dt
    if t_final is None:
        t_final = 2.0 * L

    n_steps = round(t_final / dt)

    t_array = np.zeros(n_steps)
    E_left = np.zeros(n_steps)
    E_right = np.zeros(n_steps)

    for i in range(n_steps):
        fdtd._step()
        t_array[i] = fdtd.t
        E_left[i] = fdtd.e[obs_left_idx]
        E_right[i] = fdtd.e[obs_right_idx]

    return {
        't_array': t_array,
        'E_left': E_left,
        'E_right': E_right,
        'dt': dt,
        'obs_left_idx': obs_left_idx,
        'obs_right_idx': obs_right_idx,
    }


def compute_RT_fdtd(panel_result, ref_result, freq_max=None):
    """
    Compute frequency-domain R(f) and T(f) from FDTD time-domain data.

    The reflected signal is: E_left(panel) - E_left(reference)
    The transmitted signal is: E_right(panel)
    The incident signal is: E_right(reference)  (same as incident passing through)

    Parameters
    ----------
    panel_result : dict from run_fdtd_panel
    ref_result : dict from run_fdtd_reference

    Returns
    -------
    freq : ndarray (positive frequencies in normalized units)
    R : complex ndarray
    T : complex ndarray
    """
    dt = panel_result['dt']
    n = len(panel_result['t_array'])

    # Reflected = total field at left obs - incident field at left obs
    E_reflected = panel_result['E_left'] - ref_result['E_left']
    # Transmitted = field at right obs (panel sim)
    E_transmitted = panel_result['E_right']
    # Incident = field at right obs (reference, no panel)
    E_incident = ref_result['E_right']

    # FFT
    E_ref_fft = np.fft.rfft(E_reflected)
    E_trans_fft = np.fft.rfft(E_transmitted)
    E_inc_fft = np.fft.rfft(E_incident)

    freq = np.fft.rfftfreq(n, d=dt)

    # Avoid division by zero at low frequencies
    mask = np.abs(E_inc_fft) > 1e-10 * np.max(np.abs(E_inc_fft))

    R = np.zeros_like(freq, dtype=complex)
    T = np.zeros_like(freq, dtype=complex)
    R[mask] = E_ref_fft[mask] / E_inc_fft[mask]
    T[mask] = E_trans_fft[mask] / E_inc_fft[mask]

    return freq, R, T
