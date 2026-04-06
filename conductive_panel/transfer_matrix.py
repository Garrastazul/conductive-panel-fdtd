"""
Analytical reflection and transmission coefficients using the Transfer Matrix Method.

For a homogeneous panel of thickness d with complex permittivity
    eps_c = eps_r * eps_0 - j * sigma / omega
the transfer (ABCD) matrix is (notes eq. 1.19):
    Phi = [[cosh(gamma*d),  eta*sinh(gamma*d)],
           [sinh(gamma*d)/eta,  cosh(gamma*d)]]
where
    gamma = j * omega * sqrt(mu * eps_c)   (complex propagation constant)
    eta   = sqrt(mu / eps_c)               (intrinsic impedance)

T and R are then obtained from (notes eqs. 1.24, 1.25):
    T = 2*eta0 / (Phi11*eta0 + Phi12 + Phi21*eta0^2 + Phi22*eta0)
    R = (Phi11*eta0 + Phi12 - Phi21*eta0^2 - Phi22*eta0) /
        (Phi11*eta0 + Phi12 + Phi21*eta0^2 + Phi22*eta0)

References:
    [1] Class notes, section 1.4.3
    [2] Orfanidis, "Electromagnetic Waves and Antennas", Chapter 4-5
"""

import numpy as np

# Free-space constants
mu_0 = 4.0 * np.pi * 1e-7        # H/m
eps_0 = 8.854187817e-12           # F/m
c_0 = 1.0 / np.sqrt(mu_0 * eps_0)  # m/s
eta_0 = np.sqrt(mu_0 / eps_0)      # ~377 Ohm


def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    """
    Compute the 2x2 ABCD transfer matrix for a single homogeneous panel.

    Parameters
    ----------
    freq : array_like
        Frequencies in Hz.
    d : float
        Panel thickness in meters.
    eps_r : float
        Relative permittivity.
    sigma : float
        Conductivity in S/m.
    mu_r : float
        Relative permeability.

    Returns
    -------
    Phi : ndarray, shape (len(freq), 2, 2)
        Transfer matrix at each frequency.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq

    mu = mu_r * mu_0
    eps_c = eps_r * eps_0 - 1j * sigma / omega  # complex permittivity

    gamma = 1j * omega * np.sqrt(mu * eps_c)    # propagation constant
    eta = np.sqrt(mu / eps_c)                    # intrinsic impedance

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch

    return Phi


def stack_transfer_matrix(freq, layers):
    """
    Compute the transfer matrix for a stack of panels.

    Parameters
    ----------
    freq : array_like
        Frequencies in Hz.
    layers : list of dict
        Each dict has keys: 'd', 'eps_r', 'sigma', 'mu_r' (optional).

    Returns
    -------
    Phi_total : ndarray, shape (len(freq), 2, 2)
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0

    for layer in layers:
        Phi_i = panel_transfer_matrix(
            freq,
            d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        # Matrix multiply at each frequency
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0] * Phi_i[:, 0, 0] + Phi_total[:, 0, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0] * Phi_i[:, 0, 1] + Phi_total[:, 0, 1] * Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0] * Phi_i[:, 0, 0] + Phi_total[:, 1, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0] * Phi_i[:, 0, 1] + Phi_total[:, 1, 1] * Phi_i[:, 1, 1]
        Phi_total = Phi_new

    return Phi_total


def RT_from_transfer_matrix(Phi, eta0=eta_0):
    """
    Compute reflection (R) and transmission (T) coefficients from the ABCD matrix.

    Uses notes eqs. (1.24) and (1.25), assuming free space on both sides.

    Parameters
    ----------
    Phi : ndarray, shape (N, 2, 2)
    eta0 : float
        Impedance of surrounding medium.

    Returns
    -------
    R : ndarray, shape (N,)
    T : ndarray, shape (N,)
    """
    A = Phi[:, 0, 0]
    B = Phi[:, 0, 1]
    C = Phi[:, 1, 0]
    D = Phi[:, 1, 1]

    denom = A * eta0 + B + C * eta0**2 + D * eta0

    T = 2.0 * eta0 / denom
    R = (A * eta0 + B - C * eta0**2 - D * eta0) / denom

    return R, T


def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    """
    Compute R(f) and T(f) for a single homogeneous conductive panel in free space.

    Parameters
    ----------
    freq : array_like
        Frequencies in Hz.
    d : float
        Panel thickness (m).
    eps_r : float
        Relative permittivity.
    sigma : float
        Conductivity (S/m).
    mu_r : float
        Relative permeability.

    Returns
    -------
    R, T : complex ndarrays
    """
    Phi = panel_transfer_matrix(freq, d, eps_r, sigma, mu_r)
    return RT_from_transfer_matrix(Phi)


def reflection_transmission_stack(freq, layers):
    """
    Compute R(f) and T(f) for a stack of panels in free space.
    """
    Phi = stack_transfer_matrix(freq, layers)
    return RT_from_transfer_matrix(Phi)
