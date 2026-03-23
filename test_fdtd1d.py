import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, C

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)


def test_fdtd_solves_basic_propagation():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 0.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()

    e_expected = 0.5 * gaussian(x, -t_final*C, sigma)      + 0.5 * gaussian(x, t_final*C, sigma)


    plt.plot(x, e_solved)
    plt.plot(x, e_expected)

    assert np.corrcoef(e_solved, e_expected)[0,1] > 0.99


def test_fdtd_PEC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PEC', 'PEC')
    
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd.load_initial_field(initial_e)
    
    fdtd = FDTD1D(x, boundaries)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = - initial_e
    h_expected = np.zeros_like(h_solved)
    
    assert np.allclose(e_solved, e_expected)
    assert np.allclose(h_solved, h_expected)


def test_fdtd_PMC_boundary_conditions():
    # Test
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PMC', 'PMC')
    
    x0 = 0.0
    sigma = 0.05
    initial_h = gaussian(x, x0, sigma)
    fdtd.load_initial_field(initial_h)
    
    fdtd = FDTD1D(x, boundaries)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = np.zeros_like(e_solved)
    h_expected = - initial_h

    assert np.allclose(e_solved, e_expected)
    assert np.allclose(h_solved, h_expected)



def test_fdtd_periodic_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('periodic', 'periodic')
    
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    initial_h = np.zeros_like(initial_e[:-1]) 

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = initial_e
    h_expected = initial_h

    assert np.corrcoef(e_solved, e_expected)[0, 1] > 0.99
    assert np.allclose(h_solved, h_expected, atol=1e-2)


def test_fdtd_TF_SF():
    xMin, xMax = -1.0, 1.0
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('periodic', 'periodic')

    sigma = 0.05

    # Define incident and scattered initial fields
    incident_e = gaussian(x, -0.3, sigma)
    scattered_e = gaussian(x, 0.3, sigma)
    total_e = incident_e + scattered_e

    t_final = 0.2

    # Run incident wave
    fdtd_incident = FDTD1D(x, boundaries)
    fdtd_incident.load_initial_field(incident_e)
    fdtd_incident.run_until(t_final)

    # Run scattered wave
    fdtd_scattered = FDTD1D(x, boundaries)
    fdtd_scattered.load_initial_field(scattered_e)
    fdtd_scattered.run_until(t_final)

    # Run total wave (incident + scattered)
    fdtd_total = FDTD1D(x, boundaries)
    fdtd_total.load_initial_field(total_e)
    fdtd_total.run_until(t_final)

    e_total = fdtd_total.get_e()
    h_total = fdtd_total.get_h()

    e_sum = fdtd_incident.get_e() + fdtd_scattered.get_e()
    h_sum = fdtd_incident.get_h() + fdtd_scattered.get_h()

    assert np.allclose(e_total, e_sum, atol=1e-8)
    assert np.allclose(h_total, h_sum, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
