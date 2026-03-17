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

    e_expected = 0.5 * gaussian(x, -t_final*C, sigma) \
     + 0.5 * gaussian(x, t_final*C, sigma)
    
    assert np.allclose(e_solved, e_expected)

<<<<<<< HEAD
def gaussian(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def test_fdtd_solves_one_wave():
    x = np.linspace(-1, 1, 100)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd = FDTD1D(initial_e)
    fdtd.load_initial_field(initial_e)

    fdtd.run_until(t_final)

    e_solved = get_e()

    e_expected = 0.5 * (gaussian(x - c * t_final, x0, sigma) + gaussian(x + c * t_final, x0, sigma))

    assert np.allclose(e_solved, e_expected)
=======

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
>>>>>>> 95cfd69385fc542087fefe8949eebcb0ac79d656


if __name__ == "__main__":
    pytest.main([__file__])
