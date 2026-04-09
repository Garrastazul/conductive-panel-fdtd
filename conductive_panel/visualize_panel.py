# %% [markdown]
# # Conductive Panel – FDTD Visualization
#
# Gaussian pulse propagating through a conductive panel (1D FDTD).
# Run each cell with **Shift+Enter** in VSCode/Jupyter.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '..'))
from fdtd1d import FDTD1D, gaussian, panel_transfer_matrix, stack_transfer_matrix, RT_from_transfer_matrix

# %% [markdown]
# ## 1. Field Animation – Pulse hitting a conductive panel

# %% Parameters
N = 2001
L = 4.0
panel_center = 2.0
panel_thickness = 0.3
eps_r = 4.0
sigma_val = 0.5
pulse_x0 = 1.2
pulse_sigma = 0.08

x = np.linspace(0, L, N)
xH = (x[1:] + x[:-1]) / 2.0

panel_left = panel_center - panel_thickness / 2
panel_right = panel_center + panel_thickness / 2

# %% Run simulation and capture frames
fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
fdtd.load_initial_field(gaussian(x, pulse_x0, pulse_sigma))
fdtd.h = gaussian(xH, pulse_x0, pulse_sigma).copy()
fdtd.eps_r = np.where((x >= panel_left) & (x <= panel_right), eps_r, 1.0)
fdtd.sig = np.where((x >= panel_left) & (x <= panel_right), sigma_val, 0.0)

n_frames = 250
dt_per_frame = 0.015

frames_e, frames_h, times = [fdtd.get_e()], [fdtd.get_h()], [fdtd.t]
for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames_e.append(fdtd.get_e())
    frames_h.append(fdtd.get_h())
    times.append(fdtd.t)

print(f"Captured {len(frames_e)} frames (t = {times[0]:.3f} ... {times[-1]:.3f})")

# %% Animate
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, L); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("Field amplitude")
ax.set_title(f"FDTD – Pulse through conductive panel "
             f"($\\varepsilon_r$={eps_r}, $\\sigma$={sigma_val}, d={panel_thickness})")
ax.grid(True, alpha=0.3)

ax.add_patch(Rectangle((panel_left, -1.1), panel_thickness, 2.2,
                        color='orange', alpha=0.2, label='Panel'))
ax.axvline(panel_left, color='orange', ls='--', lw=1)
ax.axvline(panel_right, color='orange', ls='--', lw=1)

(line_e,) = ax.plot([], [], lw=2, color='royalblue', label='E(x,t)')
(line_h,) = ax.plot([], [], lw=1.5, color='darkorange', alpha=0.7, label='H(x,t)')
ax.legend(loc='upper right')
time_txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)

def init():
    line_e.set_data([], []); line_h.set_data([], []); time_txt.set_text("")
    return line_e, line_h, time_txt

def update(i):
    line_e.set_data(x, frames_e[i]); line_h.set_data(xH, frames_h[i])
    time_txt.set_text(f"t = {times[i]:.3f}")
    return line_e, line_h, time_txt

anim = FuncAnimation(fig, update, frames=len(frames_e), init_func=init,
                     interval=40, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())

# %% [markdown]
# ## 2. R(f) and T(f) – FDTD vs Analytical (single panel)

# %% Helper: run FDTD and extract R,T via FFT
def run_panel_fdtd(N, L, d, eps_r, sigma, pulse_sigma, pulse_x0=0.8):
    """Run FDTD with and without panel, return freq, R, T."""
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0
    pc = L / 2
    pl, pr = pc - d/2, pc + d/2
    obs_l = np.argmin(np.abs(x - (pl - 0.4)))
    obs_r = np.argmin(np.abs(x - (pr + 0.4)))

    e0 = gaussian(x, pulse_x0, pulse_sigma)
    h0 = gaussian(xH, pulse_x0, pulse_sigma)
    t_final = 2.5 * L

    # With panel
    fdtd_p = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_p.load_initial_field(e0); fdtd_p.h = h0.copy()
    fdtd_p.eps_r = np.where((x >= pl) & (x <= pr), eps_r, 1.0)
    fdtd_p.sig = np.where((x >= pl) & (x <= pr), sigma, 0.0)

    n_steps = round(t_final / fdtd_p.dt)
    El_p, Er_p = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_p._step()
        El_p[i] = fdtd_p.e[obs_l]; Er_p[i] = fdtd_p.e[obs_r]

    # Free space reference
    fdtd_r = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd_r.load_initial_field(e0); fdtd_r.h = h0.copy()
    El_r, Er_r = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_r._step()
        El_r[i] = fdtd_r.e[obs_l]; Er_r[i] = fdtd_r.e[obs_r]

    # FFT
    dt = fdtd_p.dt
    Eref_fft = np.fft.rfft(El_p - El_r)
    Etrans_fft = np.fft.rfft(Er_p)
    Einc_fft = np.fft.rfft(Er_r)
    freq = np.fft.rfftfreq(n_steps, d=dt)

    valid = np.abs(Einc_fft) > 1e-10 * np.max(np.abs(Einc_fft))
    R = np.zeros_like(freq, dtype=complex)
    T = np.zeros_like(freq, dtype=complex)
    R[valid] = Eref_fft[valid] / Einc_fft[valid]
    T[valid] = Etrans_fft[valid] / Einc_fft[valid]
    return freq, R, T

# %% Compute R,T
print("Running FDTD simulations...")
freq_fdtd, R_fdtd, T_fdtd = run_panel_fdtd(4001, 4.0, panel_thickness, eps_r, sigma_val, 0.06)

f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
R_anal, T_anal = RT_from_transfer_matrix(panel_transfer_matrix(f_anal, panel_thickness, eps_r, sigma_val))

f_bw = 1.0 / (2.0 * np.pi * 0.06)
f_max = min(3.0 * f_bw, freq_fdtd.max())

# %% Plot R,T comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Single conductive panel: $\\varepsilon_r$={eps_r}, '
             f'$\\sigma$={sigma_val}, d={panel_thickness}', fontsize=13)

mask = (freq_fdtd > 0.05) & (freq_fdtd < f_max)
mask_a = (f_anal > 0.05) & (f_anal < f_max)

axes[0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analytical (TMM)')
axes[0].set_xlabel('Frequency (normalized)'); axes[0].set_ylabel('|R|')
axes[0].set_title('Reflection |R(f)|'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analytical (TMM)')
axes[1].set_xlabel('Frequency (normalized)'); axes[1].set_ylabel('|T|')
axes[1].set_title('Transmission |T(f)|'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
             'b-', alpha=0.6, lw=1, label='FDTD')
axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
             'r--', lw=2, label='Analytical (TMM)')
axes[2].axhline(1.0, color='gray', ls=':', alpha=0.5)
axes[2].set_xlabel('Frequency (normalized)'); axes[2].set_ylabel('$|R|^2 + |T|^2$')
axes[2].set_title('Energy conservation (< 1 for lossy)')
axes[2].legend(); axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1.15)

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Multi-layer Panel (Bonus)

# %% Multi-layer: analytical
layers = [
    {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
    {'d': 0.08, 'eps_r': 6.0, 'sigma': 1.0},
    {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
]

f_ml = np.linspace(0.01, 10.0, 1000)
R_ml, T_ml = RT_from_transfer_matrix(stack_transfer_matrix(f_ml, layers))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Multi-layer panel: 3 layers', fontsize=12)
axes[0].plot(f_ml, np.abs(R_ml), 'b-', lw=1.5); axes[0].set_ylabel('|R|')
axes[0].set_xlabel('Frequency'); axes[0].set_title('|R(f)|'); axes[0].grid(True, alpha=0.3)
axes[1].plot(f_ml, np.abs(T_ml), 'r-', lw=1.5); axes[1].set_ylabel('|T|')
axes[1].set_xlabel('Frequency'); axes[1].set_title('|T(f)|'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Parameter Study – Effect of conductivity on R and T

# %% Parameter sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Effect of conductivity on R,T ($\\varepsilon_r$={eps_r}, d={panel_thickness})', fontsize=13)

f_sweep = np.linspace(0.01, 8.0, 1000)
for sig in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    R_s, T_s = RT_from_transfer_matrix(panel_transfer_matrix(f_sweep, panel_thickness, eps_r, sig))
    axes[0].plot(f_sweep, np.abs(R_s), label=f'$\\sigma$={sig}')
    axes[1].plot(f_sweep, np.abs(T_s), label=f'$\\sigma$={sig}')

axes[0].set_xlabel('Frequency (normalized)'); axes[0].set_ylabel('|R|')
axes[0].set_title('|R(f)| for different $\\sigma$'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Frequency (normalized)'); axes[1].set_ylabel('|T|')
axes[1].set_title('|T(f)| for different $\\sigma$'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
# %%