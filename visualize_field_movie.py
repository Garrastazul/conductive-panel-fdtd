# %% [markdown]
# # 1-D FDTD Field Movie
#
# Run this file in VSCode with the Jupyter extension (or any IPython-compatible
# interactive environment) by executing each cell with **Shift+Enter**.
#
# The initial condition is a narrow Gaussian pulse centred in the domain.
# Because CFL = 1 the pulse splits into two exact half-amplitude copies that
# travel in opposite directions.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from fdtd1d import FDTD1D, gaussian, panel_transfer_matrix, RT_from_transfer_matrix
from matplotlib.patches import Rectangle

# %% [markdown]
# ## Simulation parameters

# %% Parameters
# Spatial grid
x = np.linspace(-1.0, 1.0, 401)

# Gaussian initial condition
x0    = 0.0   # centre
sigma = 0.08  # width
e0    = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

# Number of animation frames and simulation time step between frames
n_frames     = 120
dt_per_frame = 0.01

# %% [markdown]
# ## Pre-compute all frames

# %% Run simulation
fdtd = FDTD1D(x)
fdtd.load_initial_field(e0)

frames = []  # E-field snapshots
times  = []  # corresponding simulation times

# Store frame 0 (initial condition)
frames.append(fdtd.get_e())
times.append(fdtd.t)

for _ in range(n_frames - 1):
    fdtd.run_until(fdtd.t + dt_per_frame)
    frames.append(fdtd.get_e())
    times.append(fdtd.t)

print(f"Captured {len(frames)} frames  "
      f"(t = {times[0]:.3f} … {times[-1]:.3f})")

# %% [markdown]
# ## Build and display the animation

# %% Animate
fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.6, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("E(x, t)")
ax.set_title("1-D FDTD – Electric field evolution")
ax.grid(True, alpha=0.3)

(line,)  = ax.plot([], [], lw=2, color="royalblue")
time_txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)


def init():
    line.set_data([], [])
    time_txt.set_text("")
    return line, time_txt


def update(frame_idx):
    line.set_data(x, frames[frame_idx])
    time_txt.set_text(f"t = {times[frame_idx]:.3f}")
    return line, time_txt


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    init_func=init,
    interval=40,   # ms between frames → ~25 fps
    blit=True,
)

plt.close(fig)         # prevent a static duplicate figure
HTML(anim.to_jshtml()) # display the animation inline

# %% [markdown]
# ## Mur Absorbing Boundary Conditions
#
# A purely **left-traveling** Gaussian pulse (initialized with H = −E)
# propagates toward the left boundary where a first-order Mur ABC is
# applied.  The pulse should be absorbed with negligible reflections.

# %% Parameters (Mur)
x_mur  = np.linspace(-1.0, 1.0, 401)
xH_mur = (x_mur[1:] + x_mur[:-1]) / 2.0

x0_mur    = 0.0
sigma_mur = 0.08

# Left-traveling wave: E = gaussian, H = -gaussian
e0_mur = np.exp(-0.5 * ((x_mur  - x0_mur) / sigma_mur) ** 2)
h0_mur = -np.exp(-0.5 * ((xH_mur - x0_mur) / sigma_mur) ** 2)

n_frames_mur     = 160
dt_per_frame_mur = 0.01

# %% Run simulation (Mur)
fdtd_mur = FDTD1D(x_mur, boundaries=('mur', 'mur'))
fdtd_mur.load_initial_field(e0_mur)
fdtd_mur.h = h0_mur.copy()

frames_e_mur = []
frames_h_mur = []
times_mur    = []

frames_e_mur.append(fdtd_mur.get_e())
frames_h_mur.append(fdtd_mur.get_h())
times_mur.append(fdtd_mur.t)

for _ in range(n_frames_mur - 1):
    fdtd_mur.run_until(fdtd_mur.t + dt_per_frame_mur)
    frames_e_mur.append(fdtd_mur.get_e())
    frames_h_mur.append(fdtd_mur.get_h())
    times_mur.append(fdtd_mur.t)

print(f"Mur ABC – Captured {len(frames_e_mur)} frames  "
      f"(t = {times_mur[0]:.3f} … {times_mur[-1]:.3f})")

# %% Animate (Mur)
fig_mur, ax_mur = plt.subplots(figsize=(8, 4))

ax_mur.set_xlim(x_mur[0], x_mur[-1])
ax_mur.set_ylim(-1.1, 1.1)
ax_mur.set_xlim(-1.1, 1.1)
ax_mur.set_xlabel("x")
ax_mur.set_ylabel("Field amplitude")
ax_mur.set_title("1-D FDTD – Mur ABC – Left-traveling wave absorption")
ax_mur.grid(True, alpha=0.3)

ax_mur.axvline(x_mur[0],  color="red", ls="--", lw=1.5, label="Mur boundary")
ax_mur.axvline(x_mur[-1], color="red", ls="--", lw=1.5)

(line_e_mur,) = ax_mur.plot([], [], lw=2, color="royalblue", label="E(x, t)")
(line_h_mur,) = ax_mur.plot([], [], lw=1.5, color="darkorange", label="H(x, t)")
ax_mur.legend(loc="upper right", fontsize=9)

time_txt_mur = ax_mur.text(0.02, 0.93, "", transform=ax_mur.transAxes, fontsize=10)


def init_mur():
    line_e_mur.set_data([], [])
    line_h_mur.set_data([], [])
    time_txt_mur.set_text("")
    return line_e_mur, line_h_mur, time_txt_mur


def update_mur(frame_idx):
    line_e_mur.set_data(x_mur, frames_e_mur[frame_idx])
    line_h_mur.set_data(xH_mur, frames_h_mur[frame_idx])
    time_txt_mur.set_text(f"t = {times_mur[frame_idx]:.3f}")
    return line_e_mur, line_h_mur, time_txt_mur


anim_mur = FuncAnimation(
    fig_mur,
    update_mur,
    frames=len(frames_e_mur),
    init_func=init_mur,
    interval=40,
    blit=True,
)

plt.close(fig_mur)
HTML(anim_mur.to_jshtml())

# %% [markdown]
# ## Conductive Panel – Reflection and Transmission
#
# A **right-traveling** Gaussian pulse (H = +E) hits a slightly conductive
# panel placed in the middle of the domain.  Part of the pulse is reflected,
# part is transmitted (attenuated), and part is absorbed.

# %% Parameters (Panel)
x_pan  = np.linspace(0.0, 4.0, 2001)
xH_pan = (x_pan[1:] + x_pan[:-1]) / 2.0

x0_pan    = 1.2
sigma_pan = 0.08

# Right-traveling wave: E = gaussian, H = +gaussian
e0_pan = gaussian(x_pan,  x0_pan, sigma_pan)
h0_pan = gaussian(xH_pan, x0_pan, sigma_pan)

# Panel properties
panel_center = 2.0
panel_d      = 0.3
eps_r_pan    = 4.0
sigma_c_pan  = 0.5   # conductivity

panel_left  = panel_center - panel_d / 2
panel_right = panel_center + panel_d / 2

n_frames_pan     = 250
dt_per_frame_pan = 0.015

# %% Run simulation (Panel)
fdtd_pan = FDTD1D(x_pan, boundaries=('mur', 'mur'))
fdtd_pan.load_initial_field(e0_pan)
fdtd_pan.h = h0_pan.copy()
fdtd_pan.eps_r = np.where((x_pan >= panel_left) & (x_pan <= panel_right), eps_r_pan, 1.0)
fdtd_pan.sig   = np.where((x_pan >= panel_left) & (x_pan <= panel_right), sigma_c_pan, 0.0)

frames_e_pan = []
frames_h_pan = []
times_pan    = []

frames_e_pan.append(fdtd_pan.get_e())
frames_h_pan.append(fdtd_pan.get_h())
times_pan.append(fdtd_pan.t)

for _ in range(n_frames_pan - 1):
    fdtd_pan.run_until(fdtd_pan.t + dt_per_frame_pan)
    frames_e_pan.append(fdtd_pan.get_e())
    frames_h_pan.append(fdtd_pan.get_h())
    times_pan.append(fdtd_pan.t)

print(f"Panel – Captured {len(frames_e_pan)} frames  "
      f"(t = {times_pan[0]:.3f} … {times_pan[-1]:.3f})")

# %% Animate (Panel)
fig_pan, ax_pan = plt.subplots(figsize=(10, 5))

ax_pan.set_xlim(x_pan[0], x_pan[-1])
ax_pan.set_ylim(-1.1, 1.1)
ax_pan.set_xlabel("x")
ax_pan.set_ylabel("Field amplitude")
ax_pan.set_title(
    f"1-D FDTD – Pulse through conductive panel "
    f"($\\varepsilon_r$={eps_r_pan}, $\\sigma$={sigma_c_pan}, d={panel_d})"
)
ax_pan.grid(True, alpha=0.3)

ax_pan.add_patch(Rectangle(
    (panel_left, -1.1), panel_d, 2.2,
    color="orange", alpha=0.2, label="Panel"
))
ax_pan.axvline(panel_left,  color="orange", ls="--", lw=1)
ax_pan.axvline(panel_right, color="orange", ls="--", lw=1)

(line_e_pan,) = ax_pan.plot([], [], lw=2, color="royalblue", label="E(x, t)")
(line_h_pan,) = ax_pan.plot([], [], lw=1.5, color="darkorange", alpha=0.7, label="H(x, t)")
ax_pan.legend(loc="upper right", fontsize=9)

time_txt_pan = ax_pan.text(0.02, 0.93, "", transform=ax_pan.transAxes, fontsize=10)


def init_pan():
    line_e_pan.set_data([], [])
    line_h_pan.set_data([], [])
    time_txt_pan.set_text("")
    return line_e_pan, line_h_pan, time_txt_pan


def update_pan(frame_idx):
    line_e_pan.set_data(x_pan, frames_e_pan[frame_idx])
    line_h_pan.set_data(xH_pan, frames_h_pan[frame_idx])
    time_txt_pan.set_text(f"t = {times_pan[frame_idx]:.3f}")
    return line_e_pan, line_h_pan, time_txt_pan


anim_pan = FuncAnimation(
    fig_pan,
    update_pan,
    frames=len(frames_e_pan),
    init_func=init_pan,
    interval=40,
    blit=True,
)

plt.close(fig_pan)
HTML(anim_pan.to_jshtml())

# %% [markdown]
# ## Conductive Panel – R(f) and T(f) comparison
#
# Run FDTD with and without the panel, extract R(f) and T(f) via FFT,
# and compare with the analytical Transfer Matrix Method.

# %% FDTD extraction of R,T
N_rt = 4001
L_rt = 4.0
x_rt  = np.linspace(0, L_rt, N_rt)
xH_rt = (x_rt[1:] + x_rt[:-1]) / 2.0

pulse_sigma_rt = 0.06
pulse_x0_rt    = 0.8
pl = panel_center - panel_d / 2
pr = panel_center + panel_d / 2
obs_l = np.argmin(np.abs(x_rt - (pl - 0.4)))
obs_r = np.argmin(np.abs(x_rt - (pr + 0.4)))

e0_rt = gaussian(x_rt,  pulse_x0_rt, pulse_sigma_rt)
h0_rt = gaussian(xH_rt, pulse_x0_rt, pulse_sigma_rt)
t_final_rt = 2.5 * L_rt

# With panel
fdtd_p = FDTD1D(x_rt, boundaries=('mur', 'mur'))
fdtd_p.load_initial_field(e0_rt)
fdtd_p.h = h0_rt.copy()
fdtd_p.eps_r = np.where((x_rt >= pl) & (x_rt <= pr), eps_r_pan, 1.0)
fdtd_p.sig   = np.where((x_rt >= pl) & (x_rt <= pr), sigma_c_pan, 0.0)

n_steps_rt = round(t_final_rt / fdtd_p.dt)
El_p = np.zeros(n_steps_rt)
Er_p = np.zeros(n_steps_rt)
for i in range(n_steps_rt):
    fdtd_p._step()
    El_p[i] = fdtd_p.e[obs_l]
    Er_p[i] = fdtd_p.e[obs_r]

# Free-space reference
fdtd_ref = FDTD1D(x_rt, boundaries=('mur', 'mur'))
fdtd_ref.load_initial_field(e0_rt)
fdtd_ref.h = h0_rt.copy()

El_ref = np.zeros(n_steps_rt)
Er_ref = np.zeros(n_steps_rt)
for i in range(n_steps_rt):
    fdtd_ref._step()
    El_ref[i] = fdtd_ref.e[obs_l]
    Er_ref[i] = fdtd_ref.e[obs_r]

# FFT
dt_rt = fdtd_p.dt
Eref_fft  = np.fft.rfft(El_p - El_ref)
Etrans_fft = np.fft.rfft(Er_p)
Einc_fft   = np.fft.rfft(Er_ref)
freq_fdtd  = np.fft.rfftfreq(n_steps_rt, d=dt_rt)

valid = np.abs(Einc_fft) > 1e-10 * np.max(np.abs(Einc_fft))
R_fdtd = np.zeros_like(freq_fdtd, dtype=complex)
T_fdtd = np.zeros_like(freq_fdtd, dtype=complex)
R_fdtd[valid] = Eref_fft[valid]  / Einc_fft[valid]
T_fdtd[valid] = Etrans_fft[valid] / Einc_fft[valid]

# Analytical
f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
R_anal, T_anal = RT_from_transfer_matrix(
    panel_transfer_matrix(f_anal, panel_d, eps_r_pan, sigma_c_pan)
)

f_bw  = 1.0 / (2.0 * np.pi * pulse_sigma_rt)
f_max = min(3.0 * f_bw, freq_fdtd.max())

print("FDTD vs Analytical – done.")

# %% Plot R,T comparison
fig_rt, axes_rt = plt.subplots(1, 3, figsize=(16, 5))
fig_rt.suptitle(
    f"Conductive panel: $\\varepsilon_r$={eps_r_pan}, "
    f"$\\sigma$={sigma_c_pan}, d={panel_d}",
    fontsize=13,
)

mask_f = (freq_fdtd > 0.05) & (freq_fdtd < f_max)
mask_a = (f_anal > 0.05) & (f_anal < f_max)

axes_rt[0].plot(freq_fdtd[mask_f], np.abs(R_fdtd[mask_f]), "b-", alpha=0.6, lw=1, label="FDTD")
axes_rt[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), "r--", lw=2, label="Analytical (TMM)")
axes_rt[0].set_xlabel("Frequency (normalized)")
axes_rt[0].set_ylabel("|R|")
axes_rt[0].set_title("Reflection |R(f)|")
axes_rt[0].legend()
axes_rt[0].grid(True, alpha=0.3)

axes_rt[1].plot(freq_fdtd[mask_f], np.abs(T_fdtd[mask_f]), "b-", alpha=0.6, lw=1, label="FDTD")
axes_rt[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), "r--", lw=2, label="Analytical (TMM)")
axes_rt[1].set_xlabel("Frequency (normalized)")
axes_rt[1].set_ylabel("|T|")
axes_rt[1].set_title("Transmission |T(f)|")
axes_rt[1].legend()
axes_rt[1].grid(True, alpha=0.3)

axes_rt[2].plot(freq_fdtd[mask_f],
    np.abs(R_fdtd[mask_f])**2 + np.abs(T_fdtd[mask_f])**2,
    "b-", alpha=0.6, lw=1, label="FDTD")
axes_rt[2].plot(f_anal[mask_a],
    np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
    "r--", lw=2, label="Analytical (TMM)")
axes_rt[2].axhline(1.0, color="gray", ls=":", alpha=0.5)
axes_rt[2].set_xlabel("Frequency (normalized)")
axes_rt[2].set_ylabel("$|R|^2 + |T|^2$")
axes_rt[2].set_title("Energy conservation (< 1 for lossy)")
axes_rt[2].legend()
axes_rt[2].grid(True, alpha=0.3)
axes_rt[2].set_ylim(0, 1.15)

plt.tight_layout()
plt.show()
# %%
