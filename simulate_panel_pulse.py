import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from fdtd1d import FDTD1D
from slightly_conductive_panel import SlightlyConductivePanel

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def build_rightward_pulse_simulation():
    x = np.linspace(-1.0, 1.0, 401)
    xH = (x[1:] + x[:-1]) / 2.0

    panel = SlightlyConductivePanel(
        x,
        xL_panel=0.0,
        d_panel=0.4,
        eps_panel=2.5,
        sigma_panel=0.08,
    )

    x0 = -0.8
    sigma = 0.05
    e0 = gaussian(x, x0, sigma)
    h0 = gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
    panel.apply_to(fdtd)
    fdtd.load_initial_field(e0)
    fdtd.h = h0.copy()

    n_frames = 240
    dt_per_frame = 0.01

    frames = []
    times = []

    frames.append(fdtd.get_e())
    times.append(fdtd.t)

    for _ in range(n_frames - 1):
        fdtd.run_until(fdtd.t + dt_per_frame)
        frames.append(fdtd.get_e())
        times.append(fdtd.t)

    return x, frames, times, panel


def save_panel_pulse_gif(filename='panel_pulse.gif'):
    x, frames, times, panel = build_rightward_pulse_simulation()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('E(x, t)')
    ax.set_title('Pulso gaussiano derecho atravesando un panel levemente conductor')
    ax.grid(True, alpha=0.3)

    ax.axvspan(panel.xL_panel, panel.xL_panel + panel.d_panel,
               color='gray', alpha=0.25, label='panel')
    ax.legend(loc='upper right', fontsize=9)

    line, = ax.plot([], [], lw=2, color='royalblue')
    time_txt = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=10)

    def init():
        line.set_data([], [])
        time_txt.set_text('')
        return line, time_txt

    def update(frame_idx):
        line.set_data(x, frames[frame_idx])
        time_txt.set_text(f't = {times[frame_idx]:.3f}')
        return line, time_txt

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                         interval=40, blit=True)

    writer = PillowWriter(fps=25)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f'Guardado GIF de la simulación en: {filename}')


if __name__ == '__main__':
    save_panel_pulse_gif('panel_pulse.gif')
