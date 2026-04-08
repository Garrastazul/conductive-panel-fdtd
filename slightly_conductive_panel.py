import numpy as np
import matplotlib.pyplot as plt
from fdtd1d import FDTD1D

class SlightlyConductivePanel:
    def __init__(self, x, xL_panel=0.0, d_panel=0.5,
                 eps_panel=2.0, sigma_panel=0.1,
                 eps_bg=1.0, sigma_bg=0.0):
        self.x = np.asarray(x)
        self.xMin = np.min(self.x)
        self.xMax = np.max(self.x)
        self.L = self.xMax - self.xMin
        self.N = len(self.x)

        self.xL_panel = xL_panel
        self.d_panel = d_panel
        self.eps_panel = eps_panel
        self.sigma_panel = sigma_panel
        self.eps_bg = eps_bg
        self.sigma_bg = sigma_bg

        self.eps_r = np.full(self.N, self.eps_bg)
        self.sig = np.full(self.N, self.sigma_bg)
        self._build_panel()

    def _build_panel(self):
        xR_panel = self.xL_panel + self.d_panel
        inside = (self.x >= self.xL_panel) & (self.x <= xR_panel)

        self.eps_r[inside] = self.eps_panel
        self.sig[inside] = self.sigma_panel
        self.eps = FDTD1D.eps0 * self.eps_r

    def apply_to(self, fdtd):
        fdtd.eps_r = self.eps_r.copy()
        fdtd.sig = self.sig.copy()
        fdtd.eps = FDTD1D.eps0 * fdtd.eps_r

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))

        ax.plot(self.x, self.eps_r, label=r'$\epsilon_r$')
        ax.plot(self.x, self.sig, label=r'$\sigma$')
        ax.axvspan(self.xL_panel, self.xL_panel + self.d_panel,
                   color='gray', alpha=0.2, label='panel')
        ax.set_xlabel('x')
        ax.set_ylabel('Material properties')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        return ax

    def summary(self):
        return {
            'xMin': self.xMin,
            'xMax': self.xMax,
            'L': self.L,
            'N': self.N,
            'xL_panel': self.xL_panel,
            'd_panel': self.d_panel,
            'eps_panel': self.eps_panel,
            'sigma_panel': self.sigma_panel,
        }


def create_panel(x, **kwargs):
    return SlightlyConductivePanel(x, **kwargs)


if __name__ == '__main__':
    x = np.linspace(-1.0, 1.0, 401)
    panel = SlightlyConductivePanel(x, xL_panel=-0.25, d_panel=0.5,
                                    eps_panel=2.0, sigma_panel=0.05)
    fig, ax = plt.subplots(figsize=(8, 3))
    panel.plot(ax=ax)
    plt.title('Panel ligeramente conductor')
    plt.show()

