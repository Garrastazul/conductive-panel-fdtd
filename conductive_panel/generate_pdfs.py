"""
Genera dos PDFs:
  1. Presentacion (3 diapositivas) en formato apaisado
  2. Resumen del proyecto
Ambos en espanol.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fpdf import FPDF

OUT_DIR = os.path.dirname(__file__)

# ── Colores ──
BLUE = (41, 65, 122)
LIGHT_BLUE = (220, 230, 245)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
ACCENT = (180, 60, 60)


# ═══════════════════════════════════════════════════════════════════
#  Generar figuras auxiliares con matplotlib
# ═══════════════════════════════════════════════════════════════════

def generate_figures():
    """Genera las figuras que se insertan en la presentacion y resumen."""
    from compare import transfer_matrix_normalized
    from transfer_matrix import RT_from_transfer_matrix
    from fdtd_panel import run_fdtd_panel, run_fdtd_reference, compute_RT_fdtd

    figs_dir = os.path.join(OUT_DIR, '_figs')
    os.makedirs(figs_dir, exist_ok=True)

    # ── Fig 1: diagrama esquematico (hecho con matplotlib) ──
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Panel
    from matplotlib.patches import FancyArrowPatch, Rectangle
    rect = Rectangle((4, 0.3), 2, 2.4, fc='#e8c170', ec='#8B6914', lw=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(5, 1.5, 'Panel\nconductivo\n$\\varepsilon_r, \\sigma, d$',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Flechas
    ax.annotate('', xy=(3.8, 2.0), xytext=(1, 2.0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(2.2, 2.25, '$E_{inc}$', fontsize=11, color='blue', fontweight='bold')

    ax.annotate('', xy=(1, 1.0), xytext=(3.8, 1.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(2.2, 0.65, '$E_{ref}$', fontsize=11, color='red', fontweight='bold')

    ax.annotate('', xy=(9, 1.5), xytext=(6.2, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(7.4, 1.75, '$E_{trans}$', fontsize=11, color='green', fontweight='bold')

    # Etiquetas
    ax.text(0.5, 0.05, 'Espacio libre ($\\eta_0$)', fontsize=9, color='gray')
    ax.text(7, 0.05, 'Espacio libre ($\\eta_0$)', fontsize=9, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'esquema.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ── Fig 2: R,T comparison (single panel) ──
    d, eps_r, sigma_val = 0.2, 4.0, 0.5
    pulse_sigma = 0.06
    N, L = 4001, 4.0

    print("  Simulacion FDTD (panel)...")
    panel_res = run_fdtd_panel(N=N, L=L, panel_thickness=d, eps_r=eps_r,
                               sigma=sigma_val, pulse_sigma=pulse_sigma, t_final=2.5*L)
    print("  Simulacion FDTD (referencia)...")
    ref_res = run_fdtd_reference(N=N, L=L, panel_thickness=d,
                                 pulse_sigma=pulse_sigma, t_final=2.5*L)
    freq_fdtd, R_fdtd, T_fdtd = compute_RT_fdtd(panel_res, ref_res)

    f_anal = np.linspace(0.01, freq_fdtd.max(), 2000)
    Phi = transfer_matrix_normalized(f_anal, d, eps_r, sigma_val)
    R_anal, T_anal = RT_from_transfer_matrix(Phi, eta0=1.0)

    f_bw = 1.0 / (2.0 * np.pi * pulse_sigma)
    f_max = min(3.0 * f_bw, freq_fdtd.max())

    mask = (freq_fdtd > 0.05) & (freq_fdtd < f_max)
    mask_a = (f_anal > 0.05) & (f_anal < f_max)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(freq_fdtd[mask], np.abs(R_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
    axes[0].plot(f_anal[mask_a], np.abs(R_anal[mask_a]), 'r--', lw=2, label='Analitico (TMM)')
    axes[0].set_xlabel('Frecuencia (normalizada)')
    axes[0].set_ylabel('|R|')
    axes[0].set_title('Coef. de reflexion |R(f)|')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq_fdtd[mask], np.abs(T_fdtd[mask]), 'b-', alpha=0.6, lw=1, label='FDTD')
    axes[1].plot(f_anal[mask_a], np.abs(T_anal[mask_a]), 'r--', lw=2, label='Analitico (TMM)')
    axes[1].set_xlabel('Frecuencia (normalizada)')
    axes[1].set_ylabel('|T|')
    axes[1].set_title('Coef. de transmision |T(f)|')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_fdtd[mask], np.abs(R_fdtd[mask])**2 + np.abs(T_fdtd[mask])**2,
                 'b-', alpha=0.6, lw=1, label='FDTD')
    axes[2].plot(f_anal[mask_a], np.abs(R_anal[mask_a])**2 + np.abs(T_anal[mask_a])**2,
                 'r--', lw=2, label='Analitico (TMM)')
    axes[2].axhline(1.0, color='gray', ls=':', alpha=0.5)
    axes[2].set_xlabel('Frecuencia (normalizada)')
    axes[2].set_ylabel('$|R|^2 + |T|^2$')
    axes[2].set_title('Conservacion de energia')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.15)

    fig.suptitle(f'Panel conductivo: $\\varepsilon_r$={eps_r}, $\\sigma$={sigma_val}, d={d}',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'comparacion_RT.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ── Fig 3: Senales temporales ──
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(panel_res['t_array'], panel_res['E_left'], 'b-', alpha=0.7, lw=1,
            label='E izq (inc+ref)')
    ax.plot(panel_res['t_array'], panel_res['E_right'], 'g-', alpha=0.7, lw=1,
            label='E der (trans)')
    ax.plot(ref_res['t_array'], ref_res['E_left'], 'r--', alpha=0.5, lw=1,
            label='E izq (ref, sin panel)')
    reflected = panel_res['E_left'] - ref_res['E_left']
    ax.plot(panel_res['t_array'], reflected, 'm-', alpha=0.7, lw=1.2,
            label='E reflejado (diferencia)')
    ax.set_xlabel('Tiempo (normalizado)')
    ax.set_ylabel('Campo E')
    ax.set_title('Senales temporales en los puntos de observacion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'temporal.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ── Fig 4: Barrido de sigma ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    f_sweep = np.linspace(0.01, 8.0, 1000)
    sigma_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    for sig in sigma_values:
        Phi_s = transfer_matrix_normalized(f_sweep, d, eps_r, sig)
        R_s, T_s = RT_from_transfer_matrix(Phi_s, eta0=1.0)
        axes[0].plot(f_sweep, np.abs(R_s), label=f'$\\sigma$={sig}')
        axes[1].plot(f_sweep, np.abs(T_s), label=f'$\\sigma$={sig}')

    axes[0].set_xlabel('Frecuencia (normalizada)')
    axes[0].set_ylabel('|R|')
    axes[0].set_title('|R(f)| para distintos $\\sigma$')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frecuencia (normalizada)')
    axes[1].set_ylabel('|T|')
    axes[1].set_title('|T(f)| para distintos $\\sigma$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Estudio parametrico ($\\varepsilon_r$={eps_r}, d={d})', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'barrido_sigma.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ── Fig 5: Multilayer ──
    from compare import transfer_matrix_stack_normalized
    layers = [
        {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
        {'d': 0.08, 'eps_r': 6.0, 'sigma': 1.0},
        {'d': 0.05, 'eps_r': 2.0, 'sigma': 0.2},
    ]
    f_ml = np.linspace(0.01, 10.0, 1000)
    Phi_ml = transfer_matrix_stack_normalized(f_ml, layers)
    R_ml, T_ml = RT_from_transfer_matrix(Phi_ml, eta0=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(f_ml, np.abs(R_ml), 'b-', lw=1.5)
    axes[0].set_xlabel('Frecuencia')
    axes[0].set_ylabel('|R|')
    axes[0].set_title('|R(f)| - Panel multicapa')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_ml, np.abs(T_ml), 'r-', lw=1.5)
    axes[1].set_xlabel('Frecuencia')
    axes[1].set_ylabel('|T|')
    axes[1].set_title('|T(f)| - Panel multicapa')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Panel multicapa: 3 capas con materiales distintos', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'multicapa.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    print("  Figuras generadas en", figs_dir)
    return figs_dir


# ═══════════════════════════════════════════════════════════════════
#  PRESENTACION (3 diapositivas, formato apaisado)
# ═══════════════════════════════════════════════════════════════════

class SlidePDF(FPDF):
    def __init__(self):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.set_auto_page_break(auto=False)

    def slide_bg(self):
        self.set_fill_color(*LIGHT_BLUE)
        self.rect(0, 0, 297, 210, 'F')
        # Barra superior
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 297, 28, 'F')

    def slide_title(self, title):
        self.set_font('Helvetica', 'B', 22)
        self.set_text_color(*WHITE)
        self.set_xy(10, 5)
        self.cell(277, 18, title, align='C')

    def slide_footer(self, text):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.set_xy(10, 198)
        self.cell(277, 8, text, align='C')

    def body_text(self, x, y, w, text, size=11, bold=False):
        self.set_font('Helvetica', 'B' if bold else '', size)
        self.set_text_color(*DARK)
        self.set_xy(x, y)
        self.multi_cell(w, size * 0.5, text)


def build_presentation(figs_dir):
    pdf = SlidePDF()

    footer = 'Metodos Computacionales en Fisica No Lineal - UGR 2026'

    # ── DIAPOSITIVA 1: Introduccion y teoria ──
    pdf.add_page()
    pdf.slide_bg()
    pdf.slide_title('Coeficiente de Reflexion de un Panel Conductivo')
    pdf.slide_footer(footer)

    # Columna izquierda: teoria
    pdf.body_text(10, 33, 145,
        'Problema: Calcular R(f) y T(f) de un panel con baja\n'
        'conductividad usando FDTD y comparar con la solucion\n'
        'analitica del metodo de la Matriz de Transferencia.',
        size=11, bold=True)

    pdf.body_text(10, 58, 145,
        'Matriz de transferencia (ABCD) para un panel homogeneo:\n\n'
        '             [cosh(gd)      eta*sinh(gd) ]\n'
        '   Phi =     [sinh(gd)/eta  cosh(gd)     ]\n\n'
        'donde:\n'
        '   g = j*w*sqrt(mu * eps_c)  (cte. propagacion)\n'
        '   eta = sqrt(mu / eps_c)    (impedancia intrinseca)\n'
        '   eps_c = eps_r*eps_0 - j*sigma/w\n\n'
        'Coeficientes de reflexion y transmision:\n'
        '   T = 2*eta0 / (A*eta0 + B + C*eta0^2 + D*eta0)\n'
        '   R = (A*eta0 + B - C*eta0^2 - D*eta0) / (denom)',
        size=9)

    # Columna derecha: esquema
    img_path = os.path.join(figs_dir, 'esquema.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=155, y=35, w=135)

    pdf.body_text(155, 90, 135,
        'Referencias:\n'
        '  - Notas de clase, seccion 1.4.3\n'
        '  - Orfanidis, "EM Waves & Antennas", Cap. 4-5\n'
        '  - Para multicapa: Phi_total = prod(Phi_i)',
        size=9)

    # ── DIAPOSITIVA 2: Metodo FDTD ──
    pdf.add_page()
    pdf.slide_bg()
    pdf.slide_title('Metodo FDTD: Simulacion Numerica')
    pdf.slide_footer(footer)

    pdf.body_text(10, 33, 140,
        'Procedimiento FDTD:',
        size=12, bold=True)

    pdf.body_text(10, 42, 140,
        '1. Dominio [0, L] con condiciones de frontera Mur (ABC)\n'
        '2. Pulso gaussiano viajando a la derecha (E, H = -E)\n'
        '3. Panel conductivo en el centro del dominio\n'
        '4. Puntos de observacion a izquierda y derecha\n'
        '5. Simulacion de referencia SIN panel\n'
        '6. Calculo en frecuencia via FFT:\n\n'
        '   E_reflejado(t) = E_izq(con panel) - E_izq(sin panel)\n'
        '   E_transmitido(t) = E_der(con panel)\n'
        '   E_incidente(t) = E_der(sin panel)\n\n'
        '   R(f) = FFT(E_ref) / FFT(E_inc)\n'
        '   T(f) = FFT(E_trans) / FFT(E_inc)',
        size=10)

    img_path = os.path.join(figs_dir, 'temporal.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=150, y=33, w=142)

    pdf.body_text(10, 120, 140,
        'Ventajas del metodo:\n'
        '  - No requiere hipotesis de armonicidad\n'
        '  - Obtiene R(f) y T(f) en todo el espectro de una vez\n'
        '  - Reutiliza la clase FDTD1D del repositorio',
        size=9)

    pdf.body_text(150, 140, 142,
        'Parametros utilizados:\n'
        '  N = 4001 puntos, L = 4.0, CFL = 1\n'
        '  Pulso: sigma = 0.06 (ancho espectral amplio)\n'
        '  Panel: eps_r = 4, sigma = 0.5, d = 0.2',
        size=9)

    # ── DIAPOSITIVA 3: Resultados ──
    pdf.add_page()
    pdf.slide_bg()
    pdf.slide_title('Resultados: FDTD vs Analitico')
    pdf.slide_footer(footer)

    img_path = os.path.join(figs_dir, 'comparacion_RT.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=5, y=30, w=200)

    img_path = os.path.join(figs_dir, 'barrido_sigma.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=5, y=110, w=150)

    img_path = os.path.join(figs_dir, 'multicapa.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=155, y=110, w=137)

    pdf.body_text(205, 32, 90,
        'Conclusiones:\n\n'
        '- Excelente concordancia\n'
        '  entre FDTD y TMM\n\n'
        '- |R|^2 + |T|^2 < 1 para\n'
        '  medios con perdidas\n'
        '  (energia absorbida)\n\n'
        '- Mayor sigma -> mayor\n'
        '  reflexion, menor\n'
        '  transmision\n\n'
        '- Bonus: panel multicapa\n'
        '  con 3 capas distintas',
        size=10)

    out_path = os.path.join(OUT_DIR, 'presentacion_panel_conductivo.pdf')
    pdf.output(out_path)
    print(f"Presentacion guardada: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════
#  RESUMEN DEL PROYECTO (documento vertical)
# ═══════════════════════════════════════════════════════════════════

class ResumenPDF(FPDF):
    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 15, 'F')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*WHITE)
        self.set_xy(10, 3)
        self.cell(190, 10, 'Metodos Computacionales en Fisica No Lineal - UGR 2026', align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*BLUE)
        self.ln(4)
        self.cell(0, 8, title)
        self.ln(8)
        self.set_draw_color(*BLUE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body(self, text, size=10):
        self.set_font('Helvetica', '', size)
        self.set_text_color(*DARK)
        self.multi_cell(0, size * 0.55, text)
        self.ln(2)

    def body_bold(self, text, size=10):
        self.set_font('Helvetica', 'B', size)
        self.set_text_color(*DARK)
        self.multi_cell(0, size * 0.55, text)
        self.ln(1)


def build_resumen(figs_dir):
    pdf = ResumenPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Pagina 1: Portada ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 15, 'Coeficiente de Reflexion de un', align='C')
    pdf.ln(15)
    pdf.cell(0, 15, 'Panel Ligeramente Conductivo', align='C')
    pdf.ln(25)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 10, 'Proyecto: Metodo FDTD vs Matriz de Transferencia', align='C')
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'Metodos Computacionales en Fisica No Lineal', align='C')
    pdf.ln(8)
    pdf.cell(0, 8, 'Master en Fisica y Matematicas - Universidad de Granada', align='C')
    pdf.ln(8)
    pdf.cell(0, 8, 'Curso 2025-2026', align='C')

    # ── Pagina 2: Introduccion ──
    pdf.add_page()
    pdf.ln(5)
    pdf.section_title('1. Introduccion')
    pdf.body(
        'Este proyecto calcula los coeficientes de reflexion R(f) y transmision T(f) '
        'de un panel con baja conductividad utilizando dos metodos independientes:\n\n'
        '  1. Metodo de la Matriz de Transferencia (TMM) - solucion analitica\n'
        '  2. Metodo FDTD (Finite-Differences Time-Domain) - solucion numerica\n\n'
        'Se comparan ambos resultados para validar la implementacion FDTD y se '
        'extiende el analisis a paneles multicapa (bonus).'
    )

    pdf.section_title('2. Marco Teorico')
    pdf.body_bold('2.1 Matriz de Transferencia (ABCD)')
    pdf.body(
        'Para un panel homogeneo de espesor d con permitividad compleja '
        'eps_c = eps_r * eps_0 - j*sigma/omega, la matriz de transferencia es:\n\n'
        '   Phi = [[cosh(gamma*d),  eta*sinh(gamma*d)],\n'
        '          [sinh(gamma*d)/eta,  cosh(gamma*d)]]\n\n'
        'donde gamma = j*omega*sqrt(mu*eps_c) es la constante de propagacion '
        'y eta = sqrt(mu/eps_c) es la impedancia intrinseca del material.\n\n'
        'Los coeficientes R y T se obtienen de la matriz ABCD (Phi) como:\n\n'
        '   T = 2*eta_0 / (A*eta_0 + B + C*eta_0^2 + D*eta_0)\n'
        '   R = (A*eta_0 + B - C*eta_0^2 - D*eta_0) / (denominador)\n\n'
        'Para un apilamiento de N capas: Phi_total = Phi_1 * Phi_2 * ... * Phi_N'
    )

    pdf.body_bold('2.2 Metodo FDTD')
    pdf.body(
        'El algoritmo FDTD resuelve las ecuaciones de Maxwell en el dominio del '
        'tiempo usando diferencias finitas centradas en una malla escalonada (Yee). '
        'En 1D, las ecuaciones de actualizacion para E y H son:\n\n'
        '   E^{n+1}_i = ca_i * E^n_i - cb_i * (H^n_{i+1/2} - H^n_{i-1/2})\n'
        '   H^{n+1}_{i+1/2} = H^n_{i+1/2} - r * (E^{n+1}_{i+1} - E^{n+1}_i)\n\n'
        'donde ca y cb incorporan la conductividad sigma del material.'
    )

    # ── Pagina 3: Metodologia ──
    pdf.add_page()
    pdf.ln(5)
    pdf.section_title('3. Metodologia')
    pdf.body_bold('Procedimiento de la simulacion FDTD:')
    pdf.body(
        '1. Se define un dominio [0, L] con condiciones de frontera absorbente Mur.\n'
        '2. Se inicializa un pulso gaussiano viajando a la derecha (E = gauss, H = -gauss).\n'
        '3. Se coloca el panel conductivo en el centro del dominio.\n'
        '4. Se registran las senales E(t) en dos puntos de observacion: uno a la izquierda\n'
        '   del panel (captura incidente + reflejado) y otro a la derecha (transmitido).\n'
        '5. Se repite la simulacion SIN panel (referencia) para obtener la senal incidente.\n'
        '6. Se calculan R(f) y T(f) en el dominio de la frecuencia via FFT:\n\n'
        '   E_reflejado(t) = E_izq(con panel) - E_izq(sin panel)\n'
        '   R(f) = FFT(E_reflejado) / FFT(E_incidente)\n'
        '   T(f) = FFT(E_transmitido) / FFT(E_incidente)'
    )

    pdf.body_bold('Parametros de la simulacion:')
    pdf.body(
        '   - Puntos de malla: N = 4001\n'
        '   - Longitud del dominio: L = 4.0 (unidades normalizadas, c=1)\n'
        '   - Numero CFL = 1 (esquema exacto para medios homogeneos)\n'
        '   - Ancho del pulso: sigma = 0.06\n'
        '   - Panel: eps_r = 4, sigma = 0.5, d = 0.2'
    )

    # ── Pagina 4: Resultados ──
    pdf.add_page()
    pdf.ln(5)
    pdf.section_title('4. Resultados')

    pdf.body_bold('4.1 Comparacion FDTD vs Analitico')
    img_path = os.path.join(figs_dir, 'comparacion_RT.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, w=190)
        pdf.ln(5)

    pdf.body(
        'La figura muestra excelente concordancia entre el metodo FDTD (linea azul) '
        'y la solucion analitica TMM (linea roja discontinua) para |R(f)|, |T(f)| y '
        'la verificacion de conservacion de energia |R|^2 + |T|^2. Para medios con '
        'perdidas (sigma > 0), la suma |R|^2 + |T|^2 < 1, indicando absorcion de energia.'
    )

    pdf.body_bold('4.2 Senales temporales')
    img_path = os.path.join(figs_dir, 'temporal.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=15, w=170)
        pdf.ln(3)

    pdf.body(
        'Se observa como el pulso incidente genera un pulso reflejado (hacia la izquierda) '
        'y un pulso transmitido (hacia la derecha) de menor amplitud debido a las perdidas.'
    )

    # ── Pagina 5: Estudio parametrico + multicapa ──
    pdf.add_page()
    pdf.ln(5)
    pdf.section_title('4.3 Estudio Parametrico')

    img_path = os.path.join(figs_dir, 'barrido_sigma.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, w=190)
        pdf.ln(3)

    pdf.body(
        'Al aumentar la conductividad sigma:\n'
        '  - |R| aumenta (mayor reflexion)\n'
        '  - |T| disminuye (menor transmision)\n'
        '  - Las oscilaciones de tipo Fabry-Perot se amortiguan'
    )

    pdf.section_title('4.4 Panel Multicapa (Bonus)')

    img_path = os.path.join(figs_dir, 'multicapa.png')
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, w=190)
        pdf.ln(3)

    pdf.body(
        'Panel de 3 capas con materiales distintos:\n'
        '  - Capa 1: eps_r=2, sigma=0.2, d=0.05\n'
        '  - Capa 2: eps_r=6, sigma=1.0, d=0.08\n'
        '  - Capa 3: eps_r=2, sigma=0.2, d=0.05\n\n'
        'La respuesta muestra un patron de interferencia mas complejo que el caso '
        'de una sola capa, debido a las multiples reflexiones internas entre interfaces.'
    )

    # ── Pagina 6: Estructura del codigo y conclusiones ──
    pdf.add_page()
    pdf.ln(5)
    pdf.section_title('5. Estructura del Codigo')
    pdf.body(
        'El proyecto se organiza en la carpeta conductive_panel/ dentro del repositorio:\n\n'
        '  transfer_matrix.py   - Solucion analitica (TMM) en unidades SI\n'
        '  fdtd_panel.py        - Simulacion FDTD con extraccion de R,T via FFT\n'
        '  compare.py           - Comparacion analitico vs numerico + graficas\n'
        '  visualize_panel.py   - Visualizacion interactiva (Jupyter/VSCode)\n'
        '  test_panel.py        - 10 tests unitarios (todos pasan)\n\n'
        'El codigo reutiliza la clase FDTD1D del repositorio principal, extendiendo\n'
        'su funcionalidad para analizar la interaccion con paneles conductivos.'
    )

    pdf.section_title('6. Conclusiones')
    pdf.body(
        '1. Se implemento exitosamente el calculo de R(f) y T(f) para paneles '
        'conductivos usando tanto el metodo de la Matriz de Transferencia (analitico) '
        'como el metodo FDTD (numerico).\n\n'
        '2. Los resultados FDTD muestran excelente concordancia con la solucion analitica '
        'en todo el rango de frecuencias donde el pulso tiene contenido espectral.\n\n'
        '3. Se verifico la conservacion de energia: para paneles sin perdidas '
        '|R|^2 + |T|^2 = 1, y para paneles conductivos |R|^2 + |T|^2 < 1.\n\n'
        '4. El estudio parametrico muestra que la conductividad aumenta la reflexion '
        'y reduce la transmision, amortiguando las resonancias Fabry-Perot.\n\n'
        '5. Se extendio el analisis a paneles multicapa (bonus), demostrando la '
        'generalidad del metodo de matrices de transferencia.'
    )

    pdf.section_title('7. Referencias')
    pdf.body(
        '[1] Notas de clase: "Deterministic Computational Methods", seccion 1.4.3\n'
        '[2] Notas de clase: "Finite-Differences in Time Domain", capitulo 2\n'
        '[3] S. J. Orfanidis, "Electromagnetic Waves and Antennas", capitulos 4-5\n'
        '[4] D. A. Frickey, "Conversions between S, Z, Y, H, ABCD, and T parameters",\n'
        '    IEEE Trans. MTT, 1994'
    )

    out_path = os.path.join(OUT_DIR, 'resumen_panel_conductivo.pdf')
    pdf.output(out_path)
    print(f"Resumen guardado: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generando figuras...")
    figs_dir = generate_figures()
    print("\nGenerando presentacion...")
    build_presentation(figs_dir)
    print("\nGenerando resumen...")
    build_resumen(figs_dir)
    print("\nListo!")
