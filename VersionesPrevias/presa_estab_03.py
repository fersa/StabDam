# -*- coding: utf-8 -*-
"""
Cálculo de estabilidad de una presa de gravedad (Categoría A, Norma Técnica ESP)
Estructura: 1 archivo .py | Enfoque funcional | Uso en IDE | Salida Excel + PNG
Unidades SI: m, kN, kN/m, kN/m2 (kPa). Cargas por metro de espesor.

Dependencias: numpy, pandas, matplotlib
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# =============================================================================
# ======================== BLOQUE DE ENTRADA (EDITABLE) =======================
# =============================================================================
CONFIG = {
    "GEOMETRIA": {
        # Definición por taludes: origen en pie de Aguas Arriba (0,0), base horizontal (marco local)
        # m_up: horizontal:vertical en AU; m_down: horizontal:vertical en AD
        "altura_H": 36.96,             # [m]
        "ancho_coronacion": 0.0,       # [m]
        "talud_AU_m": 0.0,             # [m:1]
        "talud_AD_m": 0.8,             # [m:1]
        "inclinacion_cimiento_deg": -6.55, # [°] negativo si contrapendiente
    },

    "MATERIALES": {
        "gamma_hormigon": 27.0,
        "gamma_agua": 9.81,
        "friccion_phi_deg": 35.0,
        "cohesion_kPa": 196.13       # [kPa] cohesión de contacto
    },

    # =========================================================================
    # DEFINICIÓN DIRECTA DE CARGAS POR ESCENARIO
    # =========================================================================
    
    "ESCENARIOS": {
        "N_1": {
            "nombre": "Normal 1",
            "H_AU": 37.97,           # Nivel aguas arriba (m) - NMN + oleaje (1.01 m)
            "H_AD": 0.0,             # Nivel aguas abajo (m)
            "H_AU_subpresion": 36.96,  # Nivel para subpresión AU (m)
            "H_AD_subpresion": 0.0,    # Nivel para subpresión AD (m)
            "eficiencia_dren": 0.00,   # η drenes (0..1)
            "kh": 0.00,              # Coeficiente sísmico horizontal
            "kv": 0.00,              # Coeficiente sísmico vertical
            "p_hielo_kPa": 0.0,      # Presión de hielo (kPa)
            "usar_sismo": False,
            "usar_westergaard": False,
            "usar_hielo": False
        },
        
        "A_1": {
            "nombre": "Accidental 1 - Sismo Proyecto, embalse vacío",
            "H_AU": 0,
            "H_AD": 0.0,
            "H_AU_subpresion": 0.0,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.108,
            "kv": 0.054,
            "p_hielo_kPa": 0.0,
            "usar_sismo": True,
            "usar_westergaard": True,
            "usar_hielo": False
        },
        
        "A_2": {
            "nombre": "Accidental 2 - Avenida de proyecto",
            "H_AU": 39.20,    # NAP + oleaje (0.7 m)
            "H_AD": 0.0,
            "H_AU_subpresion": 36.96,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.00,
            "kv": 0.00,
            "p_hielo_kPa": 0.0,
            "usar_sismo": False,
            "usar_westergaard": False,
            "usar_hielo": False
        },
        
        "A_3": {
            "nombre": "Accidental 3 - Sismo Proyecto, embalse lleno",
            "H_AU": 37.66,     # NMN + oleaje (0.7 m)
            "H_AD": 0.0,
            "H_AU_subpresion": 36.96,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.108,
            "kv": 0.054,
            "p_hielo_kPa": 0.00,
            "usar_sismo": True,
            "usar_westergaard": True,
            "usar_hielo": False
        },
        
        "E_1": {
            "nombre": "Extrema 1 - Sismo extremo, embalse vacío",
            "H_AU": 0.00,
            "H_AD": 0.00,
            "H_AU_subpresion": 0.00,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.245,
            "kv": 0.123,
            "p_hielo_kPa": 0.0,
            "usar_sismo": True,
            "usar_westergaard": True,
            "usar_hielo": False
        },
        
        "E_2": {
            "nombre": "Extrema 2 - Avenida extrema",
            "H_AU": 39.39,    # NAE + oleaje (0.7 m)
            "H_AD": 0.0,
            "H_AU_subpresion": 36.96,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.00,
            "kv": 0.00,
            "p_hielo_kPa": 0.0,
            "usar_sismo": False,
            "usar_westergaard": False,
            "usar_hielo": False
        },
        
        "E_3": {
            "nombre": "Extrema 3 - Sismo extremo, embalse lleno",
            "H_AU": 37.66,    # NMN + oleaje (0.7 m)
            "H_AD": 0.0,
            "H_AU_subpresion": 36.96,
            "H_AD_subpresion": 0.0,
            "eficiencia_dren": 0.00,
            "kh": 0.245,
            "kv": 0.123,
            "p_hielo_kPa": 0.00,
            "usar_sismo": True,
            "usar_westergaard": True,
            "usar_hielo": False
        }
    },

    # Reglas de uso de cohesión por tipo de escenario
    "REGLAS_COHESION": {
        "N_1": "with_c",
        "A_1": "with_c",
        "A_2": "with_c",
        "A_3": "with_c",
        "E_1": "with_c",
        "E_2": "with_c",
        "E_3": "with_c"
    },

    # Factores mínimos requeridos de FS por escenario
    "FACTORES_MINIMOS_FS": {
        "N_1": {"desliz": 1.40, "vuelco": 1.40},
        "A_1": {"desliz": 1.30, "vuelco": 1.30},
        "A_2": {"desliz": 1.30, "vuelco": 1.30},
        "A_3": {"desliz": 1.30, "vuelco": 1.30},
        "E_1": {"desliz": 1.00, "vuelco": 1.00},
        "E_2": {"desliz": 1.00, "vuelco": 1.00},
        "E_3": {"desliz": 1.00, "vuelco": 1.00}
    },

    "SALIDA": {
        "ruta_excel": "resultados.xlsx",
        "ruta_png_base": "fig_",
        "dpi": 280,
        "dibujar_fuerzas": True
    }
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Crear directorio de plots si no existe
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIG["SALIDA"]["ruta_excel"] = os.path.join(SCRIPT_DIR, "resultados.xlsx")
CONFIG["SALIDA"]["ruta_png_base"] = os.path.join(PLOTS_DIR, "fig_")

# =============================================================================
# ============================== CÓDIGO DEL CÁLCULO ===========================
# =============================================================================

@dataclass
class Geometry:
    H: float
    bc: float
    m_up: float
    m_down: float
    beta_deg: float
    toe_ad: tuple
    B_len: float
    poly: np.ndarray

def build_geometry_from_slopes(altura_H, ancho_coronacion, m_up, m_down, beta_deg=0.0) -> Geometry:
    """
    Geometría conforme a:
      - toe AU en (0,0)
      - coro AU: (m_up*H, H)
      - coro AD: (m_up*H + bc, H)
      - toe AD: intersección de cimiento y paramento AD
    """
    H  = float(altura_H)
    bc = float(ancho_coronacion)
    mup, mdown = float(m_up), float(m_down)
    beta = math.radians(float(beta_deg))

    x_cAU = mup * H
    x_cAD = mup * H + bc

    denom = (1.0/mdown) - math.tan(beta)
    if abs(denom) < 1e-12:
        raise ValueError("Paramento AD y línea de cimiento casi paralelos.")
    x_toe = (H + x_cAD/mdown) / denom
    y_toe = -math.tan(beta) * x_toe

    poly = np.array([
        [0.0,   0.0],
        [x_toe, y_toe],
        [x_cAD, H],
        [x_cAU, H]
    ], dtype=float)

    B_len = float(np.hypot(x_toe - 0.0, y_toe - 0.0))

    return Geometry(H=H, bc=bc, m_up=mup, m_down=mdown, beta_deg=float(beta_deg),
                    toe_ad=(x_toe, y_toe), B_len=B_len, poly=poly)

def polygon_area_and_centroid(poly):
    x = np.asarray(poly[:,0], dtype=float)
    y = np.asarray(poly[:,1], dtype=float)
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)

    cross = x*y1 - x1*y
    A2 = np.sum(cross)
    A  = 0.5 * abs(A2)

    if abs(A2) < 1e-12:
        return 0.0, float('nan'), float('nan')

    Cx = np.sum((x + x1) * cross) / (3.0 * A2)
    Cy = np.sum((y + y1) * cross) / (3.0 * A2)

    return A, Cx, Cy

def moment_about_point(F, x, y, xr, yr):
    """Momento escalar de una fuerza F=(Fx,Fy) respecto al punto r=(xr,yr)"""
    Fx, Fy = F
    rx, ry = x - xr, y - yr
    return rx*Fy - ry*Fx

def weight_force(geom: Geometry, gamma_c):
    """Peso propio"""
    A, Cx, Cy = polygon_area_and_centroid(geom.poly)
    W = gamma_c * A
    return np.array([0.0, -W]), (Cx, Cy)

def hydrostatic_upstream(geom: Geometry, gamma_w, H_AU):
    H = max(0.0, H_AU)
    if H <= 0.0:
        return np.array([0.0, 0.0]), (0.0, 0.0)
   
    m_up = float(geom.m_up)
    s = math.sqrt(1.0 + m_up*m_up)

    if H > geom.H:    # nivel de embalse mayor que la cota sup. presa (labio vertedero)
        Rn = 0.5 * gamma_w * H*H * s - 0.5 * gamma_w * s * (H-geom.H) ** 2
    else:
        Rn = 0.5 * gamma_w * H*H * s
        
    n  = np.array([1.0, -m_up], dtype=float) / s
    F  = Rn * n

    y_app = H/3.0
    x_app = m_up * y_app

    return F, (x_app, y_app)

def ice_pressure_force(geom: Geometry, p_hielo_kPa: float, H_AU: float, h_ef_m: float = 1.0):
    """Carga de hielo en la cara de aguas arriba"""
    H = max(0.0, min(float(H_AU), float(geom.H)))
    p = float(p_hielo_kPa)
    if H <= 0.0 or p <= 0.0 or h_ef_m <= 0.0:
        return np.array([0.0, 0.0], dtype=float), (0.0, 0.0)

    Fh = p * float(h_ef_m)
    y_app = H
    x_app = float(geom.m_up) * y_app

    F = np.array([Fh, 0.0], dtype=float)
    return F, (x_app, y_app)

def hydrostatic_downstream(geom: Geometry, gamma_w, H_AD):
    H = max(0.0, min(H_AD, geom.H))
    Hx = 0.5 * gamma_w * H*H
    x_app = geom.toe_ad[0]
    y_app = H/3.0 if H > 0 else 0.0
    return np.array([-Hx, 0.0]), (x_app, y_app)

def uplift_force(geom: Geometry, gamma_w, H_AU, H_AD, eficiencia):
    """Subpresión normal a la base"""
    # En esta versión, se calcula como una ley triangular
    H_AU = max(0.0, min(H_AU, geom.H))
    H_AD = max(0.0, min(H_AD, geom.H))

    p_AU = gamma_w * H_AU
    p_AD = gamma_w * H_AD

    B_len = geom.B_len
    U_raw = 0.5 * (p_AU + p_AD) * B_len
    U = (1.0 - eficiencia) * U_raw

    beta = math.radians(geom.beta_deg)
    #n = np.array([math.sin(beta), math.cos(beta)], dtype=float)
    n = np.array([0.0, 1.0], dtype=float) 
    F = U * n

    if (p_AU + p_AD) > 1e-12:
        s_app = B_len * (p_AU + 2.0*p_AD) / (3.0*(p_AU + p_AD))
    else:
        s_app = 0.5 * B_len

    toe_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]], dtype=float)
    t = toe_vec / (B_len if B_len > 0 else 1.0)
    x_app, y_app = (t * s_app).tolist()

    return F, (x_app, y_app)

# def seismic_inertial_force(W_vector, kh, kv, cog):
#     """Fuerza inercial pseudoestática"""
#     W = -W_vector[1]
#     Fh = kh * W
#     Fv = -kv * W
#     return np.array([Fh, Fv]), cog

def seismic_inertial_force(W_vector, kh, kv, cog):
    """Fuerza inercial pseudoestática"""
    W = -W_vector[1]     # W_vector es negativo
    Fh = kh * W
    Fv = kv * W
    return np.array([Fh, Fv]), cog

def westergaard_force(geom: Geometry, gamma_w, H_AU, kh):
    """Fuerza hidrodinámica (Westergaard)"""
    H = max(0.0, min(float(H_AU), float(geom.H)))
    if H <= 0.0 or kh == 0.0:
        return np.array([0.0, 0.0]), (0.0, 0.0)

    Cw = 7.0 / 12.0
    Wh = Cw * float(gamma_w) * float(kh) * H * H

    y_app = 0.4 * H
    x_app = float(geom.m_up) * y_app

    F = np.array([Wh, 0.0], dtype=float)
    return F, (x_app, y_app)

def combine_actions(actions):
    """Suma fuerzas y momentos"""
    Fx = Fy = M_toe = 0.0
    xr = actions["_xr"]; yr = actions["_yr"]
    for k, val in actions.items():
        if k.startswith("_"): 
            continue
        F, (x, y) = val
        Fx += F[0]; Fy += F[1]
        M_toe += moment_about_point(F, x, y, xr, yr)
    return np.array([Fx, Fy], dtype=float), M_toe

def base_resultant_and_stress(geom: Geometry, R, M_toeAD):
    """Proyección sobre la base y cálculo de tensiones"""
    Fx, Fy = R
    beta = math.radians(geom.beta_deg)

    t = np.array([math.cos(beta), -math.sin(beta)])
    n = np.array([math.sin(beta),  math.cos(beta)])

    T = Fx*t[0] + Fy*t[1]
    N = -(Fx*n[0] + Fy*n[1])

    B = geom.B_len
    
    if abs(N) < 1e-12:
        e = float('inf')
        sigma_AU = float('-inf')
        sigma_AD = float('-inf')
        sigma_min = float('-inf')
        sigma_max = float('inf')
        no_traccion = False
    else:
        s = M_toeAD / N
        e = s - B/2.0
        sigma_med = N / B
    
        sigma_AU = sigma_med * (1.0 + 6.0*e/B)
        sigma_AD = sigma_med * (1.0 - 6.0*e/B)
    
        sigma_min = min(sigma_AU, sigma_AD)
        sigma_max = max(sigma_AU, sigma_AD)
    
        no_traccion = (sigma_AU >= 0.0) and (sigma_AD >= 0.0)
    
    return {
        "N_kN": N, "T_kN": T, "e_m": e,
        "sigma_min_kPa": sigma_min, "sigma_max_kPa": sigma_max,
        "sigma_AU_kPa": sigma_AU, "sigma_AD_kPa": sigma_AD,
        "no_traccion": no_traccion
    }

def sliding_FS(N, T, phi_deg, c_kPa, B_contacto,
               sigma_AU_kPa=None, sigma_AD_kPa=None,
               usar_cohesion=True, anular_cohesion_si_traccion=True):
    """Factor de seguridad al deslizamiento"""
    demand = abs(float(T))
    if demand <= 1e-12:
        return float('inf'), float(N)*math.tan(math.radians(phi_deg)) + (float(c_kPa)*float(B_contacto) if usar_cohesion else 0.0), 0.0

    phi = math.radians(float(phi_deg))
    Rf = float(N) * math.tan(phi)

    B_eff = float(B_contacto)
    hay_traccion = False
    if (sigma_AU_kPa is not None) and (sigma_AD_kPa is not None):
        sigma_AU = float(sigma_AU_kPa)
        sigma_AD = float(sigma_AD_kPa)
        if sigma_AU < 0.0 and sigma_AD > 0.0:
            if sigma_AD > 1e-12:
                B_eff = B_contacto - (-sigma_AU * B_contacto) / (sigma_AD - sigma_AU)
                #B_eff = 2.0*float(N) / sigma_AU
                hay_traccion = True
        elif sigma_AD < 0.0 and sigma_AU > 0.0:
            if sigma_AU > 1e-12:
                B_eff = (-sigma_AD * B_contacto) / (sigma_AU - sigma_AD)
                hay_traccion = True
        elif sigma_AU < 0.0 and sigma_AD < 0.0:
            B_eff = 0.0
            hay_traccion = True

        B_eff = max(0.0, min(B_eff, float(B_contacto)))
    if usar_cohesion and not (anular_cohesion_si_traccion and hay_traccion):
        Rc = float(c_kPa) * (B_eff)
    else:
        Rc = 0.0

    R_cap = Rf + Rc
    FS = R_cap / demand if demand > 1e-12 else float('inf')
    L_grieta = B_contacto - B_eff
    return FS, R_cap, B_eff, L_grieta
    # return FS, R_cap, demand

def overturning_FS(M_stab, M_destab):
    """Factor de seguridad al vuelco"""
    if M_destab <= 1e-9:
        return np.inf
    return M_stab / M_destab

def split_moments_by_sign(actions, xr, yr):
    Mplus = Mminus = 0.0
    for k, val in actions.items():
        if k.startswith("_"): 
            continue
        F, (x, y) = val
        M = moment_about_point(F, x, y, xr, yr)
        if M >= 0: Mplus += M
        else:      Mminus += -M
    return Mplus, Mminus

def plot_dam_and_resultant(geom: Geometry, R_vec, e_m, title, ruta_png, dpi=140):
    """Dibuja el perfil de la presa y la resultante"""
    plt.figure()
    poly = geom.poly.copy()
    closed = np.vstack([poly, poly[0]])
    plt.plot(closed[:,0], closed[:,1], 'b-', linewidth=2, label="Perfil")
    
    plt.plot([0, geom.toe_ad[0]], [0, geom.toe_ad[1]], 'k--', linewidth=1)
    
    base_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]])
    base_len = geom.B_len
    base_unit = base_vec / base_len
    
    mid_point = 0.5 * base_vec
    offset = e_m * base_unit
    app_point = mid_point + offset
    
    R = np.array(R_vec)
    R_mag = np.linalg.norm(R)
    if R_mag > 1e-6:
        scale = 0.2 * max(geom.H, geom.B_len)
        R_unit = R / R_mag
        
        p1 = app_point - scale * R_unit
        p2 = app_point + scale * R_unit
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, 
                label=f'R = {R_mag:.1f} kN/m')
        
        plt.arrow(app_point[0], app_point[1],
                scale * 0.5 * R_unit[0], scale * 0.5 * R_unit[1],
                head_width=0.02*scale, head_length=0.05*scale,
                fc='r', ec='r')
        
        plt.plot(app_point[0], app_point[1], 'ro')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    
    plt.savefig(ruta_png, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_base_pressures(geom: Geometry, N_kN, e_m, title, ruta_png, dpi=140):
    """Dibuja el diagrama de presiones en la base"""
    B = geom.B_len
    if B <= 0:
        return

    sigma_med = N_kN / B
    sigma_AU = sigma_med * (1.0 + 6.0 * e_m / B)
    sigma_AD = sigma_med * (1.0 - 6.0 * e_m / B)

    s = np.array([0.0, B])
    sig = np.array([sigma_AU, sigma_AD])

    plt.figure()
    plt.plot(s, sig, linewidth=2)
    plt.axhline(0, linestyle="--")
    plt.xlabel("s a lo largo de base (m)  [0=toe AU → B=toe AD]")
    plt.ylabel("σ normal (kPa)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=dpi)
    plt.close()

def plot_profile_with_forces(geom: Geometry, actions: dict, R_vec, e_m, title, ruta_png, dpi=140):
    """Dibuja el perfil, las fuerzas individuales y la resultante"""
    plt.figure()
    poly = geom.poly.copy()
    closed = np.vstack([poly, poly[0]])
    plt.plot(closed[:,0], closed[:,1], linewidth=2, label="Perfil")
    plt.plot([0.0, geom.toe_ad[0]], [0.0, geom.toe_ad[1]], 
             linestyle="--", linewidth=1.2, label="Base")

    mags = [np.linalg.norm(val[0]) for k, val in actions.items() if not k.startswith("_")]
    mags.append(np.linalg.norm(R_vec))
    mag_max = max(mags) if mags else 1.0
    
    L_ref = max(geom.H, geom.B_len)
    arrow_scale = 0.15 * L_ref

    color_map = {
        "W": "#1f77b4", "H": "#ff7f0e", "H_AD": "#ffbb78",
        "U": "#2ca02c", "S": "#d62728", "Wh": "#9467bd", "I": "#17becf"
    }
    
    for k, val in actions.items():
        if k.startswith("_"):
            continue
        F, (x0, y0) = val
        F_mag = np.linalg.norm(F)
        if F_mag < 1e-12:
            continue
            
        F_unit = F / F_mag
        L = arrow_scale * (F_mag / mag_max)**0.8
        x1, y1 = x0 + F_unit[0]*L, y0 + F_unit[1]*L
        
        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=2, 
                                  color=color_map.get(k, None)))
        plt.text(x1, y1, f" {k} ({F_mag:.1f})", fontsize=8, va="center")

    R_mag = np.linalg.norm(R_vec)
    if R_mag > 1e-12:
        base_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]])
        base_unit = base_vec / geom.B_len
        
        mid_point = 0.5 * base_vec
        offset = e_m * base_unit
        app_point = mid_point + offset
        x_cross, y_cross = app_point[0], app_point[1]
        
        R_unit = R_vec / R_mag
        scale = 0.2 * L_ref
        
        p1 = np.array([x_cross, y_cross]) - scale * R_unit
        p2 = np.array([x_cross, y_cross]) + scale * R_unit
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, 
                label=f'R = {R_mag:.1f} kN/m')
        
        plt.annotate("", xy=(p2[0], p2[1]), xytext=(x_cross, y_cross),
                    arrowprops=dict(arrowstyle="->", lw=2, color='red'))
        
        plt.plot(x_cross, y_cross, 'ro', markersize=8)

    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.4)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=dpi, bbox_inches='tight')
    plt.close()

def augment_summary_with_actions(resumen: dict, actions: dict, geom: Geometry, R_vec, e_m):
    """Añade información de acciones individuales al resumen"""
    beta = math.radians(geom.beta_deg)
    t_vec = np.array([math.cos(beta), -math.sin(beta)], dtype=float)
    n_vec = np.array([math.sin(beta),  math.cos(beta)], dtype=float)

    # Resultante y punto de cruce
    Rmag = float(np.linalg.norm(R_vec))
    B = geom.B_len
    s_cross = np.clip(B/2.0 + (e_m if np.isfinite(e_m) else 0.0), 0.0, B)
    toe_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]], dtype=float)
    t_base = toe_vec / (B if B > 0 else 1.0)
    xR, yR = (t_base * s_cross).tolist()
    resumen.update({"R_kN": Rmag, "R_x_m": xR, "R_y_m": yR})

    wanted = ["W","H","H_AD","U","S","Wh","I"]
    for k in wanted:
        if k not in actions:
            resumen[f"{k}_kN"]=0.0; resumen[f"{k}_x_m"]=0.0; resumen[f"{k}_y_m"]=0.0
            resumen[f"{k}_T_kN"]=0.0; resumen[f"{k}_N_kN"]=0.0
        else:
            F, (x, y) = actions[k]
            mag = float(np.linalg.norm(F))
            Tk  = float(np.array(F) @ t_vec)
            Nk  = float(-np.array(F) @ n_vec)
            resumen[f"{k}_kN"]=mag; resumen[f"{k}_x_m"]=float(x); resumen[f"{k}_y_m"]=float(y)
            resumen[f"{k}_T_kN"]=Tk; resumen[f"{k}_N_kN"]=Nk
    return resumen

def compute_for_scenario(geom: Geometry, cfg: dict, scen_name: str):
    """Calcula estabilidad para un escenario específico"""
    gamma_c = cfg["MATERIALES"]["gamma_hormigon"]
    gamma_w = cfg["MATERIALES"]["gamma_agua"]
    phi_deg = cfg["MATERIALES"]["friccion_phi_deg"]
    c_kPa   = cfg["MATERIALES"]["cohesion_kPa"]

    # Obtener parámetros del escenario
    escenario = cfg["ESCENARIOS"][scen_name]
    
    H_AU = float(escenario.get("H_AU", 0.0))
    H_AD = float(escenario.get("H_AD", 0.0))
    H_AU_subp = float(escenario.get("H_AU_subpresion", H_AU))
    H_AD_subp = float(escenario.get("H_AD_subpresion", H_AD))
    eta = float(escenario.get("eficiencia_dren", 0.0))
    kh = float(escenario.get("kh", 0.0))
    kv = float(escenario.get("kv", 0.0))
    p_hielo = float(escenario.get("p_hielo_kPa", 0.0))
    
    usar_sismo = escenario.get("usar_sismo", False)
    usar_wh = escenario.get("usar_westergaard", False)
    usar_hielo = escenario.get("usar_hielo", False)

    # Construcción de acciones
    actions = {}
    
    # Peso propio (siempre activo)
    F_W, (xW, yW) = weight_force(geom, gamma_c)
    actions["W"] = (F_W, (xW, yW))
    
    # Hidrostática AU
    if H_AU > 0:
        F_H, (xH, yH) = hydrostatic_upstream(geom, gamma_w, H_AU)
        if np.linalg.norm(F_H) > 0:
            actions["H"] = (F_H, (xH, yH))
    
    # Hidrostática AD
    if H_AD > 0:
        F_HAD, (xHAD, yHAD) = hydrostatic_downstream(geom, gamma_w, H_AD)
        if np.linalg.norm(F_HAD) > 0:
            actions["H_AD"] = (F_HAD, (xHAD, yHAD))
    
    # Subpresión
    F_U, (xU, yU) = uplift_force(geom, gamma_w, H_AU_subp, H_AD_subp, eta)
    if np.linalg.norm(F_U) > 0:
        actions["U"] = (F_U, (xU, yU))
    
    # Sismo pseudoestático
    if usar_sismo and (kh > 0 or kv > 0):
        A, Cx, Cy = polygon_area_and_centroid(geom.poly)
        F_S, (xS, yS) = seismic_inertial_force(np.array([0.0, -gamma_c*A]), kh, kv, (Cx, Cy))
        if np.linalg.norm(F_S) > 0:
            actions["S"] = (F_S, (xS, yS))
    
    # Westergaard
    if usar_wh and kh > 0:
        F_Wh, (xWh, yWh) = westergaard_force(geom, gamma_w, H_AU, kh)
        if np.linalg.norm(F_Wh) > 0:
            actions["Wh"] = (F_Wh, (xWh, yWh))
    
    # Hielo
    if usar_hielo and p_hielo > 0:
        h_ef_m = 1.0  # Altura efectiva de contacto
        F_I, (xI, yI) = ice_pressure_force(geom, p_hielo, H_AU, h_ef_m=h_ef_m)
        if np.linalg.norm(F_I) > 0:
            actions["I"] = (F_I, (xI, yI))

    # Punto de referencia para momentos (pie AD)
    actions["_xr"], actions["_yr"] = geom.toe_ad

    # Resultante y momentos
    R, M_toeAD = combine_actions(actions)

    # Proyección y tensiones en base
    base = base_resultant_and_stress(geom, R, M_toeAD)

    # FS deslizamiento
    usar_c = (cfg["REGLAS_COHESION"].get(scen_name, "with_c") == "with_c")
    FS_desliz, Rcap, B_eff, L_grieta = sliding_FS(
        base["N_kN"], base["T_kN"],
        phi_deg, c_kPa, geom.B_len,
        sigma_AU_kPa=base.get("sigma_AU_kPa"),
        sigma_AD_kPa=base.get("sigma_AD_kPa"),
        usar_cohesion=usar_c,
        anular_cohesion_si_traccion=False
    )
    
    # FS vuelco
    xr, yr = geom.toe_ad
    Mplus, Mminus = split_moments_by_sign(actions, xr, yr)
    FS_vuelco = (Mplus/Mminus) if Mminus > 1e-12 else float("inf")

    # FS mínimos requeridos
    FS_req = cfg["FACTORES_MINIMOS_FS"].get(scen_name, {"desliz": 1.0, "vuelco": 1.0})
    FS_req_desliz = FS_req["desliz"]
    FS_req_vuelco = FS_req["vuelco"]

    # Generación de gráficos
    ruta_base = cfg["SALIDA"].get("ruta_png_base", "fig_")
    dpi = int(cfg["SALIDA"].get("dpi", 140))
    out_dir = os.path.dirname(ruta_base)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    png_pres = f"{ruta_base}{scen_name}_presiones.png"
    png_perf = f"{ruta_base}{scen_name}_perfil.png"
    png_forc = f"{ruta_base}{scen_name}_fuerzas.png"

    titulo = f"Escenario: {scen_name} - {escenario.get('nombre', '')}"

    # 1) Diagrama de presiones
    try:
        plot_base_pressures(geom, base["N_kN"], base["e_m"], titulo, png_pres, dpi=dpi)
    except Exception as e:
        print(f"[{scen_name}] Aviso: no se pudo generar 'presiones': {e}")
        png_pres = ""

    # 2) Perfil + resultante
    try:
        plot_dam_and_resultant(geom, R, base["e_m"], titulo, png_perf, dpi=dpi)
    except Exception as e:
        print(f"[{scen_name}] Aviso: no se pudo generar 'perfil': {e}")
        png_perf = ""

    # 3) Perfil + fuerzas
    dibujar_fuerzas = bool(cfg["SALIDA"].get("dibujar_fuerzas", True))
    if dibujar_fuerzas:
        try:
            plot_profile_with_forces(geom, actions, R, base["e_m"], titulo, png_forc, dpi=dpi)
        except Exception as e:
            print(f"[{scen_name}] Aviso: no se pudo generar 'fuerzas': {e}")
            png_forc = ""
    else:
        png_forc = ""

    # Resumen
    resumen = {
        "Escenario": scen_name,
        "Nombre": escenario.get("nombre", ""),
        "H_AU_m": H_AU,
        "H_AD_m": H_AD,
        "kh": kh,
        "kv": kv,
        "p_hielo_kPa": p_hielo,
        "B_m_contacto": geom.B_len,
        "L_comp": B_eff,
        "L_grieta": L_grieta,
        "N_kN": base["N_kN"], 
        "T_kN": base["T_kN"], 
        "e_m": base["e_m"], 
        "R_cap": Rcap,
        "sigma_AU_kPa": base["sigma_AU_kPa"],
        "sigma_AD_kPa": base["sigma_AD_kPa"],
        "sigma_min_kPa": base["sigma_min_kPa"], 
        "sigma_max_kPa": base["sigma_max_kPa"],
        "no_traccion": base["no_traccion"],
        "FS_desliz": FS_desliz, 
        "FS_vuelco": FS_vuelco,
        "FS_req_desliz": FS_req_desliz, 
        "FS_req_vuelco": FS_req_vuelco,
        "cumple_desliz": FS_desliz >= FS_req_desliz,
        "cumple_vuelco": FS_vuelco >= FS_req_vuelco,
        "PNG_presiones": png_pres,
        "PNG_perfil": png_perf,
        "PNG_fuerzas": png_forc
    }

    # Añadir información de acciones individuales
    resumen = augment_summary_with_actions(resumen, actions, geom, R, base["e_m"])

    return resumen

def main(CONFIG):
    """Función principal"""
    # 1) Construir geometría
    G = build_geometry_from_slopes(
        altura_H=CONFIG["GEOMETRIA"]["altura_H"],
        ancho_coronacion=CONFIG["GEOMETRIA"]["ancho_coronacion"],
        m_up=CONFIG["GEOMETRIA"]["talud_AU_m"],
        m_down=CONFIG["GEOMETRIA"]["talud_AD_m"],
        beta_deg=CONFIG["GEOMETRIA"]["inclinacion_cimiento_deg"]
    )

    print(f"Geometría construida: H={G.H:.2f} m, B={G.B_len:.2f} m")
    print(f"Pie AD en: ({G.toe_ad[0]:.2f}, {G.toe_ad[1]:.2f})")
    print(f"\nCalculando {len(CONFIG['ESCENARIOS'])} escenarios...")

    # 2) Ejecutar todos los escenarios
    resultados = []
    for scen_name in CONFIG["ESCENARIOS"].keys():
        print(f"  - Procesando {scen_name}...")
        res = compute_for_scenario(G, CONFIG, scen_name)
        resultados.append(res)

    # 3) Crear DataFrame
    df = pd.DataFrame(resultados)

    # Ordenar columnas
    cols_pref = [c for c in [
        "Escenario", "Nombre",
        "H_AU_m", "H_AD_m", "kh", "kv", "p_hielo_kPa",
        "B_m_contacto", "L_comp", "L_grieta",
        "N_kN","T_kN","e_m", "R_cap",
        "sigma_AU_kPa","sigma_AD_kPa","sigma_min_kPa","sigma_max_kPa","no_traccion",
        "FS_desliz","FS_req_desliz","cumple_desliz",
        "FS_vuelco","FS_req_vuelco","cumple_vuelco",
        "R_kN","R_x_m","R_y_m",
        "W_kN","W_x_m","W_y_m","W_T_kN","W_N_kN",
        "H_kN","H_x_m","H_y_m","H_T_kN","H_N_kN",
        "H_AD_kN","H_AD_x_m","H_AD_y_m","H_AD_T_kN","H_AD_N_kN",
        "U_kN","U_x_m","U_y_m","U_T_kN","U_N_kN",
        "S_kN","S_x_m","S_y_m","S_T_kN","S_N_kN",
        "Wh_kN","Wh_x_m","Wh_y_m","Wh_T_kN","Wh_N_kN",
        "I_kN","I_x_m","I_y_m","I_T_kN","I_N_kN",
        "PNG_presiones", "PNG_perfil", "PNG_fuerzas"
    ] if c in df.columns]
    
    if cols_pref:
        df = df[cols_pref]

    # 4) Guardar Excel
    ruta_xlsx = CONFIG["SALIDA"]["ruta_excel"]
    try:
        with pd.ExcelWriter(ruta_xlsx, engine="xlsxwriter") as xw:
            df.to_excel(xw, index=False, sheet_name="Resultados")
        print(f"\n✓ Resultados guardados en: {ruta_xlsx}")
    except Exception as e:
        ruta_csv = ruta_xlsx.rsplit(".", 1)[0] + ".csv"
        df.to_csv(ruta_csv, index=False)
        print(f"\n⚠ No se pudo escribir Excel ({e})")
        print(f"✓ Resultados guardados en CSV: {ruta_csv}")

    # 5) Resumen en consola
    print("\n" + "="*80)
    print("RESUMEN DE FACTORES DE SEGURIDAD")
    print("="*80)
    for _, row in df.iterrows():
        print(f"\n{row['Escenario']} - {row['Nombre']}")
        print(f"  FS deslizamiento: {row['FS_desliz']:.2f} (req: {row['FS_req_desliz']:.2f}) {'✓' if row['cumple_desliz'] else '✗'}")
        print(f"  FS vuelco:        {row['FS_vuelco']:.2f} (req: {row['FS_req_vuelco']:.2f}) {'✓' if row['cumple_vuelco'] else '✗'}")
        print(f"  Excentricidad:    {row['e_m']:.3f} m")
        print(f"  Sin tracción:     {'Sí' if row['no_traccion'] else 'No'}")

    print("\n" + "="*80)
    print(f"✓ Gráficos guardados en: {PLOTS_DIR}")
    print("="*80)

if __name__ == "__main__":
    main(CONFIG)

