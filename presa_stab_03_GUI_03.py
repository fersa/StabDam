# -*- coding: utf-8 -*-
"""
GUI Streamlit para Análisis de Estabilidad de Presas de Gravedad
Permite configurar parámetros y calcular factores de seguridad por escenarios
"""

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import io

# =============================================================================
# CLASES Y FUNCIONES DE CÁLCULO
# =============================================================================

@dataclass
class Geometry:
    H: float
    # bc: float
    m_up: float
    m_down: float
    beta_deg: float
    toe_ad: tuple
    B_len: float
    poly: np.ndarray

def build_geometry_from_slopes(altura_H, m_up, m_down, beta_deg=0.0) -> Geometry:
    # def build_geometry_from_slopes(altura_H, ancho_coronacion, m_up, m_down, beta_deg=0.0) -> Geometry:
    H = float(altura_H)
    # bc = float(ancho_coronacion)
    mup, mdown = float(m_up), float(m_down)
    beta = math.radians(float(beta_deg))

    x_cAU = mup * H
    x_cAD = mup * H #+ bc

    denom = (1.0/mdown) - math.tan(beta)
    if abs(denom) < 1e-12:
        raise ValueError("Paramento AD y línea de cimiento casi paralelos.")
    x_toe = (H + x_cAD/mdown) / denom
    y_toe = -math.tan(beta) * x_toe

    poly = np.array([
        [0.0, 0.0],
        [x_toe, y_toe],
        [x_cAD, H],
        [x_cAU, H]
    ], dtype=float)

    B_len = float(np.hypot(x_toe - 0.0, y_toe - 0.0))

    return Geometry(H=H, m_up=mup, m_down=mdown, beta_deg=float(beta_deg),
                    toe_ad=(x_toe, y_toe), B_len=B_len, poly=poly)
    # return Geometry(H=H, bc=bc, m_up=mup, m_down=mdown, beta_deg=float(beta_deg),
                    # toe_ad=(x_toe, y_toe), B_len=B_len, poly=poly)

def polygon_area_and_centroid(poly):
    x = np.asarray(poly[:,0], dtype=float)
    y = np.asarray(poly[:,1], dtype=float)
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)

    cross = x*y1 - x1*y
    A2 = np.sum(cross)
    A = 0.5 * abs(A2)

    if abs(A2) < 1e-12:
        return 0.0, float('nan'), float('nan')

    Cx = np.sum((x + x1) * cross) / (3.0 * A2)
    Cy = np.sum((y + y1) * cross) / (3.0 * A2)  

    return A, Cx, Cy

def moment_about_point(F, x, y, xr, yr):
    Fx, Fy = F
    rx, ry = x - xr, y - yr
    return rx*Fy - ry*Fx

def weight_force(geom: Geometry, gamma_c):
    A, Cx, Cy = polygon_area_and_centroid(geom.poly)
    W = gamma_c * A
    return np.array([0.0, -W]), (Cx, Cy)

def hydrostatic_upstream(geom: Geometry, gamma_w, H_AU):
    H = max(0.0, H_AU)
    if H <= 0.0:
        return np.array([0.0, 0.0]), (0.0, 0.0)
   
    m_up = float(geom.m_up)
    s = math.sqrt(1.0 + m_up*m_up)

    if H > geom.H:
        Rn = 0.5 * gamma_w * H*H * s - 0.5 * gamma_w * s * (H-geom.H) ** 2
    else:
        Rn = 0.5 * gamma_w * H*H * s
        
    n = np.array([1.0, -m_up], dtype=float) / s
    F = Rn * n

    y_app = H/3.0
    x_app = m_up * y_app

    return F, (x_app, y_app)

def ice_pressure_force(geom: Geometry, p_hielo_kPa: float, H_AU: float, h_ef_m: float = 1.0):
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

def uplift_force(geom: Geometry, gamma_w, H_AU, H_AD, eficiencia, L_grieta=0.0):
    """
    Calcula la fuerza de subpresión considerando posible zona de tracción.
    - Si no hay tracción (L_grieta = 0): ley triangular clásica.
    - Si hay tracción: presión constante p_AU en zona traccionada y triangular en zona comprimida.
    """

    import math
    H_AU = max(0.0, min(H_AU, geom.H))
    H_AD = max(0.0, min(H_AD, geom.H))

    p_AU = gamma_w * H_AU
    p_AD = gamma_w * H_AD
    eficiencia = min(max(eficiencia, 0.0), 1.0)
    B_len = geom.B_len

    # === Caso 1: base totalmente comprimida ===
    if L_grieta <= 1e-6:
        U_raw = 0.5 * (p_AU + p_AD) * B_len
        U = (1.0 - eficiencia) * U_raw

        n = np.array([0.0, 1.0], dtype=float)
        F = U * n

        if (p_AU + p_AD) > 1e-12:
            s_app = B_len * (p_AU + 2.0*p_AD) / (3.0*(p_AU + p_AD))
        else:
            s_app = 0.5 * B_len

    # === Caso 2: hay zona traccionada ===
    else:
        L_comp = max(B_len - L_grieta, 0.0)

        area_traccion = p_AU * L_grieta
        area_comp = 0.5 * (p_AU + p_AD) * L_comp

        U_raw = area_traccion + area_comp
        U = (1.0 - eficiencia) * U_raw
        n = np.array([0.0, 1.0], dtype=float)
        F = U * n

        # Centroide compuesto
        if U_raw > 1e-9:
            x_rect = 0.5 * L_grieta
            if (p_AU + p_AD) > 1e-12:
                x_trap = L_grieta + L_comp * (p_AU + 2.0*p_AD) / (3.0*(p_AU + p_AD))
            else:
                x_trap = L_grieta + 0.5 * L_comp
            s_app = (area_traccion * x_rect + area_comp * x_trap) / U_raw
        else:
            s_app = 0.5 * B_len

    toe_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]], dtype=float)
    t = toe_vec / (B_len if B_len > 0 else 1.0)
    x_app, y_app = (t * s_app).tolist()

    return F, (x_app, y_app)

def seismic_inertial_force(W_vector, kh, kv, cog):
    W = -W_vector[1]
    Fh = kh * W
    Fv = kv * W
    return np.array([Fh, Fv]), cog

def westergaard_force(geom: Geometry, gamma_w, H_AU, kh):
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
    Fx, Fy = R
    beta = math.radians(geom.beta_deg)

    t = np.array([math.cos(beta), -math.sin(beta)])
    n = np.array([math.sin(beta), math.cos(beta)])

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
    demand = abs(float(T))
    if demand <= 1e-12:
        return float('inf'), float(N)*math.tan(math.radians(phi_deg)) + (float(c_kPa)*float(B_contacto) if usar_cohesion else 0.0), 0.0, 0.0

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

def split_moments_by_sign(actions, xr, yr):
    Mplus = Mminus = 0.0
    for k, val in actions.items():
        if k.startswith("_"): 
            continue
        F, (x, y) = val
        M = moment_about_point(F, x, y, xr, yr)
        if M >= 0: Mplus += M
        else: Mminus += -M
    return Mplus, Mminus

def plot_profile_with_forces(geom: Geometry, actions: dict, R_vec, e_m, title):
    """Dibuja el perfil, las fuerzas individuales y la resultante"""
    fig, ax = plt.subplots(figsize=(12, 10))
    poly = geom.poly.copy()
    closed = np.vstack([poly, poly[0]])
    ax.plot(closed[:,0], closed[:,1], linewidth=2, label="Perfil", color='blue')
    ax.plot([0.0, geom.toe_ad[0]], [0.0, geom.toe_ad[1]], 
             linestyle="--", linewidth=1.2, label="Base", color='black')

    # Calcular escala para flechas
    mags = [np.linalg.norm(val[0]) for k, val in actions.items() if not k.startswith("_")]
    mags.append(np.linalg.norm(R_vec))
    mag_max = max(mags) if mags else 1.0
    
    L_ref = max(geom.H, geom.B_len)
    arrow_scale = 0.15 * L_ref

    # Colores para cada tipo de fuerza
    color_map = {
        "W": "#1f77b4", "H": "#ff7f0e", "H_AD": "#ffbb78",
        "U": "#2ca02c", "S": "#d62728", "Wh": "#9467bd", "I": "#17becf"
    }
    
    # Dibujar fuerzas individuales
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
        
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=2, 
                                  color=color_map.get(k, 'gray')))
        ax.text(x1, y1, f" {k}\n({F_mag:.1f})", fontsize=9, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Dibujar resultante
    R_mag = np.linalg.norm(R_vec)
    if R_mag > 1e-12:
        base_vec = np.array([geom.toe_ad[0], geom.toe_ad[1]])
        base_unit = base_vec / geom.B_len
        
        mid_point = 0.5 * base_vec
        offset = e_m * base_unit if np.isfinite(e_m) else np.array([0.0, 0.0])
        app_point = mid_point + offset
        x_cross, y_cross = app_point[0], app_point[1]
        
        R_unit = R_vec / R_mag
        scale = 0.2 * L_ref
        
        p1 = np.array([x_cross, y_cross]) - scale * R_unit
        p2 = np.array([x_cross, y_cross]) + scale * R_unit
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3, 
                label=f'R = {R_mag:.1f} kN/m', alpha=0.8)
        
        ax.annotate("", xy=(p2[0], p2[1]), xytext=(x_cross, y_cross),
                    arrowprops=dict(arrowstyle="->", lw=3, color='red'))
        
        ax.plot(x_cross, y_cross, 'ro', markersize=10, label='Punto aplicación')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    return fig

def plot_base_pressures(
    geom,
    N_kN,
    e_m,
    title,
    ruta_png=None,
    dpi=140,
    gamma_w=None,
    H_AU=None,
    H_AD=None,
    eficiencia=0.0,
    L_grieta=0.0
):
    """
    Dibuja el diagrama de tensiones normales, subpresión y tensión neta.
    Devuelve figura si ruta_png es None (modo Streamlit).
    """

    import matplotlib.pyplot as plt
    B = geom.B_len
    if B <= 0:
        return None

    s_vals = np.linspace(0, B, 100)

    # Tensiones transmitidas
    sigma_med = N_kN / B
    sigma_AU = sigma_med * (1.0 + 6.0 * e_m / B)
    sigma_AD = sigma_med * (1.0 - 6.0 * e_m / B)
    sigma_transmitida = - (sigma_AU + (sigma_AD - sigma_AU) * (s_vals / B))

    # Subpresión
    uplift = np.zeros_like(s_vals)
    if gamma_w is not None and H_AU is not None and H_AD is not None:
        H_AU = max(0.0, min(H_AU, geom.H))
        H_AD = max(0.0, min(H_AD, geom.H))
        p_AU = gamma_w * H_AU
        p_AD = gamma_w * H_AD
        eficiencia = min(max(eficiencia, 0.0), 1.0)

        if L_grieta <= 1e-6:
            uplift = (p_AU + (p_AD - p_AU) * (s_vals / B))
        else:
            L_comp = B - L_grieta
            for i, s in enumerate(s_vals):
                if s <= L_grieta:
                    uplift[i] = p_AU
                else:
                    t = (s - L_grieta) / L_comp if L_comp > 0 else 0.0
                    uplift[i] = p_AU + t * (p_AD - p_AU)
        uplift *= (1 - eficiencia)
    uplift_plot = -uplift  # 👉 Mostrar subpresión como tensión negativa
    # sigma_neta = sigma_transmitida - uplift

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(s_vals, sigma_transmitida, label="σ (compresión negativa)", linewidth=2)
    if gamma_w is not None:
        ax.plot(s_vals, uplift_plot, label="Subpresión", linestyle="--", linewidth=2)
        # ax.plot(s_vals, sigma_neta, label="σ neta", linestyle=":", linewidth=2)

    ax.axhline(0, color="gray", linestyle=":")
    # === Sombreado de la zona en compresión (σ < 0) ===
    ax.fill_between(
        s_vals,
        sigma_transmitida,
        0,
        where=(sigma_transmitida < 0),
        color="lightblue",
        alpha=0.4,
        interpolate=True,
        label="Zona en compresión"
    )
    ax.set_xlabel("s a lo largo de la base (m)")
    ax.set_ylabel("Tensión normal (kPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    # Salida: figura o archivo
    if ruta_png:
        fig.savefig(ruta_png, dpi=dpi)
        plt.close(fig)
        return ruta_png
    else:
        return fig



def compute_for_scenario(geom: Geometry, config_mat, escenario_data, usar_c=True):
    gamma_c = config_mat["gamma_hormigon"]
    gamma_w = config_mat["gamma_agua"]
    phi_deg = config_mat["friccion_phi_deg"]
    c_kPa = config_mat["cohesion_kPa"]

    H_AU = float(escenario_data.get("H_AU", 0.0))
    H_AD = float(escenario_data.get("H_AD", 0.0))
    altura_ola = float(escenario_data.get("altura_ola_m", 0.0))

    # ✅ Efecto del oleaje: se suma a H_AU
    H_AU_total = H_AU + altura_ola
    H_AU_subp = float(escenario_data.get("H_AU_subpresion", H_AU))
    H_AD_subp = float(escenario_data.get("H_AD_subpresion", H_AD))
    eta = float(escenario_data.get("eficiencia_dren", 0.0))
    kh = float(escenario_data.get("kh", 0.0))
    kv = float(escenario_data.get("kv", 0.0))
    p_hielo = float(escenario_data.get("p_hielo_kPa", 0.0))
    
    usar_sismo = escenario_data.get("usar_sismo", False)
    usar_wh = escenario_data.get("usar_westergaard", False)
    usar_hielo = escenario_data.get("usar_hielo", False)

    actions = {}
    
    # === Carga puntual de coronación ===
    P_cor = config_mat.get("P_cor_kN_m", 0.0)
    x_cor = config_mat.get("x_cor_m", 0.0)
    if P_cor > 0.0:
        F_cor = np.array([0.0, -P_cor], dtype=float)
        # La coronación está en la parte superior del paramento aguas arriba
        # Coordenadas: (x_cor, geom.H)
        actions["Pcor"] = (F_cor, (x_cor, geom.H))


    F_W, (xW, yW) = weight_force(geom, gamma_c)
    actions["W"] = (F_W, (xW, yW))
    
    if H_AU_total > 0:
        F_H, (xH, yH) = hydrostatic_upstream(geom, gamma_w, H_AU_total)
        if np.linalg.norm(F_H) > 0:
            actions["H"] = (F_H, (xH, yH))
    
    if H_AD > 0:
        F_HAD, (xHAD, yHAD) = hydrostatic_downstream(geom, gamma_w, H_AD)
        if np.linalg.norm(F_HAD) > 0:
            actions["H_AD"] = (F_HAD, (xHAD, yHAD))
    
    # F_U, (xU, yU) = uplift_force(geom, gamma_w, H_AU_subp, H_AD_subp, eta)
    # if np.linalg.norm(F_U) > 0:
    #     actions["U"] = (F_U, (xU, yU))
    
    if usar_sismo and (kh > 0 or kv > 0):
        A, Cx, Cy = polygon_area_and_centroid(geom.poly)
        F_S, (xS, yS) = seismic_inertial_force(np.array([0.0, -gamma_c*A]), kh, kv, (Cx, Cy))
        if np.linalg.norm(F_S) > 0:
            actions["S"] = (F_S, (xS, yS))
    
    if usar_wh and kh > 0 and H_AU > 0:
        F_Wh, (xWh, yWh) = westergaard_force(geom, gamma_w, H_AU_total, kh)
        if np.linalg.norm(F_Wh) > 0:
            actions["Wh"] = (F_Wh, (xWh, yWh))
    
    if usar_hielo and p_hielo > 0:
        h_ef_m = 1.0
        F_I, (xI, yI) = ice_pressure_force(geom, p_hielo, H_AU, h_ef_m=h_ef_m)
        if np.linalg.norm(F_I) > 0:
            actions["I"] = (F_I, (xI, yI))

    # === Subpresión iterativa ===
    L_grieta = 0.0
    tol = 1e-3  # [m] tolerancia de convergencia
    max_iter = 20
    prev_L = None

    for i in range(max_iter):
        # Calcular fuerza de subpresión con L_grieta actual
        F_U, (xU, yU) = uplift_force(geom, gamma_w, H_AU_subp, H_AD_subp, eta, L_grieta=L_grieta)
        actions["U"] = (F_U, (xU, yU))

        # Punto de referencia para momentos
        actions["_xr"], actions["_yr"] = geom.toe_ad

        # Sumar acciones y obtener resultante
        R, M_toeAD = combine_actions(actions)

        # Tensiones en la base
        base = base_resultant_and_stress(geom, R, M_toeAD)

        # Calcular nueva zona de contacto
        # usar_c = (cfg["REGLAS_COHESION"].get(scen_name, "with_c") == "with_c")
        FS_desliz, Rcap, B_eff, L_new = sliding_FS(
            base["N_kN"], base["T_kN"],
            phi_deg, c_kPa, geom.B_len,
            sigma_AU_kPa=base.get("sigma_AU_kPa"),
            sigma_AD_kPa=base.get("sigma_AD_kPa"),
            usar_cohesion=usar_c,
            anular_cohesion_si_traccion=False
        )

        if prev_L is not None and abs(L_new - L_grieta) < tol:
            break

        prev_L = L_grieta
        L_grieta = L_new

    # Al final de la iteración, los valores finales de base, FS_desliz, etc. quedan consistentes

    # actions["_xr"], actions["_yr"] = geom.toe_ad

    # R, M_toeAD = combine_actions(actions)
    # base = base_resultant_and_stress(geom, R, M_toeAD)

    # FS_desliz, Rcap, B_eff, L_grieta = sliding_FS(
    #     base["N_kN"], base["T_kN"],
    #     phi_deg, c_kPa, geom.B_len,
    #     sigma_AU_kPa=base.get("sigma_AU_kPa"),
    #     sigma_AD_kPa=base.get("sigma_AD_kPa"),
    #     usar_cohesion=usar_c,
    #     anular_cohesion_si_traccion=False
    # )
    
    xr, yr = geom.toe_ad
    Mplus, Mminus = split_moments_by_sign(actions, xr, yr)
    FS_vuelco = (Mplus/Mminus) if Mminus > 1e-12 else float("inf")

    return {
        "R": R,
        "M_toeAD": M_toeAD,
        "base": base,
        "FS_desliz": FS_desliz,
        "FS_vuelco": FS_vuelco,
        "B_eff": B_eff,
        "L_grieta": L_grieta,
        "Rcap": Rcap,
        "actions": actions
    }

def parametric_analysis(geom: Geometry, config_mat, escenario_data, usar_c=True,
                        coh_range=None, phi_range=None):
    """Realiza un análisis paramétrico variando cohesión y ángulo de fricción"""
    if coh_range is None or phi_range is None:
        return None, None, None

    coh_vals = np.linspace(coh_range[0], coh_range[1], num=coh_range[2])
    phi_vals = np.linspace(phi_range[0], phi_range[1], num=phi_range[2])

    FS_desliz_grid = np.zeros((len(phi_vals), len(coh_vals)))
    FS_vuelco_grid = np.zeros((len(phi_vals), len(coh_vals)))

    for i, phi_deg in enumerate(phi_vals):
        for j, coh_kgcm2 in enumerate(coh_vals):
            config_temp = config_mat.copy()
            config_temp["friccion_phi_deg"] = phi_deg
            config_temp["cohesion_kPa"] = coh_kgcm2 * 98.0665  # conversión a kPa

            result = compute_for_scenario(geom, config_temp, escenario_data, usar_c=usar_c)
            FS_desliz_grid[i, j] = result["FS_desliz"]
            FS_vuelco_grid[i, j] = result["FS_vuelco"]

    return phi_vals, coh_vals, (FS_desliz_grid, FS_vuelco_grid)

# =============================================================================
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# =============================================================================

def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = {
            "GEOMETRIA": {
                "altura_H": 36.96,
                # "ancho_coronacion": 0.0,
                "talud_AU_m": 0.0,
                "talud_AD_m": 0.8,
                "inclinacion_cimiento_deg": -6.55,
            },
            "MATERIALES": {
                "gamma_hormigon": 27.0,
                "gamma_agua": 9.81,
                "friccion_phi_deg": 35.0,
                "cohesion_kPa": 196.13
            },
            "ESCENARIOS": {
                # "N_1": {
                #     "nombre": "Normal 1",
                #     "H_AU": 37.97,
                #     "H_AD": 0.0,
                #     "H_AU_subpresion": 36.96,
                #     "H_AD_subpresion": 0.0,
                #     "eficiencia_dren": 0.00,
                #     "kh": 0.00,
                #     "kv": 0.00,
                #     "p_hielo_kPa": 0.0,
                #     "usar_sismo": False,
                #     "usar_westergaard": False,
                #     "usar_hielo": False
                # },
                # "A_1": {
                #     "nombre": "Accidental 1",
                #     "H_AU": 0.0,
                #     "H_AD": 0.0,
                #     "H_AU_subpresion": 0.0,
                #     "H_AD_subpresion": 0.0,
                #     "eficiencia_dren": 0.00,
                #     "kh": 0.108,
                #     "kv": 0.054,
                #     "p_hielo_kPa": 0.0,
                #     "usar_sismo": True,
                #     "usar_westergaard": True,
                #     "usar_hielo": False
                # }
            },
            "REGLAS_COHESION": {
                # "N_1": "with_c",
                # "A_1": "with_c"
            },
            "FACTORES_MINIMOS_FS": {
                # "N_1": {"desliz": 1.40, "vuelco": 1.40},
                # "A_1": {"desliz": 1.30, "vuelco": 1.30}
            }
        }
    
    if 'geometry' not in st.session_state:
        st.session_state.geometry = None
    
    if 'results' not in st.session_state:
        st.session_state.results = {}

def main():
    st.set_page_config(page_title="Análisis de Estabilidad de Presas", layout="wide")
    
    init_session_state()
    
    st.title("🏗️ Análisis de Estabilidad de Presas de Gravedad")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        with st.expander("🧱 Geometría", expanded=True): 
            altura_H = st.number_input("Altura H (m)", value=st.session_state.config["GEOMETRIA"]["altura_H"], 
                                        min_value=0.0, format="%.2f")
            # ancho_coronacion = st.number_input("Ancho coronación (m)", 
            #                                 value=st.session_state.config["GEOMETRIA"]["ancho_coronacion"], 
            #                                 min_value=0.0, format="%.2f")
            talud_AU_m = st.number_input("Talud AU m:1", value=st.session_state.config["GEOMETRIA"]["talud_AU_m"], 
                                        min_value=0.0, format="%.2f")
            talud_AD_m = st.number_input("Talud AD m:1", value=st.session_state.config["GEOMETRIA"]["talud_AD_m"], 
                                        min_value=0.0, format="%.2f")
            st.markdown("### ⚖️ Carga en coronación")

            P_cor = st.number_input(
                "Peso coronación (kN/m)", 
                value=st.session_state.config["GEOMETRIA"].get("P_cor_kN_m", 0.0),
                min_value=0.0, format="%.2f"
            )
            x_cor = st.number_input(
                "Distancia horizontal al pie AU (m)", 
                value=st.session_state.config["GEOMETRIA"].get("x_cor_m", 0.0),
                min_value=0.0, format="%.2f"
            )

            inclinacion_cimiento = st.number_input("Inclinación cimiento (°)", 
                                                    value=st.session_state.config["GEOMETRIA"]["inclinacion_cimiento_deg"], 
                                                    format="%.2f")
        
        with st.expander("🧪 Materiales", expanded=False): 
            gamma_hormigon = st.number_input("γ hormigón (kN/m³)", 
                                            value=st.session_state.config["MATERIALES"]["gamma_hormigon"], 
                                            min_value=0.0, format="%.2f")
            gamma_agua = st.number_input("γ agua (kN/m³)", 
                                        value=st.session_state.config["MATERIALES"]["gamma_agua"], 
                                        min_value=0.0, format="%.2f")
            friccion_phi = st.number_input("Ángulo de fricción φ (°)", 
                                        value=st.session_state.config["MATERIALES"]["friccion_phi_deg"], 
                                        min_value=0.0, max_value=90.0, format="%.2f")
            cohesion_kgcm2 = st.number_input(
                "Cohesión (kg/cm²)",
                value=st.session_state.config["MATERIALES"]["cohesion_kPa"] / 98.0665,
                min_value=0.0,
                format="%.3f"
            )
            cohesion_kPa = cohesion_kgcm2 * 98.0665
        with st.expander("📈 Análisis Paramétrico", expanded=False):
            st.markdown("Define el rango de variación de cohesión y ángulo de fricción para el análisis paramétrico:")

            coh_min = st.number_input("Cohesión mínima (kg/cm²)", 0.0, 100.0, 0.0, 0.1)
            coh_max = st.number_input("Cohesión máxima (kg/cm²)", 0.1, 200.0, 3.0, 0.1)
            phi_min = st.number_input("Ángulo de fricción mínimo (°)", 0.0, 90.0, 25.0, 0.5)
            phi_max = st.number_input("Ángulo de fricción máximo (°)", 0.0, 90.0, 45.0, 0.5)
            n_points = st.number_input("Número de puntos por variable", 3, 50, 10, 1)

            st.session_state.parametric = {
                "coh_min": coh_min, "coh_max": coh_max,
                "phi_min": phi_min, "phi_max": phi_max,
                "n_points": n_points
            }
        
        st.markdown("---")
        if st.button("🔄 Guardar", type="primary", use_container_width=True):
            try:
                st.session_state.config["GEOMETRIA"]["altura_H"] = altura_H
                # st.session_state.config["GEOMETRIA"]["ancho_coronacion"] = ancho_coronacion
                st.session_state.config["GEOMETRIA"]["P_cor_kN_m"] = P_cor
                st.session_state.config["GEOMETRIA"]["x_cor_m"] = x_cor
                st.session_state.config["GEOMETRIA"]["talud_AU_m"] = talud_AU_m
                st.session_state.config["GEOMETRIA"]["talud_AD_m"] = talud_AD_m
                st.session_state.config["GEOMETRIA"]["inclinacion_cimiento_deg"] = inclinacion_cimiento
                
                st.session_state.config["MATERIALES"]["gamma_hormigon"] = gamma_hormigon
                st.session_state.config["MATERIALES"]["gamma_agua"] = gamma_agua
                st.session_state.config["MATERIALES"]["friccion_phi_deg"] = friccion_phi
                st.session_state.config["MATERIALES"]["cohesion_kPa"] = cohesion_kPa
                
                st.session_state.geometry = build_geometry_from_slopes(
                    altura_H=altura_H,
                    # ancho_coronacion=ancho_coronacion,
                    m_up=talud_AU_m,
                    m_down=talud_AD_m,
                    beta_deg=inclinacion_cimiento
                )
                st.success("✅ Geometría actualizada")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Tabs
    escenarios_list = list(st.session_state.config["ESCENARIOS"].keys())
    tab_names = ["📊 Gestión"] + escenarios_list
    tabs = st.tabs(tab_names)
    
    # Tab de gestión
    with tabs[0]:
        st.header("Gestión de Escenarios")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("➕ Agregar Escenario")
            new_esc_id = st.text_input("ID (ej: N_2, A_4)", "")
            new_esc_name = st.text_input("Nombre", "")
            
            if st.button("Agregar"):
                if new_esc_id and new_esc_name and new_esc_id not in st.session_state.config["ESCENARIOS"]:
                    st.session_state.config["ESCENARIOS"][new_esc_id] = {
                        "nombre": new_esc_name,
                        "H_AU": 0.0,
                        "H_AD": 0.0,
                        "H_AU_subpresion": 0.0,
                        "H_AD_subpresion": 0.0,
                        "eficiencia_dren": 0.0,
                        "altura_ola_m": 0.0,   # ✅ nueva carga de oleaje
                        "kh": 0.0,
                        "kv": 0.0,
                        "p_hielo_kPa": 0.0,
                        "usar_sismo": False,
                        "usar_westergaard": False,
                        "usar_hielo": False
                    }
                    st.session_state.config["REGLAS_COHESION"][new_esc_id] = "with_c"
                    st.session_state.config["FACTORES_MINIMOS_FS"][new_esc_id] = {"desliz": 1.4, "vuelco": 1.4}
                    st.success(f"✅ {new_esc_id} agregado")
                    st.rerun()
        
        with col2:
            st.subheader("📋 Escenarios")
            for esc_id in st.session_state.config["ESCENARIOS"].keys():
                esc_name = st.session_state.config["ESCENARIOS"][esc_id].get("nombre", "")
                st.write(f"**{esc_id}**: {esc_name}")
    
    # Tabs de escenarios
    for idx, esc_id in enumerate(escenarios_list):
        with tabs[idx + 1]:
            st.header(f"Escenario: {esc_id}")
            esc_data = st.session_state.config["ESCENARIOS"][esc_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💧 Niveles de Agua")
                nombre = st.text_input("Nombre", value=esc_data.get("nombre", ""), key=f"{esc_id}_nombre")
                H_AU = st.number_input("Nivel de embalse (m)", value=esc_data.get("H_AU", 0.0), 
                                       min_value=0.0, format="%.2f", key=f"{esc_id}_H_AU")
                altura_ola_m = st.number_input("Altura de ola (m)", 
                               value=esc_data.get("altura_ola_m", 0.0),
                               min_value=0.0, format="%.2f",
                               key=f"{esc_id}_altura_ola")
                H_AD = st.number_input("Nivel aguas abajo (m)", value=esc_data.get("H_AD", 0.0), 
                                       min_value=0.0, format="%.2f", key=f"{esc_id}_H_AD")
                H_AU_subp = st.number_input("Carga subpresión aguas arriba (m)", value=esc_data.get("H_AU_subpresion", 0.0), 
                                            min_value=0.0, format="%.2f", key=f"{esc_id}_H_AU_subp")
                H_AD_subp = st.number_input("Carga subpresión aguas abajo (m)", value=esc_data.get("H_AD_subpresion", 0.0), 
                                            min_value=0.0, format="%.2f", key=f"{esc_id}_H_AD_subp")
                eficiencia_dren = st.slider("Eficiencia drenes", min_value=0.0, max_value=1.0, 
                                            value=esc_data.get("eficiencia_dren", 0.0), 
                                            step=0.01, key=f"{esc_id}_eta")
            
            with col2:
                st.subheader("🌊 Cargas Sísmicas")
                usar_sismo = st.checkbox("Usar sismo", value=esc_data.get("usar_sismo", False), 
                                        key=f"{esc_id}_usar_sismo")
                kh = st.number_input("kh", value=esc_data.get("kh", 0.0), 
                                    min_value=0.0, max_value=1.0, format="%.3f", 
                                    key=f"{esc_id}_kh", disabled=not usar_sismo)
                kv = st.number_input("kv", value=esc_data.get("kv", 0.0), 
                                    min_value=0.0, max_value=1.0, format="%.3f", 
                                    key=f"{esc_id}_kv", disabled=not usar_sismo)
                usar_wh = st.checkbox("Usar Westergaard", value=esc_data.get("usar_westergaard", False), 
                                     key=f"{esc_id}_usar_wh", disabled=not usar_sismo)
                
                st.subheader("❄️ Hielo")
                usar_hielo = st.checkbox("Usar hielo", value=esc_data.get("usar_hielo", False), 
                                        key=f"{esc_id}_usar_hielo")
                p_hielo = st.number_input("Presión hielo (kPa)", value=esc_data.get("p_hielo_kPa", 0.0), 
                                         min_value=0.0, format="%.2f", 
                                         key=f"{esc_id}_p_hielo", disabled=not usar_hielo)
            
            with col3:
                st.subheader("📐 Factores Mínimos")
                fs_min = st.session_state.config["FACTORES_MINIMOS_FS"].get(esc_id, {"desliz": 1.0, "vuelco": 1.0})
                FS_desliz_min = st.number_input("FS deslizamiento mín", 
                                                value=fs_min.get("desliz", 1.0), 
                                                min_value=0.0, format="%.2f", 
                                                key=f"{esc_id}_fs_desliz_min")
                FS_vuelco_min = st.number_input("FS vuelco mín", 
                                               value=fs_min.get("vuelco", 1.0), 
                                               min_value=0.0, format="%.2f", 
                                               key=f"{esc_id}_fs_vuelco_min")
                
                st.subheader("🔧 Cohesión")
                regla_c = st.session_state.config["REGLAS_COHESION"].get(esc_id, "with_c")
                # Mapeo entre lo que ve el usuario y el valor interno
                label_to_value = {"Sí": "with_c", "No": "without_c"}
                value_to_label = {v: k for k, v in label_to_value.items()}
                usar_cohesion_label = st.selectbox(
                    "Usar cohesión",
                    options=list(label_to_value.keys()),
                    index=0 if regla_c == "with_c" else 1,
                    key=f"{esc_id}_cohesion_label"
                )
                usar_cohesion = label_to_value[usar_cohesion_label]
            st.markdown("---")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button(f"💾 Guardar {esc_id}", type="primary", use_container_width=True):
                    st.session_state.config["ESCENARIOS"][esc_id] = {
                        "nombre": nombre,
                        "H_AU": H_AU,
                        "H_AD": H_AD,
                        "H_AU_subpresion": H_AU_subp,
                        "H_AD_subpresion": H_AD_subp,
                        "eficiencia_dren": eficiencia_dren,
                        "altura_ola_m": altura_ola_m,   # ✅ guardar altura de ola
                        "kh": kh,
                        "kv": kv,
                        "p_hielo_kPa": p_hielo,
                        "usar_sismo": usar_sismo,
                        "usar_westergaard": usar_wh,
                        "usar_hielo": usar_hielo
                    }
                    st.session_state.config["FACTORES_MINIMOS_FS"][esc_id] = {
                        "desliz": FS_desliz_min,
                        "vuelco": FS_vuelco_min
                    }
                    st.session_state.config["REGLAS_COHESION"][esc_id] = usar_cohesion
                    st.success(f"✅ {esc_id} guardado")
            
            with col_btn2:
                if st.button(f"🔢 Calcular {esc_id}", use_container_width=True):
                    if st.session_state.geometry is None:
                        st.error("⚠️ Actualice la geometría primero")
                    else:
                        try:
                            with st.spinner(f"Calculando {esc_id}..."):
                                usar_c = (usar_cohesion == "with_c")
                                # Unir materiales y geometría para pasar todo
                                config_total = {**st.session_state.config["MATERIALES"], **st.session_state.config["GEOMETRIA"]}

                                result = compute_for_scenario(
                                    st.session_state.geometry,
                                    config_total,
                                    st.session_state.config["ESCENARIOS"][esc_id],
                                    usar_c=usar_c
                                )
                                st.session_state.results[esc_id] = result
                                st.success(f"✅ {esc_id} calculado")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col_btn3:
                if st.button(f"🗑️ Limpiar {esc_id}", use_container_width=True):
                    if esc_id in st.session_state.results:
                        del st.session_state.results[esc_id]
                        st.success(f"{esc_id} limpiado")
                        st.rerun()
            
            # Mostrar resultados
            if esc_id in st.session_state.results:
                st.markdown("---")
                st.subheader("📊 Resultados")
                
                result = st.session_state.results[esc_id]
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                
                with col_r1:
                    fs_d = result['FS_desliz']
                    fs_d_str = f"{fs_d:.3f}" if np.isfinite(fs_d) else "∞"
                    st.metric("FS Deslizamiento", fs_d_str)
                    fs_desliz_ok = fs_d >= FS_desliz_min
                    if fs_desliz_ok:
                        st.success("✅ Cumple")
                    else:
                        st.error("❌ No cumple")
                
                with col_r2:
                    fs_v = result['FS_vuelco']
                    fs_v_str = f"{fs_v:.3f}" if np.isfinite(fs_v) else "∞"
                    st.metric("FS Vuelco", fs_v_str)
                    fs_vuelco_ok = fs_v >= FS_vuelco_min
                    if fs_vuelco_ok:
                        st.success("✅ Cumple")
                    else:
                        st.error("❌ No cumple")
                
                with col_r3:
                    e_val = result['base']['e_m']
                    e_str = f"{e_val:.3f}" if np.isfinite(e_val) else "N/A"
                    st.metric("Excentricidad (m)", e_str)
                    if result['base']['no_traccion']:
                        st.success("✅ Sin tracción")
                    else:
                        st.warning("⚠️ Con tracción")
                
                with col_r4:
                    st.metric("Long. comprimida (m)", f"{result['B_eff']:.3f}")
                    st.metric("Long. grieta (m)", f"{result['L_grieta']:.3f}")
                
                # Tabla de detalles
                st.markdown("#### Detalles de las Cargas")
                col_t1, col_t2 = st.columns(2)
                
                with col_t1:
                    st.write(f"**Normal N (kN/m)**: {result['base']['N_kN']:.2f}")
                    st.write(f"**Tangencial T (kN/m)**: {result['base']['T_kN']:.2f}")
                    st.write(f"**Capacidad (kN/m)**: {result['Rcap']:.2f}")
                    st.write(f"**σ AU (kPa)**: {result['base']['sigma_AU_kPa']:.2f}")
                    st.write(f"**σ AD (kPa)**: {result['base']['sigma_AD_kPa']:.2f}")
                
                with col_t2:
                    st.write(f"**σ min (kPa)**: {result['base']['sigma_min_kPa']:.2f}")
                    st.write(f"**σ max (kPa)**: {result['base']['sigma_max_kPa']:.2f}")
                    st.write(f"**Momento (kN·m/m)**: {result['M_toeAD']:.2f}")
                    st.write(f"**Resultante X (kN/m)**: {result['R'][0]:.2f}")
                    st.write(f"**Resultante Y (kN/m)**: {result['R'][1]:.2f}")
                
                # Gráficos
                st.markdown("#### Visualizaciones")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("**Perfil con Fuerzas**")
                    try:
                        fig_perfil = plot_profile_with_forces(
                            st.session_state.geometry,
                            result['actions'],
                            result['R'],
                            result['base']['e_m'],
                            f"{esc_id} - {nombre}"
                        )
                        st.pyplot(fig_perfil)
                        plt.close(fig_perfil)
                    except Exception as e:
                        st.error(f"Error en gráfico: {e}")
                    st.markdown("**Presiones en la Base**")
                    try:
                        # fig_presiones = plot_base_pressures(
                        #     st.session_state.geometry,
                        #     result['base']['N_kN'],
                        #     result['base']['e_m'],
                        #     f"{esc_id} - Presiones"
                        # )
                        fig_presiones = plot_base_pressures(
                            st.session_state.geometry,
                            result['base']['N_kN'],
                            result['base']['e_m'],
                            f"{esc_id} - Presiones",
                            ruta_png=None,
                            gamma_w=st.session_state.config["MATERIALES"]["gamma_agua"],  # ✅ corregido también
                            H_AU=esc_data.get("H_AU_subpresion", 0.0),
                            H_AD=esc_data.get("H_AD_subpresion", 0.0),
                            eficiencia=esc_data.get("eficiencia_dren", 0.0),
                            L_grieta=result.get("L_grieta", 0.0)
                        )


                        if fig_presiones:
                            st.pyplot(fig_presiones)
                            plt.close(fig_presiones)
                    except Exception as e:
                        st.error(f"Error en gráfico: {e}")
                with col_g2:
                     # --- Análisis paramétrico de deslizamiento ---
                    st.markdown("**Análisis Paramétrico – FS Deslizamiento**")
                    param_conf = st.session_state.parametric
                    try:
                        phi_vals, coh_vals, (FSd, FSv) = parametric_analysis(
                            st.session_state.geometry,
                            st.session_state.config["MATERIALES"],
                            st.session_state.config["ESCENARIOS"][esc_id],
                            usar_c=(usar_cohesion == "with_c"),
                            coh_range=(param_conf["coh_min"], param_conf["coh_max"], param_conf["n_points"]),
                            phi_range=(param_conf["phi_min"], param_conf["phi_max"], param_conf["n_points"])
                        )

                        fig1, ax1 = plt.subplots(figsize=(6, 5))
                        im1 = ax1.imshow(FSd, extent=[coh_vals[0], coh_vals[-1], phi_vals[0], phi_vals[-1]],
                                        origin="lower", aspect="auto", cmap="viridis")
                        plt.colorbar(im1, ax=ax1, label="FS deslizamiento")

                        # --- Curvas de nivel para FS = 1, 2, 3 ---
                        coh_grid, phi_grid = np.meshgrid(coh_vals, phi_vals)
                        CS1 = ax1.contour(coh_grid, phi_grid, FSd, levels=[1, 2, 3], colors='white', linewidths=1.2)
                        ax1.clabel(CS1, inline=True, fontsize=8, fmt='%1.0f')

                        # --- Curva destacada para FS requerido ---
                        fs_req = st.session_state.config["FACTORES_MINIMOS_FS"][esc_id]["desliz"]
                        CS_req = ax1.contour(coh_grid, phi_grid, FSd, levels=[fs_req], colors='red', linewidths=2.0)
                        ax1.clabel(CS_req, inline=True, fontsize=9, fmt=f"FS={fs_req:.2f}", colors='red')

                        # --- Asterisco para el valor actual ---
                        phi_actual = st.session_state.config["MATERIALES"]["friccion_phi_deg"]
                        coh_actual = st.session_state.config["MATERIALES"]["cohesion_kPa"] / 98.0665
                        ax1.plot(coh_actual, phi_actual, marker='*', color='black', markersize=12, label='Valor actual')
                        ax1.legend(loc='lower right', fontsize=8)

                        ax1.set_xlabel("Cohesión (kg/cm²)")
                        ax1.set_ylabel("Ángulo de fricción (°)")
                        ax1.set_title(f"{esc_id} - FS deslizamiento")
                        st.pyplot(fig1, use_container_width=True)
                        plt.close(fig1)
                    except Exception as e:
                        st.error(f"Error en análisis paramétrico (desliz): {e}")
                        # --- Análisis paramétrico de vuelco ---
                    st.markdown("**Análisis Paramétrico – FS Vuelco**")
                    try:
                        # (usa los mismos phi_vals, coh_vals y FSv del cálculo anterior)
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        im2 = ax2.imshow(FSv, extent=[coh_vals[0], coh_vals[-1], phi_vals[0], phi_vals[-1]],
                                        origin="lower", aspect="auto", cmap="plasma")
                        plt.colorbar(im2, ax=ax2, label="FS vuelco")

                        # --- Curvas de nivel para FS = 1, 2, 3 ---
                        coh_grid, phi_grid = np.meshgrid(coh_vals, phi_vals)
                        CS2 = ax2.contour(coh_grid, phi_grid, FSv, levels=[1, 2, 3], colors='white', linewidths=1.2)
                        ax2.clabel(CS2, inline=True, fontsize=8, fmt='%1.0f')

                        # --- Curva destacada para FS requerido ---
                        fs_req_v = st.session_state.config["FACTORES_MINIMOS_FS"][esc_id]["vuelco"]
                        CS_req_v = ax2.contour(coh_grid, phi_grid, FSv, levels=[fs_req_v], colors='red', linewidths=2.0)
                        ax2.clabel(CS_req_v, inline=True, fontsize=9, fmt=f"FS={fs_req_v:.2f}", colors='red')

                        # --- Asterisco para el valor actual ---
                        phi_actual = st.session_state.config["MATERIALES"]["friccion_phi_deg"]
                        coh_actual = st.session_state.config["MATERIALES"]["cohesion_kPa"] / 98.0665
                        ax2.plot(coh_actual, phi_actual, marker='*', color='black', markersize=12, label='Valor actual')
                        ax2.legend(loc='lower right', fontsize=8)

                        ax2.set_xlabel("Cohesión (kg/cm²)")
                        ax2.set_ylabel("Ángulo de fricción (°)")
                        ax2.set_title(f"{esc_id} - FS vuelco")
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    except Exception as e:
                        st.error(f"Error en análisis paramétrico (vuelco): {e}")
    
    # Comparación de resultados
    st.markdown("---")
    st.header("📈 Comparación de Resultados")
    
    if len(st.session_state.results) > 0:
        comp_data = []
        for esc_id, result in st.session_state.results.items():
            fs_min = st.session_state.config["FACTORES_MINIMOS_FS"].get(esc_id, {"desliz": 1.0, "vuelco": 1.0})
            fs_d = result['FS_desliz']
            fs_v = result['FS_vuelco']
            
            comp_data.append({
                "Escenario": esc_id,
                "Nombre": st.session_state.config["ESCENARIOS"][esc_id].get("nombre", ""),
                "FS Desliz": f"{fs_d:.3f}" if np.isfinite(fs_d) else "∞",
                "FS Req Desliz": f"{fs_min['desliz']:.2f}",
                "Cumple Desliz": "✅" if fs_d >= fs_min['desliz'] else "❌",
                "FS Vuelco": f"{fs_v:.3f}" if np.isfinite(fs_v) else "∞",
                "FS Req Vuelco": f"{fs_min['vuelco']:.2f}",
                "Cumple Vuelco": "✅" if fs_v >= fs_min['vuelco'] else "❌",
                "e (m)": f"{result['base']['e_m']:.3f}" if np.isfinite(result['base']['e_m']) else "N/A",
                "Sin Tracción": "✅" if result['base']['no_traccion'] else "❌"
            })
        
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
        # Exportar Excel
        col_exp1, col_exp2 = st.columns([1, 4])
        
        with col_exp1:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_comp.to_excel(writer, sheet_name='Resumen', index=False)
                
                for esc_id, result in st.session_state.results.items():
                    detalle = {
                        "Parámetro": [
                            "Normal N (kN/m)", "Tangencial T (kN/m)", 
                            "Excentricidad (m)", "Capacidad (kN/m)",
                            "σ AU (kPa)", "σ AD (kPa)", "σ min (kPa)", "σ max (kPa)",
                            "B efectiva (m)", "Long. grieta (m)",
                            "FS Deslizamiento", "FS Vuelco", "Sin tracción"
                        ],
                        "Valor": [
                            f"{result['base']['N_kN']:.2f}",
                            f"{result['base']['T_kN']:.2f}",
                            f"{result['base']['e_m']:.3f}" if np.isfinite(result['base']['e_m']) else "N/A",
                            f"{result['Rcap']:.2f}",
                            f"{result['base']['sigma_AU_kPa']:.2f}",
                            f"{result['base']['sigma_AD_kPa']:.2f}",
                            f"{result['base']['sigma_min_kPa']:.2f}",
                            f"{result['base']['sigma_max_kPa']:.2f}",
                            f"{result['B_eff']:.3f}",
                            f"{result['L_grieta']:.3f}",
                            f"{result['FS_desliz']:.3f}" if np.isfinite(result['FS_desliz']) else "∞",
                            f"{result['FS_vuelco']:.3f}" if np.isfinite(result['FS_vuelco']) else "∞",
                            "Sí" if result['base']['no_traccion'] else "No"
                        ]
                    }
                    df_detalle = pd.DataFrame(detalle)
                    df_detalle.to_excel(writer, sheet_name=esc_id[:31], index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Exportar Excel",
                data=excel_data,
                file_name="resultados_presa.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("ℹ️ No hay resultados para comparar. Configure y calcule al menos un escenario.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Análisis de Estabilidad de Presas de Gravedad - Categoría A (Norma Técnica ESP)</p>
        <p>Unidades SI: m, kN, kN/m, kPa | Cargas por metro de espesor</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()