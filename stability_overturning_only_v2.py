import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from io import BytesIO

"""
Overturning stability Streamlit app

This script reads two GiD-style stress files (.grf): one containing
self-weight stresses and another with applied loads only. It rotates
stresses into a contact-aligned coordinate system, computes compressive
equivalent tractions, evaluates stabilizing moments from self-weight,
overturning moments from loads (including optional uplift), and
reports a factor of safety against overturning.

Sign conventions:
- In the rotated frame `sigma_n` is NEGATIVE for compression and
    POSITIVE for tension/opening. The code converts compression to
    positive traction when computing stabilizing forces/moments.

This file adds clarifying comments to make the handling of traction
vs tension and uplift clearer.
"""

st.set_page_config(page_title="Overturning Stability Analysis", layout="wide")

st.title("🏗️ Overturning Stability Analysis Tool")
st.markdown("Compute factor of safety against overturning from two GiD stress files (self-weight vs loads).")

# -------------------------
# Utilities (from original app)
# -------------------------
def parse_three_graph_blocks(content: bytes):
    """
    Parse GiD .grf-like file with three blocks (Sxx, Syy, Sxy).
    Returns three dataframes with columns [x, y].
    """
    blocks, current = [], []
    lines = content.decode("utf-8", errors="ignore").split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if line.startswith("# End") and current:
                blocks.append(pd.DataFrame(current, columns=["x", "y"]))
                current = []
            continue
        parts = line.split()
        if len(parts) >= 2:
            current.append((float(parts[0]), float(parts[1])))
    if current:
        blocks.append(pd.DataFrame(current, columns=["x", "y"]))
    if len(blocks) < 3:
        raise ValueError(f"Expected 3 data blocks (Sxx, Syy, Sxy). Found {len(blocks)}.")
    return blocks[0], blocks[1], blocks[2]

def merge_blocks_on_union_x(df_sxx, df_syy, df_sxy):
    x_union = np.unique(np.concatenate([df_sxx["x"].values, df_syy["x"].values, df_sxy["x"].values]))
    x_union.sort()

    def interp(df):
        return np.interp(x_union, df["x"].values, df["y"].values)

    return pd.DataFrame({
        "x": x_union,
        "Sxx": interp(df_sxx),
        "Syy": interp(df_syy),
        "Sxy": interp(df_sxy),
    })

def rotate_to_normal_tangential(df, theta_deg: float):
    """
    Rotate global stresses (Sxx,Syy,Sxy) into contact normal/tangential stresses.
    Sign convention inherited from the original app:
      - compression is NEGATIVE sigma_n
      - tension/opening is POSITIVE sigma_n
    """
    th = math.radians(theta_deg)
    c2 = math.cos(2 * th)
    s2 = math.sin(2 * th)
    sxx = df["Sxx"].values
    syy = df["Syy"].values
    txy = df["Sxy"].values

    sigma_n = 0.5 * (sxx + syy) - 0.5 * (sxx - syy) * c2 - txy * s2
    tau = 0.5 * (sxx - syy) * s2 - txy * c2

    out = df.copy()
    out["sigma_n"] = sigma_n
    out["tau"] = tau
    return out

def integrate_trapezoid(x, y):
    return float(np.trapz(y, x))

def compute_uplift_pressure_from_crack(x, crack_length, Hw, crack_ref="upstream"):
    """
    Deterministic uplift pressure p(x) based on a user-defined crack length.
    crack_length: length from upstream heel (default) or from toe (if crack_ref="toe")
    Hw: water head (m)
    Returns p(x) >= 0 (Pa).
    """
    gamma_w = 9810.0
    x = np.asarray(x)
    x0 = x.min()
    x_toe = x.max()
    L_total = x_toe - x0
    if L_total <= 0:
        return np.zeros_like(x)

    Lc = float(max(0.0, min(crack_length, L_total)))

    if crack_ref == "toe":
        crack_start = x_toe - Lc
    else:
        crack_start = x0 + Lc

    p0 = gamma_w * Hw
    p = np.zeros_like(x, dtype=float)

    for i, xi in enumerate(x):
        if crack_ref == "toe":
            # crack region is [crack_start, x_toe]
            if xi >= crack_start:
                p[i] = p0
            else:
                # upstream of crack: linear decay from p0 at crack_start to 0 at x0
                Lc_up = crack_start - x0
                if Lc_up > 1e-12:
                    p[i] = p0 * max(0.0, 1.0 - (crack_start - xi) / Lc_up)
                else:
                    p[i] = 0.0
        else:
            # crack region is [x0, crack_start]
            if xi <= crack_start:
                p[i] = p0
            else:
                # downstream of crack: linear decay from p0 at crack_start to 0 at x_toe
                Lc_down = x_toe - crack_start
                if Lc_down > 1e-12:
                    p[i] = p0 * max(0.0, 1.0 - (xi - crack_start) / Lc_down)
                else:
                    p[i] = p0
    return p

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("📥 Input files")

file_sw = st.sidebar.file_uploader("SELF-WEIGHT only (.grf)", type=["grf"], key="sw_file")
file_ld = st.sidebar.file_uploader("LOADS only (.grf)", type=["grf"], key="ld_file")

st.sidebar.markdown("---")
st.sidebar.header("📐 Geometry")
theta_deg = st.sidebar.number_input("Contact inclination θ (°)", value=6.550, step=0.1, format="%.3f")
toe_convention = st.sidebar.selectbox(
    "Downstream toe location",
    options=["toe at x max (default)", "toe at x min"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("💧 Uplift (overturning only)")
include_uplift = st.sidebar.checkbox("Include uplift in overturning demand?", value=True)
Hw = st.sidebar.number_input("Water head Hw (m)", value=36.96, step=0.1, format="%.2f")
crack_ref = st.sidebar.selectbox("Crack length reference", options=["upstream (heel → toe)", "from toe (toe → heel)"], index=0)
crack_length = st.sidebar.number_input("Crack length Lc (m)", value=0.0, min_value=0.0, step=0.1)

st.sidebar.markdown("---")
FS_required = st.sidebar.number_input("Required FS (overturning)", value=1.4, min_value=0.1, step=0.1)

run = st.sidebar.button("🧮 Run Overturning Analysis", type="primary", use_container_width=True)

# -------------------------
# Main
# -------------------------
if not (file_sw and file_ld):
    st.info("👈 Upload BOTH files (self-weight only and loads only) to run the overturning check.")
    st.stop()

if run:
    try:
        # Read + parse SW file
        content_sw = file_sw.read()
        file_sw.seek(0)
        df_sw_sxx, df_sw_syy, df_sw_sxy = parse_three_graph_blocks(content_sw)
        df_sw = merge_blocks_on_union_x(df_sw_sxx, df_sw_syy, df_sw_sxy)
        df_sw_rot = rotate_to_normal_tangential(df_sw, theta_deg)

        # Read + parse Loads file
        content_ld = file_ld.read()
        file_ld.seek(0)
        df_ld_sxx, df_ld_syy, df_ld_sxy = parse_three_graph_blocks(content_ld)
        df_ld = merge_blocks_on_union_x(df_ld_sxx, df_ld_syy, df_ld_sxy)
        df_ld_rot = rotate_to_normal_tangential(df_ld, theta_deg)

        # Align x grids on union for consistent integration/plots
        x_union = np.unique(np.concatenate([df_sw_rot["x"].values, df_ld_rot["x"].values]))
        x_union.sort()

        def interp_series(df, col):
            return np.interp(x_union, df["x"].values, df[col].values)

        sig_sw = interp_series(df_sw_rot, "sigma_n")
        sig_ld = interp_series(df_ld_rot, "sigma_n")

        # Toe position
        x_min, x_max = float(x_union.min()), float(x_union.max())
        if toe_convention == "toe at x min":
            x_toe = x_min
        else:
            x_toe = x_max

        lever = (x_toe - x_union)

        # --- Stabilizing (Self-weight): compression-only, no uplift ---
        t_sw_eq = np.maximum(-sig_sw, 0.0)  # equivalent compressive traction (>=0)
        N_sw = integrate_trapezoid(x_union, t_sw_eq)
        M_sw = integrate_trapezoid(x_union, t_sw_eq * lever)

        xR_sw = (M_sw / N_sw) if N_sw > 1e-12 else float("nan")

        # --- Loads: moment must include FEM tension (signed traction) ---
        t_ld_signed = -sig_ld  # signed traction; tension => negative
        N_ld_signed = integrate_trapezoid(x_union, t_ld_signed)
        M_ld_signed = integrate_trapezoid(x_union, t_ld_signed * lever)

        M_ld = abs(M_ld_signed)  # overturning demand from FEM loads

        # Resultant location for loads must be physically meaningful -> compressive-equivalent
        t_ld_eq = np.maximum(-sig_ld, 0.0)
        N_ld_eq = integrate_trapezoid(x_union, t_ld_eq)
        M_ld_eq = integrate_trapezoid(x_union, t_ld_eq * lever)
        xR_ld_eq = (M_ld_eq / N_ld_eq) if N_ld_eq > 1e-12 else float("nan")

        # --- Uplift (optional, overturning demand only) ---
        if include_uplift:
            # Map the user-friendly selectbox string to the function's crack_ref
            # expected values ('toe' or 'upstream'). The selectbox options are
            # intentionally verbose, so we check the string start for clarity.
            cref = "toe" if crack_ref.startswith("from toe") else "upstream"
            p_u = compute_uplift_pressure_from_crack(x_union, crack_length, Hw, crack_ref=cref)
            M_u = integrate_trapezoid(x_union, p_u * lever)
        else:
            p_u = np.zeros_like(x_union)
            M_u = 0.0

        M_overturn = M_ld + M_u
        FS_ot = (M_sw / M_overturn) if M_overturn > 1e-12 else float("inf")

        st.success("✅ Overturning analysis completed")

        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("FS (Overturning)", f"{FS_ot:.2f}",
                      delta="OK" if FS_ot >= FS_required else "FAIL",
                      delta_color="normal" if FS_ot >= FS_required else "inverse")
        with col2:
            st.metric("M_stabilizing (SW)", f"{M_sw/1000:.2f} kN·m")
        with col3:
            st.metric("M_loads (|signed|)", f"{M_ld/1000:.2f} kN·m")
        with col4:
            st.metric("M_uplift", f"{M_u/1000:.2f} kN·m" if include_uplift else "0.00 kN·m")
        with col5:
            st.metric("Toe x", f"{x_toe:.3f}")

        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("x_R SW (from toe)", f"{xR_sw:.3f}" if np.isfinite(xR_sw) else "N/A")
        with col7:
            st.metric("x_R loads (compressive eq., from toe)", f"{xR_ld_eq:.3f}" if np.isfinite(xR_ld_eq) else "N/A")
        with col8:
            st.metric("N_SW (eq.)", f"{N_sw/1000:.2f} kN")

        with st.expander("Loads-only signed resultants (from FEM, includes tension)"):
            st.write(f"Signed N_loads = {N_ld_signed:.6e} N")
            st.write(f"Signed M_loads(toe) = {M_ld_signed:.6e} N·m")
            st.write("Note: x_R is not computed from signed N (it can be near zero or tensile).")

        tab1, tab2, tab3 = st.tabs(["📈 Traction & Uplift", "📋 Detailed arrays", "📥 Export"])

        with tab1:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(x_union, t_sw_eq, label="SW: traction eq. (compression only)")
            ax.plot(x_union, t_ld_signed, label="Loads: signed traction (-σn)")
            ax.plot(x_union, t_ld_eq, label="Loads: traction eq. (compression only)", linestyle="--")
            if include_uplift:
                ax.plot(x_union, p_u, label="Uplift pressure p(x)")
            ax.axhline(0.0, linewidth=1)
            ax.set_xlabel("x along contact")
            ax.set_ylabel("Traction / pressure (Pa)")
            ax.set_title("Contact tractions and uplift pressure")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        with tab2:
            df_out = pd.DataFrame({
                "x": x_union,
                "sigma_n_SW": sig_sw,
                "traction_SW_eq": t_sw_eq,
                "sigma_n_Loads": sig_ld,
                "traction_Loads_signed": t_ld_signed,
                "traction_Loads_eq": t_ld_eq,
                "uplift_p": p_u,
                "lever_arm_(x_toe-x)": lever
            })
            st.dataframe(df_out, use_container_width=True)

        with tab3:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_out.to_excel(writer, sheet_name="PointData", index=False)
                summary = pd.DataFrame([{
                    "file_self_weight": file_sw.name,
                    "file_loads": file_ld.name,
                    "theta_deg": theta_deg,
                    "toe_convention": toe_convention,
                    "toe_x": x_toe,
                    "include_uplift": include_uplift,
                    "Hw_m": Hw,
                    "crack_length_m": crack_length,
                    "crack_reference": crack_ref,
                    "M_stabilizing_Nm": M_sw,
                    "M_loads_abs_Nm": M_ld,
                    "M_uplift_Nm": M_u,
                    "M_overturn_Nm": M_overturn,
                    "FS_overturning": FS_ot,
                    "FS_required": FS_required,
                    "xR_SW_from_toe": xR_sw,
                    "xR_Loads_from_toe_compressive_eq": xR_ld_eq,
                    "N_SW_eq_N": N_sw,
                    "N_Loads_signed_N": N_ld_signed,
                    "M_Loads_signed_Nm": M_ld_signed
                }])
                summary.to_excel(writer, sheet_name="Summary", index=False)

            st.download_button(
                "📥 Download Excel",
                data=output.getvalue(),
                file_name="overturning_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
else:
    st.info("Configure inputs in the sidebar and click **Run Overturning Analysis**.")
