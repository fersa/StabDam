import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from pathlib import Path

st.set_page_config(page_title="Contact Stability Analysis", layout="wide")

st.title("🏗️ Contact Stability Analysis Tool")
st.markdown("Analysis of contact stresses and sliding stability from GiD stress results")

# Sidebar for inputs
st.sidebar.header("📊 Input Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload GiD stress file (.grf)", type=['grf'])

st.sidebar.markdown("---")
st.sidebar.subheader("Geometric Parameters")
theta_deg = st.sidebar.number_input("Contact inclination θ (°)", value=6.550, step=0.1, format="%.3f")

st.sidebar.subheader("Uplift Options")
apply_uplift = st.sidebar.checkbox("Include uplift pressure?", value=True)
H_w = st.sidebar.number_input("Water height for upift Hw (m)", value=36.96, step=0.1, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Material Parameters")
phi_deg = st.sidebar.number_input("Friction angle φ (°)", value=35.0, min_value=0.0, max_value=90.0, step=1.0)
c_kgcm2 = st.sidebar.number_input("Cohesion c (kg/cm²)", value=2.0, min_value=0.0, step=0.1, format="%.3f")
FS_required = st.sidebar.number_input("Required FS", value=1.4, min_value=0.1, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Parametric Study Ranges")
phi_min = st.sidebar.number_input("φ min (°)", value=0, min_value=0, max_value=90)
phi_max = st.sidebar.number_input("φ max (°)", value=50, min_value=0, max_value=90)
phi_step = st.sidebar.number_input("φ step (°)", value=5, min_value=1, max_value=10)

c_min = st.sidebar.number_input("c min (kg/cm²)", value=0.0, min_value=0.0, step=0.1)
c_max = st.sidebar.number_input("c max (kg/cm²)", value=3.0, min_value=0.0, step=0.1)
c_points = st.sidebar.number_input("c points", value=13, min_value=2, max_value=50)

# Analysis functions
def parse_three_graph_blocks(content):
    blocks, current = [], []
    lines = content.decode('utf-8', errors='ignore').split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if line.startswith("# End") and current:
                blocks.append(pd.DataFrame(current, columns=["x","y"]))
                current = []
            continue
        parts = line.split()
        if len(parts) >= 2:
            current.append((float(parts[0]), float(parts[1])))
    if current: 
        blocks.append(pd.DataFrame(current, columns=["x","y"]))
    if len(blocks) < 3:
        raise ValueError(f"Expected 3 data blocks (Sxx, Syy, Sxy). Found {len(blocks)}.")
    return blocks[0], blocks[1], blocks[2]

def merge_blocks_on_union_x(df_sxx, df_syy, df_sxy):
    x_union = np.unique(np.concatenate([df_sxx["x"].values, df_syy["x"].values, df_sxy["x"].values]))
    x_union.sort()
    def interp(df): 
        return np.interp(x_union, df["x"].values, df["y"].values)
    return pd.DataFrame({"x": x_union, "Sxx": interp(df_sxx), "Syy": interp(df_syy), "Sxy": interp(df_sxy)})

def rotate_to_normal_tangential(df, theta_deg):
    th = math.radians(theta_deg)
    c2 = math.cos(2*th)
    s2 = math.sin(2*th)
    sxx = df["Sxx"].values
    syy = df["Syy"].values
    txy = df["Sxy"].values
    sigma_n = 0.5*(sxx + syy) - 0.5*(sxx - syy)*c2 - txy*s2
    tau = 0.5*(sxx - syy)*s2 - txy*c2
    out = df.copy()
    out["sigma_n"] = sigma_n
    out["tau"] = tau
    return out

def integrate_trapezoid(x, y): 
    return float(np.trapz(y, x))

def compute_uplift_stress(x_union, L_total, H_w):
    gamma_w = 9810
    syy_hydro = gamma_w * H_w * (1 - x_union/L_total)
    return pd.DataFrame({
        "x": x_union,
        "Sxx": np.zeros_like(x_union),
        "Syy": syy_hydro,
        "Sxy": np.zeros_like(x_union)
    })

# Main analysis
if uploaded_file is not None:
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if run_analysis:
        with st.spinner("Processing data..."):
            try:
                # Convert cohesion to Pa
                c = c_kgcm2 * 98066.5
                
                # Parse file
                content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset for potential re-read
                df_sxx, df_syy, df_sxy = parse_three_graph_blocks(content)
                df = merge_blocks_on_union_x(df_sxx, df_syy, df_sxy)
                
                # Compute hydrostatic stress
                x_union = df["x"].values
                L_total = x_union.max() - x_union.min()
                df_uplift = compute_uplift_stress(x_union, L_total, H_w)
                
                # Rotate both stress states
                df_fem = rotate_to_normal_tangential(df, theta_deg)
                df_uplift_rotated = rotate_to_normal_tangential(df_uplift, theta_deg)
                
                # Add stresses
                df = df_fem.copy()
                df["sigma_n"] += df_uplift_rotated["sigma_n"]
                df["tau"] += df_uplift_rotated["tau"]
                
                closed = df["sigma_n"].values < 0.0
                df["closed"] = closed
                sigma_n_comp = np.where(closed, -df["sigma_n"].values, 0.0)
                
                x = df["x"].values
                N = integrate_trapezoid(x, sigma_n_comp)
                T = integrate_trapezoid(x, df["tau"].values)
                Lc = integrate_trapezoid(x, closed.astype(float))
                
                tan_phi = math.tan(math.radians(phi_deg))
                R = c*Lc + N*tan_phi
                FS = R / max(abs(T), 1e-12)
                
                # Display results
                st.success("✅ Analysis completed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Factor of Safety", f"{FS:.2f}", 
                             delta="OK" if FS >= FS_required else "FAIL",
                             delta_color="normal" if FS >= FS_required else "inverse")
                with col2:
                    st.metric("Normal Force N", f"{N:.2e}")
                with col3:
                    st.metric("Tangential Force T", f"{T:.2e}")
                with col4:
                    st.metric("Resistance R", f"{R:.2e}")
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Closed Length Lc", f"{Lc:.2f}")
                with col6:
                    st.metric("Open Length", f"{L_total - Lc:.2f}")
                with col7:
                    st.metric("Open %", f"{100.0*((L_total - Lc)/L_total):.1f}%")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Stress Distribution", "🔥 Parametric Study", "📋 Detailed Data"])
                
                with tab1:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(x, df["sigma_n"], 'b-', label='σn', linewidth=2)
                    ax1.plot(x, df["tau"], 'r-', label='τ', linewidth=2)
                    ax1.fill_between(x, df["sigma_n"], 0, where=(df["sigma_n"] < 0), 
                                    color='blue', alpha=0.2, label='Compression')
                    
                    text_x = x.min() + 0.05 * (x.max() - x.min())
                    text_y = df["sigma_n"].min()
                    ax1.text(text_x, text_y*0.85, f'N = {N:.2e}', fontsize=10)
                    ax1.text(text_x, text_y*0.70, f'T = {T:.2e}', fontsize=10)
                    ax1.text(text_x, text_y*0.55, f'FS = {FS:.2f}', fontsize=10,
                            color='green' if FS >= FS_required else 'red', weight='bold')
                    
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlabel('Position x (length units)', fontsize=11)
                    ax1.set_ylabel('Stress (stress units)', fontsize=11)
                    ax1.set_title(f'Normal and Shear Stresses Along Contact\n{uploaded_file.name}', fontsize=12)
                    ax1.legend()
                    st.pyplot(fig1)
                
                with tab2:
                    # Parametric study
                    phi_deg_range = np.arange(phi_min, phi_max + phi_step, phi_step)
                    c_kgcm2_range = np.linspace(c_min, c_max, c_points)
                    c_range = c_kgcm2_range * 98066.5
                    
                    FS_matrix = np.zeros((len(phi_deg_range), len(c_range)))
                    
                    for i, phi in enumerate(phi_deg_range):
                        for j, c_val in enumerate(c_range):
                            tan_phi = math.tan(math.radians(phi))
                            R = c_val*Lc + N*tan_phi
                            FS_matrix[i,j] = R / max(abs(T), 1e-12)
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    im = ax2.imshow(FS_matrix, origin='lower', aspect='auto', 
                                   extent=[c_kgcm2_range[0], c_kgcm2_range[-1], 
                                          phi_deg_range[0], phi_deg_range[-1]])
                    
                    plt.colorbar(im, ax=ax2, label='Factor de Seguridad')
                    
                    # Define two sets of levels and colors
                    levels_req = [FS_required]  # Required FS level
                    levels_std = [1.0, 2.0, 3.0]  # Standard levels

                    # Plot standard contours in white
                    cs_std = ax2.contour(c_kgcm2_range, phi_deg_range, FS_matrix, 
                                         levels=levels_std, colors='white', linewidths=1.5)
                    ax2.clabel(cs_std, inline=True, fmt='%.1f', fontsize=10)

                    # Plot required FS contour in yellow with greater line width
                    cs_req = ax2.contour(c_kgcm2_range, phi_deg_range, FS_matrix, 
                                         levels=levels_req, colors='yellow', linewidths=2.5)
                    ax2.clabel(cs_req, inline=True, fmt='Required FS=%.1f', fontsize=10)
                    
                    ax2.set_xlabel('Cohesión (kg/cm²)', fontsize=11)
                    ax2.set_ylabel('Ángulo de fricción (°)', fontsize=11)
                    ax2.set_title(f'Factor de Seguridad al Deslizamiento\n(theta = {theta_deg}°)', fontsize=12)
                    
                    ax2.plot(c_kgcm2, phi_deg, 'w*', markersize=15, label='Current case')
                    ax2.legend()
                    st.pyplot(fig2)
                
                with tab3:
                    df_out = df.copy()
                    df_out["sigma_n_comp(magnitude)"] = sigma_n_comp
                    st.dataframe(df_out, use_container_width=True)
                    
                    # Download button for Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_out.to_excel(writer, sheet_name='Detailed_Results', index=False)
                        
                        summary = {
                            "input_file": uploaded_file.name,
                            "theta_deg_contact": theta_deg,
                            "phi_deg": phi_deg,
                            "c (stress units)": c,
                            "FS_required": FS_required,
                            "Lc (length units)": Lc,
                            "L_total (length units)": L_total,
                            "open_length (length units)": L_total - Lc,
                            "open_%": 100.0*((L_total - Lc)/L_total),
                            "N (force per unit thickness)": N,
                            "T (force per unit thickness)": T,
                            "R (force per unit thickness)": R,
                            "FS": FS,
                            "FS_ok": FS >= FS_required
                        }
                        pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download Excel Results",
                        data=output.getvalue(),
                        file_name="contact_stability_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
else:
    st.info("👈 Please upload a GiD stress file (.grf) to begin the analysis")
    
    st.markdown("""
    ### 📖 Instructions
    
    1. **Upload your file**: Use the sidebar to upload a GiD stress results file (.grf format)
    2. **Set parameters**: Adjust geometric and material parameters as needed
    3. **Configure ranges**: Set ranges for the parametric study
    4. **Run analysis**: Click the "Run Analysis" button
    5. **Review results**: Explore stress distributions, parametric study, and detailed data
    6. **Download**: Export results to Excel for further processing
    
    ### 📝 File Format
    The input file should contain three data blocks (Sxx, Syy, Sxy) in GiD graph format.
    """)
