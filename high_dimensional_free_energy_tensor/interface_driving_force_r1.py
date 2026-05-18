"""
CoCrFeNi Gibbs Free Energy Explorer
Enhanced with composition-dependent molar volume and driving force plots
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

# ================= CONFIGURATION =================
st.set_page_config(page_title="CoCrFeNi Phase Stability", layout="wide", page_icon="⚛️")
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure
""")

# ================= CONSTANTS =================
# Pure element molar volumes at 1000 K (m³/mol) – from literature
# Values: Co (FCC), Cr (BCC but use same), Fe (FCC), Ni (FCC)
PURE_VM = {
    "Co": 6.80e-6,
    "Cr": 7.23e-6,
    "Fe": 7.09e-6,
    "Ni": 6.59e-6
}
DEFAULT_VM = 7.2e-6  # constant fallback

# ================= DATA LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        return None, []
    data = {}
    for f in files:
        try:
            basename = Path(f).stem
            T = int(basename.replace("Gibbs_", "").replace("K", ""))
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
            df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
            data[T] = df[["Co", "Cr", "Fe", "G_LIQ", "G_FCC"]]
        except Exception as e:
            st.warning(f"⚠️ Skipping {os.path.basename(f)}: {e}")
    return data, sorted(data.keys())

data_by_T, temperatures = load_temperature_data(CSV_FILES_DIR)

if not data_by_T:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("Expected format: `Gibbs_300K.csv` with columns Co, Cr, Fe, Ni, G_LIQ, G_FCC")
    st.stop()

# ================= INTERPOLATORS =================
@st.cache_resource
def build_interpolators(data_dict, temp_list):
    liq_interp, fcc_interp = {}, {}
    for T in temp_list:
        df = data_dict[T]
        if len(df) < 4:
            continue
        points = df[["Co", "Cr", "Fe"]].values
        liq_interp[T] = LinearNDInterpolator(points, df["G_LIQ"].values, fill_value=np.nan)
        fcc_interp[T] = LinearNDInterpolator(points, df["G_FCC"].values, fill_value=np.nan)
    return liq_interp, fcc_interp

liq_interp, fcc_interp = build_interpolators(data_by_T, temperatures)

# ================= HELPER FUNCTIONS =================
def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
    """Rule of mixtures: Vm = Σ x_i * V_i,pure"""
    return (x_co * PURE_VM["Co"] + x_cr * PURE_VM["Cr"] +
            x_fe * PURE_VM["Fe"] + x_ni * PURE_VM["Ni"])

def evaluate_point(x_co, x_cr, x_fe, T, liq_int, fcc_int):
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq = liq_int[T](point)
        g_fcc = fcc_int[T](point)
        # Check for nan (outside convex hull)
        if np.isnan(g_liq) or np.isnan(g_fcc):
            return None, None
        return float(g_liq), float(g_fcc)
    except:
        return None, None

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Input Parameters")
T = st.sidebar.select_slider("Temperature (K)", options=temperatures,
                              value=1000 if 1000 in temperatures else temperatures[0])

col1, col2, col3 = st.sidebar.columns(3)
x_co = col1.number_input("x_Co", 0.0, 1.0, 0.25, 0.01)
x_cr = col2.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01)
x_fe = col3.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01)
x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid: x_Ni = {x_ni:.4f}")
    st.stop()
st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

# Molar volume model selection
st.sidebar.subheader("📐 Molar Volume Model")
vm_model = st.sidebar.radio("Model", ["Constant", "Composition‑dependent (1000 K values)"])
if vm_model == "Constant":
    V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e")
    st.sidebar.caption("Typical: 7.0–7.5 × 10⁻⁶ m³/mol")
else:
    V_m = composition_dependent_vm(x_co, x_cr, x_fe, x_ni)
    st.sidebar.metric("Calculated Vₘ", f"{V_m:.2e} m³/mol")
    st.sidebar.caption("Using rule of mixtures: Σ xᵢ Vᵢ⁰")

# ================= EVALUATION =================
g_liq, g_fcc = evaluate_point(x_co, x_cr, x_fe, T, liq_interp, fcc_interp)

# ================= RESULTS DISPLAY =================
st.header(f"📊 Results at T = {T} K")

if g_liq is None or g_fcc is None:
    st.warning("⚠️ Composition outside convex hull of training data. Try nearby values.")
    # Show available composition range hint
    df_sample = data_by_T[T]
    st.info(f"Available data: Co [{df_sample['Co'].min():.2f}, {df_sample['Co'].max():.2f}], "
            f"Cr [{df_sample['Cr'].min():.2f}, {df_sample['Cr'].max():.2f}], "
            f"Fe [{df_sample['Fe'].min():.2f}, {df_sample['Fe'].max():.2f}]")
else:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("G_LIQUID", f"{g_liq:,.1f} J/mol")
    col_b.metric("G_FCC", f"{g_fcc:,.1f} J/mol")

    delta_G = g_fcc - g_liq
    col_c.metric("ΔG = G_FCC − G_LIQ", f"{delta_G:,.1f} J/mol",
                 delta="FCC favored" if delta_G < 0 else "LIQUID favored")

    stable_phase = "FCC" if g_fcc < g_liq else "LIQUID"
    if stable_phase == "FCC":
        st.success(f"🏆 Most stable: {stable_phase}")
    else:
        st.warning(f"🏆 Most stable: {stable_phase}")

    st.divider()

    # ===== MECHANICAL CONVERSION =====
    st.subheader("⚙️ Interface Driving Force (Mechanical)")
    delta_G_v = delta_G / V_m  # Pa
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Driving Pressure ΔGᵥ", f"{delta_G_v/1e6:.3f} MPa")
    col_p2.metric("In SI units", f"{delta_G_v:.2e} N/m²")

    direction = "→ FCC grows" if delta_G < 0 else "→ LIQUID grows"
    col_p3.metric("Interface motion", direction)

    # Force calculator
    st.markdown("### 🔧 Force on Interface of Area A")
    A = st.number_input("Interface area A (m²)", 1e-12, 1e-2, 1e-8, 1e-10, format="%.2e")
    F_net = delta_G_v * A
    st.metric("Net force F = ΔGᵥ × A", f"{F_net:.3e} N")
    st.info(f"**Interpretation**: {abs(F_net):.2e} N per {A:.2e} m² – maximum thermodynamic driving force (kinetics depend on mobility).")

# ================= VISUALIZATION TOOLS =================
st.divider()
st.header("🗺️ Exploration Tools")
tab1, tab2, tab3, tab4 = st.tabs(["📈 G vs Composition", "🌡️ Phase Map vs T",
                                  "📊 ΔGᵥ vs T", "📋 Raw Data"])

with tab1:
    st.markdown("### Gibbs Energy along composition axis")
    scan_var = st.radio("Vary", ["x_Co", "x_Cr", "x_Fe"], horizontal=True)
    fixed_val = st.slider("Fixed value for other two components", 0.0, 0.4, 0.2, 0.01)
    max_val = 1.0 - 2*fixed_val - 0.01
    if max_val < 0.01:
        st.error("Fixed values too large. Reduce them.")
    else:
        x_vals = np.linspace(0.01, max_val, 100)
        g_liq_scan, g_fcc_scan = [], []
        for xv in x_vals:
            if scan_var == "x_Co":
                gl, gf = evaluate_point(xv, fixed_val, fixed_val, T, liq_interp, fcc_interp)
            elif scan_var == "x_Cr":
                gl, gf = evaluate_point(fixed_val, xv, fixed_val, T, liq_interp, fcc_interp)
            else:
                gl, gf = evaluate_point(fixed_val, fixed_val, xv, T, liq_interp, fcc_interp)
            g_liq_scan.append(gl if gl is not None else np.nan)
            g_fcc_scan.append(gf if gf is not None else np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=g_liq_scan, name="G_LIQUID",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=x_vals, y=g_fcc_scan, name="G_FCC",
                                 line=dict(color="#d62728", width=2)))
        fig.update_layout(title=f"Gibbs Energy vs {scan_var} at T={T} K (others={fixed_val})",
                          xaxis_title=scan_var, yaxis_title="G (J/mol)", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Phase stability vs Temperature (fixed composition)")
    T_scan = temperatures
    delta_G_list, delta_Gv_list = [], []
    valid_T = []
    for T_val in T_scan:
        gl, gf = evaluate_point(x_co, x_cr, x_fe, T_val, liq_interp, fcc_interp)
        if gl is not None and gf is not None:
            dG = gf - gl
            delta_G_list.append(dG)
            # Use composition-dependent Vm for each T? For simplicity use constant or current Vm model.
            if vm_model == "Composition‑dependent (1000 K values)":
                # We use the same pure Vm values (they are for 1000K) – ignore thermal expansion for clarity
                vm_local = composition_dependent_vm(x_co, x_cr, x_fe, x_ni)
            else:
                vm_local = V_m  # constant
            delta_Gv_list.append(dG / vm_local)
            valid_T.append(T_val)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=valid_T, y=delta_G_list, mode="lines+markers",
                              name="ΔG (J/mol)", yaxis="y1",
                              line=dict(color="blue", width=2)))
    fig2.add_trace(go.Scatter(x=valid_T, y=[v/1e6 for v in delta_Gv_list], mode="lines+markers",
                              name="ΔGᵥ (MPa)", yaxis="y2",
                              line=dict(color="red", width=2, dash="dot")))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", yaxis="y1")
    fig2.update_layout(
        title=f"Driving Force vs Temperature for Co{x_co:.2f}Cr{x_cr:.2f}Fe{x_fe:.2f}Ni{x_ni:.2f}",
        xaxis_title="Temperature (K)",
        yaxis=dict(title="ΔG (J/mol)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="ΔGᵥ (MPa)", titlefont=dict(color="red"), tickfont=dict(color="red"),
                    overlaying="y", side="right"),
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Driving Pressure vs Composition (ΔGᵥ in MPa)")
    # Scan along one axis and compute ΔGᵥ
    scan_var2 = st.radio("Scan along", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan_dgv")
    fixed_val2 = st.slider("Fixed other components", 0.0, 0.4, 0.2, 0.01, key="fixed_dgv")
    max_val2 = 1.0 - 2*fixed_val2 - 0.01
    if max_val2 < 0.01:
        st.error("Fixed values too large.")
    else:
        x_vals2 = np.linspace(0.01, max_val2, 100)
        dGv_vals = []
        for xv in x_vals2:
            if scan_var2 == "x_Co":
                gl, gf = evaluate_point(xv, fixed_val2, fixed_val2, T, liq_interp, fcc_interp)
            elif scan_var2 == "x_Cr":
                gl, gf = evaluate_point(fixed_val2, xv, fixed_val2, T, liq_interp, fcc_interp)
            else:
                gl, gf = evaluate_point(fixed_val2, fixed_val2, xv, T, liq_interp, fcc_interp)
            if gl is not None and gf is not None:
                dG = gf - gl
                if vm_model == "Composition‑dependent (1000 K values)":
                    # Need full composition for Vm
                    if scan_var2 == "x_Co":
                        vm_local = composition_dependent_vm(xv, fixed_val2, fixed_val2, 1-xv-2*fixed_val2)
                    elif scan_var2 == "x_Cr":
                        vm_local = composition_dependent_vm(fixed_val2, xv, fixed_val2, 1-xv-2*fixed_val2)
                    else:
                        vm_local = composition_dependent_vm(fixed_val2, fixed_val2, xv, 1-xv-2*fixed_val2)
                else:
                    vm_local = V_m
                dGv_vals.append(dG / vm_local / 1e6)  # MPa
            else:
                dGv_vals.append(np.nan)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x_vals2, y=dGv_vals, mode="lines",
                                  line=dict(color="green", width=3),
                                  fill="tozeroy"))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(title=f"Driving Pressure ΔGᵥ vs {scan_var2} at T={T} K",
                           xaxis_title=scan_var2, yaxis_title="ΔGᵥ (MPa)",
                           height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Positive ΔGᵥ → LIQUID grows; negative → FCC grows.")

with tab4:
    st.dataframe(data_by_T[T].head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(data_by_T[T])} data points at T={T} K")

st.divider()
st.caption("""
**References**:
- Driving force conversion: Porter & Easterling, *Phase Transformations in Metals and Alloys*
- Pure element molar volumes at 1000 K: Mills, *The International Journal of Thermophysics* (2002)
- Interpolation: Scipy `LinearNDInterpolator`
""")
