"""
CoCrFeNi Gibbs Free Energy Explorer
Optimized with RegularGridInterpolator + Build Button
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from pathlib import Path

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(page_title="CoCrFeNi Phase Stability", layout="wide", page_icon="⚛️")
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure
""")

# ================= CONSTANTS =================
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6

# ================= DATA LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        return None, []
    data = {}
    for f in files:
        basename = Path(f).stem
        T = int(basename.replace("Gibbs_", "").replace("K", ""))
        df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
        data[T] = df
    return data, sorted(data.keys())

data_by_T, temperatures = load_temperature_data(CSV_FILES_DIR)
if not data_by_T:
    st.error(f"❌ No valid CSV files in `{CSV_FILES_DIR}`")
    st.stop()

# ================= CHECK GRID REGULARITY =================
def is_regular_grid(df):
    """Check if composition points form a regular grid (all combinations)."""
    n_co = df["Co"].nunique()
    n_cr = df["Cr"].nunique()
    n_fe = df["Fe"].nunique()
    expected = n_co * n_cr * n_fe
    return len(df) == expected

# Store grid status per temperature
grid_regular = {T: is_regular_grid(df) for T, df in data_by_T.items()}
if not any(grid_regular.values()):
    st.warning("⚠️ Data not on regular grid – using slower LinearNDInterpolator.")
    USE_REGULAR = False
else:
    USE_REGULAR = True
    st.info("✅ Data on regular grid – using fast RegularGridInterpolator.")

# ================= INTERPOLATOR BUILD (with button) =================
@st.cache_resource(ttl=3600)
def build_regular_interpolator(T, phase):
    """Build a RegularGridInterpolator for a given T and phase."""
    df = data_by_T[T]
    co_vals = np.sort(df["Co"].unique())
    cr_vals = np.sort(df["Cr"].unique())
    fe_vals = np.sort(df["Fe"].unique())
    # Reshape values: order must match the sorting above
    df_sorted = df.sort_values(["Co", "Cr", "Fe"])
    values = df_sorted[f"G_{phase}"].values.reshape(len(co_vals), len(cr_vals), len(fe_vals))
    return RegularGridInterpolator((co_vals, cr_vals, fe_vals), values,
                                   bounds_error=False, fill_value=np.nan)

@st.cache_resource(ttl=3600)
def build_linearnd_interpolator(T, phase):
    """Fallback: LinearNDInterpolator."""
    df = data_by_T[T]
    points = df[["Co", "Cr", "Fe"]].values
    values = df[f"G_{phase}"].values
    return LinearNDInterpolator(points, values, fill_value=np.nan)

# Initialize storage for interpolators
if "interpolators" not in st.session_state:
    st.session_state.interpolators = {"LIQ": {}, "FCC": {}}
if "interpolators_built" not in st.session_state:
    st.session_state.interpolators_built = False

def build_all_interpolators():
    """Build all interpolators for all temperatures (called by button)."""
    progress_bar = st.progress(0, text="Building interpolators...")
    total = len(temperatures) * 2
    for i, T in enumerate(temperatures):
        for phase in ["LIQ", "FCC"]:
            if USE_REGULAR:
                interp = build_regular_interpolator(T, phase)
            else:
                interp = build_linearnd_interpolator(T, phase)
            st.session_state.interpolators[phase][T] = interp
            progress_bar.progress((i*2 + (1 if phase=="FCC" else 0)) / total)
    time.sleep(0.5)
    st.session_state.interpolators_built = True
    progress_bar.empty()

# Sidebar: Build button
with st.sidebar:
    st.header("⚡ Performance")
    if st.button("🚀 Build All Interpolators", type="primary", use_container_width=True):
        with st.spinner("Building interpolators (this may take 10-30 seconds)..."):
            build_all_interpolators()
        st.success("✅ Interpolators ready!")
    st.caption("Build once; subsequent queries will be instant.")

# Evaluation function (uses built interpolators if available)
#
def evaluate_point(x_co, x_cr, x_fe, T):
    """Return (G_LIQ, G_FCC) as floats, or (None, None) on failure."""
    # Obtain interpolators (built or on‑the‑fly)
    if not st.session_state.interpolators_built:
        if USE_REGULAR:
            interp_liq = build_regular_interpolator(T, "LIQ")
            interp_fcc = build_regular_interpolator(T, "FCC")
        else:
            interp_liq = build_linearnd_interpolator(T, "LIQ")
            interp_fcc = build_linearnd_interpolator(T, "FCC")
    else:
        interp_liq = st.session_state.interpolators["LIQ"].get(T)
        interp_fcc = st.session_state.interpolators["FCC"].get(T)
        if interp_liq is None or interp_fcc is None:
            # Rebuild on demand
            if USE_REGULAR:
                interp_liq = build_regular_interpolator(T, "LIQ")
                interp_fcc = build_regular_interpolator(T, "FCC")
            else:
                interp_liq = build_linearnd_interpolator(T, "LIQ")
                interp_fcc = build_linearnd_interpolator(T, "FCC")
    
    # If interpolators are still None, bail out
    if interp_liq is None or interp_fcc is None:
        return None, None
    
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq = interp_liq(point)
        g_fcc = interp_fcc(point)
    except Exception:
        return None, None
    
    # Convert numpy arrays to scalar (e.g., array(123.4) → 123.4)
    if hasattr(g_liq, 'item'):
        g_liq = g_liq.item()
    if hasattr(g_fcc, 'item'):
        g_fcc = g_fcc.item()
    
    # Validate values
    if g_liq is None or g_fcc is None or np.isnan(g_liq) or np.isnan(g_fcc):
        return None, None
    
    return g_liq, g_fcc

# ================= HELPER FUNCTIONS =================
def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
    return (x_co * PURE_VM["Co"] + x_cr * PURE_VM["Cr"] +
            x_fe * PURE_VM["Fe"] + x_ni * PURE_VM["Ni"])

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Composition & Temperature")
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

# Molar volume model
st.sidebar.subheader("📐 Molar Volume Model")
vm_model = st.sidebar.radio("Model", ["Constant", "Composition‑dependent"], index=1)
if vm_model == "Constant":
    V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e")
else:
    V_m = composition_dependent_vm(x_co, x_cr, x_fe, x_ni)
    st.sidebar.metric("Calculated Vₘ", f"{V_m:.2e} m³/mol")

# ================= RESULTS DISPLAY =================
st.header(f"📊 Results at T = {T} K")
g_liq, g_fcc = evaluate_point(x_co, x_cr, x_fe, T)

if g_liq is None or g_fcc is None:
    st.warning("⚠️ Composition outside convex hull of training data. Try nearby values.")
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
    st.success(f"🏆 Most stable: {stable_phase}")

    st.divider()
    st.subheader("⚙️ Interface Driving Force (Mechanical)")
    delta_G_v = delta_G / V_m
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Driving Pressure ΔGᵥ", f"{delta_G_v/1e6:.3f} MPa")
    col_p2.metric("SI units", f"{delta_G_v:.2e} N/m²")
    direction = "→ FCC grows" if delta_G < 0 else "→ LIQUID grows"
    col_p3.metric("Interface motion", direction)

    st.markdown("### 🔧 Force on Interface Area A")
    A = st.number_input("Area A (m²)", 1e-12, 1e-2, 1e-8, 1e-10, format="%.2e")
    st.metric("Net force F = ΔGᵥ × A", f"{delta_G_v * A:.3e} N")
    st.info(f"**Interpretation**: Maximum thermodynamic driving force; actual kinetics depend on mobility.")

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
        st.error("Fixed values too large.")
    else:
        x_vals = np.linspace(0.01, max_val, 100)
        g_liq_scan, g_fcc_scan = [], []
        for xv in x_vals:
            if scan_var == "x_Co":
                gl, gf = evaluate_point(xv, fixed_val, fixed_val, T)
            elif scan_var == "x_Cr":
                gl, gf = evaluate_point(fixed_val, xv, fixed_val, T)
            else:
                gl, gf = evaluate_point(fixed_val, fixed_val, xv, T)
            g_liq_scan.append(gl if gl is not None else np.nan)
            g_fcc_scan.append(gf if gf is not None else np.nan)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=g_liq_scan, name="G_LIQUID"))
        fig.add_trace(go.Scatter(x=x_vals, y=g_fcc_scan, name="G_FCC"))
        fig.update_layout(title=f"Gibbs Energy vs {scan_var} at T={T} K (others={fixed_val})",
                          xaxis_title=scan_var, yaxis_title="G (J/mol)", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Phase stability vs Temperature (fixed composition)")
    T_scan = temperatures
    delta_G_list, delta_Gv_list = [], []
    valid_T = []
    for T_val in T_scan:
        gl, gf = evaluate_point(x_co, x_cr, x_fe, T_val)
        if gl is not None and gf is not None:
            dG = gf - gl
            delta_G_list.append(dG)
            vm_local = composition_dependent_vm(x_co, x_cr, x_fe, x_ni) if vm_model == "Composition‑dependent" else V_m
            delta_Gv_list.append(dG / vm_local / 1e6)  # MPa
            valid_T.append(T_val)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=valid_T, y=delta_G_list, mode="lines+markers",
                              name="ΔG (J/mol)", yaxis="y1"))
    fig2.add_trace(go.Scatter(x=valid_T, y=delta_Gv_list, mode="lines+markers",
                              name="ΔGᵥ (MPa)", yaxis="y2", line=dict(dash="dot")))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", yaxis="y1")
    fig2.update_layout(
        title=f"Driving Force vs Temperature for Co{x_co:.2f}Cr{x_cr:.2f}Fe{x_fe:.2f}Ni{x_ni:.2f}",
        xaxis_title="Temperature (K)",
        yaxis=dict(title="ΔG (J/mol)", titlefont=dict(color="blue")),
        yaxis2=dict(title="ΔGᵥ (MPa)", titlefont=dict(color="red"), overlaying="y", side="right"),
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Driving Pressure vs Composition (ΔGᵥ in MPa)")
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
                gl, gf = evaluate_point(xv, fixed_val2, fixed_val2, T)
                if gl is not None:
                    vm_local = composition_dependent_vm(xv, fixed_val2, fixed_val2, 1-xv-2*fixed_val2) if vm_model == "Composition‑dependent" else V_m
            elif scan_var2 == "x_Cr":
                gl, gf = evaluate_point(fixed_val2, xv, fixed_val2, T)
                if gl is not None:
                    vm_local = composition_dependent_vm(fixed_val2, xv, fixed_val2, 1-xv-2*fixed_val2) if vm_model == "Composition‑dependent" else V_m
            else:
                gl, gf = evaluate_point(fixed_val2, fixed_val2, xv, T)
                if gl is not None:
                    vm_local = composition_dependent_vm(fixed_val2, fixed_val2, xv, 1-xv-2*fixed_val2) if vm_model == "Composition‑dependent" else V_m
            if gl is not None and gf is not None:
                dGv_vals.append((gf - gl) / vm_local / 1e6)
            else:
                dGv_vals.append(np.nan)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x_vals2, y=dGv_vals, mode="lines", fill="tozeroy"))
        fig3.add_hline(y=0, line_dash="dash")
        fig3.update_layout(title=f"Driving Pressure ΔGᵥ vs {scan_var2} at T={T} K",
                           xaxis_title=scan_var2, yaxis_title="ΔGᵥ (MPa)", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Positive → LIQUID grows; negative → FCC grows.")

with tab4:
    st.dataframe(data_by_T[T].head(50), use_container_width=True)

st.divider()
st.caption("""
**References**:
- Porter & Easterling, *Phase Transformations in Metals and Alloys*
- Pure element molar volumes at 1000 K: Mills, *Int. J. Thermophys.* (2002)
- Grid interpolation: SciPy `RegularGridInterpolator`
""")
