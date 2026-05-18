"""
CoCrFeNi Gibbs Free Energy Explorer
Calculates phase stability and interface driving forces
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
# PATH CONFIGURATION (FIXED)
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# Ensure directories exist
os.makedirs(CSV_FILES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Add script dir to path for imports if needed
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ================= CONFIGURATION =================
st.set_page_config(page_title="CoCrFeNi Phase Stability", layout="wide", page_icon="⚛️")
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure
""")

# Show debug info in sidebar (optional, remove in production)
with st.sidebar.expander("🔍 Path Debug Info"):
    st.code(f"SCRIPT_DIR: {SCRIPT_DIR}")
    st.code(f"CSV_FILES_DIR: {CSV_FILES_DIR}")
    files_found = glob.glob(os.path.join(CSV_FILES_DIR, "Gibbs_*.csv"))
    st.write(f"📁 CSV files found: {len(files_found)}")
    if files_found:
        st.write("First 3 files:")
        for f in files_found[:3]:
            st.text(os.path.basename(f))

# ================= CONSTANTS =================
DEFAULT_VM = 7.2e-6  # m³/mol (typical for CoCrFeNi HEA)

# ================= DATA LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    """Load all Gibbs_*.csv files into a dictionary keyed by temperature."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    
    if not files:
        return None, []
    
    data = {}
    for f in files:
        try:
            # Extract temperature from filename: Gibbs_1000K.csv -> 1000
            basename = Path(f).stem  # "Gibbs_1000K"
            T = int(basename.replace("Gibbs_", "").replace("K", ""))
            
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            
            # Validate mole fractions sum to ~1 (tolerance for floating point)
            df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
            df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
            
            data[T] = df[["Co", "Cr", "Fe", "G_LIQ", "G_FCC"]]
        except Exception as e:
            st.warning(f"⚠️ Skipping {os.path.basename(f)}: {e}")
    
    if not data:
        return None, []
    
    return data, sorted(data.keys())

# Try to load data
data_by_T, temperatures = load_temperature_data(CSV_FILES_DIR)

# Handle no data case
if data_by_T is None or len(temperatures) == 0:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("""
    **Expected file format**: `Gibbs_300K.csv`, `Gibbs_400K.csv`, ..., `Gibbs_3300K.csv`
    
    **Required columns**: `Co`, `Cr`, `Fe`, `Ni`, `G_LIQ`, `G_FCC`
    
    **Directory structure**:
    ```
    your_project/
    ├── app.py
    └── csv_files/
        ├── Gibbs_300K.csv
        ├── Gibbs_400K.csv
        └── ...
    ```
    """)
    st.stop()

# ================= INTERPOLATORS =================
@st.cache_resource
def build_interpolators(data_dict, temp_list):
    """Build 3D linear interpolators for each temperature slice."""
    liq_interp, fcc_interp = {}, {}
    
    for T in temp_list:
        df = data_dict[T]
        if len(df) < 4:  # Need minimum points for interpolation
            continue
        points = df[["Co", "Cr", "Fe"]].values
        liq_interp[T] = LinearNDInterpolator(points, df["G_LIQ"].values)
        fcc_interp[T] = LinearNDInterpolator(points, df["G_FCC"].values)
    
    return liq_interp, fcc_interp

liq_interp, fcc_interp = build_interpolators(data_by_T, temperatures)

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Input Parameters")

T = st.sidebar.select_slider("Temperature (K)", options=temperatures, value=1000 if 1000 in temperatures else temperatures[0])

col1, col2, col3 = st.sidebar.columns(3)
x_co = col1.number_input("x_Co", 0.0, 1.0, 0.25, 0.01)
x_cr = col2.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01)
x_fe = col3.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01)

x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid composition: x_Ni = {x_ni:.4f} (must be 0–1)")
    st.stop()

st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

# Molar volume input
st.sidebar.subheader("📐 Molar Volume")
V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e")
st.sidebar.caption("Typical: 7.0–7.5 × 10⁻⁶ m³/mol for CoCrFeNi")

# ================= EVALUATION =================
def evaluate_point(x_co, x_cr, x_fe, T, liq_int, fcc_int):
    """Evaluate Gibbs energies at a composition point."""
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq = float(liq_int[T](point))
        g_fcc = float(fcc_int[T](point))
        return g_liq, g_fcc
    except:
        return None, None

g_liq, g_fcc = evaluate_point(x_co, x_cr, x_fe, T, liq_interp, fcc_interp)

# ================= RESULTS DISPLAY =================
st.header(f"📊 Results at T = {T} K")

if g_liq is None or g_fcc is None:
    st.warning("⚠️ Composition outside convex hull of training data. Try nearby values.")
else:
    # Molar quantities
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("G_LIQUID", f"{g_liq:,.1f} J/mol")
    col_b.metric("G_FCC", f"{g_fcc:,.1f} J/mol")
    
    delta_G = g_fcc - g_liq  # J/mol
    col_c.metric("ΔG = G_FCC − G_LIQ", f"{delta_G:,.1f} J/mol",
                 delta="FCC favored" if delta_G < 0 else "LIQUID favored")
    
    # Phase stability
    stable_phase = "FCC" if g_fcc < g_liq else "LIQUID"
    if stable_phase == "FCC":
        st.success(f"🏆 **Most stable phase: {stable_phase}**")
    else:
        st.warning(f"🏆 **Most stable phase: {stable_phase}**")
    
    st.divider()
    
    # ===== MECHANICAL CONVERSION =====
    st.subheader("⚙️ Interface Driving Force (Mechanical)")
    
    delta_G_v = delta_G / V_m  # Pa = N/m²
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Driving Pressure ΔGᵥ", f"{delta_G_v/1e6:.3f} MPa")
    col_p2.metric("In SI units", f"{delta_G_v:.2e} N/m²")
    
    # Direction indicator
    direction = "→ FCC grows" if delta_G < 0 else "→ LIQUID grows"
    col_p3.metric("Interface motion", direction)
    
    # Force calculator
    st.markdown("### 🔧 Force on Interface of Area A")
    A = st.number_input("Interface area A (m²)", 1e-12, 1e-2, 1e-8, 1e-10, format="%.2e")
    F_net = delta_G_v * A  # Newtons
    
    st.metric("Net force F = ΔGᵥ × A", f"{F_net:.3e} N")
    
    # Physical interpretation
    st.info(f"""
    **Interpretation**:  
    - A **negative** ΔGᵥ means the interface experiences a thermodynamic "pressure" 
      pushing it toward the **higher-energy phase** ({'LIQUID' if g_liq > g_fcc else 'FCC'}).  
    - For A = {A:.2e} m², the net force is {abs(F_net):.2e} N.  
    - This is the **maximum thermodynamic driving force**; actual kinetics depend on 
      mobility, diffusion, and interface structure.
    """)

# ================= VISUALIZATION =================
st.divider()
st.header("🗺️ Exploration Tools")

tab1, tab2, tab3 = st.tabs(["📈 G vs Composition", "🌡️ Phase Map vs T", "📋 Raw Data"])

with tab1:
    st.markdown("### Scan along one composition axis")
    
    scan_var = st.radio("Vary", ["x_Co", "x_Cr", "x_Fe"], horizontal=True)
    fixed_val = st.slider("Fixed value for other two components", 0.0, 0.4, 0.2, 0.01)
    
    max_val = 1.0 - 2*fixed_val - 0.01
    if max_val < 0.01:
        st.error("Fixed values too large for valid scan. Reduce them.")
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
            g_liq_scan.append(gl)
            g_fcc_scan.append(gf)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=g_liq_scan, name="G_LIQUID", 
                                line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=x_vals, y=g_fcc_scan, name="G_FCC", 
                                line=dict(color="#d62728", width=2)))
        
        fig.update_layout(
            title=f"Gibbs Energy vs {scan_var} at T={T} K (others={fixed_val})",
            xaxis_title=scan_var, yaxis_title="G (J/mol)",
            hovermode="x unified", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Phase stability vs Temperature (fixed composition)")
    
    x_co_fix, x_cr_fix, x_fe_fix = x_co, x_cr, x_fe
    T_scan = temperatures
    stable_phases, delta_G_scan = [], []
    
    for T_scan_val in T_scan:
        gl, gf = evaluate_point(x_co_fix, x_cr_fix, x_fe_fix, T_scan_val, liq_interp, fcc_interp)
        if gl is not None and gf is not None:
            stable_phases.append("FCC" if gf < gl else "LIQUID")
            delta_G_scan.append(gf - gl)
        else:
            stable_phases.append(None)
            delta_G_scan.append(None)
    
    fig = go.Figure()
    colors = ["red" if p == "FCC" else "blue" if p == "LIQUID" else "gray" for p in stable_phases]
    
    fig.add_trace(go.Scatter(x=T_scan, y=delta_G_scan, mode="markers+lines",
                            marker=dict(color=colors, size=6),
                            name="ΔG = G_FCC − G_LIQ",
                            line=dict(width=1)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"Phase Stability vs T for Co{x_co_fix:.2f}Cr{x_cr_fix:.2f}Fe{x_fe_fix:.2f}Ni{1-x_co_fix-x_cr_fix-x_fe_fix:.2f}",
        xaxis_title="Temperature (K)", yaxis_title="ΔG (J/mol)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(data_by_T[T].head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(data_by_T[T])} data points at T={T} K")

# ================= FOOTER =================
st.divider()
st.caption("""
**References**:  
- Driving force conversion: Porter & Easterling, *Phase Transformations in Metals and Alloys*  
- Molar volume estimate: Otto et al., *Acta Materialia* 61 (2013) 5743–5755  
- Interpolation: Scipy `LinearNDInterpolator` (Delaunay triangulation in composition space)
""")
