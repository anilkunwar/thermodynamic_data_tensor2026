"""
CoCrFeNi Gibbs Free Energy Explorer
Robust interpolation with convex hull handling & CSV parsing fixes
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from pathlib import Path

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ================= CONFIGURATION =================
st.set_page_config(page_title="CoCrFeNi Phase Stability", layout="wide", page_icon="⚛️")
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure
""")

# ================= CONSTANTS =================
DEFAULT_VM = 7.2e-6  # m³/mol

# ================= ROBUST CSV LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    """Load CSV files with robust parsing for malformed headers."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    
    if not files:
        return None, [], {}
    
    data = {}
    bounds_info = {}
    
    for f in files:
        try:
            T = int(Path(f).stem.replace("Gibbs_", "").replace("K", ""))
            
            # Read raw file to handle header issues
            with open(f, 'r') as file:
                first_line = file.readline().strip()
            
            # Check if header is merged with data (common issue)
            if "G_LIQ" in first_line and "," in first_line.split("G_LIQ")[1][:20]:
                # Header is OK, use normal read
                df = pd.read_csv(f)
            else:
                # Try reading with explicit header inference
                df = pd.read_csv(f, header=0, engine='python', on_bad_lines='skip')
            
            # Ensure required columns exist
            required_cols = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]
            if not all(col in df.columns for col in required_cols):
                # Try to fix column names (strip whitespace)
                df.columns = df.columns.str.strip()
            
            if not all(col in df.columns for col in required_cols):
                st.warning(f"⚠️ Skipping {os.path.basename(f)}: Missing required columns")
                continue
            
            # Clean data
            df = df[required_cols].copy()
            df = df.dropna()
            
            # Validate mole fractions sum to ~1
            df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
            df = df[np.abs(df["sum_x"] - 1.0) < 1e-4].copy()
            
            if len(df) < 10:
                continue
                
            data[T] = df[["Co", "Cr", "Fe", "G_LIQ", "G_FCC"]]
            
            # Store bounds for this temperature
            bounds_info[T] = {
                "Co": (df["Co"].min(), df["Co"].max()),
                "Cr": (df["Cr"].min(), df["Cr"].max()),
                "Fe": (df["Fe"].min(), df["Fe"].max()),
                "Ni": (df["Ni"].min(), df["Ni"].max()),
            }
            
        except Exception as e:
            st.warning(f"⚠️ Error loading {os.path.basename(f)}: {e}")
            continue
    
    if not data:
        return None, [], {}
    
    return data, sorted(data.keys()), bounds_info

data_by_T, temperatures, bounds_info = load_temperature_data(CSV_FILES_DIR)

# Handle no data case
if data_by_T is None or len(temperatures) == 0:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("""
    **Expected file format**: `Gibbs_300K.csv`, `Gibbs_400K.csv`, ..., `Gibbs_3300K.csv`
    
    **Required columns**: `Co`, `Cr`, `Fe`, `Ni`, `G_LIQ`, `G_FCC`
    
    **Tip**: Ensure your CSV has proper newlines between header and data rows.
    """)
    st.stop()

# ================= ROBUST INTERPOLATION =================
@st.cache_resource
def build_interpolators(data_dict, temp_list):
    """Build interpolators with nearest-neighbor fallback for out-of-hull points."""
    interpolators = {}
    
    for T in temp_list:
        df = data_dict[T]
        if len(df) < 4:
            continue
            
        points = df[["Co", "Cr", "Fe"]].values
        g_liq_vals = df["G_LIQ"].values
        g_fcc_vals = df["G_FCC"].values
        
        # Primary: Linear interpolation (accurate inside hull)
        liq_linear = LinearNDInterpolator(points, g_liq_vals)
        fcc_linear = LinearNDInterpolator(points, g_fcc_vals)
        
        # Fallback: Nearest neighbor (works everywhere)
        liq_nearest = NearestNDInterpolator(points, g_liq_vals)
        fcc_nearest = NearestNDInterpolator(points, g_fcc_vals)
        
        interpolators[T] = {
            "liq_linear": liq_linear,
            "fcc_linear": fcc_linear,
            "liq_nearest": liq_nearest,
            "fcc_nearest": fcc_nearest,
            "points": points,
        }
    
    return interpolators

interpolators = build_interpolators(data_by_T, temperatures)

def evaluate_point_robust(x_co, x_cr, x_fe, T, interps, use_nearest_fallback=True):
    """Evaluate with automatic fallback to nearest-neighbor if outside convex hull."""
    point = np.array([[x_co, x_cr, x_fe]])
    
    try:
        g_liq = float(interps[T]["liq_linear"](point))
        g_fcc = float(interps[T]["fcc_linear"](point))
        
        # Check for NaN (outside convex hull)
        if np.isnan(g_liq) or np.isnan(g_fcc):
            if use_nearest_fallback:
                g_liq = float(interps[T]["liq_nearest"](point))
                g_fcc = float(interps[T]["fcc_nearest"](point))
                return g_liq, g_fcc, "nearest"
            return None, None, "outside"
        return g_liq, g_fcc, "linear"
    except:
        if use_nearest_fallback:
            g_liq = float(interps[T]["liq_nearest"](point))
            g_fcc = float(interps[T]["fcc_nearest"](point))
            return g_liq, g_fcc, "nearest"
        return None, None, "error"

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Input Parameters")

T = st.sidebar.select_slider("Temperature (K)", options=temperatures, value=1000 if 1000 in temperatures else temperatures[0])

# Show composition bounds for current T
if T in bounds_info:
    with st.sidebar.expander("📏 Valid Composition Ranges at T={}K".format(T)):
        for elem in ["Co", "Cr", "Fe", "Ni"]:
            vmin, vmax = bounds_info[T][elem]
            st.text(f"{elem}: {vmin:.3f} – {vmax:.3f}")

col1, col2, col3 = st.sidebar.columns(3)

# Set smart defaults based on data bounds
if T in bounds_info:
    def smart_default(elem):
        vmin, vmax = bounds_info[T][elem]
        return round((vmin + vmax) / 2, 2)
    
    default_co = smart_default("Co")
    default_cr = smart_default("Cr")
    default_fe = smart_default("Fe")
else:
    default_co, default_cr, default_fe = 0.25, 0.25, 0.25

x_co = col1.number_input("x_Co", 0.0, 1.0, default_co, 0.01)
x_cr = col2.number_input("x_Cr", 0.0, 1.0, default_cr, 0.01)
x_fe = col3.number_input("x_Fe", 0.0, 1.0, default_fe, 0.01)

x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid: x_Ni = {x_ni:.4f}")
    st.stop()

st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

# Molar volume
st.sidebar.subheader("📐 Molar Volume")
V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e")

# ================= EVALUATION =================
if T not in interpolators:
    st.error(f"No interpolator available for T={T}K")
    st.stop()

g_liq, g_fcc, method = evaluate_point_robust(x_co, x_cr, x_fe, T, interpolators)

# ================= RESULTS DISPLAY =================
st.header(f"📊 Results at T = {T} K")

if g_liq is None or g_fcc is None:
    st.error("⚠️ Could not evaluate at this composition. Try values within the data bounds shown in sidebar.")
else:
    # Show interpolation method used
    if method == "nearest":
        st.warning("🔍 Using nearest-neighbor interpolation (point outside convex hull of training data)")
    else:
        st.success("✨ Using linear interpolation (point inside convex hull)")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("G_LIQUID", f"{g_liq:,.1f} J/mol")
    col_b.metric("G_FCC", f"{g_fcc:,.1f} J/mol")
    
    delta_G = g_fcc - g_liq
    col_c.metric("ΔG = G_FCC − G_LIQ", f"{delta_G:,.1f} J/mol",
                 delta="FCC favored" if delta_G < 0 else "LIQUID favored")
    
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
    
    direction = "→ FCC grows" if delta_G < 0 else "→ LIQUID grows"
    col_p3.metric("Interface motion", direction)
    
    # Force calculator
    st.markdown("### 🔧 Force on Interface of Area A")
    A = st.number_input("Interface area A (m²)", 1e-12, 1e-2, 1e-8, 1e-10, format="%.2e")
    F_net = delta_G_v * A
    
    st.metric("Net force F = ΔGᵥ × A", f"{F_net:.3e} N")
    
    st.info(f"""
    **Interpretation**:  
    - ΔGᵥ < 0 → thermodynamic pressure pushes interface toward LIQUID  
    - For A = {A:.2e} m², net force = {abs(F_net):.2e} N  
    - This is the *maximum* driving force; actual kinetics depend on mobility & diffusion.
    """)

# ================= VISUALIZATION =================
st.divider()
st.header("🗺️ Exploration Tools")

tab1, tab2, tab3 = st.tabs(["📈 G vs Composition", "🌡️ Phase Map vs T", "📋 Raw Data"])

with tab1:
    st.markdown("### Scan along one composition axis")
    
    scan_var = st.radio("Vary", ["x_Co", "x_Cr", "x_Fe"], horizontal=True)
    fixed_val = st.slider("Fixed value for other two", 0.0, 0.4, 0.2, 0.01)
    
    max_val = 1.0 - 2*fixed_val - 0.01
    if max_val < 0.01:
        st.error("Reduce fixed values for valid scan range.")
    else:
        x_vals = np.linspace(0.01, max_val, 100)
        g_liq_scan, g_fcc_scan, methods_used = [], [], []
        
        for xv in x_vals:
            if scan_var == "x_Co":
                gl, gf, meth = evaluate_point_robust(xv, fixed_val, fixed_val, T, interpolators)
            elif scan_var == "x_Cr":
                gl, gf, meth = evaluate_point_robust(fixed_val, xv, fixed_val, T, interpolators)
            else:
                gl, gf, meth = evaluate_point_robust(fixed_val, fixed_val, xv, T, interpolators)
            g_liq_scan.append(gl)
            g_fcc_scan.append(gf)
            methods_used.append(meth)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=g_liq_scan, name="G_LIQUID", 
                                line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=x_vals, y=g_fcc_scan, name="G_FCC", 
                                line=dict(color="#d62728", width=2)))
        
        # Add stability shading
        for i in range(len(x_vals)-1):
            if g_fcc_scan[i] is not None and g_liq_scan[i] is not None:
                color = "rgba(214,39,40,0.1)" if g_fcc_scan[i] < g_liq_scan[i] else "rgba(31,119,180,0.1)"
                fig.add_vrect(x0=x_vals[i], x1=x_vals[i+1], fillcolor=color, layer="below", line_width=0)
        
        fig.update_layout(
            title=f"Gibbs Energy vs {scan_var} at T={T} K",
            xaxis_title=scan_var, yaxis_title="G (J/mol)",
            hovermode="x unified", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Phase stability vs Temperature")
    
    T_scan = temperatures
    stable_phases, delta_G_scan = [], []
    
    for T_val in T_scan:
        gl, gf, _ = evaluate_point_robust(x_co, x_cr, x_fe, T_val, interpolators)
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
                            name="ΔG = G_FCC − G_LIQ"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"Phase Stability vs T for Co-{x_co:.2f}Cr-{x_cr:.2f}Fe-{x_fe:.2f}Ni",
        xaxis_title="Temperature (K)", yaxis_title="ΔG (J/mol)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(data_by_T[T].head(50), use_container_width=True)
    st.caption(f"First 50 of {len(data_by_T[T])} points at T={T} K")

# ================= FOOTER =================
st.divider()
st.caption("""
**References**:  
- Porter & Easterling, *Phase Transformations in Metals and Alloys*  
- Otto et al., *Acta Materialia* 61 (2013) — molar volume estimates  
- Interpolation: Scipy `LinearNDInterpolator` + `NearestNDInterpolator` fallback
""")
