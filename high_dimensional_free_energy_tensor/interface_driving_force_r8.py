"""
CoCrFeNi Gibbs Free Energy Explorer
OPTIMIZED FOR STREAMLIT CLOUD (1GB RAM limit)
WITH: Explicit Run Buttons, Lazy temperature loading, cached interpolators
PLUS: Sunburst Charts, Radar Charts, LaTeX Theory Documentation
PLUS: Grain Size Derived Interfacial Area Density (Sv) & Net Force
PLUS: Capillary Pressure Correction & Differential Force Model
PLUS: Literature-Based Parameter Ranges & Uncertainty Quantification

LITERATURE SOURCES FOR PARAMETERS:
- Interfacial Energy γ: Kaptay model, Turnbull relation, experimental groove methods
- Shape Factor k: Smith-Guttman stereology, DeHoff & Rhines tables, Underwood
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION - Cloud Safe
# =============================================
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_FILES_DIR = SCRIPT_DIR / "csv_files"
CSV_FILES_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CoCrFeNi Phase Stability",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure  
**New:** Grain size → $S_v$ → Total area $A_{total}$ → Net force $F_{total}$  
**Advanced:** Capillary correction → $P_{net}$ → Differential force $dF_{net}$ on μm³ element  
**Enhanced:** Literature-based parameter ranges with uncertainty quantification  
**Optimized:** Explicit execution buttons + lazy loading for Streamlit Cloud
""")

# ================= CONSTANTS - LITERATURE-BASED RANGES =================
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6
T_MIN_NORMALIZE, T_MAX_NORMALIZE = 300, 3300

LIT_GAMMA_RANGES = {
    "Pure metals (Turnbull)": {"min": 0.10, "max": 0.40, "typical": 0.25, "ref": "Turnbull, J. Appl. Phys. 1950"},
    "Metallic alloys (Kaptay)": {"min": 0.30, "max": 1.20, "typical": 0.65, "ref": "Kaptay, Calphad 2012"},
    "Fe-Cr binary (MD)": {"min": 0.18, "max": 0.32, "typical": 0.25, "ref": "Zhang et al., Comp. Mater. Sci. 2022"},
    "Al-Ag-Cu ternary (exp)": {"min": 0.15, "max": 0.35, "typical": 0.22, "ref": "Jones, Acta Metall. 1970"},
    "CoCrFeNi HEA (estimated)": {"min": 0.40, "max": 0.90, "typical": 0.60, "ref": "Constituent averaging + CALPHAD"},
    "Carbides M7C3 (nucleation)": {"min": 2.50, "max": 4.00, "typical": 3.31, "ref": "Li et al., J. Alloys Compd. 2020"}
}

GAMMA_LIQUID_FCC_DEFAULT = 0.60
SHAPE_FACTOR_DEFAULT = 3.00
DEFAULT_DV = 1e-18
DEFAULT_DV_UM3 = 1.0
DEFAULT_MC_SAMPLES = 200

GRAIN_SHAPE_FACTORS = {
    "Spherical (k=2.0)": 2.0,
    "Tetrakaidecahedron (k=3.0) ⭐": 3.0,
    "Equiaxed polycrystal (k=2.6)": 2.6,
    "Columnar (k=1.8)": 1.8,
    "Equiaxed cubic (k=6.0)": 6.0
}

# ================= CLOUD-OPTIMIZED: LAZY LOADING FUNCTIONS =================

@st.cache_data(ttl=3600, max_entries=5)
def get_available_temperatures(csv_dir):
    csv_path = Path(csv_dir)
    files = sorted(csv_path.glob("Gibbs_*.csv"))
    temperatures = []
    for f in files:
        try:
            T = int(f.stem.replace("Gibbs_", "").replace("K", ""))
            temperatures.append(T)
        except ValueError:
            continue
    return sorted(temperatures)

@st.cache_data(ttl=3600, max_entries=5)
def load_single_temperature_csv(T, csv_dir):
    csv_path = Path(csv_dir) / f"Gibbs_{T}K.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
        return df if len(df) > 0 else None
    except Exception:
        return None

def is_regular_grid(df):
    if df is None or len(df) == 0: return False
    n_co, n_cr, n_fe = df["Co"].nunique(), df["Cr"].nunique(), df["Fe"].nunique()
    return len(df) == n_co * n_cr * n_fe and n_co > 0

@st.cache_resource(ttl=3600, max_entries=3)
def build_interpolator_for_T_phase(T, phase, csv_dir):
    df = load_single_temperature_csv(T, csv_dir)
    if df is None: return None
    
    if is_regular_grid(df):
        co_vals, cr_vals, fe_vals = np.sort(df["Co"].unique()), np.sort(df["Cr"].unique()), np.sort(df["Fe"].unique())
        df_sorted = df.sort_values(["Co", "Cr", "Fe"])
        values = df_sorted[f"G_{phase}"].values.reshape(len(co_vals), len(cr_vals), len(fe_vals))
        interp = RegularGridInterpolator((co_vals, cr_vals, fe_vals), values, bounds_error=False, fill_value=np.nan)
    else:
        points = df[["Co", "Cr", "Fe"]].values
        values = df[f"G_{phase}"].values
        interp = LinearNDInterpolator(points, values, fill_value=np.nan)
    del df
    return interp

def evaluate_point_lazy(x_co, x_cr, x_fe, T, csv_dir):
    interp_liq = build_interpolator_for_T_phase(T, "LIQ", csv_dir)
    interp_fcc = build_interpolator_for_T_phase(T, "FCC", csv_dir)
    if interp_liq is None or interp_fcc is None: return None, None
    
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq, g_fcc = float(interp_liq(point)[0]), float(interp_fcc(point)[0])
    except Exception:
        return None, None
    return (g_liq, g_fcc) if not (np.isnan(g_liq) or np.isnan(g_fcc)) else (None, None)

# Initialize temperature list
try:
    temperatures = get_available_temperatures(CSV_FILES_DIR)
except Exception as e:
    st.error(f"❌ Error scanning CSV directory: {e}")
    temperatures = []

if not temperatures:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("💡 Expected format: `Gibbs_1000K.csv` in `csv_files/` folder")
    st.stop()

# ================= HELPER FUNCTIONS =================
def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
    return (x_co * PURE_VM["Co"] + x_cr * PURE_VM["Cr"] + x_fe * PURE_VM["Fe"] + x_ni * PURE_VM["Ni"])

def normalize_temperature(T): return (T - T_MIN_NORMALIZE) / (T_MAX_NORMALIZE - T_MIN_NORMALIZE)
def get_phase_preference(delta_G): return ("FCC favored", "#1f77b4", "🔵") if delta_G < 0 else ("LIQUID favored", "#ff7f0e", "🟠")
def compute_Sv(d, k): return k / d
def compute_total_area(Sv, V): return Sv * V
def compute_curvature_radius(d, geom=0.25): return d * geom
def compute_capillary_pressure(g, r): return np.inf if r <= 0 else (2.0 * g) / r
def compute_net_pressure(dGv, Pcap): return dGv - Pcap
def compute_differential_force(P, Sv, dV): return P * Sv * dV

def propagate_uncertainty_gamma_k(g_nom, k_nom, d_m, dGv, n=DEFAULT_MC_SAMPLES, g_range=None, k_range=None):
    if g_range is None: g_range = (g_nom*0.8, g_nom*1.2)
    if k_range is None: k_range = (k_nom*0.85, k_nom*1.15)
    
    g_s = np.random.uniform(g_range[0], g_range[1], n)
    k_s = np.random.uniform(k_range[0], k_range[1], n)
    r = compute_curvature_radius(d_m)
    P_cap_s = compute_capillary_pressure(g_s, r)
    P_net_s = compute_net_pressure(dGv, P_cap_s)
    Sv_s = compute_Sv(d_m, k_s)
    dF_s = compute_differential_force(P_net_s, Sv_s, DEFAULT_DV)
    
    return {
        "P_net_mean": np.mean(P_net_s), "P_net_std": np.std(P_net_s),
        "P_net_95ci": norm.interval(0.95, loc=np.mean(P_net_s), scale=np.std(P_net_s)),
        "dF_net_mean": np.mean(dF_s), "dF_net_std": np.std(dF_s),
        "dF_net_95ci": norm.interval(0.95, loc=np.mean(dF_s), scale=np.std(dF_s)),
        "samples": {"gamma": g_s, "P_net": P_net_s, "dF": dF_s}  # kept minimal for plotting
    }

# ================= LATEX THEORY =================
def display_latex_theory():
    st.markdown("## 📚 Thermodynamic Theory Reference")
    with st.expander("📋 View Theory (Rendered Equations)", expanded=True):
        st.markdown(r"""
        | **Concept** | **Mathematical Formulation** |
        |:---|:---|
        | **Gibbs Free Energy** | $G_{\text{phase}}(x_{\text{Co}},x_{\text{Cr}},x_{\text{Fe}},x_{\text{Ni}},T)$ <br> $\sum_i x_i = 1$ |
        | **Driving Force** | $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}} \quad [\text{J/mol}]$ |
        | **Volumetric Pressure** | $\Delta G_v = \Delta G / V_m \quad [\text{Pa}]$ |
        | **Area Density** | $S_v = k/d \quad [\text{m}^2/\text{m}^3]$ |
        | **Capillary Pressure** | $P_{\text{cap}} = 2\gamma / r \quad [\text{Pa}]$ |
        | **Net Pressure** | $P_{\text{net}} = \Delta G_v - P_{\text{cap}}$ |
        | **Differential Force** | $dF_{\text{net}} = P_{\text{net}} \cdot S_v \cdot dV$ |
        """)
        st.markdown("### 📖 References")
        st.markdown("""1. Porter & Easterling, *Phase Transformations* | 2. Kaptay, *Calphad* 2012 | 3. Smith-Guttman, *Trans. AIME* 1953 | 4. Turnbull, *J. Appl. Phys.* 1950 | 5. Christian, *Transformations* | 6. Underwood, *Stereology*""")

display_latex_theory()
st.divider()

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Composition & Temperature")
T = st.sidebar.select_slider("Temperature (K)", options=temperatures, value=1000 if 1000 in temperatures else temperatures[0])

col1, col2, col3 = st.sidebar.columns(3)
x_co = col1.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_cr = col2.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_fe = col3.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid: x_Ni = {x_ni:.4f} (must sum to 1.0)")
    st.stop()
else:
    st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

st.sidebar.subheader("📐 Molar Volume Model")
vm_model = st.sidebar.radio("Model", ["Constant", "Composition‑dependent"], index=1)
V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e") if vm_model == "Constant" else composition_dependent_vm(x_co, x_cr, x_fe, x_ni)

st.sidebar.divider()
st.sidebar.subheader("🔧 Interface / Grain Boundary Parameters")
area_mode = st.sidebar.radio("Area Calculation Mode", ["Direct Input (A)", "Grain Size Derived (Sv x V)"], index=1)

if area_mode == "Direct Input (A)":
    interface_area = st.sidebar.number_input("Interface Area A (m²)", 1e-20, 1e2, 1e-8, 1e-10, format="%.2e")
    grain_size_um, shape_factor, Sv, sample_volume_m3 = None, None, None, None
else:
    grain_size_um = st.sidebar.number_input("Average Grain Size d (μm)", 0.001, 10000.0, 10.0, 0.1, format="%.3f")
    grain_size_m = grain_size_um * 1e-6
    shape_choice = st.sidebar.selectbox("Grain Shape Factor", list(GRAIN_SHAPE_FACTORS.keys()), index=1)
    shape_factor = GRAIN_SHAPE_FACTORS[shape_choice]
    sample_volume_cm3 = st.sidebar.number_input("Sample Volume V (cm³)", 1e-9, 1e6, 1.0, 0.1, format="%.3f")
    sample_volume_m3 = sample_volume_cm3 * 1e-6
    Sv = compute_Sv(grain_size_m, shape_factor)
    interface_area = compute_total_area(Sv, sample_volume_m3)
    st.sidebar.metric("Sv", f"{Sv:.2e} m²/m³")
    st.sidebar.metric("A_total", f"{interface_area:.2e} m²")

st.sidebar.divider()
st.sidebar.subheader("🌊 Capillary Pressure Correction")
use_capillary = st.sidebar.checkbox("Enable Capillary Correction", value=True)

if use_capillary:
    with st.sidebar.expander("📚 Literature Ranges for γ", expanded=False):
        for name, vals in LIT_GAMMA_RANGES.items():
            st.markdown(f"- **{name}**: {vals['min']:.2f}–{vals['max']:.2f} N/m")
    
    gamma_method = st.sidebar.radio("γ Selection", ["Manual input", "Preset from literature"], index=1)
    if gamma_method == "Preset from literature":
        gamma_preset = st.sidebar.selectbox("System/Preset", list(LIT_GAMMA_RANGES.keys()), index=4)
        gamma_info = LIT_GAMMA_RANGES[gamma_preset]
        gamma = st.sidebar.slider(f"γ (N/m) - {gamma_preset}", gamma_info["min"], gamma_info["max"], gamma_info["typical"], 0.01, format="%.2f")
    else:
        gamma = st.sidebar.number_input("γ (N/m)", 0.01, 5.0, GAMMA_LIQUID_FCC_DEFAULT, 0.01, format="%.2f")
    
    dV_um3 = st.sidebar.number_input("dV (μm³)", 0.001, 1000.0, DEFAULT_DV_UM3, 0.1, format="%.3f")
    dV = dV_um3 * 1e-18
    
    enable_uncertainty = st.sidebar.checkbox("Enable Monte Carlo Uncertainty", value=False)
    if enable_uncertainty:
        gamma_uncertainty_pct = st.sidebar.slider("γ uncertainty (%)", 5, 50, 20)
        k_uncertainty_pct = st.sidebar.slider("k uncertainty (%)", 5, 30, 15)
        n_mc_samples = st.sidebar.selectbox("MC samples", [100, 200, 500, 1000], index=1)
else:
    gamma, dV, dV_um3, enable_uncertainty = 0.0, 0.0, 0.0, False

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear All Results & Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        if key.startswith(("res_", "mc_", "plot_")):
            del st.session_state[key]
    st.rerun()

# ================= MAIN COMPUTATION (EXPLICIT BUTTON) =================
st.header(f"📊 Results at T = {T} K")

# Detect input changes to invalidate old results
current_inputs = (x_co, x_cr, x_fe, T, V_m, grain_size_um, shape_factor, gamma, dV_um3, use_capillary, area_mode)
inputs_changed = st.session_state.get("last_main_inputs") != current_inputs

if inputs_changed and "res_main" in st.session_state:
    for key in list(st.session_state.keys()):
        if key.startswith(("res_", "mc_")):
            del st.session_state[key]

col_btn, col_status = st.columns([3, 1])
run_main = col_btn.button("🚀 Compute Gibbs & Interface Forces", type="primary", width="stretch")

if run_main or inputs_changed:
    with st.spinner("⏳ Computing Gibbs energies & interface forces..."):
        g_liq, g_fcc = evaluate_point_lazy(x_co, x_cr, x_fe, T, CSV_FILES_DIR)
        delta_G = g_fcc - g_liq
        phase_pref, phase_color, phase_emoji = get_phase_preference(delta_G)
        delta_G_v = delta_G / V_m
        delta_G_v_MPa = delta_G_v / 1e6
        
        curvature_r = compute_curvature_radius(grain_size_m) if grain_size_m else None
        P_capillary = compute_capillary_pressure(gamma, curvature_r) if curvature_r else 0
        P_capillary_MPa = P_capillary / 1e6
        P_net = compute_net_pressure(delta_G_v, P_capillary) if use_capillary and curvature_r else delta_G_v
        P_net_MPa = P_net / 1e6
        dF_net = compute_differential_force(P_net, Sv, dV) if (use_capillary and Sv) else None
        net_force = P_net * interface_area if use_capillary else delta_G_v * interface_area
        
        st.session_state["res_main"] = {
            "g_liq": g_liq, "g_fcc": g_fcc, "delta_G": delta_G, "delta_G_v_MPa": delta_G_v_MPa,
            "P_capillary_MPa": P_capillary_MPa, "P_net_MPa": P_net_MPa, "dF_net": dF_net,
            "net_force": net_force, "phase_pref": phase_pref, "phase_color": phase_color,
            "phase_emoji": phase_emoji, "Sv": Sv, "interface_area": interface_area,
            "curvature_r": curvature_r, "V_m": V_m, "grain_size_um": grain_size_um,
            "shape_factor": shape_factor, "delta_G_v": delta_G_v, "P_net": P_net
        }
        st.session_state["last_main_inputs"] = current_inputs
        st.success("✅ Calculation complete!")

# ================= DISPLAY RESULTS =================
if "res_main" in st.session_state:
    res = st.session_state["res_main"]
    g_liq, g_fcc = res["g_liq"], res["g_fcc"]
    delta_G = res["delta_G"]
    
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Composition outside convex hull of training data")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("G_LIQUID", f"{g_liq:,.1f} J/mol")
        col_b.metric("G_FCC", f"{g_fcc:,.1f} J/mol")
        col_c.metric("ΔG = G_FCC − G_LIQ", f"{delta_G:,.1f} J/mol", delta=res["phase_pref"], delta_color="normal" if delta_G < 0 else "inverse")
        st.success(f"🏆 Most stable phase: **{res['phase_pref'].split()[0]}** {res['phase_emoji']}")
        
        st.divider()
        st.subheader("⚙️ Interface Driving Force (Mechanical)")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("ΔGᵥ", f"{res['delta_G_v_MPa']:.3f} MPa")
        if use_capillary:
            col_p2.metric("P_capillary", f"{res['P_capillary_MPa']:.3f} MPa", delta="resists growth", delta_color="inverse")
            col_p3.metric("P_net", f"{res['P_net_MPa']:.3f} MPa")
            col_p4.metric("Motion", "→ FCC grows" if res["P_net"] > 0 else "→ LIQUID grows")
        else:
            col_p2.metric("SI units", f"{res['delta_G_v']:.2e} N/m²")
            col_p3.metric("Motion", "→ FCC grows" if delta_G < 0 else "→ LIQUID grows")
            
        st.markdown(f"""
        | Parameter | Value |
        |:---|:---|
        | Grain size $d$ | {res['grain_size_um']:.2f} μm |
        | Shape factor $k$ | {res['shape_factor']:.0f} |
        | $S_v$ | {res['Sv']:.2e} m²/m³ |
        | $A_{total}$ | {res['interface_area']:.2e} m² |
        """)
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Net Force $F_{total}$", f"{res['net_force']:.3e} N")
        if use_capillary and res["dF_net"] is not None:
            col_f2.metric("dF_net (μm³)", f"{res['dF_net']:.3e} N")

    # ================= UNCERTAINTY SECTION =================
    if enable_uncertainty and "res_main" in st.session_state and res["g_liq"] is not None:
        st.divider()
        st.subheader("📊 Monte Carlo Uncertainty Propagation")
        run_mc = st.button("📊 Run Monte Carlo Uncertainty", type="secondary", width="stretch")
        
        mc_inputs = (res["delta_G_v"], gamma, shape_factor, res["grain_size_um"], n_mc_samples, gamma_uncertainty_pct, k_uncertainty_pct)
        if run_mc or st.session_state.get("last_mc_inputs") != mc_inputs:
            with st.spinner("⏳ Running Monte Carlo simulation..."):
                g_range = (gamma*(1-gamma_uncertainty_pct/100), gamma*(1+gamma_uncertainty_pct/100))
                k_range = (shape_factor*(1-k_uncertainty_pct/100), shape_factor*(1+k_uncertainty_pct/100))
                mc_results = propagate_uncertainty_gamma_k(gamma, shape_factor, res["grain_size_um"]*1e-6, res["delta_G_v"], n_mc_samples, g_range, k_range)
                st.session_state["mc_results"] = mc_results
                st.session_state["last_mc_inputs"] = mc_inputs
                st.success("✅ Monte Carlo complete!")
        
        if "mc_results" in st.session_state:
            mc = st.session_state["mc_results"]
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            col_mc1.metric("P_net (95% CI)", f"{mc['P_net_mean']/1e6:.3f} MPa", delta=f"±{mc['P_net_std']/1e6:.3f} MPa")
            col_mc2.metric("dF_net (95% CI)", f"{mc['dF_net_mean']:.2e} N", delta=f"±{mc['dF_net_std']:.2e} N")
            col_mc3.metric("Relative Uncertainty", f"P: {mc['P_net_std']/abs(mc['P_net_mean'])*100:.1f}%, F: {mc['dF_net_std']/abs(mc['dF_net_mean'])*100:.1f}%")
            
            # Plot distributions on demand to save memory
            if st.checkbox("📈 Show Monte Carlo Distributions", key="show_mc_dist"):
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    fig_p = go.Figure(go.Histogram(x=mc["samples"]["P_net"]/1e6, nbinsx=30, marker_color=res["phase_color"], opacity=0.7))
                    fig_p.update_layout(title="P_net Distribution", xaxis_title="P_net (MPa)", height=300)
                    st.plotly_chart(fig_p, width="stretch")
                with col_plot2:
                    fig_f = go.Figure(go.Histogram(x=mc["samples"]["dF"], nbinsx=30, marker_color="#9467bd", opacity=0.7))
                    fig_f.update_layout(title="dF_net Distribution", xaxis_title="dF_net (N)", height=300)
                    st.plotly_chart(fig_f, width="stretch")
    else:
        st.caption("💡 Enable Monte Carlo in sidebar to run uncertainty analysis")

# ================= VISUALIZATION TOOLS (EXPLICIT BUTTONS PER TAB) =================
st.divider()
st.header("🗺️ Exploration Tools")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 G vs Comp", "🌡️ Phase Map vs T", "📊 ΔGᵥ vs Comp", "📋 Raw Data",
    "🌞 Sunburst", "🕸️ Radar", "📐 Grain Scaling", "📊 Sensitivity"
])

# Helper to clear tab cache if inputs change
def check_tab_input(key, val):
    if st.session_state.get(f"last_{key}") != val:
        st.session_state.pop(f"plot_{key}", None)
        st.session_state[f"last_{key}"] = val

with tab1:
    st.markdown("### Gibbs Energy vs Composition")
    scan_var1 = st.radio("Vary", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="tab1_var")
    fixed_val1 = st.slider("Fixed", 0.0, 0.4, 0.2, 0.01, key="tab1_fixed")
    if st.button("📈 Generate G Scan", key="btn_tab1"):
        check_tab_input("tab1", (T, scan_var1, fixed_val1))
        with st.spinner("⏳ Scanning compositions..."):
            max_v = 1.0 - 2*fixed_val1 - 0.01
            if max_v < 0.01: st.error("❌ Fixed values too large")
            else:
                xs = np.linspace(0.01, max_v, 100)
                g_liq_s, g_fcc_s, valid = [], [], []
                for xv in xs:
                    args = (xv, fixed_val1, fixed_val1) if scan_var1=="x_Co" else ((fixed_val1, xv, fixed_val1) if scan_var1=="x_Cr" else (fixed_val1, fixed_val1, xv))
                    gl, gf = evaluate_point_lazy(*args, T, CSV_FILES_DIR)
                    if gl is not None and gf is not None: g_liq_s.append(gl); g_fcc_s.append(gf); valid.append(xv)
                    else: g_liq_s.append(np.nan); g_fcc_s.append(np.nan)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid, y=g_liq_s, name="G_LIQ", line=dict(color="#ff7f0e", width=2)))
                fig.add_trace(go.Scatter(x=valid, y=g_fcc_s, name="G_FCC", line=dict(color="#1f77b4", width=2)))
                fig.update_layout(title=f"G vs {scan_var1} at {T}K", xaxis_title=scan_var1, yaxis_title="G (J/mol)", height=400, hovermode="x unified")
                st.session_state["plot_tab1"] = fig
    if "plot_tab1" in st.session_state: st.plotly_chart(st.session_state["plot_tab1"], width="stretch")

with tab2:
    st.markdown("### Phase Stability vs Temperature")
    if st.button("🌡️ Generate T Scan", key="btn_tab2"):
        check_tab_input("tab2", (x_co, x_cr, x_fe, x_ni))
        with st.spinner("⏳ Scanning temperatures..."):
            dG_list, dGv_list, valid_T = [], [], []
            for Tv in temperatures:
                gl, gf = evaluate_point_lazy(x_co, x_cr, x_fe, Tv, CSV_FILES_DIR)
                if gl is not None and gf is not None:
                    dG = gf - gl
                    dG_list.append(dG)
                    vm = composition_dependent_vm(x_co, x_cr, x_fe, x_ni) if vm_model=="Composition‑dependent" else V_m
                    dGv_list.append(dG/vm/1e6)
                    valid_T.append(Tv)
            
            if not valid_T: st.warning("⚠️ No valid data points")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid_T, y=dG_list, name="ΔG (J/mol)", yaxis="y1", line=dict(color="#2ca02c")))
                fig.add_trace(go.Scatter(x=valid_T, y=dGv_list, name="ΔGᵥ (MPa)", yaxis="y2", line=dict(color="#d62728", dash="dot")))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Driving Force vs T", xaxis_title="T (K)", yaxis=dict(title="ΔG", color="#2ca02c"), yaxis2=dict(title="ΔGᵥ", color="#d62728", overlaying="y", side="right"), height=450)
                st.session_state["plot_tab2"] = fig
    if "plot_tab2" in st.session_state: st.plotly_chart(st.session_state["plot_tab2"], width="stretch")

with tab3:
    st.markdown("### ΔGᵥ vs Composition")
    scan_var3 = st.radio("Scan", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="tab3_var")
    fixed_val3 = st.slider("Fixed", 0.0, 0.4, 0.2, 0.01, key="tab3_fixed")
    if st.button("📊 Generate ΔGᵥ Scan", key="btn_tab3"):
        check_tab_input("tab3", (T, scan_var3, fixed_val3))
        with st.spinner("⏳ Computing ΔGᵥ scan..."):
            max_v = 1.0 - 2*fixed_val3 - 0.01
            if max_v < 0.01: st.error("❌ Fixed values too large")
            else:
                xs = np.linspace(0.01, max_v, 100)
                dGv_vals, valid = [], []
                for xv in xs:
                    if scan_var3=="x_Co": xc,xr,xf = xv,fixed_val3,fixed_val3
                    elif scan_var3=="x_Cr": xc,xr,xf = fixed_val3,xv,fixed_val3
                    else: xc,xr,xf = fixed_val3,fixed_val3,xv
                    xn = 1.0-(xc+xr+xf)
                    if 0<=xn<=1:
                        gl, gf = evaluate_point_lazy(xc, xr, xf, T, CSV_FILES_DIR)
                        if gl and gf:
                            vm = composition_dependent_vm(xc,xr,xf,xn) if vm_model=="Composition‑dependent" else V_m
                            dGv_vals.append((gf-gl)/vm/1e6)
                            valid.append(xv)
                fig = go.Figure(go.Scatter(x=valid, y=dGv_vals, fill="tozeroy", line=dict(color="#9467bd")))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title=f"ΔGᵥ vs {scan_var3} at {T}K", height=400)
                st.session_state["plot_tab3"] = fig
    if "plot_tab3" in st.session_state: st.plotly_chart(st.session_state["plot_tab3"], width="stretch")

with tab4:
    st.markdown("### 📋 Raw Data")
    if st.button("📥 Load Raw Data Table", key="btn_tab4"):
        check_tab_input("tab4", T)
        with st.spinner("⏳ Loading data..."):
            df = load_single_temperature_csv(T, CSV_FILES_DIR)
            st.session_state["plot_tab4"] = df
    if "plot_tab4" in st.session_state:
        st.dataframe(st.session_state["plot_tab4"].style.format({"Co":"{:.3f}","Cr":"{:.3f}","Fe":"{:.3f}","Ni":"{:.3f}","G_LIQ":"{:.1f}","G_FCC":"{:.1f}"}), height=500, width="stretch")
        st.download_button("📥 Download CSV", st.session_state["plot_tab4"].to_csv(index=False), f"data_T{T}K.csv")

with tab5:
    st.markdown("### 🌞 Sunburst Hierarchy")
    if st.button("🌞 Generate Sunburst", key="btn_tab5") and "res_main" in st.session_state:
        check_tab_input("tab5", T)
        res = st.session_state["res_main"]
        ids, parents, labels, values, colors = ["root"], [""], ["CoCrFeNi System"], [1], ["lightgray"]
        ids.extend([f"T_{T}", f"T_{T}_Co", f"T_{T}_Cr", f"T_{T}_Fe", f"T_{T}_Ni"])
        parents.extend(["root", "root", f"T_{T}", f"T_{T}", f"T_{T}"])
        labels.extend([f"T={T} K", "Co", "Cr", "Fe", "Ni"])
        values.extend([1, x_co, x_cr, x_fe, x_ni])
        colors.extend(["lightblue", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        force_val = abs(res["net_force"])
        ids.extend([f"f_{i}" for i in range(4)])
        parents.extend([p for p in [f"T_{T}_Co", f"T_{T}_Cr", f"T_{T}_Fe", f"T_{T}_Ni"]])
        labels.extend([f"Force: {force_val:.2e} N"]*4)
        values.extend([force_val]*4)
        colors.extend(["gold"]*4)
        
        fig = go.Figure(go.Sunburst(ids=ids, parents=parents, labels=labels, values=values, marker=dict(colors=colors), branchvalues="total", maxdepth=3))
        fig.update_layout(title="Hierarchy: T → Composition → Force", margin=dict(t=30,b=0,l=0,r=0), height=600)
        st.session_state["plot_tab5"] = fig
    if "plot_tab5" in st.session_state: st.plotly_chart(st.session_state["plot_tab5"], width="stretch")

with tab6:
    st.markdown("### 🕸️ Radar Chart")
    if st.button("🕸️ Generate Radar", key="btn_tab6") and "res_main" in st.session_state:
        check_tab_input("tab6", T)
        res = st.session_state["res_main"]
        cats = ["x_Co", "x_Cr", "x_Fe", "x_Ni", "T_norm", "|ΔGᵥ|"]
        vals = [x_co, x_cr, x_fe, x_ni, normalize_temperature(T), min(1.0, abs(res["delta_G_v_MPa"])/100)]
        fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself", name="State", line=dict(color=res["phase_color"], width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Thermodynamic State Radar", height=500)
        st.session_state["plot_tab6"] = fig
    if "plot_tab6" in st.session_state: st.plotly_chart(st.session_state["plot_tab6"], width="stretch")

with tab7:
    st.markdown("### 📐 Grain Size Scaling")
    if st.button("📐 Generate Scaling Plots", key="btn_tab7") and "res_main" in st.session_state:
        check_tab_input("tab7", gamma)
        res = st.session_state["res_main"]
        gs_um = np.linspace(0.5, 50, 100)
        gs_m = gs_um * 1e-6
        P_caps, P_nets = [], []
        for g in gs_m:
            r = compute_curvature_radius(g)
            Pc = compute_capillary_pressure(gamma, r)
            P_caps.append(Pc/1e6)
            P_nets.append(compute_net_pressure(res["delta_G_v"], Pc)/1e6)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=gs_um, y=P_caps, name="P_cap", line=dict(color="#d62728", dash="dash")))
        fig1.add_trace(go.Scatter(x=gs_um, y=P_nets, name="P_net", line=dict(color="#2ca02c")))
        fig1.add_hline(y=res["delta_G_v_MPa"], line_dash="dot", line_color="blue")
        fig1.update_layout(title="Pressure vs Grain Size", height=400)
        
        fig2 = go.Figure()
        dF_vals = [compute_differential_force(compute_net_pressure(res["delta_G_v"], compute_capillary_pressure(gamma, compute_curvature_radius(g))), compute_Sv(g, res["shape_factor"]), dV) for g in gs_m]
        fig2.add_trace(go.Scatter(x=gs_um, y=dF_vals, fill="tozeroy", line=dict(color="#9467bd")))
        fig2.update_layout(title="dF_net vs Grain Size", height=400)
        st.session_state["plot_tab7"] = (fig1, fig2)
    if "plot_tab7" in st.session_state:
        f1, f2 = st.session_state["plot_tab7"]
        st.plotly_chart(f1, width="stretch")
        st.plotly_chart(f2, width="stretch")

with tab8:
    st.markdown("### 📊 Sensitivity Analysis")
    if st.button("📊 Generate Sensitivity Plots", key="btn_tab8") and "res_main" in st.session_state:
        check_tab_input("tab8", (gamma, shape_factor))
        res = st.session_state["res_main"]
        gs = res["grain_size_um"]*1e-6 if res["grain_size_um"] else 10e-6
        r = compute_curvature_radius(gs)
        
        g_sweep = np.linspace(0.3, 1.2, 50)
        p_net_g = [(res["delta_G_v"] - compute_capillary_pressure(g, r))/1e6 for g in g_sweep]
        fig1 = go.Figure(go.Scatter(x=g_sweep, y=p_net_g, fill="tozeroy", line=dict(color="#d62728")))
        fig1.add_vline(x=gamma, line_dash="dash", line_color="black")
        fig1.update_layout(title="P_net vs γ", height=350)
        
        k_sweep = np.linspace(2.0, 4.0, 50)
        dF_k = [compute_differential_force(res["P_net"], compute_Sv(gs, k), dV) for k in k_sweep]
        fig2 = go.Figure(go.Scatter(x=k_sweep, y=dF_k, fill="tozeroy", line=dict(color="#9467bd")))
        fig2.add_vline(x=shape_factor, line_dash="dash", line_color="black")
        fig2.update_layout(title="dF_net vs k", height=350)
        st.session_state["plot_tab8"] = (fig1, fig2)
    if "plot_tab8" in st.session_state:
        f1, f2 = st.session_state["plot_tab8"]
        st.plotly_chart(f1, width="stretch")
        st.plotly_chart(f2, width="stretch")

# ================= FOOTER =================
st.divider()
st.caption("""
**Cloud Optimization**: Explicit buttons + lazy loading + cached interpolators = ✅ Streamlit Cloud ready (1GB RAM limit)  
**Units**: Energy [J/mol], Pressure [Pa], Force [N], Length [m]  
**References**: Porter & Easterling | Kaptay 2012 | Smith-Guttman 1953 | Turnbull 1950 | Christian 2002 | Underwood 1970
""")
col_f1, col_f2 = st.columns(2)
with col_f1: st.caption("🔄 Auto-refreshes on explicit button click")
with col_f2: st.caption(f"📍 Working dir: `{SCRIPT_DIR}`")
