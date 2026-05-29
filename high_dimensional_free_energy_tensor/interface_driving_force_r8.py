"""
CoCrFeNi Gibbs Free Energy Explorer
OPTIMIZED FOR STREAMLIT CLOUD (1GB RAM limit)
FIXED: Thermodynamic Sign Convention & Motion Direction
PLUS: Comprehensive Theory on dF, Kinetics, and Phase Transformation
WITH: Explicit Run Buttons, Lazy loading, cached interpolators
PLUS: Grain Size Derived Interfacial Area Density (Sv) & Net Force
PLUS: Capillary Pressure Correction & Differential Force Model
PLUS: Literature-Based Parameter Ranges & Uncertainty Quantification
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
ΔG (J/mol) → $P_{\text{chem}} = (G_{\text{LIQ}} - G_{\text{FCC}})/V_m$ (Pa) → Interface driving pressure  
**Physics Corrected:** Positive $P_{\text{net}}$ drives LIQUID → FCC solidification.  
$F_{\text{local}} = P_{\text{net}} \cdot dA_{\text{ref}}$ (Local force on $1 \, \mu m^2$ facet).  
**Advanced:** Capillary correction → $P_{\text{net}}$ → Differential force $dF_{\text{net}}$ on μm³ element  
**Enhanced:** Literature-based parameter ranges, uncertainty, & transformation kinetics
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
DEFAULT_DV = 1e-18  # m³ = 1 μm³
DEFAULT_DV_UM3 = 1.0
DEFAULT_MC_SAMPLES = 200

# Reference Area for Force Calculation (Physical Correction)
dA_REF = 1e-12  # 1 square micron (1 μm²)

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
    if not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
        return df if len(df) > 0 else None
    except Exception: return None

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
    except Exception: return None, None
    return (g_liq, g_fcc) if not (np.isnan(g_liq) or np.isnan(g_fcc)) else (None, None)

try:
    temperatures = get_available_temperatures(CSV_FILES_DIR)
except Exception as e:
    st.error(f"❌ Error scanning CSV directory: {e}")
    temperatures = []

if not temperatures:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
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
def compute_differential_force(P, Sv, dV): return P * Sv * dV

def propagate_uncertainty_gamma_k(g_nom, k_nom, d_m, P_chem, n=DEFAULT_MC_SAMPLES, g_range=None, k_range=None):
    if g_range is None: g_range = (g_nom*0.8, g_nom*1.2)
    if k_range is None: k_range = (k_nom*0.85, k_nom*1.15)
    g_s = np.random.uniform(g_range[0], g_range[1], n)
    k_s = np.random.uniform(k_range[0], k_range[1], n)
    r = compute_curvature_radius(d_m)
    P_cap_s = compute_capillary_pressure(g_s, r)
    P_net_s = P_chem - P_cap_s
    Sv_s = compute_Sv(d_m, k_s)
    dF_s = compute_differential_force(P_net_s, Sv_s, DEFAULT_DV)
    return {
        "P_net_mean": np.mean(P_net_s), "P_net_std": np.std(P_net_s),
        "P_net_95ci": norm.interval(0.95, loc=np.mean(P_net_s), scale=np.std(P_net_s)),
        "dF_net_mean": np.mean(dF_s), "dF_net_std": np.std(dF_s),
        "dF_net_95ci": norm.interval(0.95, loc=np.mean(dF_s), scale=np.std(dF_s)),
        "samples": {"gamma": g_s.tolist(), "P_net": P_net_s.tolist(), "dF": dF_s.tolist()}
    }

# ================= LATEX THEORY & KINETICS DESCRIPTION =================
def display_latex_theory():
    st.markdown("## 📚 Thermodynamic Theory & Transformation Kinetics")
    with st.expander("📖 Read Full Theory: Sign, Magnitude & Kinetics of dF", expanded=True):
        st.markdown(r"""
        ### 🔬 Physical Meaning of Differential Force ($dF$) and Net Pressure ($P_{\text{net}}$)
        In solid-state phase transformations, the **driving force** is an interface-localized mechanical pressure.
        The chemical driving pressure for LIQUID → FCC transformation is defined as:
        $$P_{\text{chem}} = \frac{G_{\text{LIQ}} - G_{\text{FCC}}}{V_m} = -\frac{\Delta G}{V_m}$$
        Capillary pressure ($P_{\text{cap}} = 2\gamma/r$) always resists curvature-driven growth. The net pressure is:
        $$P_{\text{net}} = P_{\text{chem}} - P_{\text{cap}}$$
        The differential force on a reference area $dA = 1\,\mu m^2$ is:
        $$F_{\text{local}} = P_{\text{net}} \cdot dA$$

        ### ⚡ Sign Convention & Phase Motion
        | Sign of $P_{\text{net}}$ | Physical Interpretation | Phase Motion |
        |:---|:---|:---|
        | **$P_{\text{net}} > 0$** | Chemical driving force exceeds capillary resistance. | **LIQUID → FCC** (Solidification) |
        | **$P_{\text{net}} < 0$** | Capillary resistance or thermodynamic instability dominates. | **FCC → LIQUID** (Remelting/Dissolution) |
        | **$P_{\text{net}} \approx 0$** | Local equilibrium at the interface. | Interface stationary |

        ### 📉 Magnitude & Relationship to Transformation Kinetics
        The magnitude $|P_{\text{net}}|$ dictates the **interface velocity** $v$ via the linear kinetic law:
        $$v = M \cdot |P_{\text{net}}|$$
        where $M$ is the **interface mobility** [m⁴/(J·s)].
        
        **In Co-Cr-Fe-Ni High-Entropy Alloys (HEAs):**
        1. **High $|P_{\text{net}}|$ (> 50 MPa):** Massive/rapid solidification. Interface moves fast, often partitionless. Common in laser/additive manufacturing.
        2. **Moderate $|P_{\text{net}}|$ (5–50 MPa):** Diffusive growth. Sluggish HEA diffusion limits velocity despite high thermodynamic drive.
        3. **Low $|P_{\text{net}}|$ (< 5 MPa):** Nucleation-controlled or capillary-blocked. Fine nuclei ($r < r^*$) cannot grow until undercooling increases $P_{\text{chem}}$ sufficiently.
        """)
        st.caption("📚 References: Porter & Easterling (2009), Christian (2002), Miracle & Senkov (2017) HEA kinetics review, Kaptay (2012) capillary models.")

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
    interface_area = compute_total_area(Sv, sample_volume_m3) if Sv else 0
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

# ================= MAIN COMPUTATION (EXPLICIT BUTTON & FIXED PHYSICS) =================
st.header(f"📊 Results at T = {T} K")

current_inputs = (x_co, x_cr, x_fe, T, V_m, grain_size_um, shape_factor, gamma, dV_um3, use_capillary, area_mode)
inputs_changed = st.session_state.get("last_main_inputs") != current_inputs

if inputs_changed and "res_main" in st.session_state:
    for key in list(st.session_state.keys()):
        if key.startswith(("res_", "mc_")):
            del st.session_state[key]

col_btn, col_status = st.columns([3, 1])
run_main = col_btn.button("🚀 Compute Gibbs & Interface Forces", type="primary", use_container_width=True)

if run_main or inputs_changed:
    with st.spinner("⏳ Computing Gibbs energies & interface forces..."):
        g_liq, g_fcc = evaluate_point_lazy(x_co, x_cr, x_fe, T, CSV_FILES_DIR)
        delta_G = (g_fcc - g_liq) if (g_liq is not None and g_fcc is not None) else 0.0
        phase_pref, phase_color, phase_emoji = get_phase_preference(delta_G)
        
        # 🔧 FIXED PHYSICS: Chemical driving pressure for LIQ -> FCC
        P_chem = -delta_G / V_m if V_m != 0 else 0.0
        
        curvature_r = compute_curvature_radius(grain_size_m) if (grain_size_m and grain_size_m > 0) else None
        P_cap = compute_capillary_pressure(gamma, curvature_r) if (use_capillary and curvature_r) else 0.0
        
        # Net pressure driving the transformation
        P_net = P_chem - P_cap
        
        # Local force on 1 μm²
        F_local = P_net * dA_REF
        F_local_nN = F_local * 1e9
        
        # Differential force on volume element dV
        dF_net = compute_differential_force(P_net, Sv, dV) if (Sv is not None) else None
        
        # 🔧 Safe State Storage (Guaranteed Keys)
        st.session_state["res_main"] = {
            "g_liq": g_liq, "g_fcc": g_fcc, "delta_G": delta_G, 
            "P_chem": P_chem, "P_cap": P_cap, "P_net": P_net,
            "dF_net": dF_net, "F_local": F_local, "F_local_nN": F_local_nN, 
            "phase_pref": phase_pref, "phase_color": phase_color, "phase_emoji": phase_emoji, 
            "Sv": Sv, "interface_area": interface_area, "curvature_r": curvature_r, 
            "V_m": V_m, "grain_size_um": grain_size_um, "shape_factor": shape_factor
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
        col_p1.metric("Chemical Drive $P_{\text{chem}}$", f"{res['P_chem']/1e6:.3f} MPa", help="Positive when FCC is thermodynamically favored")
        col_p2.metric("Capillary Resistance", f"{res['P_cap']/1e6:.3f} MPa", delta="resists growth", delta_color="inverse")
        col_p3.metric("Net Pressure $P_{\text{net}}$", f"{res['P_net']/1e6:.3f} MPa")
        
        # 🔧 FIXED: Correct motion direction based on P_net sign
        motion_dir = "→ LIQUID → FCC (Growth)" if res["P_net"] > 0 else "→ FCC → LIQUID (Remelting/Blocked)"
        col_p4.metric("Interface Motion", motion_dir)
            
        st.markdown(f"""
        | Parameter | Value |
        |:---|:---|
        | Grain size $d$ | {res.get('grain_size_um', 'N/A')} μm |
        | Shape factor $k$ | {res.get('shape_factor', 'N/A')} |
        | $S_v$ | {res.get('Sv', 0):.2e} m²/m³ |
        | Reference Area $dA$ | $1.0 \times 10^{-12}$ m² (1 μm²) |
        """)
        col_f1, col_f2 = st.columns(2)
        f_loc = res.get("F_local", 0.0)
        col_f1.metric("Local Force ($F_{\text{local}}$)", f"{f_loc:.2e} N", help="Force on 1 μm² interface facet")
        if res.get("dF_net") is not None:
            col_f2.metric("Differential Force ($dF_{\text{net}}$)", f"{res['dF_net']:.2e} N", help="Force on 1 μm³ volume element")

# ================= VISUALIZATION TOOLS =================
st.divider()
st.header("🗺️ Exploration Tools")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_theory = st.tabs([
    "📈 G vs Comp", "🌡️ Phase Map vs T", "📊 ΔGᵥ vs Comp", "📋 Raw Data",
    "🌞 Sunburst", "🕸️ Radar", "⚡ Driving Force", "📊 Sensitivity", "📖 Kinetics"
])

with tab_theory:
    st.markdown("### 📖 Detailed Transformation Kinetics & Force Significance")
    st.markdown(r"""
    This section details how the sign and magnitude of the calculated differential force ($dF$) and net pressure ($P_{\text{net}}$) govern the **kinetics of phase transformation** between the FCC solid solution and LIQUID phases in equiatomic and off-stoichiometric CoCrFeNi alloys.
    """)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 1. Positive $P_{\text{net}}$ ($F_{\text{local}} > 0$): FCC Growth")
        st.markdown(r"""
        - **Thermodynamic Condition:** $G_{\text{FCC}} < G_{\text{LIQ}} \Rightarrow P_{\text{chem}} > 0$.
        - **Physical Meaning:** The system reduces its free energy by converting liquid to solid.
        - **Kinetics:** $v = M \cdot P_{\text{net}}$. In HEAs, $M$ is low due to sluggish diffusion. High $P_{\text{net}}$ (deep undercooling) is required to achieve measurable growth velocities.
        - **Morphology:** High $P_{\text{net}}$ favors dendritic or cellular growth. Low $P_{\text{net}}$ near equilibrium yields faceted or planar interfaces.
        """)
    with col_t2:
        st.markdown("#### 2. Negative $P_{\text{net}}$ ($F_{\text{local}} < 0$): Liquid Growth / Remelting")
        st.markdown(r"""
        - **Thermodynamic Condition:** $G_{\text{FCC}} > G_{\text{LIQ}}$ or $P_{\text{cap}} > P_{\text{chem}}$.
        - **Physical Meaning:** Solid is unstable; boundaries retreat into the liquid phase.
        - **Kinetics:** Dissolution rate is controlled by solute diffusion away from the interface. In CoCrFeNi, Cr and Co often segregate, creating diffusion barriers that slow remelting even with high negative $P_{\text{net}}$.
        - **Grain Refinement:** Local remelting of unstable nuclei reduces $S_v$, shifting the system towards larger, more stable grains.
        """)
    st.divider()
    st.markdown("#### 3. Magnitude & Critical Driving Force")
    st.markdown(r"""
    | $|P_{\text{net}}|$ Range | Transformation Regime | HEA Context |
    |:---|:---|:---|
    | $< 10$ MPa | **Equilibrium / Nucleation** | Thermal fluctuations dominate. Nucleation rate $I \propto \exp(-1/\Delta G^2)$. Extremely slow. |
    | $10 - 100$ MPa | **Diffusive Growth** | Standard solidification. Interface controlled by atomic attachment. Sluggish HEA diffusion limits velocity to $10^{-5} - 10^{-2}$ m/s. |
    | $> 100$ MPa | **Massive / Athermal** | Interface moves faster than diffusion. "Partitionless" solidification. Common in additive manufacturing (laser cooling rates $>10^6$ K/s). |
    
    **Capillary Suppression:** For small grains ($d < 1 \mu m$), $P_{\text{cap}} = 2\gamma/r$ can exceed $200$ MPa. If $P_{\text{net}} \le 0$, grains cannot grow despite $\Delta G < 0$. This explains why HEAs often exhibit a **grain size threshold** during rapid solidification.
    """)

with tab1:
    st.markdown("### Gibbs Energy vs Composition")
    scan_var1 = st.radio("Vary", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="tab1_var")
    fixed_val1 = st.slider("Fixed", 0.0, 0.4, 0.2, 0.01, key="tab1_fixed")
    if st.button("📈 Generate G Scan", key="btn_tab1"):
        with st.spinner("⏳ Scanning..."):
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
                fig.update_layout(title=f"G vs {scan_var1} at {T}K", height=400, hovermode="x unified")
                st.session_state["plot_tab1"] = fig
    if "plot_tab1" in st.session_state: st.plotly_chart(st.session_state["plot_tab1"], use_container_width=True)

with tab2:
    st.markdown("### Phase Stability vs Temperature")
    if st.button("🌡️ Generate T Scan", key="btn_tab2"):
        with st.spinner("⏳ Scanning temperatures..."):
            dG_list, P_chem_list, valid_T = [], [], []
            for Tv in temperatures:
                gl, gf = evaluate_point_lazy(x_co, x_cr, x_fe, Tv, CSV_FILES_DIR)
                if gl is not None and gf is not None:
                    dG = gf - gl; dG_list.append(dG)
                    vm = composition_dependent_vm(x_co, x_cr, x_fe, x_ni) if vm_model=="Composition‑dependent" else V_m
                    P_chem_list.append(-dG/vm/1e6); valid_T.append(Tv)
            if valid_T:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid_T, y=dG_list, name="ΔG (J/mol)", yaxis="y1", line=dict(color="#2ca02c")))
                fig.add_trace(go.Scatter(x=valid_T, y=P_chem_list, name="$P_{chem}$ (MPa)", yaxis="y2", line=dict(color="#d62728", dash="dot")))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Driving Force vs T", height=450)
                st.session_state["plot_tab2"] = fig
            else: st.warning("⚠️ No valid data points")
    if "plot_tab2" in st.session_state: st.plotly_chart(st.session_state["plot_tab2"], use_container_width=True)

with tab3:
    st.markdown("### $P_{chem}$ vs Composition")
    scan_var3 = st.radio("Scan", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="tab3_var")
    fixed_val3 = st.slider("Fixed", 0.0, 0.4, 0.2, 0.01, key="tab3_fixed")
    if st.button("📊 Generate $P_{chem}$ Scan", key="btn_tab3"):
        with st.spinner("⏳ Computing..."):
            max_v = 1.0 - 2*fixed_val3 - 0.01
            if max_v < 0.01: st.error("❌ Fixed values too large")
            else:
                xs = np.linspace(0.01, max_v, 100)
                P_chem_vals, valid = [], []
                for xv in xs:
                    if scan_var3=="x_Co": xc,xr,xf = xv,fixed_val3,fixed_val3
                    elif scan_var3=="x_Cr": xc,xr,xf = fixed_val3,xv,fixed_val3
                    else: xc,xr,xf = fixed_val3,fixed_val3,xv
                    xn = 1.0-(xc+xr+xf)
                    if 0<=xn<=1:
                        gl, gf = evaluate_point_lazy(xc, xr, xf, T, CSV_FILES_DIR)
                        if gl and gf:
                            vm = composition_dependent_vm(xc,xr,xf,xn) if vm_model=="Composition‑dependent" else V_m
                            P_chem_vals.append(-(gf-gl)/vm/1e6); valid.append(xv)
                fig = go.Figure(go.Scatter(x=valid, y=P_chem_vals, fill="tozeroy", line=dict(color="#9467bd")))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title=f"$P_{chem}$ vs {scan_var3} at {T}K", height=400)
                st.session_state["plot_tab3"] = fig
    if "plot_tab3" in st.session_state: st.plotly_chart(st.session_state["plot_tab3"], use_container_width=True)

with tab4:
    st.markdown("### 📋 Raw Data")
    if st.button("📥 Load Raw Data Table", key="btn_tab4"):
        with st.spinner("⏳ Loading..."):
            df = load_single_temperature_csv(T, CSV_FILES_DIR)
            st.session_state["plot_tab4"] = df
    if "plot_tab4" in st.session_state:
        st.dataframe(st.session_state["plot_tab4"].style.format({"Co":"{:.3f}","Cr":"{:.3f}","Fe":"{:.3f}","Ni":"{:.3f}","G_LIQ":"{:.1f}","G_FCC":"{:.1f}"}), height=500, use_container_width=True)
        st.download_button("📥 Download CSV", st.session_state["plot_tab4"].to_csv(index=False), f"data_T{T}K.csv")

with tab5:
    st.markdown("### 🌞 Hierarchical Sunburst Visualization")
    if st.button("🌞 Generate Sunburst", key="btn_tab5") and "res_main" in st.session_state:
        res = st.session_state["res_main"]
        F_scale = max(abs(res.get("F_local", 0)), 1e-12) * 1e9
        ids = ["root", "T", "Co", "Cr", "Fe", "Ni", "GL", "GF", "dG", "F_loc", "dF"]
        parents = ["", "root", "T", "T", "T", "T", "T", "T", "T", "T", "T"]
        labels = ["System", f"T={T}K", "Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC", "ΔG", "F_local (1μm²)", "dF_net (dV)"]
        vals = [1.0, 1.0, max(x_co,0.0), max(x_cr,0.0), max(x_fe,0.0), max(x_ni,0.0),
                abs(res["g_liq"])/10000, abs(res["g_fcc"])/10000, abs(res["delta_G"])/10000,
                abs(res.get("F_local",0))*1e9/F_scale if F_scale>0 else 0,
                abs(res.get("dF_net",0))*1e9/F_scale if F_scale>0 else 0]
        colors = ["#f5f5f5", "#a8dadc", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#ffaa80", "#80b3ff", "#80ffb3", "#ffb380", "#8080ff"]
        hover = ["Root", f"T={T}K", f"x_Co={x_co:.3f}", f"x_Cr={x_cr:.3f}", f"x_Fe={x_fe:.3f}", f"x_Ni={x_ni:.3f}",
                 f"G_LIQ={res['g_liq']:.1f}", f"G_FCC={res['g_fcc']:.1f}", f"ΔG={res['delta_G']:.1f}",
                 f"F_loc={res.get('F_local',0):.2e} N", f"dF_net={res.get('dF_net',0):.2e} N"]
        fig = go.Figure(go.Sunburst(ids=ids, parents=parents, labels=labels, values=vals, marker=dict(colors=colors, line=dict(width=0.5, color='white')), hovertext=hover, hovertemplate='<b>%{label}</b><br>%{hovertext}<extra></extra>', maxdepth=3))
        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=600)
        st.session_state["plot_tab5"] = fig
    if "plot_tab5" in st.session_state: st.plotly_chart(st.session_state["plot_tab5"], use_container_width=True)

with tab6:
    st.markdown("### 🕸️ Multivariate Radar Chart")
    if st.button("🕸️ Generate Radar", key="btn_tab6") and "res_main" in st.session_state:
        res = st.session_state["res_main"]
        G_REF, F_REF, P_REF = 15000.0, 1e-9, 200.0
        cats = ["x_Co", "x_Cr", "x_Fe", "x_Ni", "T_norm", "|ΔG|/G", "|P_net|/P_ref", "|F_loc|/F_ref"]
        vals = [x_co, x_cr, x_fe, x_ni, T/3300.0, abs(res["delta_G"])/G_REF, abs(res["P_net"])/1e6/P_REF, abs(res.get("F_local",0))/F_REF]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='State', line=dict(color=res["phase_color"], width=2)))
        fig.add_trace(go.Scatterpolar(r=[0.25]*4 + [0.15]*4, theta=cats, fill='none', name='Ref', line=dict(dash='dot', color='gray')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=500)
        st.session_state["plot_tab6"] = fig
    if "plot_tab6" in st.session_state: st.plotly_chart(st.session_state["plot_tab6"], use_container_width=True)

with tab7:
    st.markdown("### ⚡ Driving Force Visualization")
    if st.button("⚡ Generate Force Plots", key="btn_tab7") and "res_main" in st.session_state:
        res = st.session_state["res_main"]
        if res.get("grain_size_um"):
            gs_um = np.linspace(0.5, 50, 200); gs_m = gs_um * 1e-6
            F_loc_s, dF_s = [], []
            P_chem = res["P_chem"]
            for g in gs_m:
                Sv_g = compute_Sv(g, res.get("shape_factor", 3.0))
                Pc_g = compute_capillary_pressure(gamma, compute_curvature_radius(g))
                Pn_g = P_chem - Pc_g
                F_loc_s.append(Pn_g * dA_REF); dF_s.append(Pn_g * Sv_g * dV)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gs_um.tolist(), y=np.array(F_loc_s)*1e9, name="F_local (nN)", line=dict(color="#1f77b4", width=3)))
            fig.add_trace(go.Scatter(x=gs_um.tolist(), y=np.array(dF_s)*1e9, name="dF_net (nN)", yaxis="y2", line=dict(color="#d62728", dash="dash")))
            fig.add_trace(go.Scatter(x=[float(res["grain_size_um"])], y=[float(res.get("F_local",0)*1e9)], mode='markers', name="Current", marker=dict(size=12, color='#1f77b4')))
            fig.update_layout(title="Driving Force vs Grain Size", height=450)
            st.session_state["plot_tab7"] = fig
        else: st.warning("⚠️ Enable Grain Size mode")
    if "plot_tab7" in st.session_state: st.plotly_chart(st.session_state["plot_tab7"], use_container_width=True)

with tab8:
    st.markdown("### 📊 Sensitivity Analysis")
    if st.button("📊 Generate Sensitivity Plots", key="btn_tab8") and "res_main" in st.session_state:
        res = st.session_state["res_main"]
        gs = res.get("grain_size_um", 10)*1e-6
        r = compute_curvature_radius(gs)
        g_sweep = np.linspace(0.3, 1.2, 50)
        p_net_g = [(res["P_chem"] - compute_capillary_pressure(g, r))/1e6 for g in g_sweep]
        fig1 = go.Figure(go.Scatter(x=g_sweep.tolist(), y=p_net_g, fill="tozeroy", line=dict(color="#d62728")))
        fig1.add_vline(x=gamma, line_dash="dash", line_color="black")
        k_sweep = np.linspace(2.0, 4.0, 50)
        dF_k = [compute_differential_force(res["P_net"], compute_Sv(gs, k), dV) for k in k_sweep]
        fig2 = go.Figure(go.Scatter(x=k_sweep.tolist(), y=dF_k, fill="tozeroy", line=dict(color="#9467bd")))
        fig2.add_vline(x=res.get("shape_factor", 3.0), line_dash="dash", line_color="black")
        st.session_state["plot_tab8"] = (fig1, fig2)
    if "plot_tab8" in st.session_state:
        st.plotly_chart(st.session_state["plot_tab8"][0], use_container_width=True)
        st.plotly_chart(st.session_state["plot_tab8"][1], use_container_width=True)

# ================= FOOTER =================
st.divider()
st.caption("""
**Cloud Optimization**: Explicit buttons + lazy loading + cached interpolators = ✅ Streamlit Cloud ready (1GB RAM limit)  
**Physics Corrected**: $P_{\text{net}} = P_{\text{chem}} - P_{\text{cap}}$. Positive $P_{\text{net}}$ drives LIQUID → FCC. Local force $F_{\text{local}} = P_{\text{net}} \cdot 1 \mu m^2$.  
**Units**: Energy [J/mol], Pressure [Pa], Force [N], Length [m]  
**References**: Porter & Easterling | Kaptay 2012 | Smith-Guttman 1953 | Turnbull 1950 | Christian 2002 | Miracle & Senkov 2017 (HEA Kinetics)
""")
col_f1, col_f2 = st.columns(2)
with col_f1: st.caption("🔄 Auto-refreshes on explicit button click")
with col_f2: st.caption(f"📍 Working dir: `{SCRIPT_DIR}`")
