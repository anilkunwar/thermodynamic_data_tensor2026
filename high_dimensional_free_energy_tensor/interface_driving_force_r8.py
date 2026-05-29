"""
CoCrFeNi Gibbs Free Energy Explorer
FULL FEATURE SET – EXPLICIT BUTTON CONTROL
FIXED: Interface motion direction sign convention.
Nothing runs automatically. All computations require button clicks.
"""

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
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_FILES_DIR = SCRIPT_DIR / "csv_files"
CSV_FILES_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CoCrFeNi Phase Stability (Button Control)",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)

st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy Explorer – Explicit Run Button")
st.markdown("""
**All calculations are manual:** change any parameter, then click **`Run Analysis`** to compute.
No automatic updates – you are in full control.
""")

# ================= CONSTANTS =================
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6
T_MIN_NORMALIZE, T_MAX_NORMALIZE = 300, 3300

LIT_GAMMA_RANGES = {
    "Pure metals (Turnbull)": {"min": 0.10, "max": 0.40, "typical": 0.25, "ref": "Turnbull, J. Appl. Phys. 1950"},
    "Metallic alloys (Kaptay)": {"min": 0.30, "max": 1.20, "typical": 0.65, "ref": "Kaptay, Calphad 2012"},
    "Fe-Cr binary (MD)": {"min": 0.18, "max": 0.32, "typical": 0.25, "ref": "Zhang et al., Comp. Mater. Sci. 2022"},
    "CoCrFeNi HEA (estimated)": {"min": 0.40, "max": 0.90, "typical": 0.60, "ref": "Constituent averaging"},
}
GAMMA_LIQUID_FCC_DEFAULT = 0.60

LIT_K_RANGES = {
    "Spherical grains": {"min": 2.00, "max": 2.00, "typical": 2.00, "ref": "Exact geometry"},
    "Tetrakaidecahedron (Kelvin)": {"min": 2.90, "max": 3.24, "typical": 3.00, "ref": "Smith-Guttman"},
    "Equiaxed polycrystals (exp)": {"min": 2.37, "max": 2.82, "typical": 2.60, "ref": "Hensler"},
    "Cubic grains": {"min": 6.00, "max": 6.00, "typical": 6.00, "ref": "Exact geometry"},
    "Columnar/dendritic": {"min": 1.50, "max": 2.50, "typical": 1.80, "ref": "Directional solidification"},
    "Nanocrystalline HEA": {"min": 2.80, "max": 3.50, "typical": 3.10, "ref": "High-entropy GB studies"}
}
SHAPE_FACTOR_DEFAULT = 3.00
DEFAULT_DV = 1e-18
DEFAULT_DV_UM3 = 1.0

GRAIN_SHAPE_FACTORS = {
    "Spherical (k=2.0)": 2.0,
    "Tetrakaidecahedron (k=3.0) ⭐": 3.0,
    "Equiaxed polycrystal (k=2.6)": 2.6,
    "Columnar (k=1.8)": 1.8,
    "Equiaxed cubic (k=6.0)": 6.0
}

# ================= HELPER FUNCTIONS =================
def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
    return (x_co * PURE_VM["Co"] + x_cr * PURE_VM["Cr"] +
            x_fe * PURE_VM["Fe"] + x_ni * PURE_VM["Ni"])

def normalize_temperature(T):
    return (T - T_MIN_NORMALIZE) / (T_MAX_NORMALIZE - T_MIN_NORMALIZE)

def get_phase_preference(delta_G):
    if delta_G < 0:
        return "FCC favored", "#1f77b4", "🔵"
    else:
        return "LIQUID favored", "#ff7f0e", "🟠"

def compute_Sv(grain_size_m, shape_factor):
    return shape_factor / grain_size_m

def compute_total_area(Sv, sample_volume_m3):
    return Sv * sample_volume_m3

def compute_curvature_radius(grain_size_m, geometry_factor=0.25):
    return grain_size_m * geometry_factor

def compute_capillary_pressure(gamma, curvature_r):
    if curvature_r <= 0:
        return np.inf
    return (2.0 * gamma) / curvature_r

def compute_net_pressure(delta_G_v, P_capillary):
    return delta_G_v - P_capillary

def compute_differential_force(P_net, Sv, dV):
    return P_net * Sv * dV

def propagate_uncertainty_gamma_k(gamma_nominal, k_nominal, grain_size_m,
                                  delta_G_v, n_samples=300, gamma_range=None, k_range=None):
    if gamma_range is None:
        gamma_range = (gamma_nominal * 0.8, gamma_nominal * 1.2)
    if k_range is None:
        k_range = (k_nominal * 0.85, k_nominal * 1.15)
    gamma_samples = np.random.uniform(gamma_range[0], gamma_range[1], n_samples)
    k_samples = np.random.uniform(k_range[0], k_range[1], n_samples)
    r = compute_curvature_radius(grain_size_m)
    P_cap_samples = compute_capillary_pressure(gamma_samples, r)
    P_net_samples = compute_net_pressure(delta_G_v, P_cap_samples)
    Sv_samples = compute_Sv(grain_size_m, k_samples)
    dV = DEFAULT_DV
    dF_samples = compute_differential_force(P_net_samples, Sv_samples, dV)
    return {
        "P_net_mean": np.mean(P_net_samples),
        "P_net_std": np.std(P_net_samples),
        "P_net_95ci": norm.interval(0.95, loc=np.mean(P_net_samples), scale=np.std(P_net_samples)),
        "dF_net_mean": np.mean(dF_samples),
        "dF_net_std": np.std(dF_samples),
        "dF_net_95ci": norm.interval(0.95, loc=np.mean(dF_samples), scale=np.std(dF_samples)),
        "P_net_samples": P_net_samples,
        "dF_samples": dF_samples
    }

# ================= DATA LOADING & INTERPOLATION (cached) =================
@st.cache_resource(ttl=3600, max_entries=5)
def get_interpolators_for_T(T):
    """Load CSV for given T and return (interp_liq, interp_fcc). Returns (None,None) on failure."""
    path = CSV_FILES_DIR / f"Gibbs_{T}K.csv"
    try:
        df = pd.read_csv(path, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
    except Exception:
        return None, None
    df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
    df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
    if len(df) == 0:
        return None, None

    n_co = df["Co"].nunique()
    n_cr = df["Cr"].nunique()
    n_fe = df["Fe"].nunique()
    is_reg = (len(df) == n_co * n_cr * n_fe)

    if is_reg:
        co_vals = np.sort(df["Co"].unique())
        cr_vals = np.sort(df["Cr"].unique())
        fe_vals = np.sort(df["Fe"].unique())
        df_sorted = df.sort_values(["Co", "Cr", "Fe"])
        g_liq_grid = df_sorted["G_LIQ"].values.reshape(len(co_vals), len(cr_vals), len(fe_vals))
        g_fcc_grid = df_sorted["G_FCC"].values.reshape(len(co_vals), len(cr_vals), len(fe_vals))
        interp_liq = RegularGridInterpolator((co_vals, cr_vals, fe_vals), g_liq_grid, bounds_error=False, fill_value=np.nan)
        interp_fcc = RegularGridInterpolator((co_vals, cr_vals, fe_vals), g_fcc_grid, bounds_error=False, fill_value=np.nan)
    else:
        points = df[["Co", "Cr", "Fe"]].values
        interp_liq = LinearNDInterpolator(points, df["G_LIQ"].values, fill_value=np.nan)
        interp_fcc = LinearNDInterpolator(points, df["G_FCC"].values, fill_value=np.nan)
    return interp_liq, interp_fcc

def evaluate_point(x_co, x_cr, x_fe, T):
    interp_liq, interp_fcc = get_interpolators_for_T(T)
    if interp_liq is None or interp_fcc is None:
        return None, None
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq = float(interp_liq(point))
        g_fcc = float(interp_fcc(point))
    except:
        return None, None
    if np.isnan(g_liq) or np.isnan(g_fcc):
        return None, None
    return g_liq, g_fcc

# ================= SIDEBAR: USER INPUTS =================
st.sidebar.header("🎛️ Parameters (change then click Run)")

# Temperature selection
available_temps = sorted([int(f.stem.replace("Gibbs_", "").replace("K", ""))
                          for f in CSV_FILES_DIR.glob("Gibbs_*.csv") if f.stem.replace("Gibbs_", "").replace("K", "").isdigit()])
if not available_temps:
    st.error("No temperature CSV files found. Please add files like Gibbs_1000K.csv")
    st.stop()

T = st.sidebar.select_slider("Temperature (K)", options=available_temps, value=1000 if 1000 in available_temps else available_temps[0])

# Composition
col1, col2, col3 = st.sidebar.columns(3)
x_co = col1.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_cr = col2.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_fe = col3.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid: x_Ni = {x_ni:.4f} (must be 0–1)")
else:
    st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

st.sidebar.subheader("📐 Molar Volume Model")
vm_model = st.sidebar.radio("Model", ["Constant", "Composition‑dependent"], index=1)
if vm_model == "Constant":
    V_m = st.sidebar.number_input("Vₘ (m³/mol)", 1e-7, 1e-4, DEFAULT_VM, 1e-7, format="%.2e")
else:
    V_m = composition_dependent_vm(x_co, x_cr, x_fe, x_ni)

st.sidebar.divider()
st.sidebar.subheader("🔧 Interface / Grain Boundary")
area_mode = st.sidebar.radio("Area Calculation Mode", ["Direct Input (A)", "Grain Size Derived (Sv x V)"], index=1)

if area_mode == "Direct Input (A)":
    interface_area = st.sidebar.number_input("Interface Area A (m²)", 1e-20, 1e2, 1e-8, step=1e-10, format="%.2e")
    grain_size_um = None
else:
    grain_size_um = st.sidebar.number_input("Avg Grain Size d (μm)", 0.001, 10000.0, 10.0, step=0.1, format="%.3f")
    grain_size_m = grain_size_um * 1e-6
    shape_choice = st.sidebar.selectbox("Grain Shape Factor", list(GRAIN_SHAPE_FACTORS.keys()), index=1)
    shape_factor = GRAIN_SHAPE_FACTORS[shape_choice]
    sample_volume_cm3 = st.sidebar.number_input("Sample Volume V (cm³)", 1e-9, 1e6, 1.0, step=0.1, format="%.3f")
    sample_volume_m3 = sample_volume_cm3 * 1e-6
    Sv = compute_Sv(grain_size_m, shape_factor)
    interface_area = compute_total_area(Sv, sample_volume_m3)

st.sidebar.divider()
st.sidebar.subheader("🌊 Capillary Correction")
use_capillary = st.sidebar.checkbox("Enable Capillary Correction", value=True)
if use_capillary:
    gamma_method = st.sidebar.radio("γ Selection", ["Manual input", "Preset from literature"], index=1)
    if gamma_method == "Preset from literature":
        gamma_preset = st.sidebar.selectbox("System/Preset", list(LIT_GAMMA_RANGES.keys()), index=3)
        gamma_info = LIT_GAMMA_RANGES[gamma_preset]
        gamma = st.sidebar.slider(f"γ (N/m) - {gamma_preset}", gamma_info["min"], gamma_info["max"], gamma_info["typical"], 0.01, format="%.2f")
    else:
        gamma = st.sidebar.number_input("γ (N/m)", 0.01, 5.0, GAMMA_LIQUID_FCC_DEFAULT, 0.01, format="%.2f")
    dV_um3 = st.sidebar.number_input("Local Volume dV (μm³)", 0.001, 1000.0, DEFAULT_DV_UM3, step=0.1, format="%.3f")
    dV = dV_um3 * 1e-18
    enable_uncertainty = st.sidebar.checkbox("Enable Monte Carlo Uncertainty", value=False)
    if enable_uncertainty:
        gamma_unc_pct = st.sidebar.slider("γ uncertainty (%)", 5, 50, 20)
        k_unc_pct = st.sidebar.slider("k uncertainty (%)", 5, 30, 15)
        n_mc = st.sidebar.selectbox("MC samples", [100, 300, 500, 1000], index=1)
else:
    gamma = 0.0; dV = 0.0; enable_uncertainty = False

# ================= MAIN RUN BUTTON =================
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Store results
if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    with st.spinner("Loading temperature data and computing..."):
        if x_ni < 0 or x_ni > 1:
            st.error("Invalid composition: Ni fraction out of [0,1].")
            st.session_state.results = None
        else:
            g_liq, g_fcc = evaluate_point(x_co, x_cr, x_fe, T)
            if g_liq is None or g_fcc is None:
                st.error("Composition outside available data range. Adjust composition.")
                st.session_state.results = None
            else:
                delta_G = g_fcc - g_liq
                delta_G_v = delta_G / V_m
                delta_G_v_MPa = delta_G_v / 1e6
                phase_pref, phase_color, phase_emoji = get_phase_preference(delta_G)

                if area_mode == "Direct Input (A)":
                    net_force = delta_G_v * interface_area
                    Sv_val = None
                else:
                    if use_capillary:
                        curvature_r = compute_curvature_radius(grain_size_m)
                        P_cap = compute_capillary_pressure(gamma, curvature_r)
                        P_net = compute_net_pressure(delta_G_v, P_cap)
                        net_force = P_net * interface_area
                        dF_net = compute_differential_force(P_net, Sv, dV) if use_capillary else None
                        P_cap_MPa = P_cap / 1e6
                        P_net_MPa = P_net / 1e6
                    else:
                        net_force = delta_G_v * interface_area
                        P_cap_MPa = None; P_net_MPa = delta_G_v_MPa; dF_net = None

                mc_results = None
                if enable_uncertainty and use_capillary and area_mode != "Direct Input (A)":
                    gamma_range = (gamma * (1 - gamma_unc_pct/100), gamma * (1 + gamma_unc_pct/100))
                    k_range = (shape_factor * (1 - k_unc_pct/100), shape_factor * (1 + k_unc_pct/100))
                    mc_results = propagate_uncertainty_gamma_k(
                        gamma_nominal=gamma, k_nominal=shape_factor,
                        grain_size_m=grain_size_m, delta_G_v=delta_G_v,
                        n_samples=n_mc, gamma_range=gamma_range, k_range=k_range
                    )

                st.session_state.results = {
                    "g_liq": g_liq, "g_fcc": g_fcc, "delta_G": delta_G, "delta_G_v_MPa": delta_G_v_MPa,
                    "phase_pref": phase_pref, "phase_color": phase_color, "phase_emoji": phase_emoji,
                    "V_m": V_m, "interface_area": interface_area, "net_force": net_force,
                    "use_capillary": use_capillary, "area_mode": area_mode,
                    "grain_size_um": grain_size_um, "shape_factor": shape_factor if area_mode != "Direct Input (A)" else None,
                    "sample_volume_cm3": sample_volume_cm3 if area_mode != "Direct Input (A)" else None,
                    "Sv": Sv if area_mode != "Direct Input (A)" else None,
                    "gamma": gamma if use_capillary else None, "dV": dV if use_capillary else None,
                    "P_cap_MPa": P_cap_MPa if use_capillary else None,
                    "P_net_MPa": P_net_MPa if use_capillary else delta_G_v_MPa,
                    "dF_net": dF_net if use_capillary and area_mode != "Direct Input (A)" else None,
                    "mc_results": mc_results,
                    "T": T, "x_co": x_co, "x_cr": x_cr, "x_fe": x_fe, "x_ni": x_ni,
                }
                st.success("Analysis complete!")

# ================= DISPLAY RESULTS (if available) =================
if st.session_state.results is None:
    st.info("👈 Click **Run Analysis** in the sidebar to compute results.")
    st.stop()

res = st.session_state.results

st.header(f"📊 Results at T = {res['T']} K")
col_a, col_b, col_c = st.columns(3)
col_a.metric("G_LIQUID", f"{res['g_liq']:,.1f} J/mol")
col_b.metric("G_FCC", f"{res['g_fcc']:,.1f} J/mol")
col_c.metric("ΔG = G_FCC − G_LIQ", f"{res['delta_G']:,.1f} J/mol",
             delta=res['phase_pref'], delta_color="normal" if res['delta_G'] < 0 else "inverse")
st.success(f"🏆 Most stable phase: **{'FCC' if res['delta_G'] < 0 else 'LIQUID'}** {res['phase_emoji']}")

st.divider()
st.subheader("⚙️ Interface Driving Force (Mechanical)")

col_p1, col_p2, col_p3 = st.columns(3)
col_p1.metric("Driving Pressure ΔGᵥ", f"{res['delta_G_v_MPa']:.3f} MPa")
if res['use_capillary'] and res.get('P_cap_MPa') is not None:
    col_p2.metric("Capillary Pressure", f"{res['P_cap_MPa']:.3f} MPa", delta="resists growth", delta_color="inverse")
    col_p3.metric("Net Pressure P_net", f"{res['P_net_MPa']:.3f} MPa")
    # ---------- FIXED SIGN CONVENTION ----------
    # Negative P_net -> FCC grows; Positive P_net -> LIQUID grows
    if res['P_net_MPa'] < 0:
        motion_direction = "→ FCC grows"
    else:
        motion_direction = "→ LIQUID grows"
    st.info(f"""
    **Capillary Correction:**  
    - Grain size: **{res['grain_size_um']:.2f} μm** → r = D/4  
    - Capillary resistance: {res['P_cap_MPa']:.2f} MPa (γ = {res['gamma']:.2f} N/m)  
    - Net pressure = {res['delta_G_v_MPa']:.2f} − {res['P_cap_MPa']:.2f} = {res['P_net_MPa']:.2f} MPa  
    - Interface motion: **{motion_direction}**
    """)
else:
    col_p2.metric("SI units", f"{res['delta_G_v_MPa']*1e6:.2e} N/m²")
    # For non-capillary case, direction follows sign of ΔG (negative -> FCC favored)
    if res['delta_G'] < 0:
        motion_direction = "→ FCC grows"
    else:
        motion_direction = "→ LIQUID grows"
    col_p3.metric("Interface motion", motion_direction)

st.markdown("### 🔧 Force on Interface")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.metric("Interface Area", f"{res['interface_area']:.2e} m²")
col_f2.metric("Net Pressure", f"{res['P_net_MPa']:.3f} MPa")
col_f3.metric("Net Force", f"{res['net_force']:.3e} N")

if res['area_mode'] == "Grain Size Derived (Sv x V)" and res.get('grain_size_um') is not None:
    st.markdown(f"""
    **Grain Size Details:**  
    - d = {res['grain_size_um']:.2f} μm, k = {res['shape_factor']:.1f} → S_v = {res['Sv']:.2e} m²/m³  
    - Sample V = {res['sample_volume_cm3']:.2f} cm³ → A_total = {res['interface_area']:.2e} m²  
    - Net force = P_net × A_total = {res['net_force']:.3e} N
    """)

if res.get('dF_net') is not None:
    st.markdown("### 🧬 Differential Force")
    st.metric("dF_net on local element", f"{res['dF_net']:.3e} N", help=f"dV = {res['dV']:.2e} m³")

# Monte Carlo results
if res.get('mc_results') is not None:
    with st.expander("📊 Monte Carlo Uncertainty Results", expanded=True):
        mc = res['mc_results']
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        col_mc1.metric("P_net (95% CI)", f"{mc['P_net_mean']/1e6:.3f} MPa", delta=f"±{mc['P_net_std']/1e6:.3f} MPa")
        col_mc2.metric("dF_net (95% CI)", f"{mc['dF_net_mean']:.2e} N", delta=f"±{mc['dF_net_std']:.2e} N")
        col_mc3.metric("Rel. uncertainty", f"P: {mc['P_net_std']/abs(mc['P_net_mean'])*100:.1f}%, F: {mc['dF_net_std']/abs(mc['dF_net_mean'])*100:.1f}%")
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            fig_p = go.Figure(data=[go.Histogram(x=mc['P_net_samples']/1e6, nbinsx=30, marker_color=res['phase_color'])])
            fig_p.update_layout(title="P_net Distribution", xaxis_title="MPa")
            st.plotly_chart(fig_p, use_container_width=True)
        with col_plot2:
            fig_f = go.Figure(data=[go.Histogram(x=mc['dF_samples'], nbinsx=30, marker_color="#9467bd")])
            fig_f.update_layout(title="dF_net Distribution", xaxis_title="N")
            st.plotly_chart(fig_f, use_container_width=True)

# ================= LAZY THEORY EXPANDER (static, no computation) =================
def display_latex_theory():
    st.markdown("## 📚 Thermodynamic Theory Reference")
    with st.expander("📋 View Theory (Rendered Equations)", expanded=False):
        st.markdown(r"""
        | **Concept** | **Mathematical Formulation** |
        |:---|:---|
        | **Gibbs Free Energy** | $G_{\text{phase}}(x_{\text{Co}},x_{\text{Cr}},x_{\text{Fe}},x_{\text{Ni}},T)$ <br> Computed from CALPHAD databases for LIQUID and FCC phases. <br> Constraint: $\sum_i x_i = 1$ |
        | **Driving Force ($\Delta G$)** | $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}} \quad [\text{J/mol}]$ <br> $\Delta G < 0$: FCC favored; $\Delta G > 0$: LIQUID favored |
        | **Volumetric Driving Pressure** | $\Delta G_v = \frac{\Delta G}{V_m} \quad [\text{Pa} = \text{N/m}^2]$ <br> $V_m$: Molar volume $[\text{m}^3/\text{mol}]$ |
        | **Molar Volume Models** | *Constant:* $V_m = V_0$ (user-defined) <br> *Composition-dependent:* $V_m = \sum_i x_i V_m^{(i)}$ |
        | **Grain Boundary Area Density** | $S_v = \frac{k}{d} \quad [\text{m}^2/\text{m}^3]$ <br> $d$: avg. FCC grain size $[\text{m}]$, $k$: shape factor |
        | **Total Interface Area** | $A_{\text{total}} = S_v \times V = \frac{k \cdot V}{d} \quad [\text{m}^2]$ <br> $V$: sample volume $[\text{m}^3]$ |
        | **Net Driving Force (Bulk)** | $F_{\text{total}} = \Delta G_v \times A_{\text{total}} = \Delta G_v \cdot \frac{k \cdot V}{d} \quad [\text{N}]$ |
        | **Interface Force (Single Area)** | $F = \Delta G_v \times A \quad [\text{N}]$ <br> $A$: single interface area $[\text{m}^2]$ |
        | **Temperature Normalization** | $T_{\text{norm}} = \frac{T - 300}{3300 - 300} \in [0,1]$ |
        """)
        st.markdown(r"""
        ### 🌊 Capillary Pressure Correction
        | **Concept** | **Mathematical Formulation** |
        |:---|:---|
        | **Local Net Pressure** | $P_{\text{net}} = \Delta G_v - \frac{2\gamma}{r}$ |
        | **Curvature Radius** | $r \approx \frac{D}{4}$ |
        | **Capillary Pressure** | $P_{\text{capillary}} = \frac{2\gamma}{r}$ |
        | **Stereological Force** | $dF_{\text{net}} = P_{\text{net}} \cdot S_v \cdot dV$ |
        """)
        st.markdown("### 📖 References")
        st.markdown("""
        1. Porter, D.A., Easterling, K.E. *Phase Transformations in Metals and Alloys*, CRC Press.
        2. Kaptay, G. *Calphad* **38**, 2012.
        3. Smith, C.S., Guttman, L. *Trans. AIME* **197**, 1953.
        4. Underwood, E.E. *Quantitative Stereology*, Addison-Wesley.
        5. Christian, J.W. *The Theory of Transformations in Metals and Alloys*, Pergamon.
        """)

st.divider()
display_latex_theory()

# ================= ALL PLOTS (button-driven) =================
st.header("🗺️ Exploration Tools (all require button clicks)")

tabs = st.tabs(["📈 G vs Composition", "🌡️ Phase Map vs T", "📊 ΔGᵥ vs Composition", 
                "📋 Raw Data", "🌞 Sunburst", "🕸️ Radar", "📐 Grain Size Scaling", "📊 Uncertainty Analysis"])

# Helper for scanning
def scan_g_vs_composition(scan_var, fixed_val, T):
    max_val = 1.0 - 2*fixed_val - 0.01
    if max_val < 0.01:
        return [], [], []
    x_vals = np.linspace(0.01, max_val, 100)
    g_l, g_f = [], []
    valid = []
    for xv in x_vals:
        if scan_var == "x_Co": gl, gf = evaluate_point(xv, fixed_val, fixed_val, T)
        elif scan_var == "x_Cr": gl, gf = evaluate_point(fixed_val, xv, fixed_val, T)
        else: gl, gf = evaluate_point(fixed_val, fixed_val, xv, T)
        if gl is not None:
            g_l.append(gl); g_f.append(gf); valid.append(xv)
        else:
            g_l.append(np.nan); g_f.append(np.nan)
    return valid, g_l, g_f

with tabs[0]:  # G vs Composition
    st.markdown("### Gibbs Energy along composition axis")
    scan_var = st.radio("Vary composition", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan1")
    fixed_val = st.slider("Fixed other components", 0.0, 0.4, 0.2, 0.01, key="fix1")
    if st.button("Update G vs Composition Plot", key="btn_gcomp"):
        xv, gl, gf = scan_g_vs_composition(scan_var, fixed_val, res['T'])
        if xv:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xv, y=gl, name="G_LIQUID", line=dict(color="#ff7f0e")))
            fig.add_trace(go.Scatter(x=xv, y=gf, name="G_FCC", line=dict(color="#1f77b4")))
            fig.update_layout(title=f"Gibbs Energy vs {scan_var} at T={res['T']} K", xaxis_title=scan_var, yaxis_title="G (J/mol)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cannot scan – fixed values too large")

with tabs[1]:  # Phase Map vs T
    st.markdown("### Phase stability vs Temperature")
    if st.button("Update Temperature Scan", key="btn_Tscan"):
        T_list = available_temps
        dG_list, dGv_list = [], []
        valid_T = []
        for Tval in T_list:
            gl, gf = evaluate_point(res['x_co'], res['x_cr'], res['x_fe'], Tval)
            if gl is not None:
                dG_list.append(gf - gl)
                Vm_local = composition_dependent_vm(res['x_co'], res['x_cr'], res['x_fe'], res['x_ni']) if vm_model == "Composition‑dependent" else V_m
                dGv_list.append((gf - gl) / Vm_local / 1e6)
                valid_T.append(Tval)
        if valid_T:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valid_T, y=dG_list, name="ΔG (J/mol)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=valid_T, y=dGv_list, name="ΔGᵥ (MPa)", yaxis="y2", line=dict(dash="dot")))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(yaxis=dict(title="ΔG (J/mol)"), yaxis2=dict(title="ΔGᵥ (MPa)", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid data for temperature scan")

with tabs[2]:  # ΔGᵥ vs Composition
    st.markdown("### Driving Pressure vs Composition")
    scan_var2 = st.radio("Scan along", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan2")
    fixed_val2 = st.slider("Fixed other components", 0.0, 0.4, 0.2, 0.01, key="fix2")
    if st.button("Update ΔGᵥ Plot", key="btn_dgv"):
        max_val = 1.0 - 2*fixed_val2 - 0.01
        if max_val < 0.01:
            st.error("Fixed values too large")
        else:
            x_vals = np.linspace(0.01, max_val, 100)
            dGv_vals = []
            valid_x = []
            for xv in x_vals:
                if scan_var2 == "x_Co": co, cr, fe = xv, fixed_val2, fixed_val2
                elif scan_var2 == "x_Cr": co, cr, fe = fixed_val2, xv, fixed_val2
                else: co, cr, fe = fixed_val2, fixed_val2, xv
                ni = 1 - (co+cr+fe)
                if ni < 0 or ni > 1:
                    continue
                gl, gf = evaluate_point(co, cr, fe, res['T'])
                if gl is not None:
                    Vm_local = composition_dependent_vm(co, cr, fe, ni) if vm_model == "Composition‑dependent" else V_m
                    dGv = (gf - gl) / Vm_local / 1e6
                    dGv_vals.append(dGv)
                    valid_x.append(xv)
            if valid_x:
                fig = go.Figure(go.Scatter(x=valid_x, y=dGv_vals, mode="lines", fill="tozeroy"))
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # Raw Data
    st.markdown("### Raw Thermodynamic Data for Current Temperature")
    if st.button("Load Raw Data", key="btn_raw"):
        df_raw = None
        try:
            path = CSV_FILES_DIR / f"Gibbs_{res['T']}K.csv"
            df_raw = pd.read_csv(path)
            st.dataframe(df_raw.style.format({c: "{:.3f}" for c in ["Co","Cr","Fe","Ni"]}), use_container_width=True, height=400)
            csv = df_raw.to_csv(index=False)
            st.download_button("Download CSV", csv, f"Gibbs_{res['T']}K.csv", "text/csv")
        except:
            st.error("Could not load raw data.")

with tabs[4]:  # Sunburst
    st.markdown("### 🌞 Sunburst (Current State)")
    if st.button("Generate Sunburst", key="btn_sunburst"):
        element_colors = {"Co": "#1f77b4", "Cr": "#ff7f0e", "Fe": "#2ca02c", "Ni": "#d62728"}
        ids = ["root", f"T_{res['T']}"]
        parents = ["", "root"]
        labels = ["CoCrFeNi System", f"Temperature: {res['T']} K"]
        values = [1, 1]
        colors = ["lightgray", "lightblue"]
        for elem, frac in [("Co", res['x_co']), ("Cr", res['x_cr']), ("Fe", res['x_fe']), ("Ni", res['x_ni'])]:
            elem_id = f"T_{res['T']}_{elem}"
            ids.append(elem_id); parents.append(f"T_{res['T']}"); labels.append(f"{elem}<br>{frac:.3f}"); values.append(frac); colors.append(element_colors[elem])
            force_id = f"{elem_id}_force"
            ids.append(force_id); parents.append(elem_id); labels.append(f"Force: {res['net_force']:.2e} N"); values.append(abs(res['net_force'])); colors.append("gold")
        fig = go.Figure(go.Sunburst(ids=ids, parents=parents, labels=labels, values=values, marker=dict(colors=colors), branchvalues='total'))
        fig.update_layout(title=f"Sunburst: {res['T']} K | Force = {res['net_force']:.2e} N", width=700, height=700)
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:  # Radar
    st.markdown("### 🕸️ Radar Chart")
    if st.button("Update Radar", key="btn_radar"):
        T_norm = normalize_temperature(res['T'])
        dGv_norm = min(1.0, abs(res['delta_G_v_MPa'])/100)
        if res['use_capillary'] and res.get('P_cap_MPa') is not None:
            categories = ['x_Co', 'x_Cr', 'x_Fe', 'x_Ni', 'T (norm)', '|ΔGᵥ|', '|P_cap|', '|P_net|']
            values = [res['x_co'], res['x_cr'], res['x_fe'], res['x_ni'], T_norm, dGv_norm,
                      min(1.0, abs(res['P_cap_MPa'])/100), min(1.0, abs(res['P_net_MPa'])/100)]
        else:
            categories = ['x_Co', 'x_Cr', 'x_Fe', 'x_Ni', 'T (norm)', '|ΔGᵥ|']
            values = [res['x_co'], res['x_cr'], res['x_fe'], res['x_ni'], T_norm, dGv_norm]
        fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', name=res['phase_pref'], line=dict(color=res['phase_color'])))
        fig.add_trace(go.Scatterpolar(r=[0.25,0.25,0.25,0.25,0.5,0.3], theta=categories[:6], fill='none', name='Equiatomic', line=dict(dash='dot', color='gray')))
        fig.update_layout(title=f"Radar Chart @ {res['T']} K", polar=dict(radialaxis=dict(range=[0,1])))
        st.plotly_chart(fig, use_container_width=True)

with tabs[6]:  # Grain Size Scaling
    st.markdown("### 📐 Grain Size Scaling Analysis")
    if st.button("Generate Grain Size Scaling Plots", key="btn_grain_scale"):
        if res['area_mode'] == "Direct Input (A)":
            st.warning("Grain size scaling requires 'Grain Size Derived' area mode.")
        elif not res['use_capillary']:
            st.warning("Grain size scaling requires capillary correction enabled.")
        else:
            grain_sizes_um = np.linspace(0.5, 50, 100)
            grain_sizes_m = grain_sizes_um * 1e-6
            P_caps, P_nets, dF_nets = [], [], []
            for gs_m in grain_sizes_m:
                r = compute_curvature_radius(gs_m)
                P_cap = compute_capillary_pressure(res['gamma'], r)
                P_net_val = compute_net_pressure(res['delta_G_v_MPa']*1e6, P_cap)
                Sv_val = compute_Sv(gs_m, res['shape_factor'])
                dF = compute_differential_force(P_net_val, Sv_val, res['dV'])
                P_caps.append(P_cap/1e6)
                P_nets.append(P_net_val/1e6)
                dF_nets.append(dF)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=grain_sizes_um, y=P_caps, name="P_capillary", line=dict(dash="dash", color="#d62728")))
            fig1.add_trace(go.Scatter(x=grain_sizes_um, y=P_nets, name="P_net", line=dict(color="#2ca02c")))
            fig1.add_hline(y=res['delta_G_v_MPa'], line_dash="dot", line_color="blue", annotation_text=f"ΔGᵥ = {res['delta_G_v_MPa']:.1f} MPa")
            fig1.update_layout(title="Net Pressure vs Grain Size", xaxis_title="Grain Size D (μm)", yaxis_title="Pressure (MPa)")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=grain_sizes_um, y=dF_nets, fill="tozeroy", line=dict(color="#9467bd"), name="dF_net"))
            fig2.update_layout(title=f"Differential Force on {res['dV']/1e-18:.1f} μm³ Element vs Grain Size", xaxis_title="Grain Size D (μm)", yaxis_title="dF_net (N)")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Key Scaling Relationships")
            st.markdown(f"""
            | Parameter | Small Grains (2.5 μm) | Large Grains (10 μm) |
            |:---|:---|:---|
            | $S_v$ | {compute_Sv(2.5e-6, res['shape_factor']):.2e} m²/m³ | {compute_Sv(10e-6, res['shape_factor']):.2e} m²/m³ |
            | $P_{{cap}}$ | {compute_capillary_pressure(res['gamma'], 2.5e-6/4)/1e6:.2f} MPa | {compute_capillary_pressure(res['gamma'], 10e-6/4)/1e6:.2f} MPa |
            | $P_{{net}}$ | {(res['delta_G_v_MPa']*1e6 - compute_capillary_pressure(res['gamma'], 2.5e-6/4))/1e6:.2f} MPa | {(res['delta_G_v_MPa']*1e6 - compute_capillary_pressure(res['gamma'], 10e-6/4))/1e6:.2f} MPa |
            | $dF_{{net}}$ | {compute_differential_force(res['delta_G_v_MPa']*1e6 - compute_capillary_pressure(res['gamma'], 2.5e-6/4), compute_Sv(2.5e-6, res['shape_factor']), res['dV']):.2e} N | {compute_differential_force(res['delta_G_v_MPa']*1e6 - compute_capillary_pressure(res['gamma'], 10e-6/4), compute_Sv(10e-6, res['shape_factor']), res['dV']):.2e} N |
            """)

with tabs[7]:  # Uncertainty Analysis
    st.markdown("### 📊 Parameter Uncertainty & Sensitivity Analysis")
    if st.button("Run Uncertainty Sensitivity", key="btn_uncertainty"):
        if not res['use_capillary'] or res['area_mode'] == "Direct Input (A)":
            st.warning("Uncertainty analysis requires capillary correction and grain size mode.")
        else:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### γ Sensitivity (fixed k)")
                gamma_sweep = np.linspace(0.3, 1.2, 50)
                P_net_sweep = []
                for g_val in gamma_sweep:
                    P_cap = compute_capillary_pressure(g_val, compute_curvature_radius(res['grain_size_um']*1e-6))
                    P_net_sweep.append(compute_net_pressure(res['delta_G_v_MPa']*1e6, P_cap)/1e6)
                fig_g = go.Figure(go.Scatter(x=gamma_sweep, y=P_net_sweep, mode="lines", fill="tozeroy", line=dict(color="#d62728")))
                fig_g.add_vline(x=res['gamma'], line_dash="dash", annotation_text=f"Current γ = {res['gamma']:.2f}")
                fig_g.update_layout(title="Net Pressure vs Interfacial Energy γ", xaxis_title="γ (N/m)", yaxis_title="P_net (MPa)")
                st.plotly_chart(fig_g, use_container_width=True)
            with col_s2:
                st.markdown("#### k Sensitivity (fixed γ)")
                k_sweep = np.linspace(2.0, 4.0, 50)
                dF_sweep = []
                for k_val in k_sweep:
                    Sv_val = compute_Sv(res['grain_size_um']*1e-6, k_val)
                    dF_val = compute_differential_force(res['P_net_MPa']*1e6, Sv_val, res['dV'])
                    dF_sweep.append(dF_val)
                fig_k = go.Figure(go.Scatter(x=k_sweep, y=dF_sweep, mode="lines", fill="tozeroy", line=dict(color="#9467bd")))
                fig_k.add_vline(x=res['shape_factor'], line_dash="dash", annotation_text=f"Current k = {res['shape_factor']:.2f}")
                fig_k.update_layout(title="Differential Force vs Shape Factor k", xaxis_title="k", yaxis_title="dF_net (N)")
                st.plotly_chart(fig_k, use_container_width=True)

            st.markdown("#### Combined Uncertainty Summary")
            # Quick Monte Carlo with default ±20% γ, ±15% k
            gamma_range = (res['gamma']*0.8, res['gamma']*1.2)
            k_range = (res['shape_factor']*0.85, res['shape_factor']*1.15)
            mc_quick = propagate_uncertainty_gamma_k(
                gamma_nominal=res['gamma'], k_nominal=res['shape_factor'],
                grain_size_m=res['grain_size_um']*1e-6, delta_G_v=res['delta_G_v_MPa']*1e6,
                n_samples=300, gamma_range=gamma_range, k_range=k_range
            )
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            col_c1.metric("P_net nominal", f"{res['P_net_MPa']:.2f} MPa", delta=f"MC mean: {mc_quick['P_net_mean']/1e6:.2f} ± {mc_quick['P_net_std']/1e6:.2f}")
            col_c2.metric("dF_net nominal", f"{res['dF_net']:.2e} N", delta=f"MC mean: {mc_quick['dF_net_mean']:.2e} ± {mc_quick['dF_net_std']:.2e}")
            col_c3.metric("Capillary contribution", f"{abs(res['P_cap_MPa']/res['delta_G_v_MPa'])*100:.1f}%", help="Fraction of driving pressure offset by capillarity")
            col_c4.metric("Rel. uncertainty (dF)", f"{mc_quick['dF_net_std']/abs(mc_quick['dF_net_mean'])*100:.1f}%")
            st.info("Uncertainty ranges: γ ±20%, k ±15% (default). Adjust in sidebar before running main analysis for custom MC.")

st.divider()
st.caption("🔘 All calculations and plots are triggered manually via buttons. No automatic updates.")
