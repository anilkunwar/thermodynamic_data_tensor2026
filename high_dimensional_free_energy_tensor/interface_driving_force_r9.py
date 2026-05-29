"""
CoCrFeNi Gibbs Free Energy Explorer
Optimized with RegularGridInterpolator + Build Button
WITH: Sunburst Charts, Radar Charts, Streamlit Markdown Theory Documentation
PLUS: Grain Size Derived Interfacial Area Density (Sv), Capillary Correction & Differential Force
SIGN CONVENTION: Aligned with standard thermodynamic theory (Porter & Easterling, Christian)
  - ΔG = G_FCC - G_LIQ  [J/mol]
  - P_chem = -ΔG / V_m = (G_LIQ - G_FCC) / V_m  [Pa]
  - P_net = P_chem - P_capillary  [Pa]
  - Positive P_net drives LIQUID → FCC (growth)
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

st.set_page_config(
    page_title="CoCrFeNi Phase Stability",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)
st.title("⚛️ Co-Cr-Fe-Ni Gibbs Energy & Interface Driving Force")
st.markdown(r"""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → $P_{chem} = -\Delta G/V_m$ (Pa = N/m²) → Interface driving pressure  
**New:** Grain size → $S_v$ → capillary-corrected $P_{net}$ → differential force $dF_{net}$
""")

# ================= CONSTANTS =================
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6
T_MIN_NORMALIZE, T_MAX_NORMALIZE = 300, 3300

# Capillary-pressure / differential-force constants
GAMMA_LIQUID_FCC = 0.6  # N/m, liquid/FCC interfacial energy
DEFAULT_DV = 1e-18      # m³, default local control volume = 1 μm³

GRAIN_SHAPE_FACTORS = {
    "Spherical (k=2)": 2.0,
    "Tetrakaidecahedron (k=3)": 3.0,
    "Equiaxed cubic (k=6)": 6.0
}

# ================= DATA LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        return None, []
    data = {}
    for f in files:
        basename = Path(f).stem
        try:
            T = int(basename.replace("Gibbs_", "").replace("K", ""))
        except ValueError:
            continue
        df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
        data[T] = df
    return data, sorted(data.keys())

data_by_T, temperatures = load_temperature_data(CSV_FILES_DIR)
if not data_by_T:
    st.error(f"❌ No valid CSV files in `{CSV_FILES_DIR}`")
    st.info("💡 Expected format: `Gibbs_1000K.csv`, `Gibbs_1500K.csv`, etc.")
    st.info("💡 Required columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC")
    st.stop()

# ================= CHECK GRID REGULARITY =================
def is_regular_grid(df):
    n_co = df["Co"].nunique()
    n_cr = df["Cr"].nunique()
    n_fe = df["Fe"].nunique()
    expected = n_co * n_cr * n_fe
    return len(df) == expected and expected > 0

grid_regular = {T: is_regular_grid(df) for T, df in data_by_T.items()}
USE_REGULAR = all(grid_regular.values())
if USE_REGULAR:
    st.info("✅ All temperature files are on a regular grid – using fast RegularGridInterpolator.")
else:
    st.warning("⚠️ One or more temperature files are not on a regular grid – using slower LinearNDInterpolator for safety.")

# ================= INTERPOLATOR BUILD =================
@st.cache_resource(ttl=3600)
def build_regular_interpolator(T, phase):
    df = data_by_T[T]
    co_vals = np.sort(df["Co"].unique())
    cr_vals = np.sort(df["Cr"].unique())
    fe_vals = np.sort(df["Fe"].unique())
    df_sorted = df.sort_values(["Co", "Cr", "Fe"])
    values = df_sorted[f"G_{phase}"].values.reshape(len(co_vals), len(cr_vals), len(fe_vals))
    return RegularGridInterpolator(
        (co_vals, cr_vals, fe_vals), values,
        bounds_error=False, fill_value=np.nan
    )

@st.cache_resource(ttl=3600)
def build_linearnd_interpolator(T, phase):
    df = data_by_T[T]
    points = df[["Co", "Cr", "Fe"]].values
    values = df[f"G_{phase}"].values
    return LinearNDInterpolator(points, values, fill_value=np.nan)

if "interpolators" not in st.session_state:
    st.session_state.interpolators = {"LIQ": {}, "FCC": {}}
if "interpolators_built" not in st.session_state:
    st.session_state.interpolators_built = False

def build_all_interpolators():
    progress_bar = st.progress(0, text="Building interpolators...")
    total = len(temperatures) * 2
    status_text = st.empty()
    for i, T in enumerate(temperatures):
        for j, phase in enumerate(["LIQ", "FCC"]):
            status_text.text(f"Building {phase} interpolator for T={T}K...")
            if USE_REGULAR:
                interp = build_regular_interpolator(T, phase)
            else:
                interp = build_linearnd_interpolator(T, phase)
            st.session_state.interpolators[phase][T] = interp
            progress_bar.progress((i * 2 + j + 1) / total)
    time.sleep(0.3)
    st.session_state.interpolators_built = True
    progress_bar.empty()
    status_text.empty()

with st.sidebar:
    st.header("⚡ Performance & Build")
    if st.session_state.interpolators_built:
        st.success("✅ Interpolators ready!")
        if st.button("🔄 Rebuild All Interpolators", type="secondary"):
            st.session_state.interpolators_built = False
            st.session_state.interpolators = {"LIQ": {}, "FCC": {}}
            st.rerun()
    else:
        if st.button("🚀 Build All Interpolators", type="primary", width='content'):
            with st.spinner("Building interpolators..."):
                build_all_interpolators()
            st.success("✅ Interpolators ready!")
            st.rerun()
    st.caption("💡 Build once; subsequent queries will be instant.")
    st.divider()
    st.subheader("📊 Data Summary")
    st.metric("Available Temperatures", len(temperatures))
    if temperatures:
        st.metric("Temperature Range", f"{min(temperatures)}–{max(temperatures)} K")
        st.metric("Compositions per T", len(data_by_T[temperatures[0]]))

# ================= EVALUATION FUNCTION =================
def evaluate_point(x_co, x_cr, x_fe, T):
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
            if USE_REGULAR:
                interp_liq = build_regular_interpolator(T, "LIQ")
                interp_fcc = build_regular_interpolator(T, "FCC")
            else:
                interp_liq = build_linearnd_interpolator(T, "LIQ")
                interp_fcc = build_linearnd_interpolator(T, "FCC")
    if interp_liq is None or interp_fcc is None:
        return None, None
    point = np.array([[x_co, x_cr, x_fe]])
    try:
        g_liq = interp_liq(point)
        g_fcc = interp_fcc(point)
    except Exception:
        return None, None
    if hasattr(g_liq, 'item'):
        g_liq = g_liq.item()
    if hasattr(g_fcc, 'item'):
        g_fcc = g_fcc.item()
    if g_liq is None or g_fcc is None or np.isnan(g_liq) or np.isnan(g_fcc):
        return None, None
    return float(g_liq), float(g_fcc)

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

def compute_curvature_radius(grain_size_m):
    """Local tip curvature radius, approximated as D/4 for growing FCC grains."""
    return grain_size_m / 4.0

def compute_capillary_pressure(gamma, curvature_r):
    """Capillary pressure: P_capillary = 2γ/r [Pa]. Always positive, opposes growth."""
    return (2.0 * gamma) / curvature_r

def compute_net_pressure(P_chem, P_capillary):
    """Net pressure: P_net = P_chem - P_cap [Pa]. Positive drives LIQ→FCC."""
    return P_chem - P_capillary

def compute_differential_force(P_net, Sv, dV):
    """Differential force on local volume element: dF_net = P_net × Sv × dV [N]."""
    return P_net * Sv * dV

# ================= STREAMLIT MARKDOWN THEORY =================
def display_latex_theory():
    st.markdown("## 📚 Thermodynamic Theory Reference")

    with st.expander("📘 View Theory Explanation", expanded=True):
        st.markdown("""
        This section explains how the app converts Gibbs free-energy differences into
        an equivalent interface driving pressure and then into a force acting on an
        interface or on a grain-boundary network.
        """)

        st.markdown("### 1. Gibbs Free Energy")
        st.markdown("""
        For every selected temperature and composition, the app evaluates the Gibbs
        free energy of the LIQUID and FCC phases:
        """)
        st.markdown(r"""
        $$
        G_{\mathrm{phase}}
        =
        G_{\mathrm{phase}}
        \left(
        x_{\mathrm{Co}},
        x_{\mathrm{Cr}},
        x_{\mathrm{Fe}},
        x_{\mathrm{Ni}},
        T
        \right)
        $$
        """)
        st.markdown(r"""
        The mole fractions must satisfy:
        $$
        x_{\mathrm{Co}} + x_{\mathrm{Cr}} + x_{\mathrm{Fe}} + x_{\mathrm{Ni}} = 1
        $$
        """)
        st.info("""
        In this app, Co, Cr, and Fe are user inputs. Ni is calculated automatically as:

        x_Ni = 1 − (x_Co + x_Cr + x_Fe)
        """)

        st.markdown("### 2. Thermodynamic Driving Force")
        st.markdown("""
        The thermodynamic driving force is calculated as the difference between FCC
        and LIQUID Gibbs free energies:
        """)
        st.markdown(r"""
        $$
        \Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQUID}}
        $$
        """)
        st.markdown("""
        | Condition | Thermodynamic meaning | Preferred phase |
        |---|---|---|
        | ΔG < 0 | FCC has lower Gibbs free energy than LIQUID | FCC favored |
        | ΔG > 0 | LIQUID has lower Gibbs free energy than FCC | LIQUID favored |
        | ΔG ≈ 0 | Both phases are close in stability | Near phase boundary |
        """)

        st.markdown("### 3. Molar Volume")
        st.markdown("""
        The Gibbs-energy difference is first obtained in J/mol. To convert it into a
        pressure-like quantity, the code divides it by molar volume.
        """)
        st.markdown(r"""
        $$
        V_m = \sum_i x_i V_m^{(i)}
        $$
        """)
        st.markdown(r"""
        Here, \(V_m^{(i)}\) is the pure-element molar volume. The app also allows a
        constant molar volume if the user does not want to use composition-dependent
        linear mixing.
        """)

        st.markdown("### 4. Chemical Driving Pressure")
        st.markdown("""
        The chemical driving pressure follows the standard thermodynamic convention
        (Porter & Easterling, Christian):
        """)
        st.markdown(r"""
        $$
        P_{\text{chem}} = -\frac{\Delta G}{V_m} = \frac{G_{\text{LIQ}} - G_{\text{FCC}}}{V_m}
        $$
        """)
        st.markdown(r"""
        Unit conversion:
        $$
        \frac{\mathrm{J/mol}}{\mathrm{m^3/mol}}
        =
        \frac{\mathrm{J}}{\mathrm{m^3}}
        =
        \mathrm{Pa}
        =
        \mathrm{N/m^2}
        $$
        """)
        st.success("""
        Positive $P_{\text{chem}}$ drives LIQUID → FCC transformation.
        Negative $P_{\text{chem}}$ drives FCC → LIQUID (remelting).
        """)

        st.markdown("### 5. Interface Force from Direct Area")
        st.markdown("""
        If the interface area is supplied directly, the force is:
        """)
        st.markdown(r"""
        $$
        F = P_{\text{chem}} \cdot A
        $$
        """)
        st.markdown(r"""
        where \(A\) is the selected interface area.
        """)

        st.markdown("### 6. Grain-Size-Derived Interface Area")
        st.markdown("""
        Instead of giving interface area manually, the app can estimate the total
        grain-boundary area using grain size and sample volume.
        """)
        st.markdown(r"""
        $$
        S_v = \frac{k}{d}
        $$
        """)
        st.markdown(r"""
        | Symbol | Meaning |
        |---|---|
        | \(S_v\) | grain-boundary area density |
        | \(k\) | grain-shape factor |
        | \(d\) | average grain size |
        """)
        st.markdown(r"""
        The total interface area is:
        $$
        A_{\mathrm{total}} = S_v V = \frac{kV}{d}
        $$
        """)
        st.markdown(r"""
        where \(V\) is the sample volume.
        """)

        st.markdown("### 7. Net Driving Force on All Grain Boundaries")
        st.markdown("""
        The total thermodynamic driving force over the estimated grain-boundary area is:
        """)
        st.markdown(r"""
        $$
        F_{\mathrm{total}}
        =
        P_{\text{chem}} \cdot A_{\mathrm{total}}
        =
        P_{\text{chem}} \frac{kV}{d}
        $$
        """)
        st.warning("""
        This total-area force is a global reference value. For local interface motion,
        the corrected calculation uses capillary-corrected pressure and the local
        differential force.
        """)

        st.markdown("### 7b. Capillary-Corrected Differential Force")
        st.markdown(r"""
        The local capillary resistance is:
        $$
        P_{\mathrm{capillary}} = \frac{2\gamma}{r}, \qquad r \approx \frac{D}{4}
        $$
        The corrected net pressure is:
        $$
        P_{\mathrm{net}} = P_{\text{chem}} - P_{\mathrm{capillary}}
        $$
        The local differential force used in the multi-ring plot is:
        $$
        dF_{\mathrm{net}} = P_{\mathrm{net}} S_v dV
        $$
        """)

        st.markdown("### 8. Temperature Normalization")
        st.markdown("""
        Some visualizations normalize temperature using:
        """)
        st.markdown(r"""
        $$
        T_{\mathrm{norm}}
        =
        \frac{T - 300}{3300 - 300}
        $$
        """)
        st.markdown("""
        This maps the temperature range 300–3300 K approximately into [0, 1].
        """)

        st.markdown("### 9. Interpolation Strategy")
        st.markdown("""
        The app uses one of two interpolation strategies depending on the composition grid:
        """)
        st.markdown("""
        | Data type | Interpolator used | Meaning |
        |---|---|---|
        | Regular Co–Cr–Fe grid | `RegularGridInterpolator` | Fast interpolation |
        | Irregular/scattered composition points | `LinearNDInterpolator` | Slower fallback method |
        """)

        st.markdown("### 10. Key Assumptions")
        st.markdown("""
        - Molar volume can be approximated either as constant or by linear mixing.
        - The calculation is thermodynamic; interface kinetics are not explicitly modeled.
        - Elastic strain energy is neglected.
        - Interpolation is reliable only inside the composition range covered by the CSV data.
        - Grain size represents an average equivalent FCC grain diameter.
        - Shape factor assumes idealized equiaxed grain geometry.
        """)

        st.markdown("### 11. References")
        st.markdown("""
        1. Porter, D.A., Easterling, K.E. *Phase Transformations in Metals and Alloys*.
        2. Mills, K.C. *International Journal of Thermophysics* **23**, 2002.
        3. Saunders, N., Miodownik, A.P. *CALPHAD: Calculation of Phase Diagrams*.
        4. Underwood, E.E. *Quantitative Stereology*.
        5. SciPy documentation for `RegularGridInterpolator` and `LinearNDInterpolator`.
        """)

display_latex_theory()
st.divider()

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("🎛️ Composition & Temperature")

T = st.sidebar.select_slider(
    "Temperature (K)",
    options=temperatures,
    value=1000 if 1000 in temperatures else temperatures[0]
)

col1, col2, col3 = st.sidebar.columns(3)
x_co = col1.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_cr = col2.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_fe = col3.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.3f")
x_ni = 1.0 - (x_co + x_cr + x_fe)

if x_ni < -1e-6 or x_ni > 1.0 + 1e-6:
    st.sidebar.error(f"⚠️ Invalid: x_Ni = {x_ni:.4f} (must be 0–1)")
    st.sidebar.warning("💡 Adjust Co, Cr, or Fe so that all mole fractions sum to 1.0")
    st.stop()
else:
    st.sidebar.success(f"✅ x_Ni = {x_ni:.4f}")

st.sidebar.markdown("##### 🧪 Current Composition")
st.sidebar.markdown(f"""
| Element | Mole Fraction |
|---------|--------------|
| Co | {x_co:.3f} |
| Cr | {x_cr:.3f} |
| Fe | {x_fe:.3f} |
| Ni | {x_ni:.3f} |
| **Σ** | **{x_co+x_cr+x_fe+x_ni:.3f}** |
""", unsafe_allow_html=True)

st.sidebar.subheader("📐 Molar Volume Model")
vm_model = st.sidebar.radio(
    "Model",
    ["Constant", "Composition‑dependent"],
    index=1,
    help="Composition-dependent uses linear mixing of pure element volumes"
)

if vm_model == "Constant":
    V_m = st.sidebar.number_input(
        "Vₘ (m³/mol)",
        1e-7, 1e-4, DEFAULT_VM, 1e-7,
        format="%.2e",
        help="Constant molar volume for all compositions"
    )
else:
    V_m = composition_dependent_vm(x_co, x_cr, x_fe, x_ni)
    st.sidebar.metric("Calculated Vₘ", f"{V_m:.2e} m³/mol")
    st.sidebar.caption("Based on linear mixing: Vₘ = Σ xᵢ·Vₘ⁽ⁱ⁾")

# ================= INTERFACE AREA / GRAIN SIZE PARAMETERS =================
st.sidebar.divider()
st.sidebar.subheader("🔧 Interface / Grain Boundary Parameters")

area_mode = st.sidebar.radio(
    "Area Calculation Mode",
    ["Direct Input (A)", "Grain Size Derived (Sv x V)"],
    index=1,
    help="Choose how to specify the total interfacial area"
)

# Initialize all grain-size variables with safe defaults to prevent NameError
grain_size_um = None
grain_size_m = None
shape_choice = None
shape_factor = None
sample_volume_cm3 = None
sample_volume_m3 = None
Sv = None
curvature_r = None
P_capillary = None
P_capillary_MPa = None
P_net = None
P_net_MPa = None
dV = DEFAULT_DV
dV_um3 = 1.0
dF_net = None

if area_mode == "Direct Input (A)":
    interface_area = st.sidebar.number_input(
        "Interface Area A (m²)",
        min_value=1e-20,
        max_value=1e2,
        value=1e-8,
        step=1e-10,
        format="%.2e",
        help="Single interface area (nucleation or micro-scale)"
    )
    st.sidebar.caption("Typical: 10⁻¹² m² (nm²) to 10⁻⁶ m² (μm²)")
else:
    st.sidebar.markdown("##### 🌾 FCC Grain Size")
    grain_size_um = st.sidebar.number_input(
        "Average Grain Size d (μm)",
        min_value=0.001,
        max_value=10000.0,
        value=2.5,
        step=0.1,
        format="%.3f",
        help="Average equivalent diameter of FCC grains"
    )
    grain_size_m = grain_size_um * 1e-6

    shape_choice = st.sidebar.selectbox(
        "Grain Shape Factor",
        list(GRAIN_SHAPE_FACTORS.keys()),
        index=1,
        help="k=2: spheres | k=3: tetrakaidecahedrons (metals) | k=6: cubes"
    )
    shape_factor = GRAIN_SHAPE_FACTORS[shape_choice]

    st.sidebar.markdown("##### 📦 Sample Volume")
    sample_volume_cm3 = st.sidebar.number_input(
        "Sample Volume V (cm³)",
        min_value=1e-9,
        max_value=1e6,
        value=1.0,
        step=0.1,
        format="%.3f",
        help="Bulk sample volume for total area calculation"
    )
    sample_volume_m3 = sample_volume_cm3 * 1e-6

    Sv = compute_Sv(grain_size_m, shape_factor)
    interface_area = compute_total_area(Sv, sample_volume_m3)

    st.sidebar.metric("Interfacial Areal Density Sv", f"{Sv:.2e} m²/m³")
    st.sidebar.metric("Total Interface Area A_total", f"{interface_area:.2e} m²")
    st.sidebar.caption(f"Derived: A = (k/d) × V = ({shape_factor}/{grain_size_um}μm) × {sample_volume_cm3}cm³")

# ================= CAPILLARY PRESSURE CONTROLS =================
st.sidebar.divider()
st.sidebar.subheader("🌊 Capillary / Differential Force")

use_capillary = st.sidebar.checkbox(
    "Enable Capillary Correction",
    value=True,
    help="Use P_net = P_chem − 2γ/r before calculating local differential force."
)

if use_capillary:
    gamma = st.sidebar.number_input(
        "Liquid/FCC Interfacial Energy γ (N/m)",
        min_value=0.01,
        max_value=5.0,
        value=GAMMA_LIQUID_FCC,
        step=0.01,
        format="%.2f",
        help="Typical value for incoherent liquid/FCC interface."
    )

    dV_um3 = st.sidebar.number_input(
        "Local Volume Element dV (μm³)",
        min_value=0.001,
        max_value=1000.0,
        value=1.0,
        step=0.1,
        format="%.3f",
        help="Local control volume for dF_net = P_net × Sv × dV."
    )
    dV = dV_um3 * 1e-18
    st.sidebar.caption(f"dV = {dV:.2e} m³")
else:
    gamma = 0.0
    dV = DEFAULT_DV

# ================= RESULTS DISPLAY =================
st.header(f"📊 Results at T = {T} K")
g_liq, g_fcc = evaluate_point(x_co, x_cr, x_fe, T)

if g_liq is None or g_fcc is None:
    st.warning("⚠️ Composition outside convex hull of training data")
    df_sample = data_by_T[T]
    col_warn1, col_warn2 = st.columns(2)
    with col_warn1:
        st.info(f"""
        **Available Composition Ranges at {T}K:**
        - Co: [{df_sample['Co'].min():.2f}, {df_sample['Co'].max():.2f}]
        - Cr: [{df_sample['Cr'].min():.2f}, {df_sample['Cr'].max():.2f}]
        - Fe: [{df_sample['Fe'].min():.2f}, {df_sample['Fe'].max():.2f}]
        """)
    with col_warn2:
        st.info("""
        **Tips:**
        - Try compositions closer to equiatomic (0.25 each)
        - Ensure Σxᵢ = 1.0
        - Check if temperature has sufficient data coverage
        """)
    with st.expander("📋 View Sample Available Data"):
        st.dataframe(df_sample.head(20), width='content')
else:
    # Display Gibbs energies
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("G_LIQUID", f"{g_liq:,.1f} J/mol", help="Gibbs free energy of liquid phase")
    col_b.metric("G_FCC", f"{g_fcc:,.1f} J/mol", help="Gibbs free energy of FCC solid phase")

    delta_G = g_fcc - g_liq
    phase_pref, phase_color, phase_emoji = get_phase_preference(delta_G)

    col_c.metric(
        "ΔG = G_FCC − G_LIQ",
        f"{delta_G:,.1f} J/mol",
        delta=phase_pref,
        delta_color="normal" if delta_G < 0 else "inverse"
    )

    stable_phase = "FCC" if g_fcc < g_liq else "LIQUID"
    st.success(f"🏆 Most stable phase: **{stable_phase}** {phase_emoji}")

    st.divider()

    # Interface driving force section
    st.subheader("⚙️ Interface Driving Force (Mechanical)")

    # ------------------------------------------------------------
    # CORRECTED: Standard thermodynamic sign convention
    # P_chem = -ΔG / V_m  [Pa]
    # Positive P_chem when FCC is favored (ΔG < 0)
    # ------------------------------------------------------------
    P_chem = -delta_G / V_m  # Chemical driving pressure [Pa]
    P_chem_MPa = P_chem / 1e6  # MPa

    # ------------------------------------------------------------
    # Corrected pressure and force calculation
    # ------------------------------------------------------------
    if use_capillary and grain_size_m is not None:
        curvature_r = compute_curvature_radius(grain_size_m)
        P_capillary = compute_capillary_pressure(gamma, curvature_r)
        P_capillary_MPa = P_capillary / 1e6
        P_net = compute_net_pressure(P_chem, P_capillary)
        P_net_MPa = P_net / 1e6
        dF_net = compute_differential_force(P_net, Sv, dV)
        force_for_display = dF_net
        force_label_for_display = "Differential Force dF_net"
    else:
        curvature_r = None
        P_capillary = 0.0
        P_capillary_MPa = 0.0
        P_net = P_chem
        P_net_MPa = P_chem_MPa
        dF_net = None
        force_for_display = P_chem * interface_area
        force_label_for_display = "Direct-Area Force F"

    if use_capillary and grain_size_m is not None:
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric(
            "Chemical Drive P_chem",
            f"{P_chem_MPa:.3f} MPa",
            help="P_chem = -ΔG/Vₘ = (G_LIQ - G_FCC)/Vₘ. Positive when FCC is thermodynamically favored."
        )
        col_p2.metric(
            "Capillary Pressure",
            f"{P_capillary_MPa:.3f} MPa",
            help=f"P_cap = 2γ/r = 2×{gamma:.2f}/{curvature_r:.2e}",
            delta="resists growth",
            delta_color="inverse"
        )
        col_p3.metric(
            "Net Pressure P_net",
            f"{P_net_MPa:.3f} MPa",
            help="P_net = P_chem − P_capillary",
            delta=f"−{P_capillary_MPa:.2f} MPa",
            delta_color="normal"
        )
        # CORRECTED: Positive P_net drives LIQUID → FCC (growth)
        direction = "→ LIQUID → FCC (Growth)" if P_net > 0 else "→ FCC → LIQUID (Remelting)"
        col_p4.metric("Interface motion", direction)

        st.info(f"""
        **Capillary-Corrected Driving Pressure:**
        - Grain size: **{grain_size_um:.3f} μm** → curvature radius r = D/4 = **{curvature_r:.2e} m**
        - Capillary resistance: **{P_capillary_MPa:.3f} MPa** using γ = {gamma:.2f} N/m
        - Net pressure: **{P_net_MPa:.3f} MPa** = {P_chem_MPa:.3f} − {P_capillary_MPa:.3f} MPa
        """)
    else:
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric(
            "Chemical Drive P_chem",
            f"{P_chem_MPa:.3f} MPa",
            help="P_chem = -ΔG/Vₘ = (G_LIQ - G_FCC)/Vₘ"
        )
        col_p2.metric(
            "SI units",
            f"{P_chem:.2e} N/m²",
            help="Equivalent to Pascals (Pa)"
        )
        # CORRECTED: Positive P_chem (or P_net) drives LIQUID → FCC
        direction = "→ LIQUID → FCC (Growth)" if P_net > 0 else "→ FCC → LIQUID (Remelting)"
        col_p3.metric("Interface motion", direction)

    # Force calculation
    st.markdown("### 🔧 Force on Interface")

    if area_mode == "Grain Size Derived (Sv x V)" and grain_size_um is not None:
        total_area_force = P_net * interface_area
        local_area = Sv * dV

        st.markdown(f"""
        **Correct Local Differential Force:** $dF_{{net}} = P_{{net}} \times S_v \times dV$

        | Parameter | Value | Unit |
        |:---|:---|:---|
        | Grain size $d$ | {grain_size_um:.3f} | μm |
        | Shape factor $k$ | {shape_factor:.0f} | — |
        | $S_v = k/d$ | {Sv:.2e} | m²/m³ |
        | Local volume $dV$ | {dV:.2e} | m³ |
        | Local area $dA = S_v dV$ | {local_area:.2e} | m² |
        | Net pressure $P_{{net}}$ | {P_net:.2e} | Pa |
        """)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Net Pressure P_net", f"{P_net_MPa:.3f} MPa")
        col_f2.metric("Local Area dA = Sv×dV", f"{local_area:.2e} m²")
        col_f3.metric("Differential Force dF_net", f"{force_for_display:.3e} N",
                      help="Correct force used for local interface element and sunburst/multi-ring plot")

        with st.expander("Reference only: total-area force", expanded=False):
            st.write({
                "A_total_m2": float(interface_area),
                "F_total_reference_N": float(total_area_force),
                "note": "This is not used for the multi-ring plot. The plot uses dF_net = P_net × Sv × dV."
            })

        st.info(f"""
        **Physical Interpretation:**
        - The plotted force is the **local differential force**, not the total grain-boundary force.
        - dF_net = P_net × Sv × dV = **{force_for_display:.3e} N**.
        - The total area $A_{{total}}$ is still shown as a reference, but it is not used for the plotted force.
        """)
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Interface Area A", f"{interface_area:.2e} m²")
        col_f2.metric("Chemical Drive P_chem", f"{P_chem_MPa:.3f} MPa")
        col_f3.metric("Direct-Area Force F = P_chem × A", f"{force_for_display:.3e} N",
                      help="Direct-area fallback; switch to Grain Size Derived mode for dF_net")

        st.warning("For the corrected differential-force calculation, use `Grain Size Derived (Sv x V)` mode.")

# ================= VISUALIZATION TOOLS =================
st.divider()
st.header("🗺️ Exploration Tools")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 G vs Composition",
    "🌡️ Phase Map vs T",
    "📊 P_chem vs Composition",
    "📋 Raw Data",
    "🌞 Sunburst Hierarchy",
    "🕸️ Radar State"
])

with tab1:
    st.markdown("### Gibbs Energy along composition axis")
    scan_var = st.radio("Vary composition", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan_var1")
    fixed_val = st.slider("Fixed value for other two components", 0.0, 0.4, 0.2, 0.01, key="fixed_scan1")
    max_val = 1.0 - 2 * fixed_val - 0.01
    if max_val < 0.01:
        st.error("❌ Fixed values too large – reduce to allow variation")
    else:
        x_vals = np.linspace(0.01, max_val, 100)
        g_liq_scan, g_fcc_scan = [], []
        valid_x = []
        for xv in x_vals:
            if scan_var == "x_Co":
                gl, gf = evaluate_point(xv, fixed_val, fixed_val, T)
            elif scan_var == "x_Cr":
                gl, gf = evaluate_point(fixed_val, xv, fixed_val, T)
            else:
                gl, gf = evaluate_point(fixed_val, fixed_val, xv, T)
            if gl is not None and gf is not None:
                g_liq_scan.append(gl)
                g_fcc_scan.append(gf)
                valid_x.append(xv)
            else:
                g_liq_scan.append(np.nan)
                g_fcc_scan.append(np.nan)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_x, y=g_liq_scan,
            name="G_LIQUID", line=dict(color="#ff7f0e", width=2),
            mode='lines', fill='tozeroy', fillcolor='rgba(255,127,14,0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=valid_x, y=g_fcc_scan,
            name="G_FCC", line=dict(color="#1f77b4", width=2),
            mode='lines', fill='tozeroy', fillcolor='rgba(31,119,180,0.1)'
        ))
        if scan_var == "x_Co":
            current_x = x_co
        elif scan_var == "x_Cr":
            current_x = x_cr
        else:
            current_x = x_fe
        fig.add_trace(go.Scatter(
            x=[current_x], y=[g_liq if g_liq else np.nan],
            name="Current: LIQUID", mode='markers',
            marker=dict(symbol='circle', size=10, color='#ff7f0e', line=dict(width=2, color='white'))
        ))
        fig.add_trace(go.Scatter(
            x=[current_x], y=[g_fcc if g_fcc else np.nan],
            name="Current: FCC", mode='markers',
            marker=dict(symbol='square', size=10, color='#1f77b4', line=dict(width=2, color='white'))
        ))
        fig.update_layout(
            title=f"Gibbs Energy vs {scan_var} at T={T} K (others={fixed_val:.2f})",
            xaxis_title=scan_var,
            yaxis_title="G (J/mol)",
            height=450,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, width='content')

with tab2:
    st.markdown("### Phase stability vs Temperature (fixed composition)")
    T_scan = temperatures
    delta_G_list, P_chem_mpa_list = [], []
    valid_T = []
    for T_val in T_scan:
        gl, gf = evaluate_point(x_co, x_cr, x_fe, T_val)
        if gl is not None and gf is not None:
            dG = gf - gl
            delta_G_list.append(dG)
            vm_local = composition_dependent_vm(x_co, x_cr, x_fe, x_ni) if vm_model == "Composition‑dependent" else V_m
            # CORRECTED: P_chem = -ΔG / V_m
            P_chem_mpa_list.append(-dG / vm_local / 1e6)
            valid_T.append(T_val)
    if not valid_T:
        st.warning("⚠️ No valid data points for temperature scan at this composition")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=valid_T, y=delta_G_list,
            mode="lines+markers", name="ΔG (J/mol)",
            yaxis="y1", line=dict(color="#2ca02c", width=2),
            marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=valid_T, y=P_chem_mpa_list,
            mode="lines+markers", name="P_chem (MPa)",
            yaxis="y2", line=dict(dash="dot", color="#d62728", width=2),
            marker=dict(symbol='square', size=6)
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="ΔG=0 boundary")
        fig2.update_layout(
            title=f"Driving Force vs Temperature<br><sup>Co:{x_co:.2f} Cr:{x_cr:.2f} Fe:{x_fe:.2f} Ni:{x_ni:.2f}</sup>",
            xaxis_title="Temperature (K)",
            yaxis=dict(title="ΔG (J/mol)", titlefont=dict(color="#2ca02c"), tickfont=dict(color="#2ca02c")),
            yaxis2=dict(title="P_chem (MPa)", titlefont=dict(color="#d62728"),
                       tickfont=dict(color="#d62728"), overlaying="y", side="right"),
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig2, width='content')
        st.markdown("##### 🔍 Interpretation")
        st.markdown("""
        - **Green curve (ΔG)**: Negative values favor FCC formation
        - **Red dotted curve (P_chem)**: Chemical driving pressure in MPa. Positive values drive LIQUID → FCC
        - **Gray dashed line**: Phase boundary (ΔG = 0, equivalently P_chem = 0)
        - Crossing points indicate phase transition temperatures
        """)

with tab3:
    st.markdown("### Chemical Driving Pressure vs Composition (P_chem in MPa)")
    scan_var2 = st.radio("Scan along", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan_pchem")
    fixed_val2 = st.slider("Fixed other components", 0.0, 0.4, 0.2, 0.01, key="fixed_pchem")
    max_val2 = 1.0 - 2 * fixed_val2 - 0.01
    if max_val2 < 0.01:
        st.error("❌ Fixed values too large")
    else:
        x_vals2 = np.linspace(0.01, max_val2, 100)
        P_chem_vals = []
        valid_x2 = []
        for xv in x_vals2:
            if scan_var2 == "x_Co":
                x_co_v, x_cr_v, x_fe_v = xv, fixed_val2, fixed_val2
            elif scan_var2 == "x_Cr":
                x_co_v, x_cr_v, x_fe_v = fixed_val2, xv, fixed_val2
            else:
                x_co_v, x_cr_v, x_fe_v = fixed_val2, fixed_val2, xv
            x_ni_v = 1.0 - (x_co_v + x_cr_v + x_fe_v)
            if x_ni_v < 0 or x_ni_v > 1:
                P_chem_vals.append(np.nan)
                continue
            gl, gf = evaluate_point(x_co_v, x_cr_v, x_fe_v, T)
            if gl is not None and gf is not None:
                vm_local = composition_dependent_vm(x_co_v, x_cr_v, x_fe_v, x_ni_v) if vm_model == "Composition‑dependent" else V_m
                # CORRECTED: P_chem = -(gf - gl) / V_m
                P_chem_mpa = -(gf - gl) / vm_local / 1e6
                P_chem_vals.append(P_chem_mpa)
                valid_x2.append(xv)
            else:
                P_chem_vals.append(np.nan)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=valid_x2, y=P_chem_vals,
            mode="lines", fill="tozeroy",
            line=dict(color="#9467bd", width=2),
            fillcolor='rgba(148,103,189,0.2)',
            name="P_chem"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Phase boundary")
        if scan_var2 == "x_Co":
            current_x2 = x_co
        elif scan_var2 == "x_Cr":
            current_x2 = x_cr
        else:
            current_x2 = x_fe
        current_P_chem = -delta_G / V_m / 1e6 if g_liq is not None else None
        if current_P_chem is not None:
            fig3.add_trace(go.Scatter(
                x=[current_x2], y=[current_P_chem],
                mode='markers', name="Current point",
                marker=dict(symbol='star', size=12, color='#d62728', line=dict(width=2, color='white'))
            ))
        fig3.update_layout(
            title=f"Chemical Driving Pressure P_chem vs {scan_var2} at T={T} K",
            xaxis_title=scan_var2,
            yaxis_title="P_chem (MPa)",
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig3, width='content')
        st.caption("💡 Positive P_chem → LIQUID → FCC (Growth) | Negative P_chem → FCC → LIQUID (Remelting)")

with tab4:
    st.markdown("### 📋 Raw Thermodynamic Data")
    col_filt1, col_filt2, col_filt3 = st.columns(3)
    with col_filt1:
        filter_col = st.selectbox("Filter by column", ["None", "Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
    with col_filt2:
        filter_op = st.selectbox("Operator", ["==", ">=", "<=", ">", "<"]) if filter_col != "None" else None
    with col_filt3:
        filter_val = st.number_input("Value", value=0.25, format="%.3f") if filter_col != "None" else None
    df_display = data_by_T[T].copy()
    if filter_col != "None" and filter_op is not None and filter_val is not None:
        if filter_op == "==":
            df_display = df_display[np.isclose(df_display[filter_col], filter_val, atol=1e-3)]
        elif filter_op == ">=":
            df_display = df_display[df_display[filter_col] >= filter_val]
        elif filter_op == "<=":
            df_display = df_display[df_display[filter_col] <= filter_val]
        elif filter_op == ">":
            df_display = df_display[df_display[filter_col] > filter_val]
        elif filter_op == "<":
            df_display = df_display[df_display[filter_col] < filter_val]
    st.dataframe(
        df_display.style.format({
            "Co": "{:.3f}", "Cr": "{:.3f}", "Fe": "{:.3f}", "Ni": "{:.3f}",
            "G_LIQ": "{:.1f}", "G_FCC": "{:.1f}"
        }),
        width='content',
        height=500
    )
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"CoCrFeNi_data_T{T}K.csv",
            mime="text/csv"
        )
    with col_exp2:
        if st.button("📊 Show Statistics"):
            st.write(df_display[["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]].describe())


with tab5:
    st.markdown("### 🌞 Multi-Ring Thermodynamic Explorer")
    st.info(
        "Inner ring = Temperature, middle ring = Composition (Co/Cr/Fe/Ni stacked to sum to 1), "
        "outer ring = Differential Force dF_net. Temperature is ordered sequentially from low to high. "
        "The visual layout is unchanged; only the force calculation is corrected."
    )

    col_sb1, col_sb2, col_sb3, col_sb4, col_sb5 = st.columns(5)

    with col_sb1:
        sb_area_max = max(1e-2, float(interface_area) * 10.0)
        sb_area_step = max(1e-12, float(interface_area) * 0.01)

        sb_area = st.number_input(
            "Reference Area A_total (m²)",
            min_value=1e-12,
            max_value=sb_area_max,
            value=float(interface_area),
            step=sb_area_step,
            format="%.2e",
            key="tab5_area_input",
            help="Reference only. Correct plotted force uses dF_net = P_net × Sv × dV in grain-size mode."
        )

    with col_sb2:
        temp_cmap = st.selectbox(
            "Temperature colorscale",
            ["thermal", "inferno", "magma", "plasma", "viridis", "Plotly3", "Portland", "Bluered", "cividis", "blackbody", "hot", "turbo", "temps"],
            index=5,
            key="tab5_temp_cmap"
        )

    with col_sb3:
        force_cmap = st.selectbox(
            "Force colorscale",
            ["Portland", "Plotly3", "ice", "haline", "deep", "dense", "teal", "tealgrn", "blues", "blugrn", "pubu", "electric"],
            index=0,
            key="tab5_force_cmap"
        )

    with col_sb4:
        sb_sample = st.slider(
            "Compositions per Temperature",
            3,
            12,
            5,
            1,
            key="tab5_sample_slider"
        )

    with col_sb5:
        sample_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key="tab5_seed_input"
        )

    with st.spinner("Building multi-ring chart..."):

        temperatures_sorted = sorted(temperatures)
        n_temp = len(temperatures_sorted)

        if n_temp == 0:
            st.warning("No temperatures available.")
            st.stop()

        # ============================================================
        # RING GEOMETRY
        # ============================================================
        temp_ring_base = 0.00
        temp_ring_thickness = 0.85

        comp_ring_base = 1.02
        comp_ring_thickness = 1.25

        force_ring_base = 2.45
        force_ring_thickness = 0.90

        temp_width = 360.0 / n_temp

        # ============================================================
        # STYLE SETTINGS
        # ============================================================
        FONT_FAMILY = "Arial Black, Arial, sans-serif"

        TITLE_FONT_SIZE = 30
        SUBTITLE_FONT_SIZE = 18
        LEGEND_FONT_SIZE = 19
        COLORBAR_TITLE_SIZE = 21
        COLORBAR_TICK_SIZE = 17
        HOVER_FONT_SIZE = 18

        RING_BORDER_COLOR = "rgba(15,15,15,0.98)"
        RING_BORDER_WIDTH = 1.9

        COMPOSITION_BORDER_COLOR = "rgba(255,255,255,0.98)"
        COMPOSITION_BORDER_WIDTH = 1.15

        PAPER_BG = "white"
        PLOT_BG = "white"

        # Vivid composition colors
        comp_colors_map = {
            "Co": "#0057B8",   # strong blue
            "Cr": "#D62828",   # strong red
            "Fe": "#2A9D3F",   # strong green
            "Ni": "#F4C430",   # strong yellow/goldd
        }

        # ============================================================
        # DATA CONTAINERS
        # ============================================================
        temp_theta = []
        temp_widths = []
        temp_r = []
        temp_base = []
        temp_colors = []
        temp_custom = []

        element_data = {
            "Co": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
            "Cr": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
            "Fe": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
            "Ni": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
        }

        force_theta = []
        force_widths = []
        force_r = []
        force_base = []
        force_colors = []
        force_custom = []

        rng = np.random.default_rng(int(sample_seed))

        # ============================================================
        # BUILD DATA
        # ============================================================
        for iT, T_sun in enumerate(temperatures_sorted):
            theta_start = iT * temp_width
            theta_center_temp = theta_start + temp_width / 2.0

            # ------------------------------
            # Temperature ring
            # ------------------------------
            temp_theta.append(theta_center_temp)
            temp_widths.append(temp_width)
            temp_r.append(temp_ring_thickness)
            temp_base.append(temp_ring_base)
            temp_colors.append(float(T_sun))
            temp_custom.append([
                T_sun,
                normalize_temperature(T_sun),
                f"{T_sun} K"
            ])

            df_temp = data_by_T[T_sun].copy()

            if df_temp.empty:
                continue

            n_samples = min(int(sb_sample), len(df_temp))

            if n_samples <= 0:
                continue

            # Random but reproducible composition sampling
            if n_samples >= len(df_temp):
                sample_df = df_temp.sample(
                    frac=1.0,
                    random_state=int(sample_seed)
                ).reset_index(drop=True)
            else:
                sample_idx = rng.choice(
                    len(df_temp),
                    size=n_samples,
                    replace=False
                )
                sample_df = df_temp.iloc[sample_idx].copy().reset_index(drop=True)

            # Stable ordering inside each temperature sector
            sample_df = sample_df.sort_values(
                ["Co", "Cr", "Fe", "Ni"]
            ).reset_index(drop=True)

            comp_width = temp_width / n_samples

            for j, row in sample_df.iterrows():
                theta_local_start = theta_start + j * comp_width
                theta_local_center = theta_local_start + comp_width / 2.0

                x_co_s = float(row["Co"])
                x_cr_s = float(row["Cr"])
                x_fe_s = float(row["Fe"])
                x_ni_s = float(row["Ni"])

                g_liq_s = float(row["G_LIQ"])
                g_fcc_s = float(row["G_FCC"])

                delta_G_s = g_fcc_s - g_liq_s

                V_m_s = composition_dependent_vm(
                    x_co_s,
                    x_cr_s,
                    x_fe_s,
                    x_ni_s
                )

                # CORRECTED: P_chem = -ΔG / V_m  [Pa]
                P_chem_pa_s = -delta_G_s / V_m_s
                P_chem_mpa_s = P_chem_pa_s / 1e6

                # Correct force calculation for the outer ring
                # Grain-size mode: dF_net = P_net × Sv × dV
                # Direct-input fallback: F = P_chem × A
                if use_capillary and grain_size_m is not None:
                    chart_curvature_r = compute_curvature_radius(grain_size_m)
                    chart_P_capillary = compute_capillary_pressure(gamma, chart_curvature_r)
                    P_net_s = compute_net_pressure(P_chem_pa_s, chart_P_capillary)
                    P_net_mpa_s = P_net_s / 1e6
                else:
                    chart_P_capillary = 0.0
                    P_net_s = P_chem_pa_s
                    P_net_mpa_s = P_chem_mpa_s

                if area_mode == "Grain Size Derived (Sv x V)" and Sv is not None and dV is not None:
                    net_force_s = compute_differential_force(P_net_s, Sv, dV)
                    force_mode_s = "dF_net = P_net × Sv × dV"
                else:
                    net_force_s = P_net_s * float(sb_area)
                    force_mode_s = "F = P_net × A"

                comp_vals = {
                    "Co": x_co_s,
                    "Cr": x_cr_s,
                    "Fe": x_fe_s,
                    "Ni": x_ni_s,
                }

                cumulative_base = comp_ring_base

                # ------------------------------
                # Composition ring
                # Each composition set sums to 1 radially
                # ------------------------------
                for el in ["Co", "Cr", "Fe", "Ni"]:
                    thickness_el = comp_vals[el] * comp_ring_thickness

                    element_data[el]["theta"].append(theta_local_center)
                    element_data[el]["width"].append(comp_width)
                    element_data[el]["r"].append(thickness_el)
                    element_data[el]["base"].append(cumulative_base)

                    element_data[el]["custom"].append([
                        T_sun,
                        x_co_s,
                        x_cr_s,
                        x_fe_s,
                        x_ni_s,
                        delta_G_s,
                        P_chem_mpa_s,    # index 6
                        net_force_s,
                        el,
                        comp_vals[el],
                        P_net_mpa_s,     # index 10
                        force_mode_s,
                    ])

                    cumulative_base += thickness_el

                # ------------------------------
                # Force ring
                # ------------------------------
                force_theta.append(theta_local_center)
                force_widths.append(comp_width)
                force_r.append(force_ring_thickness)
                force_base.append(force_ring_base)
                force_colors.append(net_force_s)

                force_custom.append([
                    T_sun,
                    x_co_s,
                    x_cr_s,
                    x_fe_s,
                    x_ni_s,
                    delta_G_s,
                    P_chem_mpa_s,    # index 6
                    net_force_s,
                    P_net_mpa_s,     # index 8
                    force_mode_s,
                ])

        # ============================================================
        # FORCE COLOR SCALE LIMITS
        # ============================================================
        if len(force_colors) > 0:
            max_abs_force = max(
                abs(float(np.nanmin(force_colors))),
                abs(float(np.nanmax(force_colors)))
            )

            if max_abs_force == 0:
                max_abs_force = 1.0
        else:
            max_abs_force = 1.0

        # ============================================================
        # BUILD FIGURE
        # ============================================================
        fig = go.Figure()

        # ============================================================
        # INNER TEMPERATURE RING
        # ============================================================
        fig.add_trace(
            go.Barpolar(
                theta=temp_theta,
                width=temp_widths,
                r=temp_r,
                base=temp_base,
                marker=dict(
                    color=temp_colors,
                    colorscale=temp_cmap,
                    cmin=min(temperatures_sorted),
                    cmax=max(temperatures_sorted),
                    colorbar=dict(
                        title=dict(
                            text="<b>Temperature<br>[K]</b>",
                            font=dict(
                                size=COLORBAR_TITLE_SIZE,
                                family=FONT_FAMILY,
                                color="black"
                            )
                        ),
                        tickfont=dict(
                            size=COLORBAR_TICK_SIZE,
                            family=FONT_FAMILY,
                            color="black"
                        ),
                        x=1.06,
                        y=0.72,
                        len=0.56,
                        thickness=28,
                        outlinewidth=2.8,
                        outlinecolor="black",
                        borderwidth=2.2,
                        bordercolor="black",
                        bgcolor="white",
                        tickformat=".0f"
                    ),
                    line=dict(
                        color=RING_BORDER_COLOR,
                        width=RING_BORDER_WIDTH
                    )
                ),
                customdata=temp_custom,
                hovertemplate=(
                    "<b>TEMPERATURE RING</b><br><br>"
                    "<b>T</b> = %{customdata[0]} K<br>"
                    "<b>Normalized T</b> = %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
                name="<b>Temperature</b>",
                showlegend=False,
                opacity=0.98
            )
        )

        # ============================================================
        # MIDDLE COMPOSITION RING
        # ============================================================
        for el in ["Co", "Cr", "Fe", "Ni"]:
            fig.add_trace(
                go.Barpolar(
                    theta=element_data[el]["theta"],
                    width=element_data[el]["width"],
                    r=element_data[el]["r"],
                    base=element_data[el]["base"],
                    marker=dict(
                        color=comp_colors_map[el],
                        line=dict(
                            color=COMPOSITION_BORDER_COLOR,
                            width=COMPOSITION_BORDER_WIDTH
                        )
                    ),
                    customdata=element_data[el]["custom"],
                    hovertemplate=(
                        f"<b>{el} COMPOSITION LAYER</b><br><br>"
                        "T = %{customdata[0]} K<br>"
                        "<b>Current element</b> = %{customdata[8]}<br>"
                        "<b>Current element fraction</b> = %{customdata[9]:.3f}<br><br>"
                        "<b>Co</b> = %{customdata[1]:.3f}<br>"
                        "<b>Cr</b> = %{customdata[2]:.3f}<br>"
                        "<b>Fe</b> = %{customdata[3]:.3f}<br>"
                        "<b>Ni</b> = %{customdata[4]:.3f}<br><br>"
                        "ΔG = %{customdata[5]:.2f} J/mol<br>"
                        "P_chem = %{customdata[6]:.3f} MPa<br>"
                        "P_net = %{customdata[10]:.3f} MPa<br>"
                        "Mode = %{customdata[11]}<br>"
                        "<b>Differential Force</b> = %{customdata[7]:.2e} N"
                        "<extra></extra>"
                    ),
                    name=f"<b>{el}</b>",
                    showlegend=True,
                    opacity=0.98
                )
            )

        # ============================================================
        # OUTER FORCE RING
        # ============================================================
        fig.add_trace(
            go.Barpolar(
                theta=force_theta,
                width=force_widths,
                r=force_r,
                base=force_base,
                marker=dict(
                    color=force_colors,
                    colorscale=force_cmap,
                    cmin=-max_abs_force,
                    cmax=max_abs_force,
                    cmid=0,
                    colorbar=dict(
                        title=dict(
                            text="<b>Differential Force<br>[N]</b>",
                            font=dict(
                                size=COLORBAR_TITLE_SIZE,
                                family=FONT_FAMILY,
                                color="black"
                            )
                        ),
                        tickfont=dict(
                            size=COLORBAR_TICK_SIZE,
                            family=FONT_FAMILY,
                            color="black"
                        ),
                        x=1.23,
                        y=0.72,
                        len=0.56,
                        thickness=28,
                        outlinewidth=2.8,
                        outlinecolor="black",
                        borderwidth=2.2,
                        bordercolor="black",
                        bgcolor="white",
                        tickformat=".1e"
                    ),
                    line=dict(
                        color=RING_BORDER_COLOR,
                        width=RING_BORDER_WIDTH
                    )
                ),
                customdata=force_custom,
                hovertemplate=(
                    "<b>FORCE RING</b><br><br>"
                    "T = %{customdata[0]} K<br>"
                    "<b>Co</b> = %{customdata[1]:.3f}<br>"
                    "<b>Cr</b> = %{customdata[2]:.3f}<br>"
                    "<b>Fe</b> = %{customdata[3]:.3f}<br>"
                    "<b>Ni</b> = %{customdata[4]:.3f}<br><br>"
                    "ΔG = %{customdata[5]:.2f} J/mol<br>"
                    "P_chem = %{customdata[6]:.3f} MPa<br>"
                    "P_net = %{customdata[8]:.3f} MPa<br>"
                    "Mode = %{customdata[9]}<br>"
                    "<b>Differential Force</b> = %{customdata[7]:.2e} N"
                    "<extra></extra>"
                ),
                name="<b>dF_net</b>",
                showlegend=False,
                opacity=0.98
            )
        )

        # ============================================================
        # ANNOTATIONS / LABELS
        # ============================================================
        fig.add_annotation(
            text="<b>INNER RING: TEMPERATURE</b>",
            x=0.02,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=17,
                family=FONT_FAMILY,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5
        )

        fig.add_annotation(
            text="<b>MIDDLE RING: Co / Cr / Fe / Ni COMPOSITION</b>",
            x=0.39,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=17,
                family=FONT_FAMILY,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5
        )

        fig.add_annotation(
            text="<b>OUTER RING: dF_net [N]</b>",
            x=0.79,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=17,
                family=FONT_FAMILY,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5
        )

        # ============================================================
        # BEAUTIFIED LAYOUT
        # ============================================================
        fig.update_layout(
            title=dict(
                text=(
                    "<b>Temperature → Composition → Interface Driving Force</b><br>"
                    f"<span style='font-size:{SUBTITLE_FONT_SIZE}px'>"
                    f"Inner ring: Temperature [K] | Middle ring: Co–Cr–Fe–Ni mole fractions | "
                    f"Outer ring: Differential Force dF_net [N] | dV = {dV:.2e} m³"
                    f"</span>"
                ),
                x=0.5,
                y=0.975,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=TITLE_FONT_SIZE,
                    family=FONT_FAMILY,
                    color="black"
                )
            ),

            template="plotly_white",
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,

            height=1080,
            width=1250,

            margin=dict(
                t=160,
                b=45,
                l=45,
                r=285
            ),

            font=dict(
                family=FONT_FAMILY,
                size=18,
                color="black"
            ),

            legend=dict(
                title=dict(
                    text="<b>Composition</b>",
                    font=dict(
                        size=22,
                        family=FONT_FAMILY,
                        color="black"
                    )
                ),
                font=dict(
                    size=LEGEND_FONT_SIZE,
                    family=FONT_FAMILY,
                    color="black"
                ),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="black",
                borderwidth=2.2,
                x=1.05,
                y=0.24,
                xanchor="left",
                yanchor="middle",
                itemsizing="constant"
            ),

            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font=dict(
                    size=HOVER_FONT_SIZE,
                    family=FONT_FAMILY,
                    color="black"
                )
            ),

            polar=dict(
                bgcolor="white",

                radialaxis=dict(
                    visible=False,
                    range=[0, force_ring_base + force_ring_thickness + 0.30],
                    showline=False,
                    showgrid=False
                ),

                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    showticklabels=False,
                    ticks="",
                    showline=False,
                    gridcolor="rgba(120,120,120,0.22)",
                    gridwidth=1.0
                )
            )
        )

        fig.update_polars(sector=[0, 360])

        st.plotly_chart(fig, width='content')

        # ============================================================
        # HIGH-RES TRANSPARENT PNG DOWNLOAD BUTTON
        # ============================================================
        try:
            # Make exported PNG transparent
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            fig.update_polars(
                bgcolor="rgba(0,0,0,0)"
            )

            png_bytes = fig.to_image(
                format="png",
                width=2400,
                height=2200,
                scale=1
            )

            st.download_button(
                label="📥 Download Transparent High-Resolution PNG",
                data=png_bytes,
                file_name="CoCrFeNi_temperature_composition_force_multiring_transparent.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.warning(
                "PNG export failed. For Plotly 5.24.1, use `kaleido==0.2.1`. "
                "For Kaleido 1.x, upgrade Plotly to >=6.1.1 and install Chrome."
            )
            st.code(str(e))

        with st.expander("Debug summary", expanded=False):
            st.write({
                "temperature_order": temperatures_sorted,
                "n_temperatures": n_temp,
                "temperature_sector_width_deg": temp_width,
                "n_force_segments": len(force_theta),
                "differential_force_min_N": float(np.min(force_colors)) if len(force_colors) > 0 else None,
                "differential_force_max_N": float(np.max(force_colors)) if len(force_colors) > 0 else None,
                "max_abs_differential_force_N": float(max_abs_force),
                "composition_colors": comp_colors_map,
                "ring_geometry": {
                    "temperature": {
                        "base": temp_ring_base,
                        "thickness": temp_ring_thickness
                    },
                    "composition": {
                        "base": comp_ring_base,
                        "thickness": comp_ring_thickness
                    },
                    "force": {
                        "base": force_ring_base,
                        "thickness": force_ring_thickness
                    }
                }
            })



with tab6:
    st.markdown("### 🕸️ Multivariate State Radar Chart")
    st.caption(f"Current: Co:{x_co:.3f} Cr:{x_cr:.3f} Fe:{x_fe:.3f} Ni:{x_ni:.3f} @ {T} K")
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Evaluate a valid composition first to generate radar chart")
        st.info("💡 Adjust composition inputs to fall within data convex hull")
    else:
        T_norm = normalize_temperature(T)
        # Use absolute P_chem for normalized magnitude display
        P_chem_MPa_abs = abs(P_chem_MPa) if 'P_chem_MPa' in locals() else 0.0
        P_chem_norm = min(1.0, P_chem_MPa_abs / 100)
        categories = ['x_Co', 'x_Cr', 'x_Fe', 'x_Ni', 'T (norm)', '|P_chem| (norm)']
        values = [x_co, x_cr, x_fe, x_ni, T_norm, P_chem_norm]
        phase_pref, phase_color, phase_emoji = get_phase_preference(delta_G)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'Current State {phase_emoji}',
            line=dict(color=phase_color, width=3),
            fillcolor=f'rgba({int(phase_color[1:3],16)}, {int(phase_color[3:5],16)}, {int(phase_color[5:7],16)}, 0.25)',
            hovertemplate=
                '<b>Thermodynamic State</b><br>' +
                'Composition: Co:%{r[0]:.3f} Cr:%{r[1]:.3f} Fe:%{r[2]:.3f} Ni:%{r[3]:.3f}<br>' +
                'Temperature: %{r[4]:.3f} (norm) ≈ ' + f'{T} K<br>' +
                '|P_chem|: %{r[5]:.3f} (norm) ≈ {P_chem_MPa_abs:.2f} MPa<br>' +
                f'Interface Force: {abs(force_for_display):.2e} N<extra></extra>'
        ))
        baseline_vals = [0.25, 0.25, 0.25, 0.25, 0.5, 0.3]
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_vals,
            theta=categories,
            fill='none',
            name='Equiatomic Reference',
            line=dict(color='gray', width=1.5, dash='dot'),
            opacity=0.6
        ))
        fig_radar.update_layout(
            title=dict(
                text=f'🕸️ Thermodynamic State Radar<br>' +
                     f'<sup>T={T} K | ΔG={delta_G:.1f} J/mol | {phase_pref}</sup>',
                font=dict(size=14),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=9),
                    gridcolor='lightgray',
                    linecolor='gray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, color='darkgray'),
                    rotation=90,
                    direction='clockwise',
                    gridcolor='lightgray'
                ),
                bgcolor='rgba(240,240,240,0.3)'
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            margin=dict(t=70, l=30, r=30, b=50),
            width=550,
            height=550,
            annotations=[
                dict(
                    text=f"P_net = {P_net_MPa:.1f} MPa<br>dF = {force_for_display:.2e} N",
                    x=0.5, y=-0.22, xref='paper', yref='paper',
                    showarrow=False,
                    font=dict(size=10, color=phase_color),
                    bgcolor=f'rgba({int(phase_color[1:3],16)}, {int(phase_color[3:5],16)}, {int(phase_color[5:7],16)}, 0.1)',
                    borderpad=4,
                    bordercolor=phase_color
                )
            ]
        )
        col_rad1, col_rad2 = st.columns([2, 1])
        with col_rad1:
            st.plotly_chart(fig_radar, width='content')
        with col_rad2:
            st.markdown("##### 📊 Interpretation Guide")
            st.markdown(f"""
            **Axes (all normalized [0,1]):**
            - **x_Co, x_Cr, x_Fe, x_Ni**: Mole fractions
            - **T (norm)**: (T-300)/3000 → 0=300K, 1=3300K
            - **|P_chem| (norm)**: |chemical driving pressure| / 100 MPa
            **Visual Encoding:**
            - {phase_emoji} **Fill color**: Phase preference
              - 🔵 Blue: FCC favored (ΔG < 0)
              - 🟠 Orange: LIQUID favored (ΔG > 0)
            - **Dotted gray**: Equiatomic baseline (0.25 each)
            - **Radial distance**: Value magnitude
            **Current Metrics:**
            - ΔG = {delta_G:.1f} J/mol
            - P_chem = {P_chem_MPa:.1f} MPa
            - Force used: {abs(force_for_display):.2e} N
            """)
            dist_from_equiatomic = np.sqrt(sum((v-0.25)**2 for v in [x_co, x_cr, x_fe, x_ni]))
            st.metric("Distance from equiatomic", f"{dist_from_equiatomic:.3f}")
            st.metric("Driving force magnitude", f"{abs(P_chem_MPa):.1f} MPa")
            if st.button("📥 Download Radar as PNG", key="btn_dl_radar"):
                try:
                    img_bytes = fig_radar.to_image(format='png', width=600, height=600, scale=2)
                    st.download_button(
                        label="Click to Download",
                        data=img_bytes,
                        file_name=f"radar_CoCrFeNi_T{T}K.png",
                        mime="image/png"
                    )
                except:
                    st.error("❌ Requires `kaleido`: `pip install kaleido`")
            if st.button("📋 Copy State Summary"):
                summary = f"""CoCrFeNi State Summary @ {T}K
Composition: Co={x_co:.3f}, Cr={x_cr:.3f}, Fe={x_fe:.3f}, Ni={x_ni:.3f}
G_LIQUID = {g_liq:.1f} J/mol
G_FCC = {g_fcc:.1f} J/mol
ΔG = {delta_G:.1f} J/mol ({phase_pref})
P_chem = {P_chem_MPa:.1f} MPa
Interface Force used = {force_for_display:.2e} N"""
                st.code(summary)
                st.success("✅ Summary copied to code block!")

# ================= FOOTER & REFERENCES =================
st.divider()
st.caption("""
**References & Resources**:
1. Porter, D.A. & Easterling, K.E. *Phase Transformations in Metals and Alloys*, CRC Press (2009)
2. Mills, K.C. *Int. J. Thermophys.* **23**, 2002 (Pure element molar volumes at elevated T)
3. Saunders, N. & Miodownik, A.P. *CALPHAD: Calculation of Phase Diagrams*, Pergamon (1998)
4. SciPy: `RegularGridInterpolator`, `LinearNDInterpolator` documentation
5. Plotly: Interactive visualization library (https://plotly.com)
6. Underwood, E.E. *Quantitative Stereology*, Addison-Wesley (grain shape factors & Sv)

**Data Format**: CSV files with columns [Co, Cr, Fe, Ni, G_LIQ, G_FCC] where Σxᵢ = 1.0  
**Interpolation**: Regular grid (fast) or Delaunay triangulation (fallback)  
**Grain Size Method**: $S_v = k/d$; $A_{{total}} = S_v \times V$; $F_{{total}} = P_{{chem}} \times A_{{total}}$  
**Units**: Energy [J/mol], Volume [m³/mol], Pressure [Pa = N/m²], Force [N], Length [m]
""")

st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
with col_foot1:
    st.caption("🔄 Auto-refreshes when inputs change")
with col_foot2:
    if st.button("♻️ Reset to Defaults"):
        for key in list(st.session_state.keys()):
            if key not in ["interpolators", "interpolators_built"]:
                del st.session_state[key]
        st.rerun()
with col_foot3:
    st.caption(f"📍 Working directory: `{SCRIPT_DIR}`")

if st.sidebar.checkbox("❓ Show Help & Troubleshooting", value=False):
    st.sidebar.markdown("""
    ### 🆘 Quick Help
    **Common Issues:**
    - ❌ "Composition outside convex hull": Try values closer to 0.25 each
    - ❌ "No CSV files found": Place `Gibbs_XXXXK.csv` files in `csv_files/` folder
    - ❌ "Interpolator build slow": Click build once, then queries are instant
    **Performance Tips:**
    - ✅ Use regular composition grids for fastest interpolation
    - ✅ Build interpolators once at startup
    - ✅ Cache is cleared after 1 hour of inactivity
    **Data Requirements:**
    ```
    csv_files/
    ├── Gibbs_800K.csv
    ├── Gibbs_1000K.csv
    ├── Gibbs_1200K.csv
    └── ...
    Each CSV must have columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC
    (mole fractions must sum to 1.0 ± 1e-6)
    ```
    **Grain Size Method:**
    - $S_v = k/d$ where $d$ is grain size [m] and $k$ is shape factor
    - $k=2$: Spherical grains | $k=3$: Tetrakaidecahedron (metals) | $k=6$: Cubic
    - $A_{{total}} = S_v \times V_{{sample}}$
    - $F_{{total}} = P_{{chem}} \times A_{{total}}$
    **Export Options:**
    - 📊 Charts: Hover → Camera icon (Plotly native)
    - 📥 Data: Use "Download CSV" buttons
    - 📄 Theory: Copy LaTeX code or download .tex file
    """)
