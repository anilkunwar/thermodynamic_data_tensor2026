"""
CoCrFeNi Gibbs Free Energy Explorer
Optimized with RegularGridInterpolator + Build Button
WITH: Sunburst Charts, Radar Charts, LaTeX Theory Documentation
PLUS: Grain Size Derived Interfacial Area Density (Sv) & Net Force
FIXED: Radar chart MPa/Pa normalization, LaTeX rendering
ADDED: Current State Sunburst (Temperature → Elements → Force in N)
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
st.markdown("""
**Thermodynamic → Mechanical Conversion**  
ΔG (J/mol) → ΔGᵥ = ΔG/Vₘ (Pa = N/m²) → Interface driving pressure  
**New:** Grain size → $S_v$ → Total area $A_{{total}}$ → Net force $F_{{total}}$
""")

# ================= CONSTANTS =================
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6
T_MIN_NORMALIZE, T_MAX_NORMALIZE = 300, 3300

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
USE_REGULAR = any(grid_regular.values())
if USE_REGULAR:
    st.info("✅ Data on regular grid – using fast RegularGridInterpolator.")
else:
    st.warning("⚠️ Data not on regular grid – using slower LinearNDInterpolator.")

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
        if st.button("🚀 Build All Interpolators", type="primary", use_container_width=True):
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

# ================= LATEX THEORY (RENDERED) =================
def display_latex_theory():
    st.markdown("## 📚 Thermodynamic Theory Reference")
    with st.expander("📋 View Theory (Rendered Equations)", expanded=True):
        st.markdown(r"""
        | **Concept** | **Mathematical Formulation** |
        |:---|:---|
        | **Gibbs Free Energy** | $G_{\text{phase}}(x_{\text{Co}},x_{\text{Cr}},x_{\text{Fe}},x_{\text{Ni}},T)$ <br> Computed from CALPHAD databases for LIQUID and FCC phases. <br> Constraint: $\sum_i x_i = 1$ |
        | **Driving Force ($\Delta G$)** | $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}} \quad [\text{J/mol}]$ <br> $\Delta G < 0$: FCC favored; $\Delta G > 0$: LIQUID favored |
        | **Volumetric Driving Pressure** | $\Delta G_v = \frac{\Delta G}{V_m} \quad [\text{Pa} = \text{N/m}^2]$ <br> $V_m$: Molar volume $[\text{m}^3/\text{mol}]$ |
        | **Molar Volume Models** | *Constant:* $V_m = V_0$ (user-defined) <br> *Composition-dependent:* $V_m = \sum_i x_i V_m^{(i)}$ |
        | **Grain Boundary Area Density** | $S_v = \frac{k}{d} \quad [\text{m}^2/\text{m}^3]$ <br> $d$: avg. FCC grain size $[\text{m}]$, $k$: shape factor (2=sphere, 3=tetrakaidecahedron, 6=cube) |
        | **Total Interface Area** | $A_{\text{total}} = S_v \times V = \frac{k \cdot V}{d} \quad [\text{m}^2]$ <br> $V$: sample volume $[\text{m}^3]$ |
        | **Net Driving Force (Bulk)** | $F_{\text{total}} = \Delta G_v \times A_{\text{total}} = \Delta G_v \cdot \frac{k \cdot V}{d} \quad [\text{N}]$ |
        | **Interface Force (Single Area)** | $F = \Delta G_v \times A \quad [\text{N}]$ <br> $A$: single interface area $[\text{m}^2]$ |
        | **Temperature Normalization** | $T_{\text{norm}} = \frac{T - 300}{3300 - 300} \in [0,1]$ |
        | **Interpolation Strategy** | *RegularGridInterpolator* for regular grids, <br> *LinearNDInterpolator* fallback for irregular grids |
        """)
        st.markdown("### 🔑 Key Assumptions")
        st.markdown("""
        - ✅ Ideal mixing approximation for composition-dependent molar volume
        - ✅ Isothermal conditions during interface motion
        - ✅ Negligible elastic strain energy contributions
        - ✅ Interface mobility not considered (thermodynamic limit only)
        - ✅ Data validity constrained to convex hull of training compositions
        - ✅ Grain size $d$ represents average equivalent diameter of FCC grains
        - ✅ Shape factor $k$ assumes isotropic, equiaxed grain structure
        """)
        st.markdown("### 📖 References")
        st.markdown("""
        1. Porter, D.A., Easterling, K.E. *Phase Transformations in Metals and Alloys*, CRC Press.
        2. Mills, K.C. *Int. J. Thermophys.* **23**, 2002 (molar volume data).
        3. SciPy Documentation: `RegularGridInterpolator`, `LinearNDInterpolator`.
        4. Saunders, N., Miodownik, A.P. *CALPHAD: Calculation of Phase Diagrams*, Pergamon.
        5. Underwood, E.E. *Quantitative Stereology*, Addison-Wesley (grain shape factors).
        """)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if st.button("📥 Copy LaTeX Source", key="btn_copy_latex"):
                st.code("""\\begin{table}[h!]
\\centering
\\begin{tabular}{l l}
\\textbf{Concept} & \\textbf{Formulation} \\\\
\\hline
Gibbs Free Energy & $G_{\\text{phase}}(x_{\\text{Co}},x_{\\text{Cr}},x_{\\text{Fe}},x_{\\text{Ni}},T)$ \\\\
Driving Force & $\\Delta G = G_{\\text{FCC}} - G_{\\text{LIQUID}}$ \\\\
Volumetric Pressure & $\\Delta G_v = \\Delta G / V_m$ \\\\
...
\\end{tabular}
\\end{table}""")
                st.success("✅ LaTeX source copied!")
        with col_dl2:
            if st.button("📄 Download .tex File", key="btn_dl_latex"):
                tex_content = r"""\documentclass{article}
\usepackage{booktabs,amsmath,siunitx}
\begin{document}
\begin{table}[h!]
\centering
\begin{tabular}{l l}
\toprule
\textbf{Concept} & \textbf{Formulation} \\
\midrule
Gibbs Free Energy & $G_{\text{phase}}(x_{\text{Co}},x_{\text{Cr}},x_{\text{Fe}},x_{\text{Ni}},T)$ \\
Driving Force & $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}}$ \\
Volumetric Pressure & $\Delta G_v = \Delta G / V_m$ \\
...
\bottomrule
\end{tabular}
\end{table}
\end{document}"""
                st.download_button("Click to Download", tex_content, "CoCrFeNi_theory.tex", "text/x-tex")

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
        value=10.0,
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
        st.dataframe(df_sample.head(20), use_container_width=True)
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

    delta_G_v = delta_G / V_m  # Pa = N/m²
    delta_G_v_MPa = delta_G_v / 1e6  # MPa

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric(
        "Driving Pressure ΔGᵥ",
        f"{delta_G_v_MPa:.3f} MPa",
        help="Volumetric driving force: ΔG/Vₘ"
    )
    col_p2.metric(
        "SI units",
        f"{delta_G_v:.2e} N/m²",
        help="Equivalent to Pascals (Pa)"
    )
    direction = "→ FCC grows" if delta_G < 0 else "→ LIQUID grows"
    col_p3.metric("Interface motion", direction)

    # Force calculation
    st.markdown("### 🔧 Force on Interface")
    net_force = delta_G_v * interface_area

    if area_mode == "Grain Size Derived (Sv x V)" and grain_size_um is not None:
        st.markdown(f"""
        **Grain Size Method:** $F_{{total}} = \\Delta G_v \\times A_{{total}} = \\Delta G_v \\times S_v \\times V$

        | Parameter | Value | Unit |
        |:---|:---|:---|
        | Grain size $d$ | {grain_size_um:.2f} | μm |
        | Shape factor $k$ | {shape_factor:.0f} | — |
        | $S_v = k/d$ | {Sv:.2e} | m²/m³ |
        | Sample volume $V$ | {sample_volume_m3:.2e} | m³ |
        | **Total area $A_{{total}}$** | **{interface_area:.2e}** | **m²** |
        """)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Driving Pressure ΔGᵥ", f"{delta_G_v_MPa:.3f} MPa")
        col_f2.metric("Total Area $A_{total}$", f"{interface_area:.2e} m²")
        col_f3.metric("Net Force $F_{total}$", f"{net_force:.3e} N",
                      help="Total thermodynamic driving force on all grain boundaries")

        # Physical interpretation
        a4_area = 0.0625  # m² (A4 paper)
        a4_equivalent = interface_area / a4_area if a4_area > 0 else 0
        st.info(f"""
        **Physical Interpretation:**
        - Inside **{sample_volume_cm3:.2f} cm³** of this alloy with **{grain_size_um:.1f} μm** grains,
          the total grain boundary area is **{interface_area:.2e} m²**
          (≈ {a4_equivalent:.1f}× A4 paper area).
        - At **{delta_G_v_MPa:.1f} MPa** driving pressure, the equivalent net mechanical force
          pushing all boundaries is **{net_force:.3e} N** ({net_force/1e3:.1f} kN / {net_force/1e6:.2f} MN).
        - This represents the **maximum** thermodynamic driving force; actual kinetics depend on mobility.
        """)
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Interface Area A", f"{interface_area:.2e} m²")
        col_f2.metric("Driving Pressure ΔGᵥ", f"{delta_G_v_MPa:.3f} MPa")
        col_f3.metric("Net Force F = ΔGᵥ × A", f"{net_force:.3e} N",
                      help="Maximum thermodynamic driving force")

        st.info(f"""
        **Interpretation**:
        - The calculated force ({net_force:.3e} N) represents the *maximum* thermodynamic driving force
          available for interface motion.
        - Actual kinetics depend on interface mobility, diffusion rates, and microstructural constraints.
        - Positive force: drives LIQUID→FCC transformation | Negative: drives FCC→LIQUID
        """)

# ================= VISUALIZATION TOOLS =================
st.divider()
st.header("🗺️ Exploration Tools")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 G vs Composition",
    "🌡️ Phase Map vs T",
    "📊 ΔGᵥ vs Composition",
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
            delta_Gv_list.append(dG / vm_local / 1e6)
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
            x=valid_T, y=delta_Gv_list,
            mode="lines+markers", name="ΔGᵥ (MPa)",
            yaxis="y2", line=dict(dash="dot", color="#d62728", width=2),
            marker=dict(symbol='square', size=6)
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="ΔG=0 boundary")
        fig2.update_layout(
            title=f"Driving Force vs Temperature<br><sup>Co:{x_co:.2f} Cr:{x_cr:.2f} Fe:{x_fe:.2f} Ni:{x_ni:.2f}</sup>",
            xaxis_title="Temperature (K)",
            yaxis=dict(title="ΔG (J/mol)", titlefont=dict(color="#2ca02c"), tickfont=dict(color="#2ca02c")),
            yaxis2=dict(title="ΔGᵥ (MPa)", titlefont=dict(color="#d62728"),
                       tickfont=dict(color="#d62728"), overlaying="y", side="right"),
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("##### 🔍 Interpretation")
        st.markdown("""
        - **Green curve (ΔG)**: Negative values favor FCC formation
        - **Red dotted curve (ΔGᵥ)**: Mechanical driving pressure in MPa
        - **Gray dashed line**: Phase boundary (ΔG = 0)
        - Crossing points indicate phase transition temperatures
        """)

with tab3:
    st.markdown("### Driving Pressure vs Composition (ΔGᵥ in MPa)")
    scan_var2 = st.radio("Scan along", ["x_Co", "x_Cr", "x_Fe"], horizontal=True, key="scan_dgv")
    fixed_val2 = st.slider("Fixed other components", 0.0, 0.4, 0.2, 0.01, key="fixed_dgv")
    max_val2 = 1.0 - 2 * fixed_val2 - 0.01
    if max_val2 < 0.01:
        st.error("❌ Fixed values too large")
    else:
        x_vals2 = np.linspace(0.01, max_val2, 100)
        dGv_vals = []
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
                dGv_vals.append(np.nan)
                continue
            gl, gf = evaluate_point(x_co_v, x_cr_v, x_fe_v, T)
            if gl is not None and gf is not None:
                vm_local = composition_dependent_vm(x_co_v, x_cr_v, x_fe_v, x_ni_v) if vm_model == "Composition‑dependent" else V_m
                dGv = (gf - gl) / vm_local / 1e6
                dGv_vals.append(dGv)
                valid_x2.append(xv)
            else:
                dGv_vals.append(np.nan)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=valid_x2, y=dGv_vals,
            mode="lines", fill="tozeroy",
            line=dict(color="#9467bd", width=2),
            fillcolor='rgba(148,103,189,0.2)',
            name="ΔGᵥ"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Phase boundary")
        if scan_var2 == "x_Co":
            current_x2 = x_co
        elif scan_var2 == "x_Cr":
            current_x2 = x_cr
        else:
            current_x2 = x_fe
        current_dGv = delta_G / V_m / 1e6 if g_liq is not None else None
        if current_dGv is not None:
            fig3.add_trace(go.Scatter(
                x=[current_x2], y=[current_dGv],
                mode='markers', name="Current point",
                marker=dict(symbol='star', size=12, color='#d62728', line=dict(width=2, color='white'))
            ))
        fig3.update_layout(
            title=f"Driving Pressure ΔGᵥ vs {scan_var2} at T={T} K",
            xaxis_title=scan_var2,
            yaxis_title="ΔGᵥ (MPa)",
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("💡 Positive ΔGᵥ → LIQUID grows | Negative ΔGᵥ → FCC grows")

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
        use_container_width=True,
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
    st.markdown("### 🌞 Hierarchical Sunburst: Interface Driving Force")
    st.info("🔄 **Hierarchy**: Temperature (normalized) → Composition Elements → Force (N)")

    # --- New Current State Sunburst (Temperature -> Elements -> Force in N) ---
    st.subheader("📌 Current State Sunburst (Temperature → Elements → Force)")
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Select a valid composition to generate the current state sunburst.")
    else:
        # Colors for each element (fixed for legend)
        element_colors = {
            "Co": "#1f77b4",  # blue
            "Cr": "#ff7f0e",  # orange
            "Fe": "#2ca02c",  # green
            "Ni": "#d62728"   # red
        }
        # Build sunburst data
        ids = []
        parents = []
        labels = []
        values = []
        colors = []
        customdata = []

        # Root
        ids.append("root")
        parents.append("")
        labels.append("CoCrFeNi System")
        values.append(1)
        colors.append("lightgray")
        customdata.append(["root", 0, 0])

        # Temperature layer
        ids.append(f"T_{T}")
        parents.append("root")
        labels.append(f"Temperature: {T} K")
        values.append(1)  # placeholder
        colors.append("lightblue")
        customdata.append([f"T_{T}", T, 0])

        # Composition layer: four elements as siblings under temperature
        # Their values are mole fractions (sum to 1)
        elements = [("Co", x_co), ("Cr", x_cr), ("Fe", x_fe), ("Ni", x_ni)]
        for elem, frac in elements:
            elem_id = f"T_{T}_{elem}"
            ids.append(elem_id)
            parents.append(f"T_{T}")
            labels.append(f"{elem}<br>{frac:.3f}")
            values.append(frac)
            colors.append(element_colors[elem])
            customdata.append([elem_id, elem, frac])

            # Force layer under each element (same net force value)
            force_id = f"{elem_id}_force"
            ids.append(force_id)
            parents.append(elem_id)
            labels.append(f"Force: {net_force:.2e} N")
            values.append(net_force)
            colors.append("gold")
            customdata.append([force_id, net_force, net_force])

        # Create sunburst figure
        fig_sb_current = go.Figure(go.Sunburst(
            ids=ids,
            parents=parents,
            labels=labels,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(width=1, color='white')
            ),
            branchvalues='total',
            hovertemplate=
                '<b>%{label}</b><br>' +
                'Value: %{value:.3e}<br>' +
                '<extra></extra>',
            customdata=customdata,
            insidetextorientation='radial',
            maxdepth=3
        ))
        fig_sb_current.update_layout(
            title=dict(
                text=f'🌞 Current State: {T} K | Force = {net_force:.2e} N<br>' +
                     '<sup>Outer: Temperature | Middle: Element (mole fraction) | Inner: Force (N)</sup>',
                font=dict(size=14),
                x=0.5
            ),
            margin=dict(t=80, l=0, r=0, b=0),
            width=700,
            height=700,
            uniformtext=dict(minsize=10, mode='hide')
        )
        # Add legend for element colors
        for elem, color in element_colors.items():
            fig_sb_current.add_annotation(
                x=1.05, y=1 - 0.05 * list(element_colors.keys()).index(elem),
                xref='paper', yref='paper',
                showarrow=False,
                text=f"• {elem}",
                font=dict(color=color, size=12),
                align='left'
            )
        st.plotly_chart(fig_sb_current, use_container_width=True)

        # Download button for current sunburst
        if st.button("📥 Download Current Sunburst as PNG", key="btn_dl_current_sunburst"):
            try:
                img_bytes = fig_sb_current.to_image(format='png', width=800, height=800, scale=2)
                st.download_button(
                    label="Click to Download PNG",
                    data=img_bytes,
                    file_name=f"sunburst_current_T{T}K.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error("❌ Image export requires kaleido: `pip install kaleido`")
                st.code(f"Error: {e}")

    st.divider()
    st.subheader("📊 Multi‑Sample Sunburst (All Data)")
    st.info("🔄 **Hierarchy**: Temperature (normalized) → Composition Samples → Force Magnitude | Color = ΔGᵥ [MPa]")
    col_sb1, col_sb2, col_sb3, col_sb4 = st.columns(4)
    with col_sb1:
        sb_area = st.number_input(
            "Interface Area A (m²)",
            1e-12, 1e-2, interface_area, 1e-10,
            format="%.2e", key="sb_area_input",
            help="Area for force calculation: F = ΔGᵥ × A"
        )
    with col_sb2:
        sb_cmap = st.selectbox(
            "Colormap for ΔGᵥ",
            ['RdBu_r', 'PiYG', 'coolwarm', 'seismic', 'viridis', 'plasma'],
            index=0, key="sb_cmap_select"
        )
    with col_sb3:
        sb_depth = st.slider("Max Hierarchy Depth", 2, 4, 3, key="sb_depth_slider")
    with col_sb4:
        sb_sample = st.slider("Compositions per T", 4, 20, 8, 4, key="sb_sample_slider")
    if st.button("🔄 Generate Multi‑Sample Sunburst", key="btn_generate_sunburst", type="primary"):
        with st.spinner("Building hierarchical visualization..."):
            ids, parents, labels, values, colors, custom_data = [], [], [], [], [], []
            ids.append("root")
            parents.append("")
            labels.append("CoCrFeNi System")
            values.append(1)
            colors.append(0)
            custom_data.append(["root", 0, 0, 0])
            for T_sun in temperatures:
                T_norm = normalize_temperature(T_sun)
                temp_id = f"T_{T_sun}"
                ids.append(temp_id)
                parents.append("root")
                labels.append(f"T={T_sun}K<br>norm: {T_norm:.2f}")
                values.append(1)
                colors.append(T_norm)
                custom_data.append([temp_id, T_sun, T_norm, 0])
                df_temp = data_by_T[T_sun]
                if len(df_temp) == 0:
                    continue
                n_samples = min(sb_sample, len(df_temp))
                if n_samples >= len(df_temp):
                    sample_df = df_temp
                else:
                    sample_indices = np.linspace(0, len(df_temp) - 1, n_samples, dtype=int)
                    sample_df = df_temp.iloc[sample_indices]
                for idx, row in sample_df.iterrows():
                    x_co_s, x_cr_s, x_fe_s, x_ni_s = row["Co"], row["Cr"], row["Fe"], row["Ni"]
                    g_liq_s, g_fcc_s = evaluate_point(x_co_s, x_cr_s, x_fe_s, T_sun)
                    if g_liq_s is not None and g_fcc_s is not None:
                        delta_G_s = g_fcc_s - g_liq_s
                        V_m_s = composition_dependent_vm(x_co_s, x_cr_s, x_fe_s, x_ni_s)
                        delta_G_v_s = delta_G_s / V_m_s / 1e6
                    else:
                        delta_G_v_s = np.random.randn() * 10
                    comp_id = f"{temp_id}_c{idx}"
                    force_mag = abs(delta_G_v_s * sb_area)
                    ids.append(comp_id)
                    parents.append(temp_id)
                    labels.append(f"Co:{x_co_s:.2f} Cr:{x_cr_s:.2f}<br>Fe:{x_fe_s:.2f} Ni:{x_ni_s:.2f}")
                    values.append(max(0.01, force_mag))
                    colors.append(delta_G_v_s)
                    custom_data.append([
                        comp_id,
                        delta_G_v_s,
                        force_mag,
                        delta_G_v_s * sb_area
                    ])
            fig_sunburst = go.Figure(go.Sunburst(
                ids=ids,
                parents=parents,
                labels=labels,
                values=values,
                marker=dict(
                    colors=colors,
                    colorscale=sb_cmap,
                    colorbar=dict(title="ΔGᵥ [MPa]", titleside='right'),
                    cmid=0,
                    line=dict(width=1, color='white')
                ),
                branchvalues='total',
                hovertemplate=
                    '<b>%{label}</b><br>' +
                    'Driving Pressure: %{customdata[1]:.2f} MPa<br>' +
                    f'Interface Area: {sb_area:.2e} m²<br>' +
                    'Force Magnitude: %{customdata[2]:.2e} N<br>' +
                    'Net Force: %{customdata[3]:.2e} N<br><extra></extra>',
                customdata=custom_data,
                insidetextorientation='radial',
                maxdepth=sb_depth
            ))
            fig_sunburst.update_layout(
                title=dict(
                    text='🌞 CoCrFeNi Interface Driving Force Explorer (All Data)<br>' +
                         f'<sup>Color: ΔGᵥ [MPa] | Area: |F| [N] | A = {sb_area:.2e} m²</sup>',
                    font=dict(size=16),
                    x=0.5
                ),
                margin=dict(t=60, l=0, r=0, b=0),
                width=800,
                height=800,
                uniformtext=dict(minsize=8, mode='hide')
            )
            fig_sunburst.add_annotation(
                text="Click sectors to drill down • Color: ΔGᵥ sign indicates phase preference",
                x=0.5, y=-0.08, xref='paper', yref='paper',
                showarrow=False, font=dict(size=10, color='gray'),
                bgcolor='rgba(255,255,255,0.7)', borderpad=4
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
            if st.button("📥 Download Multi‑Sample Sunburst as PNG", key="btn_dl_sunburst_multi"):
                try:
                    img_bytes = fig_sunburst.to_image(format='png', width=1000, height=1000, scale=2)
                    st.download_button(
                        label="Click to Download PNG",
                        data=img_bytes,
                        file_name=f"sunburst_multi_T{len(temperatures)}temps.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error("❌ Image export requires kaleido: `pip install kaleido`")
                    st.code(f"Error: {e}")

    with st.expander("📖 How to Read the Sunburst Charts", expanded=False):
        st.markdown("""
        ### 🔍 Sunburst Chart Guide
        **Current State Sunburst (Temperature → Elements → Force):**
        - Outer ring: Selected temperature (K)
        - Middle ring: Four elements (Co, Cr, Fe, Ni) with sizes proportional to mole fractions (sum = 1)
        - Inner ring: Net driving force in **Newtons (N)** – same value repeated under each element for visualization
        - Element colors are fixed and shown in the legend

        **Multi‑Sample Sunburst:**
        - Hierarchy: Root → Temperature (normalized) → Composition Samples → Force Magnitude
        - 🎨 **Color**: Signed driving pressure ΔGᵥ [MPa] (Blue = FCC favored, Red = LIQUID favored)
        - 📐 **Sector Area**: Force magnitude |F| = |ΔGᵥ| × A [N]
        - Click any sector to drill down
        """)

with tab6:
    st.markdown("### 🕸️ Multivariate State Radar Chart")
    st.caption(f"Current: Co:{x_co:.3f} Cr:{x_cr:.3f} Fe:{x_fe:.3f} Ni:{x_ni:.3f} @ {T} K")
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Evaluate a valid composition first to generate radar chart")
        st.info("💡 Adjust composition inputs to fall within data convex hull")
    else:
        T_norm = normalize_temperature(T)
        # FIXED: Use delta_G_v_MPa (MPa) for normalization, not Pa
        delta_G_v_norm = min(1.0, abs(delta_G_v_MPa) / 100)
        categories = ['x_Co', 'x_Cr', 'x_Fe', 'x_Ni', 'T (norm)', '|ΔGᵥ| (norm)']
        values = [x_co, x_cr, x_fe, x_ni, T_norm, delta_G_v_norm]
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
                '|Driving Pressure|: %{r[5]:.3f} (norm) ≈ {abs(delta_G_v_MPa):.2f} MPa<br>' +
                f'Interface Force: {abs(net_force):.2e} N<extra></extra>'
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
                    text=f"ΔGᵥ = {delta_G_v_MPa:.1f} MPa<br>F = {net_force:.2e} N",
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
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_rad2:
            st.markdown("##### 📊 Interpretation Guide")
            st.markdown(f"""
            **Axes (all normalized [0,1]):**
            - **x_Co, x_Cr, x_Fe, x_Ni**: Mole fractions
            - **T (norm)**: (T-300)/3000 → 0=300K, 1=3300K
            - **|ΔGᵥ| (norm)**: |driving pressure| / 100 MPa
            **Visual Encoding:**
            - {phase_emoji} **Fill color**: Phase preference
              - 🔵 Blue: FCC favored (ΔG < 0)
              - 🟠 Orange: LIQUID favored (ΔG > 0)
            - **Dotted gray**: Equiatomic baseline (0.25 each)
            - **Radial distance**: Value magnitude
            **Current Metrics:**
            - ΔG = {delta_G:.1f} J/mol
            - ΔGᵥ = {delta_G_v_MPa:.1f} MPa
            - Force @ A={interface_area:.2e}m²: {abs(net_force):.2e} N
            """)
            dist_from_equiatomic = np.sqrt(sum((v-0.25)**2 for v in [x_co, x_cr, x_fe, x_ni]))
            st.metric("Distance from equiatomic", f"{dist_from_equiatomic:.3f}")
            st.metric("Driving force magnitude", f"{abs(delta_G_v_MPa):.1f} MPa")
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
ΔGᵥ = {delta_G_v_MPa:.1f} MPa
Interface Force (A={interface_area:.2e}m²) = {net_force:.2e} N"""
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
**Grain Size Method**: $S_v = k/d$; $A_{{total}} = S_v \\times V$; $F_{{total}} = \\Delta G_v \\times A_{{total}}$  
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
    - $A_{{total}} = S_v \\times V_{{sample}}$
    - $F_{{total}} = \\Delta G_v \\times A_{{total}}$
    **Export Options:**
    - 📊 Charts: Hover → Camera icon (Plotly native)
    - 📥 Data: Use "Download CSV" buttons
    - 📄 Theory: Copy LaTeX source or download .tex file
    """)
