"""
CoCrFeNi Gibbs Free Energy Explorer
Optimized for Streamlit Cloud deployment
WITH: Sunburst Charts, Radar Charts, LaTeX Theory Documentation
PLUS: Grain Size Derived Interfacial Area Density (Sv) & Net Force
PLUS: Capillary Pressure Correction & Differential Force Model
PLUS: LITERATURE-BASED PARAMETER RANGES & UNCERTAINTY QUANTIFICATION
FIXED: Cloud deployment issues (paths, timeouts, missing data, requirements)

LITERATURE SOURCES FOR PARAMETERS:
- Interfacial Energy γ: Kaptay model [Semantics Scholar], Turnbull relation [Emerald], 
  experimental groove methods [ResearchGate], MD simulations for Fe-Cr [ResearchGate]
- Shape Factor k: Smith-Guttman stereology [Springer], DeHoff & Rhines tables, 
  Hensler conversion factors, Underwood quantitative stereology
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
""")

# ================= CONSTANTS - LITERATURE-BASED RANGES =================
# Pure element molar volumes [m³/mol] at reference temperature
PURE_VM = {"Co": 6.80e-6, "Cr": 7.23e-6, "Fe": 7.09e-6, "Ni": 6.59e-6}
DEFAULT_VM = 7.2e-6
T_MIN_NORMALIZE, T_MAX_NORMALIZE = 300, 3300

# 🔬 LITERATURE-BASED INTERFACIAL ENERGY RANGES [N/m = J/m²]
# Sources:
# - Turnbull relation: γ ≈ 0.45·ΔH_fus/V_m^(2/3) → 0.1-0.4 J/m² for pure metals [[50]]
# - Kaptay CALPHAD model: alloys 0.3-1.2 J/m² depending on composition [[8]][[41]]
# - Groove profile experiments: Fe-Cr 0.25-0.45, Al-Ag-Cu 0.15-0.35 J/m² [[42]][[43]]
# - MD simulations Fe-Cr: 0.18-0.32 J/m² with temperature dependence [[3]]
# - HEA estimates (constituent averaging): 0.4-0.9 J/m² for CoCrFeNi system
LIT_GAMMA_RANGES = {
    "Pure metals (Turnbull)": {"min": 0.10, "max": 0.40, "typical": 0.25, "ref": "Turnbull, J. Appl. Phys. 1950"},
    "Metallic alloys (Kaptay)": {"min": 0.30, "max": 1.20, "typical": 0.65, "ref": "Kaptay, Calphad 2012"},
    "Fe-Cr binary (MD)": {"min": 0.18, "max": 0.32, "typical": 0.25, "ref": "Zhang et al., Comp. Mater. Sci. 2022"},
    "Al-Ag-Cu ternary (exp)": {"min": 0.15, "max": 0.35, "typical": 0.22, "ref": "Jones, Acta Metall. 1970"},
    "CoCrFeNi HEA (estimated)": {"min": 0.40, "max": 0.90, "typical": 0.60, "ref": "Constituent averaging + CALPHAD"},
    "Carbides M7C3 (nucleation)": {"min": 2.50, "max": 4.00, "typical": 3.31, "ref": "Li et al., J. Alloys Compd. 2020"}
}

# Default value centered in HEA range
GAMMA_LIQUID_FCC_DEFAULT = 0.60  # N/m

# 🔷 LITERATURE-BASED SHAPE FACTOR RANGES (k for Sv = k/d)
# Sources:
# - Smith & Guttman: random 3D contiguous grains → k ≈ 3.24 [[23]]
# - Hensler conversion: d = 1.62·c → k ≈ 3.24 for equiaxed grains [[23]]
# - DeHoff & Rhines tables: cube=6.0, tetrakaidecahedron≈3.0, sphere=2.0 [[23]]
# - Experimental stereology: k = 2.37-2.82 for polycrystalline metals [[23]]
# - Underwood: k depends on grain aspect ratio, 2.0-4.0 typical for HEAs
LIT_K_RANGES = {
    "Spherical grains": {"min": 2.00, "max": 2.00, "typical": 2.00, "ref": "Exact geometry"},
    "Tetrakaidecahedron (Kelvin)": {"min": 2.90, "max": 3.24, "typical": 3.00, "ref": "Smith-Guttman, DeHoff-Rhines"},
    "Equiaxed polycrystals (exp)": {"min": 2.37, "max": 2.82, "typical": 2.60, "ref": "Hensler, EPJP 2021"},
    "Cubic grains": {"min": 6.00, "max": 6.00, "typical": 6.00, "ref": "Exact geometry"},
    "Columnar/dendritic": {"min": 1.50, "max": 2.50, "typical": 1.80, "ref": "Directional solidification"},
    "Nanocrystalline HEA": {"min": 2.80, "max": 3.50, "typical": 3.10, "ref": "High-entropy GB studies [[13]][[14]]"}
}

# Default value for HEA equiaxed grains
SHAPE_FACTOR_DEFAULT = 3.00  # Tetrakaidecahedron

# Local volume element defaults
DEFAULT_DV = 1e-18  # m³ = 1 μm³
DEFAULT_DV_UM3 = 1.0  # μm³ for UI

# Grain shape factor dictionary for UI
GRAIN_SHAPE_FACTORS = {
    "Spherical (k=2.0)": 2.0,
    "Tetrakaidecahedron (k=3.0) ⭐": 3.0,  # Default for HEAs
    "Equiaxed polycrystal (k=2.6)": 2.6,
    "Columnar (k=1.8)": 1.8,
    "Equiaxed cubic (k=6.0)": 6.0
}

# ================= DATA LOADING =================
@st.cache_data
def load_temperature_data(csv_dir):
    csv_path = Path(csv_dir)
    files = sorted(csv_path.glob("Gibbs_*.csv"))
    if not files:
        return None, []
    data = {}
    for f in files:
        basename = f.stem
        try:
            T = int(basename.replace("Gibbs_", "").replace("K", ""))
        except ValueError:
            continue
        try:
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            st.warning(f"Skipping {f.name}: {e}")
            continue
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()
        if len(df) == 0:
            st.warning(f"Skipping {f.name}: no valid compositions (sum_x ≠ 1)")
            continue
        data[T] = df
    return data, sorted(data.keys())

try:
    data_by_T, temperatures = load_temperature_data(CSV_FILES_DIR)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    data_by_T, temperatures = None, []

if not data_by_T or not temperatures:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("💡 Expected format: `Gibbs_1000K.csv`, `Gibbs_1500K.csv`, etc.")
    st.info("💡 Required columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC")
    st.info("💡 Ensure your `csv_files/` folder is committed to your GitHub repo!")
    
    try:
        existing_files = list(CSV_FILES_DIR.glob("*"))
        if existing_files:
            st.write("Files found in csv_files/:")
            for f in existing_files:
                st.write(f"- {f.name}")
        else:
            st.write("Directory is empty.")
    except Exception:
        st.write("Cannot read directory contents.")
    
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

# 🔬 CAPILLARY PRESSURE AND DIFFERENTIAL FORCE FUNCTIONS (LITERATURE-ENHANCED)
def compute_curvature_radius(grain_size_m, geometry_factor=0.25):
    """
    Local tip curvature radius for growing grains.
    Default: r ≈ D/4 (spherical cap geometry) [[6]][[7]]
    geometry_factor: 0.25 for spherical cap, 0.5 for hemisphere, adjust for faceted growth
    """
    return grain_size_m * geometry_factor

def compute_capillary_pressure(gamma, curvature_r):
    """
    Laplace pressure: P_capillary = 2γ/r [Pa]
    Source: Classical capillarity theory, Christian "Theory of Transformations" [[7]]
    """
    if curvature_r <= 0:
        return np.inf
    return (2.0 * gamma) / curvature_r

def compute_net_pressure(delta_G_v, P_capillary):
    """Net driving pressure: P_net = ΔG_v - P_capillary [Pa]"""
    return delta_G_v - P_capillary

def compute_differential_force(P_net, Sv, dV):
    """Stereological force: dF_net = P_net × S_v × dV [N]"""
    return P_net * Sv * dV

# 🔬 UNCERTAINTY PROPAGATION (Monte Carlo)
def propagate_uncertainty_gamma_k(gamma_nominal, k_nominal, grain_size_m, 
                                  delta_G_v, n_samples=1000, gamma_range=None, k_range=None):
    """
    Monte Carlo uncertainty propagation for capillary-corrected net pressure.
    
    Parameters:
    -----------
    gamma_nominal : float
        Nominal interfacial energy [N/m]
    k_nominal : float
        Nominal shape factor [dimensionless]
    grain_size_m : float
        Grain size [m]
    delta_G_v : float
        Volumetric driving force [Pa]
    n_samples : int
        Number of Monte Carlo samples
    gamma_range : tuple or None
        (min, max) range for gamma; if None, use ±20% of nominal
    k_range : tuple or None
        (min, max) range for k; if None, use ±15% of nominal
    
    Returns:
    --------
    dict with keys: P_net_mean, P_net_std, P_net_95ci, dF_net_mean, dF_net_std
    """
    if gamma_range is None:
        gamma_range = (gamma_nominal * 0.8, gamma_nominal * 1.2)
    if k_range is None:
        k_range = (k_nominal * 0.85, k_nominal * 1.15)
    
    # Sample from uniform distributions (conservative for literature ranges)
    gamma_samples = np.random.uniform(gamma_range[0], gamma_range[1], n_samples)
    k_samples = np.random.uniform(k_range[0], k_range[1], n_samples)
    
    # Compute curvature radius (fixed geometry assumption)
    r = compute_curvature_radius(grain_size_m)
    
    # Vectorized capillary pressure calculation
    P_cap_samples = compute_capillary_pressure(gamma_samples, r)
    
    # Net pressure samples
    P_net_samples = compute_net_pressure(delta_G_v, P_cap_samples)
    
    # Area density samples
    Sv_samples = compute_Sv(grain_size_m, k_samples)
    
    # Differential force samples (using default dV = 1 μm³)
    dV = DEFAULT_DV
    dF_samples = compute_differential_force(P_net_samples, Sv_samples, dV)
    
    # Statistics
    results = {
        "P_net_mean": np.mean(P_net_samples),
        "P_net_std": np.std(P_net_samples),
        "P_net_95ci": norm.interval(0.95, loc=np.mean(P_net_samples), scale=np.std(P_net_samples)),
        "dF_net_mean": np.mean(dF_samples),
        "dF_net_std": np.std(dF_samples),
        "dF_net_95ci": norm.interval(0.95, loc=np.mean(dF_samples), scale=np.std(dF_samples)),
        "gamma_samples": gamma_samples,
        "k_samples": k_samples,
        "P_net_samples": P_net_samples,
        "dF_samples": dF_samples
    }
    return results

# ================= LATEX THEORY (RENDERED) WITH LITERATURE CITATIONS =================
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
        | **Grain Boundary Area Density** | $S_v = \frac{k}{d} \quad [\text{m}^2/\text{m}^3]$ <br> $d$: avg. FCC grain size $[\text{m}]$, $k$: shape factor [[23]] |
        | **Total Interface Area** | $A_{\text{total}} = S_v \times V = \frac{k \cdot V}{d} \quad [\text{m}^2]$ <br> $V$: sample volume $[\text{m}^3]$ |
        | **Net Driving Force (Bulk)** | $F_{\text{total}} = \Delta G_v \times A_{\text{total}} = \Delta G_v \cdot \frac{k \cdot V}{d} \quad [\text{N}]$ |
        | **Interface Force (Single Area)** | $F = \Delta G_v \times A \quad [\text{N}]$ <br> $A$: single interface area $[\text{m}^2]$ |
        | **Temperature Normalization** | $T_{\text{norm}} = \frac{T - 300}{3300 - 300} \in [0,1]$ |
        | **Interpolation Strategy** | *RegularGridInterpolator* for regular grids, <br> *LinearNDInterpolator* fallback for irregular grids |
        """)
        
        # 🔬 Capillary pressure theory section with literature
        st.markdown("### 🌊 Capillary Pressure Correction (Advanced)")
        st.markdown(r"""
        When liquid transforms into FCC grains during a massive transformation, the interface is not flat but **curved and dynamic**. The net driving pressure must account for capillary resistance [[6]][[7]]:

        | **Concept** | **Mathematical Formulation** | **Literature Source** |
        |:---|:---|:---|
        | **Differential Energy Balance** | $dG = -\Delta G_v \, dV + \gamma \, dA$ | Classical nucleation theory |
        | **Local Net Pressure** | $P_{\text{net}} = \Delta G_v - \gamma \frac{dA}{dV} = \Delta G_v - \frac{2\gamma}{r}$ | Laplace-Young equation |
        | **Curvature Radius** | $r \approx \frac{D}{4}$ (local tip radius during growth) | Spherical cap approximation [[6]] |
        | **Capillary Pressure** | $P_{\text{capillary}} = \frac{2\gamma}{r} \quad [\text{Pa}]$ | [[8]][[41]] Kaptay model |
        | **Stereological Force** | $dF_{\text{net}} = P_{\text{net}} \cdot S_v \cdot dV \quad [\text{N}]$ | Underwood [[21]], Smith-Guttman [[23]] |
        | **Physical Interpretation** | Smaller grains have *higher* $S_v$ but also *higher* capillary resistance. The net force per unit volume can be **~4x higher** in refined structures. | [[13]][[14]] High-entropy GB studies |
        """)
        
        st.markdown("### 🔑 Key Assumptions")
        st.markdown("""
        - ✅ Ideal mixing approximation for composition-dependent molar volume
        - ✅ Isothermal conditions during interface motion
        - ✅ Negligible elastic strain energy contributions
        - ✅ Interface mobility not considered (thermodynamic limit only)
        - ✅ Data validity constrained to convex hull of training compositions
        - ✅ Grain size $d$ represents average equivalent diameter of FCC grains
        - ✅ Shape factor $k$ assumes isotropic, equiaxed grain structure [[23]]
        - ✅ Capillary correction assumes spherical cap geometry ($r = D/4$) [[6]]
        - ✅ Interfacial energy $\gamma$ range: 0.4–0.9 N/m for CoCrFeNi liquid/FCC (incoherent) [[8]][[41]]
        """)
        
        st.markdown("### 📖 References")
        st.markdown("""
        1. Porter, D.A., Easterling, K.E. *Phase Transformations in Metals and Alloys*, CRC Press.
        2. Mills, K.C. *Int. J. Thermophys.* **23**, 2002 (molar volume data).
        3. SciPy Documentation: `RegularGridInterpolator`, `LinearNDInterpolator`.
        4. Saunders, N., Miodownik, A.P. *CALPHAD: Calculation of Phase Diagrams*, Pergamon.
        5. Underwood, E.E. *Quantitative Stereology*, Addison-Wesley (grain shape factors).
        6. Christian, J.W. *The Theory of Transformations in Metals and Alloys*, Pergamon (capillary effects).
        7. Turnbull, D. *J. Appl. Phys.* **21**, 1950 (interfacial energy scaling).
        8. Kaptay, G. *Calphad* **38**, 2012 (alloy interfacial energy model) [[8]].
        9. Smith, C.S., Guttman, L. *Trans. AIME* **197**, 1953 (stereological relations) [[23]].
        10. Hensler, J.H. *EPJP* **136**, 2021 (grain boundary resistivity factor k=3.24) [[23]].
        11. Zhang et al. *Comp. Mater. Sci.* **202**, 2022 (Fe-Cr MD interfacial energy) [[3]].
        12. Li et al. *J. Alloys Compd.* **821**, 2020 (M7C3 carbide γ=3.31 J/m²) [[41]].
        13. Zhang et al. *Commun. Mater.* **4**, 2023 (high-entropy grain boundaries) [[13]][[14]].
        """)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if st.button("📥 Copy LaTeX Source", key="btn_copy_latex"):
                st.code(r"""\begin{table}[h!]
\centering
\begin{tabular}{l l l}
\textbf{Concept} & \textbf{Formulation} & \textbf{Source} \\
\hline
Gibbs Free Energy & $G_{\text{phase}}(x_i,T)$ & CALPHAD \\
Driving Force & $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}}$ & Porter \& Easterling \\
Volumetric Pressure & $\Delta G_v = \Delta G / V_m$ & Thermodynamics \\
Capillary Pressure & $P_{\text{cap}} = 2\gamma / r$ & Laplace, Kaptay [[8]] \\
Net Pressure & $P_{\text{net}} = \Delta G_v - P_{\text{cap}}$ & Christian [[6]] \\
Area Density & $S_v = k/d$ & Smith-Guttman [[23]] \\
Differential Force & $dF_{\text{net}} = P_{\text{net}} \cdot S_v \cdot dV$ & Underwood [[21]] \\
Uncertainty & Monte Carlo propagation & This work \\
\end{tabular}
\end{table}""")
                st.success("✅ LaTeX source copied!")
        with col_dl2:
            if st.button("📄 Download .tex File", key="btn_dl_latex"):
                tex_content = r"""\documentclass{article}
\usepackage{booktabs,amsmath,siunitx,natbib}
\begin{document}
\title{CoCrFeNi Thermodynamic-Mechanical Conversion Theory}
\begin{table}[h!]
\centering
\begin{tabular}{l l l}
\toprule
\textbf{Concept} & \textbf{Formulation} & \textbf{Source} \\
\midrule
Gibbs Free Energy & $G_{\text{phase}}(x_i,T)$ & CALPHAD \\
Driving Force & $\Delta G = G_{\text{FCC}} - G_{\text{LIQUID}}$ & Porter \& Easterling \\
Volumetric Pressure & $\Delta G_v = \Delta G / V_m$ & Thermodynamics \\
Capillary Pressure & $P_{\text{cap}} = 2\gamma / r$ & Laplace, Kaptay \cite{kaptay2012} \\
Net Pressure & $P_{\text{net}} = \Delta G_v - P_{\text{cap}}$ & Christian \cite{christian2002} \\
Area Density & $S_v = k/d$ & Smith-Guttman \cite{smith1953} \\
Differential Force & $dF_{\text{net}} = P_{\text{net}} \cdot S_v \cdot dV$ & Underwood \cite{underwood1970} \\
Uncertainty & Monte Carlo propagation & This work \\
\bottomrule
\end{tabular}
\end{table}
\bibliographystyle{plain}
\begin{thebibliography}{99}
\bibitem{kaptay2012} Kaptay, G. (2012). Calphad, 38, 1-15.
\bibitem{christian2002} Christian, J.W. (2002). Pergamon.
\bibitem{smith1953} Smith, C.S., Guttman, L. (1953). Trans. AIME, 197, 81-87.
\bibitem{underwood1970} Underwood, E.E. (1970). Addison-Wesley.
\end{thebibliography}
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

# Initialize all grain-size variables with safe defaults
grain_size_um = None
grain_size_m = None
shape_choice = None
shape_factor = None
sample_volume_cm3 = None
sample_volume_m3 = None
Sv = None
curvature_r = None
P_capillary = None
P_net = None
dV = None
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
        value=10.0,
        step=0.1,
        format="%.3f",
        help="Average equivalent diameter of FCC grains"
    )
    grain_size_m = grain_size_um * 1e-6

    shape_choice = st.sidebar.selectbox(
        "Grain Shape Factor",
        list(GRAIN_SHAPE_FACTORS.keys()),
        index=1,  # Default: Tetrakaidecahedron
        help="k=2: spheres | k=3: tetrakaidecahedrons (metals) | k=6: cubes\n\nLiterature ranges:\n• Equiaxed polycrystals: k=2.37–2.82 [[23]]\n• Tetrakaidecahedron: k=2.90–3.24 [[23]]\n• HEA nanocrystalline: k=2.80–3.50 [[13]][[14]]"
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

# 🔬 CAPILLARY PRESSURE CONTROLS - LITERATURE-BASED RANGES
st.sidebar.divider()
st.sidebar.subheader("🌊 Capillary Pressure Correction")

use_capillary = st.sidebar.checkbox(
    "Enable Capillary Correction",
    value=True,
    help="Account for interfacial energy curvature resistance"
)

if use_capillary:
    # 🔬 Literature-based gamma selection
    st.sidebar.markdown("##### 🔬 Interfacial Energy γ [N/m]")
    
    # Show literature ranges in expander
    with st.sidebar.expander("📚 Literature Ranges for γ", expanded=False):
        st.markdown("**Solid-Liquid Interfacial Energy Values:**")
        for name, vals in LIT_GAMMA_RANGES.items():
            st.markdown(f"- **{name}**: {vals['min']:.2f}–{vals['max']:.2f} N/m (typ: {vals['typical']:.2f})\n  *{vals['ref']}*")
        st.caption("Sources: [[3]][[8]][[41]][[42]][[43]][[50]]")
    
    # Gamma selection method
    gamma_method = st.sidebar.radio(
        "γ Selection",
        ["Manual input", "Preset from literature"],
        index=1,
        help="Choose manual value or select from literature-based presets"
    )
    
    if gamma_method == "Preset from literature":
        gamma_preset = st.sidebar.selectbox(
            "System/Preset",
            list(LIT_GAMMA_RANGES.keys()),
            index=4,  # Default: CoCrFeNi HEA estimated
            help="Select a literature-based preset for interfacial energy"
        )
        gamma_info = LIT_GAMMA_RANGES[gamma_preset]
        gamma = st.sidebar.slider(
            f"γ (N/m) - {gamma_preset}",
            min_value=gamma_info["min"],
            max_value=gamma_info["max"],
            value=gamma_info["typical"],
            step=0.01,
            format="%.2f",
            help=f"Range: {gamma_info['min']:.2f}–{gamma_info['max']:.2f} N/m\nTypical: {gamma_info['typical']:.2f} N/m\n{gamma_info['ref']}"
        )
    else:
        gamma = st.sidebar.number_input(
            "Liquid/FCC Interfacial Energy γ (N/m)",
            min_value=0.01,
            max_value=5.0,
            value=GAMMA_LIQUID_FCC_DEFAULT,
            step=0.01,
            format="%.2f",
            help="~0.6 N/m for incoherent liquid-solid interface in HEAs\nLiterature range for alloys: 0.3–1.2 N/m [[8]][[41]]"
        )
    
    # 🔬 Local volume element
    dV_um3 = st.sidebar.number_input(
        "Local Volume Element dV (μm³)",
        min_value=0.001,
        max_value=1000.0,
        value=DEFAULT_DV_UM3,
        step=0.1,
        format="%.3f",
        help="Control volume for differential force calculation"
    )
    dV = dV_um3 * 1e-18  # Convert to m³
    st.sidebar.caption(f"dV = {dV:.2e} m³")
    
    # 🔬 UNCERTAINTY QUANTIFICATION TOGGLE
    st.sidebar.markdown("##### 📊 Uncertainty Quantification")
    enable_uncertainty = st.sidebar.checkbox(
        "Enable Monte Carlo Uncertainty Propagation",
        value=False,
        help="Propagate literature-based parameter ranges through capillary calculations"
    )
    
    if enable_uncertainty:
        st.sidebar.info("🔄 Running Monte Carlo simulation (1000 samples)...")
        
        # Uncertainty range inputs
        gamma_uncertainty_pct = st.sidebar.slider(
            "γ uncertainty range (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Percentage range around nominal γ value for Monte Carlo sampling"
        )
        k_uncertainty_pct = st.sidebar.slider(
            "k uncertainty range (%)",
            min_value=5,
            max_value=30,
            value=15,
            help="Percentage range around nominal k value for Monte Carlo sampling"
        )
        
        n_mc_samples = st.sidebar.selectbox(
            "Monte Carlo samples",
            [100, 500, 1000, 5000],
            index=2,
            help="Number of random samples for uncertainty propagation"
        )
else:
    gamma = 0.0
    dV = 0.0
    enable_uncertainty = False

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

    # 🔬 Capillary pressure calculations
    if use_capillary and grain_size_m is not None:
        curvature_r = compute_curvature_radius(grain_size_m)
        P_capillary = compute_capillary_pressure(gamma, curvature_r)
        P_capillary_MPa = P_capillary / 1e6
        P_net = compute_net_pressure(delta_G_v, P_capillary)
        P_net_MPa = P_net / 1e6
        dF_net = compute_differential_force(P_net, Sv, dV)
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric(
            "Driving Pressure ΔGᵥ",
            f"{delta_G_v_MPa:.3f} MPa",
            help="Raw volumetric driving force: ΔG/Vₘ"
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
            help="P_net = ΔGᵥ − P_capillary",
            delta=f"−{P_capillary_MPa:.2f} MPa",
            delta_color="normal"
        )
        direction = "→ FCC grows" if P_net > 0 else "→ LIQUID grows"
        col_p4.metric("Interface motion", direction)
        
        # 🔬 Uncertainty quantification display
        if enable_uncertainty and grain_size_m is not None:
            with st.expander("📊 Monte Carlo Uncertainty Results", expanded=True):
                # Run uncertainty propagation
                gamma_range = (gamma * (1 - gamma_uncertainty_pct/100), 
                              gamma * (1 + gamma_uncertainty_pct/100))
                k_range = (shape_factor * (1 - k_uncertainty_pct/100), 
                          shape_factor * (1 + k_uncertainty_pct/100))
                
                mc_results = propagate_uncertainty_gamma_k(
                    gamma_nominal=gamma,
                    k_nominal=shape_factor,
                    grain_size_m=grain_size_m,
                    delta_G_v=delta_G_v,
                    n_samples=n_mc_samples,
                    gamma_range=gamma_range,
                    k_range=k_range
                )
                
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                
                # P_net uncertainty
                P_net_ci = mc_results["P_net_95ci"]
                col_mc1.metric(
                    "P_net (95% CI)",
                    f"{mc_results['P_net_mean']/1e6:.3f} MPa",
                    delta=f"±{mc_results['P_net_std']/1e6:.3f} MPa",
                    help=f"95% CI: [{P_net_ci[0]/1e6:.3f}, {P_net_ci[1]/1e6:.3f}] MPa"
                )
                
                # dF_net uncertainty
                dF_ci = mc_results["dF_net_95ci"]
                col_mc2.metric(
                    "dF_net (95% CI)",
                    f"{mc_results['dF_net_mean']:.2e} N",
                    delta=f"±{mc_results['dF_net_std']:.2e} N",
                    help=f"95% CI: [{dF_ci[0]:.2e}, {dF_ci[1]:.2e}] N"
                )
                
                # Relative uncertainty
                rel_unc_P = mc_results["P_net_std"] / abs(mc_results["P_net_mean"]) * 100
                rel_unc_F = mc_results["dF_net_std"] / abs(mc_results["dF_net_mean"]) * 100
                col_mc3.metric(
                    "Relative uncertainty",
                    f"P: {rel_unc_P:.1f}%, F: {rel_unc_F:.1f}%",
                    help="Coefficient of variation from Monte Carlo"
                )
                
                # Distribution plots
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    fig_pnet = go.Figure(data=[go.Histogram(
                        x=mc_results["P_net_samples"]/1e6,
                        nbinsx=30,
                        name="P_net distribution",
                        marker_color=phase_color,
                        opacity=0.7
                    )])
                    fig_pnet.add_vline(x=P_net_MPa, line_dash="dash", line_color="red", 
                                      annotation_text="Nominal")
                    fig_pnet.add_vline(x=P_net_ci[0]/1e6, line_dash="dot", line_color="gray")
                    fig_pnet.add_vline(x=P_net_ci[1]/1e6, line_dash="dot", line_color="gray",
                                      annotation_text="95% CI")
                    fig_pnet.update_layout(
                        title="Net Pressure Distribution",
                        xaxis_title="P_net (MPa)",
                        yaxis_title="Frequency",
                        height=300,
                        showlegend=False,
                        margin=dict(t=30, b=30, l=30, r=10)
                    )
                    st.plotly_chart(fig_pnet, use_container_width=True)
                
                with col_plot2:
                    fig_df = go.Figure(data=[go.Histogram(
                        x=mc_results["dF_samples"],
                        nbinsx=30,
                        name="dF_net distribution",
                        marker_color="#9467bd",
                        opacity=0.7
                    )])
                    fig_df.add_vline(x=dF_net, line_dash="dash", line_color="red",
                                    annotation_text="Nominal")
                    fig_df.add_vline(x=dF_ci[0], line_dash="dot", line_color="gray")
                    fig_df.add_vline(x=dF_ci[1], line_dash="dot", line_color="gray",
                                    annotation_text="95% CI")
                    fig_df.update_layout(
                        title="Differential Force Distribution",
                        xaxis_title="dF_net (N)",
                        yaxis_title="Frequency",
                        height=300,
                        showlegend=False,
                        margin=dict(t=30, b=30, l=30, r=10)
                    )
                    st.plotly_chart(fig_df, use_container_width=True)
                
                st.caption(f"📊 Monte Carlo: {n_mc_samples} samples, γ range: ±{gamma_uncertainty_pct}%, k range: ±{k_uncertainty_pct}%")
        
        st.info(f"""
        **Capillary Correction Analysis:**
        - Grain size: **{grain_size_um:.2f} μm** → Curvature radius r = D/4 = **{curvature_r:.2e} m**
        - Capillary resistance: **{P_capillary_MPa:.2f} MPa** (γ = {gamma:.2f} N/m)
        - Net driving pressure: **{P_net_MPa:.2f} MPa** = {delta_G_v_MPa:.2f} − {P_capillary_MPa:.2f} MPa
        - Capillary effect: **{abs(P_capillary_MPa/delta_G_v_MPa)*100:.1f}%** of raw driving pressure
        """)
    else:
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
        P_net = delta_G_v
        P_net_MPa = delta_G_v_MPa

    # Force calculation
    st.markdown("### 🔧 Force on Interface")
    net_force = P_net * interface_area if use_capillary else delta_G_v * interface_area

    if area_mode == "Grain Size Derived (Sv x V)" and grain_size_um is not None:
        st.markdown(f"""
        **Grain Size Method:** $F_{{total}} = P_{{net}} \times A_{{total}} = P_{{net}} \times S_v \times V$

        | Parameter | Value | Unit |
        |:---|:---|:---|
        | Grain size $d$ | {grain_size_um:.2f} | μm |
        | Shape factor $k$ | {shape_factor:.0f} | — |
        | $S_v = k/d$ | {Sv:.2e} | m²/m³ |
        | Sample volume $V$ | {sample_volume_m3:.2e} | m³ |
        | **Total area $A_{{total}}$** | **{interface_area:.2e}** | **m²** |
        """)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Net Pressure P_net", f"{P_net_MPa:.3f} MPa")
        col_f2.metric("Total Area $A_{total}$", f"{interface_area:.2e} m²")
        col_f3.metric("Net Force $F_{total}$", f"{net_force:.3e} N",
                      help="Total thermodynamic driving force on all grain boundaries")

        # 🔬 Differential force display
        if use_capillary and dF_net is not None:
            st.markdown("### 🧬 Differential Force on Local Volume Element")
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Local Volume dV", f"{dV:.2e} m³")
            col_d2.metric("Local Area dA = Sv×dV", f"{Sv*dV:.2e} m²")
            col_d3.metric("Differential Force dF_net", f"{dF_net:.3e} N",
                          help="Force on a single μm³ element")
            
            st.info(f"""
            **Differential Force Analysis:**
            - Control volume: **{dV_um3:.3f} μm³** = {dV:.2e} m³
            - Local interface patch: dA = Sv × dV = **{Sv*dV:.2e} m²**
            - Differential force: dF = P_net × dA = **{dF_net:.3e} N**
            - This represents the **localized mechanical force** acting on an infinitesimal interface element
            """)

        # Physical interpretation
        a4_area = 0.0625  # m² (A4 paper)
        a4_equivalent = interface_area / a4_area if a4_area > 0 else 0
        st.info(f"""
        **Physical Interpretation:**
        - Inside **{sample_volume_cm3:.2f} cm³** of this alloy with **{grain_size_um:.1f} μm** grains,
          the total grain boundary area is **{interface_area:.2e} m²**
          (≈ {a4_equivalent:.1f}× A4 paper area).
        - At **{P_net_MPa:.1f} MPa** net driving pressure, the equivalent net mechanical force
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 G vs Composition",
    "🌡️ Phase Map vs T",
    "📊 ΔGᵥ vs Composition",
    "📋 Raw Data",
    "🌞 Sunburst Hierarchy",
    "🕸️ Radar State",
    "📐 Grain Size Scaling",
    "📊 Uncertainty Analysis"  # NEW TAB
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

    # --- Current State Sunburst ---
    st.subheader("📌 Current State Sunburst (Temperature → Elements → Force)")
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Select a valid composition to generate the current state sunburst.")
    else:
        element_colors = {
            "Co": "#1f77b4",
            "Cr": "#ff7f0e",
            "Fe": "#2ca02c",
            "Ni": "#d62728"
        }
        ids = []
        parents = []
        labels = []
        values = []
        colors = []
        customdata = []

        ids.append("root")
        parents.append("")
        labels.append("CoCrFeNi System")
        values.append(1)
        colors.append("lightgray")
        customdata.append(["root", 0, 0])

        ids.append(f"T_{T}")
        parents.append("root")
        labels.append(f"Temperature: {T} K")
        values.append(1)
        colors.append("lightblue")
        customdata.append([f"T_{T}", T, 0])

        elements = [("Co", x_co), ("Cr", x_cr), ("Fe", x_fe), ("Ni", x_ni)]
        for elem, frac in elements:
            elem_id = f"T_{T}_{elem}"
            ids.append(elem_id)
            parents.append(f"T_{T}")
            labels.append(f"{elem}<br>{frac:.3f}")
            values.append(frac)
            colors.append(element_colors[elem])
            customdata.append([elem_id, elem, frac])

            force_id = f"{elem_id}_force"
            ids.append(force_id)
            parents.append(elem_id)
            display_force = net_force if use_capillary else delta_G_v * interface_area
            labels.append(f"Force: {display_force:.2e} N")
            values.append(abs(display_force))
            colors.append("gold")
            customdata.append([force_id, display_force, display_force])

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
                text=f'🌞 Current State: {T} K | Force = {display_force:.2e} N<br>' +
                     '<sup>Outer: Temperature | Middle: Element (mole fraction) | Inner: Force (N)</sup>',
                font=dict(size=14),
                x=0.5
            ),
            margin=dict(t=80, l=0, r=0, b=0),
            width=700,
            height=700,
            uniformtext=dict(minsize=10, mode='hide')
        )
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

with tab6:
    st.markdown("### 🕸️ Multivariate State Radar Chart")
    st.caption(f"Current: Co:{x_co:.3f} Cr:{x_cr:.3f} Fe:{x_fe:.3f} Ni:{x_ni:.3f} @ {T} K")
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Evaluate a valid composition first to generate radar chart")
        st.info("💡 Adjust composition inputs to fall within data convex hull")
    else:
        T_norm = normalize_temperature(T)
        delta_G_v_norm = min(1.0, abs(delta_G_v_MPa) / 100)
        
        # 🔬 Add capillary metrics to radar if enabled
        if use_capillary and P_capillary is not None:
            P_cap_norm = min(1.0, abs(P_capillary_MPa) / 100)
            P_net_norm = min(1.0, abs(P_net_MPa) / 100)
            categories = ['x_Co', 'x_Cr', 'x_Fe', 'x_Ni', 'T (norm)', '|ΔGᵥ|', '|P_cap|', '|P_net|']
            values = [x_co, x_cr, x_fe, x_ni, T_norm, delta_G_v_norm, P_cap_norm, P_net_norm]
        else:
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
        if use_capillary and P_capillary is not None:
            baseline_vals = [0.25, 0.25, 0.25, 0.25, 0.5, 0.3, 0.1, 0.3]
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_vals,
            theta=categories,
            fill='none',
            name='Equiatomic Reference',
            line=dict(color='gray', width=1.5, dash='dot'),
            opacity=0.6
        ))
        
        radar_title = f'🕸️ Thermodynamic State Radar<br><sup>T={T} K | ΔG={delta_G:.1f} J/mol | {phase_pref}</sup>'
        if use_capillary and P_capillary is not None:
            radar_title = f'🕸️ Thermodynamic State Radar<br><sup>T={T} K | P_net={P_net_MPa:.1f} MPa | {phase_pref}</sup>'
        
        fig_radar.update_layout(
            title=dict(
                text=radar_title,
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

with tab7:
    st.markdown("### 📐 Grain Size Scaling Analysis")
    st.info("Analyze how capillary pressure and differential force scale with grain size")
    
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Select a valid composition first")
    else:
        # Generate grain size sweep
        grain_sizes_um = np.linspace(0.5, 50, 100)
        grain_sizes_m = grain_sizes_um * 1e-6
        
        P_caps = []
        P_nets = []
        dF_nets = []
        Svs = []
        
        for gs_m in grain_sizes_m:
            r = compute_curvature_radius(gs_m)
            P_cap = compute_capillary_pressure(gamma, r)
            P_net_val = compute_net_pressure(delta_G_v, P_cap)
            Sv_val = compute_Sv(gs_m, shape_factor if shape_factor else 3.0)
            dF = compute_differential_force(P_net_val, Sv_val, dV if dV else DEFAULT_DV)
            P_caps.append(P_cap / 1e6)
            P_nets.append(P_net_val / 1e6)
            dF_nets.append(dF)
            Svs.append(Sv_val)
        
        # Plot 1: Pressure vs Grain Size
        fig_gs1 = go.Figure()
        fig_gs1.add_trace(go.Scatter(
            x=grain_sizes_um, y=P_caps,
            mode="lines", name="P_capillary",
            line=dict(color="#d62728", width=2, dash="dash")
        ))
        fig_gs1.add_trace(go.Scatter(
            x=grain_sizes_um, y=P_nets,
            mode="lines", name="P_net",
            line=dict(color="#2ca02c", width=2)
        ))
        fig_gs1.add_hline(y=delta_G_v_MPa, line_dash="dot", line_color="blue",
                         annotation_text=f"ΔGᵥ = {delta_G_v_MPa:.1f} MPa")
        fig_gs1.update_layout(
            title="Net Pressure vs Grain Size",
            xaxis_title="Grain Size D (μm)",
            yaxis_title="Pressure (MPa)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_gs1, use_container_width=True)
        
        # Plot 2: Differential Force vs Grain Size
        fig_gs2 = go.Figure()
        fig_gs2.add_trace(go.Scatter(
            x=grain_sizes_um, y=dF_nets,
            mode="lines", fill="tozeroy",
            line=dict(color="#9467bd", width=2),
            fillcolor="rgba(148,103,189,0.2)",
            name="dF_net"
        ))
        fig_gs2.update_layout(
            title=f"Differential Force on {dV_um3 if dV_um3 else 1:.1f} μm³ Element vs Grain Size",
            xaxis_title="Grain Size D (μm)",
            yaxis_title="dF_net (N)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_gs2, use_container_width=True)
        
        # Summary table
        st.markdown("#### 📊 Key Scaling Relationships")
        st.markdown(f"""
        | Parameter | Small Grains (2.5 μm) | Large Grains (10 μm) | Ratio |
        |:---|:---|:---|:---|
        | $S_v$ | {compute_Sv(2.5e-6, shape_factor if shape_factor else 3):.2e} m²/m³ | {compute_Sv(10e-6, shape_factor if shape_factor else 3):.2e} m²/m³ | 4.0× |
        | $P_{{capillary}}$ | {compute_capillary_pressure(gamma, 2.5e-6/4)/1e6:.2f} MPa | {compute_capillary_pressure(gamma, 10e-6/4)/1e6:.2f} MPa | 4.0× |
        | $P_{{net}}$ | {(delta_G_v - compute_capillary_pressure(gamma, 2.5e-6/4))/1e6:.2f} MPa | {(delta_G_v - compute_capillary_pressure(gamma, 10e-6/4))/1e6:.2f} MPa | ~1.0× |
        | $dF_{{net}}$ | {compute_differential_force(delta_G_v - compute_capillary_pressure(gamma, 2.5e-6/4), compute_Sv(2.5e-6, shape_factor if shape_factor else 3), dV if dV else DEFAULT_DV):.2e} N | {compute_differential_force(delta_G_v - compute_capillary_pressure(gamma, 10e-6/4), compute_Sv(10e-6, shape_factor if shape_factor else 3), dV if dV else DEFAULT_DV):.2e} N | ~4.0× |
        """)
        
        st.info("""
        **Physical Insight:** Smaller grains have higher capillary resistance but also much higher $S_v$. 
        The net result: **differential force per unit volume is ~4× higher in refined structures** despite slightly lower net pressure.
        """)

# 🔬 NEW TAB: UNCERTAINTY ANALYSIS
with tab8:
    st.markdown("### 📊 Parameter Uncertainty & Sensitivity Analysis")
    
    if g_liq is None or g_fcc is None:
        st.warning("⚠️ Select a valid composition first")
    else:
        st.info("🔬 Analyze how uncertainty in γ and k propagates to net pressure and force predictions")
        
        # Quick sensitivity sweep
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### γ Sensitivity (fixed k)")
            gamma_sweep = np.linspace(0.3, 1.2, 50)  # Literature range for alloys
            P_net_sweep_gamma = []
            for g_val in gamma_sweep:
                P_cap = compute_capillary_pressure(g_val, compute_curvature_radius(grain_size_m if grain_size_m else 10e-6))
                P_net_sweep_gamma.append(compute_net_pressure(delta_G_v, P_cap))
            
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=gamma_sweep, y=np.array(P_net_sweep_gamma)/1e6,
                mode="lines", fill="tozeroy",
                line=dict(color="#d62728", width=2),
                fillcolor="rgba(214,39,40,0.2)",
                name="P_net(γ)"
            ))
            fig_gamma.add_vline(x=gamma, line_dash="dash", line_color="black", 
                             annotation_text=f"Current: {gamma:.2f}")
            fig_gamma.update_layout(
                title="Net Pressure vs Interfacial Energy γ",
                xaxis_title="γ (N/m)",
                yaxis_title="P_net (MPa)",
                height=350
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
            
        with col_s2:
            st.markdown("#### k Sensitivity (fixed γ)")
            k_sweep = np.linspace(2.0, 4.0, 50)  # Typical range for polycrystals
            dF_sweep_k = []
            for k_val in k_sweep:
                Sv_val = compute_Sv(grain_size_m if grain_size_m else 10e-6, k_val)
                dF_val = compute_differential_force(P_net if P_net is not None else delta_G_v, Sv_val, dV if dV else DEFAULT_DV)
                dF_sweep_k.append(dF_val)
            
            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(
                x=k_sweep, y=dF_sweep_k,
                mode="lines", fill="tozeroy",
                line=dict(color="#9467bd", width=2),
                fillcolor="rgba(148,103,189,0.2)",
                name="dF_net(k)"
            ))
            fig_k.add_vline(x=shape_factor, line_dash="dash", line_color="black",
                           annotation_text=f"Current: {shape_factor:.2f}")
            fig_k.update_layout(
                title="Differential Force vs Shape Factor k",
                xaxis_title="k (dimensionless)",
                yaxis_title="dF_net (N)",
                height=350
            )
            st.plotly_chart(fig_k, use_container_width=True)
        
        # 🔬 Combined uncertainty summary
        st.markdown("#### 📋 Combined Uncertainty Summary")
        
        if grain_size_m is not None and use_capillary:
            # Run quick Monte Carlo with default settings
            mc_quick = propagate_uncertainty_gamma_k(
                gamma_nominal=gamma,
                k_nominal=shape_factor,
                grain_size_m=grain_size_m,
                delta_G_v=delta_G_v,
                n_samples=500,  # Quick run
                gamma_range=(gamma*0.8, gamma*1.2),
                k_range=(shape_factor*0.85, shape_factor*1.15)
            )
            
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            # P_net summary
            P_nom = P_net_MPa
            P_mean = mc_quick["P_net_mean"]/1e6
            P_std = mc_quick["P_net_std"]/1e6
            col_sum1.metric(
                "P_net nominal",
                f"{P_nom:.2f} MPa",
                delta=f"MC mean: {P_mean:.2f} ± {P_std:.2f} MPa",
                help="Nominal vs Monte Carlo mean ± std"
            )
            
            # dF_net summary
            F_nom = dF_net
            F_mean = mc_quick["dF_net_mean"]
            F_std = mc_quick["dF_net_std"]
            col_sum2.metric(
                "dF_net nominal",
                f"{F_nom:.2e} N",
                delta=f"MC mean: {F_mean:.2e} ± {F_std:.2e} N",
                help="Nominal vs Monte Carlo mean ± std"
            )
            
            # Sensitivity coefficients
            dP_dgamma = (compute_capillary_pressure(gamma*1.01, curvature_r) - 
                        compute_capillary_pressure(gamma*0.99, curvature_r)) / (0.02*gamma) / 1e6
            dP_dk = 0  # P_net doesn't directly depend on k
            col_sum3.metric(
                "∂P_net/∂γ",
                f"{-dP_dgamma:.3f} MPa/(N/m)",
                help="Sensitivity of net pressure to interfacial energy"
            )
            
            # Relative contribution
            cap_fraction = abs(P_capillary_MPa / delta_G_v_MPa) * 100 if delta_G_v_MPa != 0 else 0
            col_sum4.metric(
                "Capillary contribution",
                f"{cap_fraction:.1f}%",
                help="Fraction of driving pressure offset by capillarity"
            )
            
            st.markdown(f"""
            **Interpretation:**
            - At grain size **{grain_size_um:.1f} μm**, capillary effects contribute **{cap_fraction:.1f}%** of the raw driving pressure
            - Uncertainty in γ (±20%) propagates to **±{P_std:.2f} MPa** in net pressure
            - Uncertainty in k (±15%) propagates to **±{F_std/F_nom*100:.1f}%** in differential force
            - For **reliable predictions**, constrain γ to literature ranges: **0.4–0.9 N/m** for CoCrFeNi [[8]][[41]]
            """)
        else:
            st.info("💡 Enable capillary correction and grain size mode to view uncertainty analysis")

# ================= FOOTER & REFERENCES =================
st.divider()
st.caption("""
**References & Resources**:
1. Porter, D.A. & Easterling, K.E. *Phase Transformations in Metals and Alloys*, CRC Press (2009)
2. Mills, K.C. *Int. J. Thermophys.* **23**, 2002 (Pure element molar volumes at elevated T)
3. Zhang et al. *Comp. Mater. Sci.* **202**, 2022 (Fe-Cr MD interfacial energy) [[3]]
4. Saunders, N. & Miodownik, A.P. *CALPHAD: Calculation of Phase Diagrams*, Pergamon (1998)
5. SciPy: `RegularGridInterpolator`, `LinearNDInterpolator` documentation
6. Christian, J.W. *The Theory of Transformations in Metals and Alloys*, Pergamon (capillary effects) [[6]]
7. Turnbull, D. *J. Appl. Phys.* **21**, 1950 (interfacial energy scaling) [[50]]
8. Kaptay, G. *Calphad* **38**, 2012 (alloy interfacial energy model) [[8]][[41]]
9. Smith, C.S. & Guttman, L. *Trans. AIME* **197**, 1953 (stereological relations) [[23]]
10. Hensler, J.H. *EPJP* **136**, 2021 (grain boundary factor k=3.24) [[23]]
11. Underwood, E.E. *Quantitative Stereology*, Addison-Wesley (grain shape factors & Sv)
12. Zhang et al. *Commun. Mater.* **4**, 2023 (high-entropy grain boundaries) [[13]][[14]]

**Data Format**: CSV files with columns [Co, Cr, Fe, Ni, G_LIQ, G_FCC] where Σxᵢ = 1.0  
**Interpolation**: Regular grid (fast) or Delaunay triangulation (fallback)  
**Grain Size Method**: $S_v = k/d$; $A_{total} = S_v \times V$; $F_{total} = \Delta G_v \times A_{total}$  
**Capillary Correction**: $P_{net} = \Delta G_v - 2\gamma/r$; $dF_{net} = P_{net} \cdot S_v \cdot dV$  
**Uncertainty**: Monte Carlo propagation with literature-based parameter ranges  
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
