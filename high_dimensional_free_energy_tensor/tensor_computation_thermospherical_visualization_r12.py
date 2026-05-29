"""
================================================================================
Co-Cr-Fe-Ni Phase Stability Explorer v2
Thermodynamic Data Tensor Analysis with Canonical Polyadic Decomposition (CPD)
================================================================================

THERMODYNAMIC DATA TENSOR (TDT) SPECIFICATION:
----------------------------------------------
The code processes CALPHAD-computed Gibbs energy data for the quaternary 
Co-Cr-Fe-Ni alloy system across 31 temperatures (700K → 3300K, ΔT=100K).

TENSOR STRUCTURE:
  G_LIQ[i, j, k, t] = Molar Gibbs energy of LIQUID phase (J/mol)
  G_FCC[i, j, k, t] = Molar Gibbs energy of FCC phase (J/mol)
  
  where:
    i ∈ [0, n_co-1]: Cobalt mole fraction index (x_Co = co_vals[i])
    j ∈ [0, n_cr-1]: Chromium mole fraction index (x_Cr = cr_vals[j])
    k ∈ [0, n_fe-1]: Iron mole fraction index (x_Fe = fe_vals[k])
    t ∈ [0, 30]: Temperature index (T = T_vals[t] ∈ {700, 800, ..., 3300} K)

COMPOSITION CONSTRAINT:
  x_Co + x_Cr + x_Fe + x_Ni = 1.0  →  x_Ni = 1 - (x_Co + x_Cr + x_Fe) ≥ 0
  
  This defines a 3-simplex (tetrahedron) in 4D composition space.
  Only ~16.7% of the full 4D hypercube contains physically valid entries.

REAL DATA CHARACTERISTICS (from Gibbs_*.csv files):
---------------------------------------------------
Temperature Grid: T_vals = [700, 800, 900, ..., 3600, 3300] K (31 points)
Composition Grid: Step ≈ 0.01 in Co/Cr/Fe, truncated by simplex constraint

THERMODYNAMIC REGIMES OBSERVED:

  1. LOW TEMPERATURE (700-1000 K): FCC-DOMINATED
     - |G| ≈ 20-35 kJ/mol (relatively small magnitude)
     - G_FCC < G_LIQ consistently → ΔG = G_LIQ - G_FCC > 0
     - Example at 700K, Co=0.13, Cr=0.4, Fe=0.2, Ni=0.27:
       G_LIQ = -21,274 J/mol, G_FCC = -28,323 J/mol → ΔG = +7,049 J/mol
     - Physical interpretation: Enthalpy dominates; ordered FCC phase favored

  2. TRANSITION REGION (1100-1600 K): PHASE COMPETITION
     - |G| ≈ 80-95 kJ/mol
     - ΔG changes sign depending on composition
     - Example at 1400K, Co=0.26, Cr=0.26, Fe=0.09, Ni=0.39:
       G_LIQ = -146,775 J/mol, G_FCC = -143,964 J/mol → ΔG = -2,811 J/mol (LIQUID)
     - Example at 1400K, Co=0.13, Cr=0.4, Fe=0.2, Ni=0.27:
       G_LIQ = -85,716 J/mol, G_FCC = -88,046 J/mol → ΔG = +2,330 J/mol (FCC)
     - Physical interpretation: Entropic driving force (-T·S) competes with enthalpy

  3. HIGH TEMPERATURE (1700-3300 K): LIQUID-DOMINATED
     - |G| ≈ 140-175 kJ/mol (large negative values from -T·S term)
     - G_LIQ < G_FCC consistently → ΔG < 0
     - Example at 2200K, Co=0.16, Cr=0.27, Fe=0.22, Ni=0.35:
       G_LIQ = -169,985 J/mol, G_FCC = -165,745 J/mol → ΔG = -4,240 J/mol
     - Physical interpretation: Configurational entropy of liquid dominates

TEMPERATURE DEPENDENCE MATHEMATICAL FORM:
-----------------------------------------
For each phase, Gibbs energy follows the CALPHAD polynomial form:

  G^phase(T) = a₀ + a₁·T + a₂·T·ln(T) + a₃·T² + a₄/T + ...

This quasi-polynomial structure enables LOW EFFECTIVE RANK for the 
temperature mode in tensor decomposition:

  Expected singular value decay for Mode-3 (Temperature):
    s_norm = [1.0, 0.08-0.15, 0.01-0.03, <0.005, ...]
    → Effective rank ≈ 3 captures >99% of temperature variation

COMPOSITION DEPENDENCE:
-----------------------
Gibbs energy follows the subregular solution model:

  G^phase(x,T) = Σᵢ xᵢ·Gᵢ^phase(T) + RT·Σᵢ xᵢ·ln(xᵢ) + G^excess(x,T)
  
  G^excess = Σᵢ<ⱼ xᵢ·xⱼ·Σₙ ⁿLᵢⱼ·(xᵢ-xⱼ)ⁿ + Σᵢ<ⱼ<ₖ xᵢ·xⱼ·xₖ·Lᵢⱼₖ + ...

This polynomial structure in composition enables MODERATE EFFECTIVE RANK:

  Expected ranks for composition modes:
    Mode-0 (Co): rank ≈ 5-7 (ferromagnetic contributions, non-ideal mixing)
    Mode-1 (Cr): rank ≈ 5-7 (ordering tendencies, miscibility effects)
    Mode-2 (Fe): rank ≈ 5-7 (magnetic Curie transition ~1043K, entropy)

CANONICAL POLYADIC DECOMPOSITION (CPD) INTERPRETATION:
------------------------------------------------------
The tensor is decomposed as:

  G[i,j,k,t] ≈ Σᵣ₌₁ᴿ λᵣ · A[i,r] · B[j,r] · C[k,r] · D[t,r]

Physical interpretation of components (for R=6):

  r=1: λ₁·A₁·B₁·C₁·D₁ → Baseline enthalpy offset (composition-weighted average)
  r=2: λ₂·A₂·B₂·C₂·D₂ → Linear entropy term (-S·T), D₂(T) ≈ linear in T
  r=3: λ₃·A₃·B₃·C₃·D₃ → Heat capacity curvature + magnetic transitions
  r=4: λ₄·A₄·B₄·C₄·D₄ → Binary interaction effects (Co-Cr, Fe-Ni pairs)
  r=5: λ₅·A₅·B₅·C₅·D₅ → Ternary non-ideal mixing contributions
  r=6: λ₆·A₆·B₆·C₆·D₆ → Fine structure: ordering, short-range effects

CRITICAL EMERGENT FEATURE: Composition-Dependent Transition Temperature
-----------------------------------------------------------------------
For each composition (x_Co, x_Cr, x_Fe), we can extract T* where ΔG=0:

  T*(x_Co, x_Cr, x_Fe) = temperature where G_LIQ = G_FCC

Observed transition temperatures from real data:
  - Ni-rich corner (Co,Cr,Fe ≈ 0.1): T* ≈ 1300-1450 K
  - Cr-rich edge (Cr ≈ 0.4): T* ≈ 1500-1700 K (Cr raises melting point)
  - Equiatomic (0.25 each): T* ≈ 1480-1550 K (consistent with HEA literature)
  - Fe-rich edge: T* shows kink near 1043 K (magnetic Curie transition)

This 3D surface T*(x) is the PRIMARY MATERIALS DESIGN OUTPUT enabled by 
the tensor representation, allowing instant "melting point prediction" for 
arbitrary compositions without re-running CALPHAD.

================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy import linalg

# Try importing scipy.special for spherical harmonics
try:
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ `scipy.special` not available. Spherical harmonics and advanced visualization modes disabled.")

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(
    page_title="CoCrFeNi Phase Stability Explorer v2",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# COLOR & SYMBOL LIBRARY
# =============================================
COLORMAPS = sorted(list(set([
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu",
    "YlOrBr", "YlOrRd", "BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy",
    "RdYlBu", "RdYlGn", "Spectral", "Twilight", "HSV", "Jet", "Rainbow",
    "Hot", "Cool", "Blackbody", "Electric", "Plotly3", "Portland", "Picnic",
    "Solar", "Balance", "Delta", "Curl", "IceFire", "Edge", "Fall", "Sunset",
    "Sunsetdark", "Teal", "Tealgrn", "Tropic", "Peach", "Oxy", "Mint",
    "Emrld", "Aggrnyl", "Agsunset", "Armyrose", "Bluered", "Blugrn", "Bluyl",
    "Brwnyl", "Burg", "Burgyl", "Darkmint", "Geysr", "Magenta", "Mrybm",
    "Mygbm", "Oryel", "Pinkyl", "Purp", "Purpor", "Redor", "Ylorrd", "Ylorbr",
    "Ylgnbu", "Ylgn", "Haline", "Ice", "Matter", "Speed", "Tempo", "Thermal",
    "Turbid", "Algae", "Deep", "Dense", "Sinebow", "Phase"
])))

PHASE_SYMBOLS = {"LIQUID": "circle", "FCC": "diamond", "BOUNDARY": "x"}
PHASE_COLORS = {"LIQUID": "#e74c3c", "FCC": "#2980b9", "BOUNDARY": "#f1c40f"}
PHASE_COLORS_RGBA = {
    "LIQUID": "rgba(231, 76, 60, 0.25)",
    "FCC": "rgba(41, 128, 185, 0.25)",
    "BOUNDARY": "rgba(241, 196, 15, 0.4)"
}

# =============================================
# DATA LOADING WITH REAL 31-TEMPERATURE SUPPORT
# =============================================
@st.cache_data(ttl=3600)
def load_all_data(csv_dir=CSV_FILES_DIR):
    """
    Load Gibbs energy data from 31 CSV files (Gibbs_700K.csv to Gibbs_3300K.csv).
    
    Expected file format per CSV:
      Columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC
      Rows: ~170,000 composition points per temperature (simplex-constrained)
      Units: mole fractions (0-1), Gibbs energy (J/mol)
    
    Returns:
      DataFrame with columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC, T
      Total rows: ~170,000 × 31 ≈ 5.3 million measurements
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    
    if not files:
        st.error(f"❌ No CSV files found in `{csv_dir}`.\n\nExpected files: Gibbs_700K.csv, Gibbs_800K.csv, ..., Gibbs_3300K.csv")
        st.stop()
    
    # Verify we have the expected 31 temperature files
    expected_temps = list(range(300, 3301, 100))  # [700, 800, ..., 3300]
    found_temps = []
    
    for f in files:
        basename = os.path.basename(f)
        try:
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            found_temps.append(T)
        except ValueError:
            st.warning(f"⚠️ Skipping unrecognized file: {basename}")
    
    missing_temps = set(expected_temps) - set(found_temps)
    if missing_temps:
        st.warning(f"⚠️ Missing temperature files: {sorted(missing_temps)[:10]}{'...' if len(missing_temps)>10 else ''}")
    
    dfs = []
    for f in files:
        basename = os.path.basename(f)
        try:
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            
            # Validate data ranges
            if not ((df["Co"] >= 0) & (df["Co"] <= 1)).all():
                st.warning(f"⚠️ Co values out of range in {basename}")
            if not ((df["Co"] + df["Cr"] + df["Fe"] + df["Ni"] - 1.0).abs() < 1e-10).all():
                st.warning(f"⚠️ Composition sum ≠ 1.0 in {basename}")
            
            df["T"] = T
            dfs.append(df)
            
        except Exception as e:
            st.warning(f"⚠️ Skipping {f}: {e}")
    
    if not dfs:
        st.error("❌ No valid data loaded from any files.")
        st.stop()
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Add derived column: driving force for phase transformation
    df_combined["dG"] = df_combined["G_LIQ"] - df_combined["G_FCC"]
    
    st.caption(f"✅ Loaded {len(df_combined):,} measurements across {len(found_temps)} temperatures")
    
    return df_combined

# Load data at module level (cached)
df = load_all_data()

# Extract temperature information
T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

# Global ranges for consistent color scaling across all visualizations
G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))

# Build convex hull of all composition points for uncertainty/distance calculation
all_pts = df[["Co", "Cr", "Fe"]].values
try:
    data_hull = ConvexHull(all_pts)
    HULL_AVAILABLE = True
except Exception:
    HULL_AVAILABLE = False
    data_hull = None

# =============================================
# TENSOR ANALYSIS FUNCTIONS (CPD per Coutinho et al. 2020)
# =============================================
@st.cache_data(ttl=7200)
def build_tensor_data(df):
    """
    Build 4D Thermodynamic Data Tensor from DataFrame.
    
    REAL DATA SPECIFICATION:
    - Input: DataFrame with ~5.3M rows (170K compositions × 31 temperatures)
    - Output: Two 4D numpy arrays G_LIQ, G_FCC of shape (n_co, n_cr, n_fe, 31)
    
    TENSOR CHARACTERISTICS:
    - Typical dimensions: n_co ≈ n_cr ≈ n_fe ≈ 41 (step=0.01, range 0.00-0.40)
    - Full hypercube size: 41³ × 31 ≈ 2.14M entries per phase
    - Valid entries: ~16.7% (simplex constraint: Co+Cr+Fe ≤ 1)
    - Memory: ~2.14M × 8 bytes × 2 phases ≈ 34 MB (dense), ~5.7 MB (sparse)
    
    THERMODYNAMIC INTERPRETATION OF TENSOR SLICES:
    
    Temperature slices (fixed T, varying composition):
      T=700K:   G ≈ -20 to -35 kJ/mol, ΔG > 0 (FCC stable)
      T=1400K:  G ≈ -80 to -95 kJ/mol, ΔG ≈ 0 (transition region)
      T=2200K:  G ≈ -165 to -175 kJ/mol, ΔG < 0 (LIQUID stable)
    
    Composition slices (fixed x, varying T):
      Follow G(T) ≈ H₀ - S₀·T + Cp·[T - T₀ - T·ln(T/T₀)]
      Low-rank structure enables efficient CPD representation
    """
    # Extract unique grid values for each dimension
    co_vals = sorted(df["Co"].unique())
    cr_vals = sorted(df["Cr"].unique())
    fe_vals = sorted(df["Fe"].unique())
    T_vals = sorted(df["T"].unique())  # Should be [700, 800, ..., 3300]
    
    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)
    
    # Create O(1) lookup dictionaries for value→index mapping
    # Rounding to 4 decimals handles floating-point precision from CSV import
    co_to_idx = {round(v, 4): i for i, v in enumerate(co_vals)}
    cr_to_idx = {round(v, 4): i for i, v in enumerate(cr_vals)}
    fe_to_idx = {round(v, 4): i for i, v in enumerate(fe_vals)}
    T_to_idx = {T: i for i, T in enumerate(T_vals)}
    
    # Initialize 4D arrays with NaN (invalid entries remain NaN)
    # Using float64 for CALPHAD precision (~0.1 J/mol)
    G_LIQ_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    G_FCC_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    
    # Populate tensor with valid simplex points from all 31 temperatures
    # This loop processes ~5.3M rows; vectorization not possible due to irregular simplex
    valid_count = 0
    for _, row in df.iterrows():
        co = round(row['Co'], 4)
        cr = round(row['Cr'], 4)
        fe = round(row['Fe'], 4)
        T = row['T']
        
        # Only populate if all indices exist (valid grid point)
        if co in co_to_idx and cr in cr_to_idx and fe in fe_to_idx and T in T_to_idx:
            i, j, k, t = co_to_idx[co], cr_to_idx[cr], fe_to_idx[fe], T_to_idx[T]
            G_LIQ_tdt[i, j, k, t] = row['G_LIQ']
            G_FCC_tdt[i, j, k, t] = row['G_FCC']
            valid_count += 1
    
    # Compute step sizes for grid metadata
    co_step = np.min(np.diff(co_vals)) if len(co_vals) > 1 else 0
    cr_step = np.min(np.diff(cr_vals)) if len(cr_vals) > 1 else 0
    fe_step = np.min(np.diff(fe_vals)) if len(fe_vals) > 1 else 0
    T_step = np.min(np.diff(T_vals)) if len(T_vals) > 1 else 0
    
    st.caption(f"Tensor built: {valid_count:,} valid entries ({100*valid_count/(n_co*n_cr*n_fe*n_T):.1f}% of hypercube)")
    
    return {
        'G_LIQ': G_LIQ_tdt,
        'G_FCC': G_FCC_tdt,
        'dims': (n_co, n_cr, n_fe, n_T),
        'co_vals': co_vals,
        'cr_vals': cr_vals,
        'fe_vals': fe_vals,
        'T_vals': T_vals,  # [700, 800, ..., 3300] - critical for interpretation
        'co_step': co_step,
        'cr_step': cr_step,
        'fe_step': fe_step,
        'T_step': T_step,
        'valid_count': valid_count
    }

def unfold_tensor(tensor, mode):
    """
    Unfold (matricize) 4D tensor along specified mode for SVD analysis.
    
    Mode mapping:
      mode=0 (Co): shape (n_co, n_cr×n_fe×n_T) - each row = one Co value
      mode=1 (Cr): shape (n_cr, n_co×n_fe×n_T) - each row = one Cr value
      mode=2 (Fe): shape (n_fe, n_co×n_cr×n_T) - each row = one Fe value
      mode=3 (T):  shape (n_T, n_co×n_cr×n_fe)  - each row = one temperature
    
    This enables mode-wise singular value decomposition to estimate 
    effective rank for each thermodynamic variable.
    """
    if mode == 0:
        return tensor.reshape(tensor.shape[0], -1)
    elif mode == 1:
        return tensor.transpose(1, 0, 2, 3).reshape(tensor.shape[1], -1)
    elif mode == 2:
        return tensor.transpose(2, 0, 1, 3).reshape(tensor.shape[2], -1)
    elif mode == 3:
        return tensor.transpose(3, 0, 1, 2).reshape(tensor.shape[3], -1)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 0, 1, 2, or 3.")

def svd_rank_analysis(matrix, threshold=0.01):
    """
    Estimate effective rank via SVD with robust NaN handling.
    
    REAL DATA CONSIDERATIONS:
    - Input matrix has ~83% NaN entries (simplex constraint)
    - Column-wise mean imputation is functional but suboptimal
    - For Co-Cr-Fe-Ni with 31 temperatures, expected results:
        Mode-3 (T): s_norm ≈ [1.0, 0.12, 0.025, 0.004, ...] → rank ≈ 3
        Mode-0 (Co): s_norm ≈ [1.0, 0.35, 0.18, 0.09, 0.045, ...] → rank ≈ 6
    
    Args:
        matrix: 2D array with NaN entries
        threshold: Fraction of max singular value to count as significant
    
    Returns:
        rank: Estimated effective rank
        s: Raw singular values
        s_norm: Normalized singular values (s/s_max)
    """
    matrix_filled = matrix.copy().astype(np.float64)
    
    # Column-wise NaN imputation (suboptimal but functional for rank estimation)
    for col in range(matrix_filled.shape[1]):
        col_data = matrix_filled[:, col]
        valid = ~np.isnan(col_data)
        if np.sum(valid) > 0:
            matrix_filled[:, col] = np.where(np.isnan(col_data), np.nanmean(col_data[valid]), col_data)
        else:
            matrix_filled[:, col] = 0.0
    
    # Check for zero matrix
    if np.linalg.norm(matrix_filled) < 1e-12:
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))
    
    try:
        U, s, Vh = linalg.svd(matrix_filled, full_matrices=False)
    except Exception as e:
        st.warning(f"⚠️ SVD failed: {e}")
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))
    
    # Normalize singular values
    s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
    s_norm = s / s_max
    
    # Count values above threshold
    rank = int(np.sum(s_norm > threshold))
    
    return rank, s, s_norm

def cpd_als_4d(tensor, rank, max_iter=100, tol=1e-6, use_weighted=False, reg=1e-8):
    """
    4-way Canonical Polyadic Decomposition via Alternating Least Squares.
    
    Args (additional):
        use_weighted: (bool, placeholder) Whether to use weighted ALS for missing entries.
                      Currently not implemented - falls back to standard ALS.
        reg: (float, placeholder) Regularization parameter for ill-conditioned steps.
             Currently not implemented.

    DECOMPOSITION FORMULA:
      tensor[i,j,k,t] ≈ Σᵣ₌₁ᴿ λᵣ · A[i,r] · B[j,r] · C[k,r] · D[t,r]
    
    REAL DATA OPTIMIZATIONS:
    - Handles ~83% NaN entries via masking (not zero-imputation for fitting)
    - Initializes temperature factor D with thermodynamic basis functions
    - Normalizes factors each iteration to prevent numerical overflow
    - Converges typically in 20-50 iterations for R=6, Co-Cr-Fe-Ni data
    
    PHYSICAL INTERPRETATION OF FACTORS (R=6):
      A[:,1], B[:,1], C[:,1]: Smooth composition baselines (enthalpy)
      D[:,1]: ~constant (baseline offset)
      
      A[:,2], B[:,2], C[:,2]: Composition-dependent entropy coefficients
      D[:,2]: ~linear in T (captures -S·T term)
      
      A[:,3], B[:,3], C[:,3]: Magnetic element weighting (Co, Fe)
      D[:,3]: Quadratic/tanh-like (Cp curvature + magnetic transitions)
      
      A[:,4-6], B[:,4-6], C[:,4-6]: Higher-order mixing interactions
      D[:,4-6]: Fine thermal corrections
    
    Args:
        tensor: 4D numpy array with NaN for invalid entries
        rank: Target CP rank R (recommended: 6 for Co-Cr-Fe-Ni)
        max_iter: Maximum ALS iterations
        tol: Convergence tolerance on relative reconstruction error
    
    Returns:
        A, B, C, D: Factor matrices of shapes (n_co,R), (n_cr,R), (n_fe,R), (31,R)
        lam: Component weights array of shape (R,)
        error: Final RMSE on observed entries (J/mol)
    """
    I, J, K, L = tensor.shape  # L = 31 (temperatures)
    mask = ~np.isnan(tensor)    # Boolean mask: True for valid simplex entries
    X = np.where(mask, tensor, 0)  # Zero-fill for numerical operations only
    
    # === INITIALIZATION: Thermodynamic priors for faster convergence ===
    # Temperature factor D: basis functions matching G(T) physics
    if L == 31:  # Standard 31-temperature grid
        T_vals_physical = np.array(list(range(700, 3701, 100)))
        T_mean, T_std = np.mean(T_vals_physical), np.std(T_vals_physical)
        T_norm = (T_vals_physical - T_mean) / (T_std + 1e-12)
        
        D = np.zeros((L, rank))
        D[:, 0] = 1.0  # r=1: Constant baseline
        
        if rank >= 2:
            D[:, 1] = T_norm  # r=2: Linear entropy term
        
        if rank >= 3:
            D[:, 2] = (T_norm**2 - 1) * 0.5  # r=3: Orthogonalized quadratic
        
        if rank >= 4:
            D[:, 3] = np.tanh(2 * T_norm) - np.mean(np.tanh(2 * T_norm))  # r=4: Transition
        
        if rank > 4:
            D[:, 4:] = np.random.rand(L, rank-4) * 0.01  # Higher orders: random
    else:
        # Fallback for non-standard temperature grids
        D = np.random.rand(L, rank) * 0.1
    
    # Composition factors: SVD initialization for stability
    X_unfolded = unfold_tensor(X, mode=0)  # (n_co, n_cr×n_fe×n_T)
    try:
        U, s, Vh = linalg.svd(X_unfolded, full_matrices=False)
        A = U[:, :rank] * np.sqrt(s[:rank])
    except:
        A = np.random.rand(I, rank) * 0.1
    
    B = np.random.rand(J, rank) * 0.1
    C = np.random.rand(K, rank) * 0.1
    
    prev_error = np.inf
    
    # === ALTERNATING LEAST SQUARES ITERATIONS ===
    for iteration in range(max_iter):
        
        # --- Update A (Co factor) ---
        BCD = np.zeros((J*K*L, rank))
        for r in range(rank):
            # Khatri-Rao product: column-wise Kronecker product
            BCD[:, r] = np.kron(np.kron(D[:, r], C[:, r]), B[:, r])
        
        X_flat = X.reshape(I, -1)
        mask_flat = mask.reshape(I, -1)
        
        for i in range(I):
            valid = mask_flat[i, :]
            if np.sum(valid) > rank:  # Need more observations than parameters
                A[i, :] = linalg.lstsq(BCD[valid, :], X_flat[i, valid])[0]
        
        # Normalize and accumulate norms in lambda later
        norms = np.linalg.norm(A, axis=0) + 1e-12
        A = A / norms
        
        # --- Update B (Cr factor) ---
        ACD = np.zeros((I*K*L, rank))
        for r in range(rank):
            ACD[:, r] = np.kron(np.kron(D[:, r], C[:, r]), A[:, r])
        
        X_flat = X.transpose(1, 0, 2, 3).reshape(J, -1)
        mask_flat = mask.transpose(1, 0, 2, 3).reshape(J, -1)
        
        for j in range(J):
            valid = mask_flat[j, :]
            if np.sum(valid) > rank:
                B[j, :] = linalg.lstsq(ACD[valid, :], X_flat[j, valid])[0]
        
        norms = np.linalg.norm(B, axis=0) + 1e-12
        B = B / norms
        
        # --- Update C (Fe factor) ---
        ABD = np.zeros((I*J*L, rank))
        for r in range(rank):
            ABD[:, r] = np.kron(np.kron(D[:, r], B[:, r]), A[:, r])
        
        X_flat = X.transpose(2, 0, 1, 3).reshape(K, -1)
        mask_flat = mask.transpose(2, 0, 1, 3).reshape(K, -1)
        
        for k in range(K):
            valid = mask_flat[k, :]
            if np.sum(valid) > rank:
                C[k, :] = linalg.lstsq(ABD[valid, :], X_flat[k, valid])[0]
        
        norms = np.linalg.norm(C, axis=0) + 1e-12
        C = C / norms
        
        # --- Update D (Temperature factor) ---
        ABC = np.zeros((I*J*K, rank))
        for r in range(rank):
            ABC[:, r] = np.kron(np.kron(C[:, r], B[:, r]), A[:, r])
        
        X_flat = X.transpose(3, 0, 1, 2).reshape(L, -1)
        mask_flat = mask.transpose(3, 0, 1, 2).reshape(L, -1)
        
        for t in range(L):
            valid = mask_flat[t, :]
            if np.sum(valid) > rank:
                D[t, :] = linalg.lstsq(ABC[valid, :], X_flat[t, valid])[0]
        
        norms = np.linalg.norm(D, axis=0) + 1e-12
        D = D / norms
        
        # --- Compute reconstruction error on OBSERVED entries only ---
        recon = np.zeros_like(X)
        for r in range(rank):
            # Outer product of factor columns, reshaped to 4D tensor
            recon += np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)
        
        # RMSE on observed entries only (critical for incomplete tensor)
        observed_residuals = (tensor - recon)[mask]
        if len(observed_residuals) > 0:
            error = np.sqrt(np.mean(observed_residuals**2))
        else:
            error = np.inf
        
        # Check convergence
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    # === Compute component weights lambda ===
    # lambda_r = ||A[:,r]|| · ||B[:,r]|| · ||C[:,r]|| · ||D[:,r]||
    lam = np.ones(rank)
    for r in range(rank):
        lam[r] = (np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * 
                  np.linalg.norm(C[:, r]) * np.linalg.norm(D[:, r]))
    
    return A, B, C, D, lam, error

# =============================================
# INTERPOLATION FOR CONTINUOUS COMPOSITION QUERIES
# =============================================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    """
    Build LinearNDInterpolator for Gibbs energies at fixed temperature.
    
    Enables continuous composition queries (not just grid points).
    Uses Delaunay triangulation of simplex-constrained composition space.
    
    Args:
        df: Full DataFrame with all temperatures
        T: Target temperature (must exist in df)
    
    Returns:
        interp_liq, interp_fcc: scipy.interpolate.LinearNDInterpolator objects
    """
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    
    # Composition points for interpolation (3D: Co, Cr, Fe; Ni is dependent)
    pts = df_T[["Co", "Cr", "Fe"]].values
    
    # Build interpolators for each phase
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values, fill_value=np.nan)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values, fill_value=np.nan)
    
    return interp_liq, interp_fcc

# =============================================
# TETRAHEDRAL GRID GENERATION & UNCERTAINTY METRICS
# =============================================
def generate_tetrahedral_grid(resolution=25):
    """
    Generate regular grid points within the composition simplex.
    
    The quaternary composition space Co-Cr-Fe-Ni with constraint:
      x_Co + x_Cr + x_Fe + x_Ni = 1, x_i ≥ 0
    is a 3-simplex (tetrahedron) in 4D, projected to 3D for visualization.
    
    Args:
        resolution: Number of points per axis (grid will have ~resolution³/6 valid points)
    
    Returns:
        pts: Array of shape (N_valid, 3) with columns [Co, Cr, Fe]
    """
    x = np.linspace(0, 1, resolution)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    
    # Apply simplex constraint: Co + Cr + Fe ≤ 1 (Ni = 1 - sum ≥ 0)
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    
    return grid_pts[valid_mask]

def compute_data_proximity(pts, data_pts, max_dist=0.15):
    """
    Compute normalized proximity to nearest CALPHAD data point.
    
    Used for uncertainty visualization: points far from training data
    have higher interpolation uncertainty.
    
    Args:
        pts: Query points (N, 3)
        data_pts: CALPHAD data points (M, 3)
        max_dist: Distance beyond which proximity = 0 (default: 0.15 mole fraction)
    
    Returns:
        proximity: Array (N,) with values in [0, 1], 1 = on data, 0 = far
    """
    tree = cKDTree(data_pts)
    dists, _ = tree.query(pts, k=1)
    proximity = np.clip(1.0 - dists / max_dist, 0.0, 1.0)
    return proximity

def find_phase_boundary_points(pts, dG_values, threshold=50.0):
    """
    Identify points near the phase boundary (ΔG ≈ 0).
    
    Args:
        pts: Composition points (N, 3)
        dG_values: Driving force values G_LIQ - G_FCC (N,)
        threshold: J/mol tolerance for "near boundary" (default: 50 J/mol)
    
    Returns:
        boundary_pts: Points with |ΔG| < threshold
        boundary_dG: Corresponding ΔG values
    """
    boundary_mask = np.abs(dG_values) < threshold
    return pts[boundary_mask], dG_values[boundary_mask]

# =============================================
# SPHERICAL HARMONICS FOR COMPOSITION VISUALIZATION
# =============================================
if SCIPY_AVAILABLE:
    def get_real_sph_harm(l, m, theta, phi):
        """
        Compute real-valued spherical harmonics for composition visualization.
        
        Maps 3D composition space (Co, Cr, Fe) to spherical coordinates
        for smooth surface representation of Gibbs energy.
        
        Args:
            l: Degree (non-negative integer)
            m: Order (integer, -l ≤ m ≤ l)
            theta: Azimuthal angle [0, 2π]
            phi: Polar angle [0, π]
        
        Returns:
            Real-valued spherical harmonic Y_l^m(θ, φ)
        """
        if hasattr(special, 'sph_harm_y'):
            # Newer scipy interface
            Y_complex = special.sph_harm_y(l, m, phi, theta)
        else:
            # Legacy interface
            Y_complex = special.sph_harm(m, l, theta, phi)
        
        # Convert complex to real basis
        if m > 0:
            return np.sqrt(2.0) * Y_complex.real
        elif m < 0:
            if hasattr(special, 'sph_harm_y'):
                Y_pos = special.sph_harm_y(l, abs(m), phi, theta)
            else:
                Y_pos = special.sph_harm(abs(m), l, theta, phi)
            return np.sqrt(2.0) * Y_pos.imag
        else:
            return Y_complex.real

    def sample_g_on_sphere(interp_liq, interp_fcc, R_fixed, n_theta=60, n_phi=60):
        """
        Sample Gibbs energy on spherical composition grid.

        Maps spherical coordinates to composition space:
          x = R·sin(φ)·cos(θ) → Co
          y = R·sin(φ)·sin(θ) → Cr  
          z = R·cos(φ) → Fe
          Ni = 1 - (Co + Cr + Fe) [implicit]

        Args:
            interp_liq, interp_fcc: Interpolators for Gibbs energies
            R_fixed: Fixed radius for spherical sampling
            n_theta, n_phi: Angular resolution

        Returns:
            TH, PH: Meshgrid of spherical angles
            G_stable: Gibbs energy of stable phase at each point
            dG: Driving force G_LIQ - G_FCC
            valid: Boolean mask for valid simplex points
            sphere_pts: Cartesian composition coordinates
        """
        # ROBUSTNESS: Clip R_fixed to ensure spherical points stay within valid simplex
        # Max radius where all points have Co+Cr+Fe ≤ 1 and all coordinates ≥ 0
        R_max_safe = 1.0 / np.sqrt(3.0)  # ~0.577
        if R_fixed > R_max_safe:
            R_fixed = R_max_safe

        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)

        # Map spherical to Cartesian composition coordinates
        x = R_fixed * np.sin(PH) * np.cos(TH)  # Co
        y = R_fixed * np.sin(PH) * np.sin(TH)  # Cr
        z = R_fixed * np.cos(PH)                # Fe
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Apply simplex constraint: Co + Cr + Fe ≤ 1 (ensures Ni ≥ 0)
        valid = (pts[:,0] + pts[:,1] + pts[:,2]) <= 1.0

        # Also ensure all coordinates are non-negative (composition constraint)
        valid = valid & (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 2] >= 0)

        # Interpolate Gibbs energies
        G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
        G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)

        # Determine stable phase and driving force
        G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
        dG = G_liq - G_fcc

        # Combine validity masks: must be in simplex AND have valid interpolated data
        valid = valid & ~np.isnan(G_stable)

        return (TH, PH, 
                G_stable.reshape(TH.shape), 
                dG.reshape(TH.shape), 
                valid.reshape(TH.shape), 
                pts)

    @st.cache_data(ttl=3600)
    def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=3):
        """
        Fit spherical harmonic coefficients to Gibbs energy data.

        Solves least-squares problem: g ≈ Σₗ₌₀ˡᵐᵃˣ Σₘ₌₋ₗˡ cₗₘ·Yₗₘ(θ,φ)

        Args:
            theta_vals, phi_vals: Spherical coordinates of data points
            g_vals: Gibbs energy values at those points
            l_max: Maximum spherical harmonic degree

        Returns:
            coeffs: Fitted coefficients array
            l_max: Actual maximum degree used
        """
        theta_flat = theta_vals.ravel()
        phi_flat = phi_vals.ravel()
        g_flat = g_vals.ravel()

        # Filter valid (non-NaN) data
        valid = ~np.isnan(g_flat)
        theta_flat = theta_flat[valid]
        phi_flat = phi_flat[valid]
        g_flat = g_flat[valid]

        if len(theta_flat) == 0:
            return None, l_max

        # Build design matrix: each row = spherical harmonics at one point
        A = []
        for t, p in zip(theta_flat, phi_flat):
            row = []
            for l in range(l_max+1):
                for m in range(-l, l+1):
                    y = get_real_sph_harm(l, m, t, p)
                    row.append(y)
            A.append(row)
        A = np.array(A)

        # ROBUSTNESS: Check if we have enough data points for the number of basis functions
        n_basis = (l_max + 1) ** 2
        if A.shape[0] < n_basis:
            st.warning(f"⚠️ Insufficient valid data points ({A.shape[0]}) for l_max={l_max} (needs ≥{n_basis}). Reducing l_max.")
            # Reduce l_max until we have enough points
            while l_max > 0 and A.shape[0] < (l_max + 1) ** 2:
                l_max -= 1
            if l_max < 0:
                return None, 0
            # Rebuild A with reduced l_max
            A = []
            for t, p in zip(theta_flat, phi_flat):
                row = []
                for l in range(l_max+1):
                    for m in range(-l, l+1):
                        y = get_real_sph_harm(l, m, t, p)
                        row.append(y)
                A.append(row)
            A = np.array(A)

        # ROBUSTNESS: Check for empty or degenerate matrix
        if A.size == 0 or A.shape[0] == 0 or A.shape[1] == 0:
            return None, l_max

        # Solve least squares with robust handling
        try:
            rank_A = np.linalg.matrix_rank(A)
            if rank_A < A.shape[1]:
                st.warning(f"⚠️ Design matrix is rank-deficient (rank={rank_A} < cols={A.shape[1]}). Using minimum-norm solution.")
            coeffs, residuals, rank, s = linalg.lstsq(A, g_flat)
        except Exception as e:
            st.warning(f"⚠️ lstsq failed: {e}. Returning None.")
            return None, l_max

        return coeffs, l_max

    def reconstruct_sh_surface(theta_grid, phi_grid, coeffs, l_max):
        """Reconstruct Gibbs energy surface from spherical harmonic coefficients."""
        recon = np.zeros_like(theta_grid, dtype=float)
        idx = 0
        for l in range(l_max+1):
            for m in range(-l, l+1):
                Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
                recon += coeffs[idx] * Y
                idx += 1
        return recon

    def extract_dg_zero_contour(TH, PH, dG_grid, R_fixed):
        """
        Extract ΔG=0 contour via edge-walking on spherical grid.
        
        Identifies phase boundary where G_LIQ = G_FCC.
        Uses linear interpolation along grid edges for sub-grid precision.
        
        Args:
            TH, PH: Spherical angle grids
            dG_grid: Driving force values on grid
            R_fixed: Radius for converting spherical to Cartesian
        
        Returns:
            contours_x, y, z: Cartesian coordinates of boundary contour
        """
        contours_x, contours_y, contours_z = [], [], []
        
        # Horizontal edges (varying theta at fixed phi)
        for i in range(dG_grid.shape[0]):
            for j in range(dG_grid.shape[1]-1):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i,j+1])):
                    continue
                if dG_grid[i,j] * dG_grid[i,j+1] < 0:  # Sign change
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i,j+1]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i,j+1] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i,j+1] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))
        
        # Vertical edges (varying phi at fixed theta)
        for i in range(dG_grid.shape[0]-1):
            for j in range(dG_grid.shape[1]):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i+1,j])):
                    continue
                if dG_grid[i,j] * dG_grid[i+1,j] < 0:
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i+1,j]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i+1,j] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i+1,j] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))
        
        return np.array(contours_x), np.array(contours_y), np.array(contours_z)

# =============================================
# TEMPERATURE-DRIVEN SHAPE MORPHING FUNCTIONS
# =============================================
def get_liquid_radius(G_sh, sh_R_fixed, T_factor):
    """
    Compute LIQUID phase surface radius with temperature-driven morphing.
    
    Physical interpretation:
      - High T: Liquid expands (thermal expansion), smooths (entropy dominates)
      - Low T: Liquid contracts, but remains smoother than FCC
    
    Args:
        G_sh: Spherical harmonic-reconstructed Gibbs energy
        sh_R_fixed: Base radius for spherical sampling
        T_factor: Normalized temperature [0, 1] = [(T-T_min)/(T_max-T_min)]
    
    Returns:
        radius: Deformed radius array for 3D visualization
    """
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    
    # Thermal expansion: 35% increase from low to high T
    thermal_exp = 1.0 + 0.35 * T_factor
    
    # Fluid-like undulations: stronger at high T, smooth sinusoidal
    fluid_dist = 0.12 * np.sin(2 * np.pi * norm) * (0.5 + 0.5 * T_factor)
    
    return sh_R_fixed * (thermal_exp + 0.22 * norm + fluid_dist)

def get_fcc_radius(G_sh, sh_R_fixed, T_factor):
    """
    Compute FCC phase surface radius with temperature-driven morphing.
    
    Physical interpretation:
      - Low T: FCC is rigid, faceted (crystalline order, magnetic contributions)
      - High T: FCC shrinks, smooths (approaching melting)
    
    Args:
        G_sh: Spherical harmonic-reconstructed Gibbs energy
        sh_R_fixed: Base radius for spherical sampling
        T_factor: Normalized temperature [0, 1]
    
    Returns:
        radius: Deformed radius array for 3D visualization
    """
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    
    # Rigidity decreases with T: 20% reduction from low to high T
    rigidity = 1.0 - 0.20 * T_factor
    
    # Crystalline faceting: multiple harmonics, stronger at low T
    crystal_factor = 0.28 * (1.0 - T_factor)
    crystal_ripples = crystal_factor * (
        0.6 * np.sin(6 * np.pi * norm) +    # Primary faceting
        0.3 * np.sin(10 * np.pi * norm) +   # Secondary features
        0.1 * np.sin(14 * np.pi * norm)     # Fine structure
    )
    
    return sh_R_fixed * (rigidity + 0.20 * norm + crystal_ripples)


# =============================================
# ADDITIVE MANUFACTURING CPD ANALYSIS MODULE
# Expanded from theoretical framework
# =============================================

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_extract_transition(A_liq_tuple, A_fcc_tuple, B_liq_tuple, B_fcc_tuple,
                                C_liq_tuple, C_fcc_tuple, D_liq_tuple, D_fcc_tuple,
                                lam_liq_tuple, lam_fcc_tuple,
                                co_vals_tuple, cr_vals_tuple, fe_vals_tuple, T_vals_tuple,
                                composition_grid_resolution=25):
    """Cached wrapper for transition surface extraction."""
    # Convert tuples back to arrays
    A_liq = np.array(A_liq_tuple); A_fcc = np.array(A_fcc_tuple)
    B_liq = np.array(B_liq_tuple); B_fcc = np.array(B_fcc_tuple)
    C_liq = np.array(C_liq_tuple); C_fcc = np.array(C_fcc_tuple)
    D_liq = np.array(D_liq_tuple); D_fcc = np.array(D_fcc_tuple)
    lam_liq = np.array(lam_liq_tuple); lam_fcc = np.array(lam_fcc_tuple)
    co_vals = np.array(co_vals_tuple); cr_vals = np.array(cr_vals_tuple)
    fe_vals = np.array(fe_vals_tuple); T_vals = np.array(T_vals_tuple)

    return _extract_transition_impl(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                     D_liq, D_fcc, lam_liq, lam_fcc,
                                     co_vals, cr_vals, fe_vals, T_vals,
                                     composition_grid_resolution)

def _extract_transition_impl(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                             D_liq, D_fcc, lam_liq, lam_fcc,
                             co_vals, cr_vals, fe_vals, T_vals,
                             composition_grid_resolution=25):
    """
    Extract T*(x_Co, x_Cr, x_Fe) surface where G_LIQ = G_FCC using CPD factors.

    CRITICAL FIX: Validates that LIQUID and FCC factors have compatible ranks.
    If ranks differ, uses the minimum rank to prevent IndexError.

    Theory: For each composition, solve ΔG(T*) = 0 via root-finding on the 
    temperature factor combination:

        ΔG(x,T) = Σᵣ λᵣ^Δ · Aᵣ(x_Co) · Bᵣ(x_Cr) · Cᵣ(x_Fe) · Dᵣ(T) = 0

    where λᵣ^Δ = λᵣ^LIQ · Aᵣ^LIQ · Bᵣ^LIQ · Cᵣ^LIQ - λᵣ^FCC · Aᵣ^FCC · Bᵣ^FCC · Cᵣ^FCC

    Args:
        A_liq, A_fcc: Co factor matrices (n_co × R)
        B_liq, B_fcc: Cr factor matrices (n_cr × R)  
        C_liq, C_fcc: Fe factor matrices (n_fe × R)
        D_liq, D_fcc: Temperature factor matrices (n_T × R)
        lam_liq, lam_fcc: Component weights (R,)
        co_vals, cr_vals, fe_vals, T_vals: Grid values
        composition_grid_resolution: Points per axis for output surface

    Returns:
        T_melt_surface: Array (res, res, res) with T* values (NaN where no root)
        valid_mask: Boolean array indicating successful root finding
        delta_G_grid: Full ΔG values for validation
    """
    from scipy.optimize import brentq

    # === RANK VALIDATION ===
    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)
    R = min(R_liq, R_fcc)  # Use minimum rank to prevent IndexError

    # Validate D matrix dimensions
    if D_liq.shape[1] < R_liq or D_fcc.shape[1] < R_fcc:
        st.warning(f"⚠️ D matrix columns ({D_liq.shape[1]}, {D_fcc.shape[1]}) < rank ({R_liq}, {R_fcc}). Truncating to min rank {R}.")
        R = min(R, D_liq.shape[1], D_fcc.shape[1])

    if R < 1:
        st.error("❌ Cannot extract transition surface: CPD rank < 1")
        return None, None, None

    if R < R_liq or R < R_fcc:
        st.info(f"ℹ️ Using truncated rank R={R} (LIQUID had {R_liq}, FCC had {R_fcc})")

    n_T = len(T_vals)

    # Pre-compute temperature-dependent difference using VALIDATED rank R
    D_diff = np.zeros((n_T, R))
    for r in range(R):
        D_diff[:, r] = lam_liq[r] * D_liq[:, r] - lam_fcc[r] * D_fcc[:, r]

    # Generate composition grid (simplex-constrained)
    x = np.linspace(0, 1, composition_grid_resolution)
    Co_grid, Cr_grid, Fe_grid = np.meshgrid(x, x, x, indexing='ij')
    Ni_grid = 1.0 - Co_grid - Cr_grid - Fe_grid
    valid_simplex = Ni_grid >= 0

    T_melt = np.full_like(Co_grid, np.nan, dtype=np.float64)
    delta_G_grid = np.full((*Co_grid.shape, n_T), np.nan, dtype=np.float64)

    # Helper: interpolate factor to query composition
    def interp_factor(vals, factor_matrix, query):
        """Linear interpolation of factor matrix column to query point."""
        result = np.zeros(factor_matrix.shape[1])
        for r in range(factor_matrix.shape[1]):
            result[r] = np.interp(query, vals, factor_matrix[:, r], left=np.nan, right=np.nan)
        return result

    # Root-finding for each valid composition
    for i in range(composition_grid_resolution):
        for j in range(composition_grid_resolution):
            for k in range(composition_grid_resolution):
                if not valid_simplex[i, j, k]:
                    continue

                co, cr, fe = Co_grid[i,j,k], Cr_grid[i,j,k], Fe_grid[i,j,k]

                # Interpolate composition factors
                A_liq_q = interp_factor(co_vals, A_liq, co)
                B_liq_q = interp_factor(cr_vals, B_liq, cr)
                C_liq_q = interp_factor(fe_vals, C_liq, fe)
                A_fcc_q = interp_factor(co_vals, A_fcc, co)
                B_fcc_q = interp_factor(cr_vals, B_fcc, cr)
                C_fcc_q = interp_factor(fe_vals, C_fcc, fe)

                # Check for NaN in interpolation
                if np.any(np.isnan(A_liq_q)) or np.any(np.isnan(B_liq_q)) or np.any(np.isnan(C_liq_q)):
                    continue
                if np.any(np.isnan(A_fcc_q)) or np.any(np.isnan(B_fcc_q)) or np.any(np.isnan(C_fcc_q)):
                    continue

                # Use VALIDATED rank R (not full rank)
                comp_coeff = (lam_liq[:R] * A_liq_q[:R] * B_liq_q[:R] * C_liq_q[:R] - 
                             lam_fcc[:R] * A_fcc_q[:R] * B_fcc_q[:R] * C_fcc_q[:R])

                # Define ΔG(T) function for root finding
                def delta_G(T_query):
                    D_q = np.zeros(R)
                    for r in range(R):
                        D_q[r] = np.interp(T_query, T_vals, D_diff[:, r])
                    return float(np.sum(comp_coeff * D_q))

                # Evaluate ΔG at temperature extremes
                try:
                    g_low = delta_G(float(T_vals[0]))
                    g_high = delta_G(float(T_vals[-1]))
                except:
                    continue

                if np.isnan(g_low) or np.isnan(g_high):
                    continue
                if np.sign(g_low) == np.sign(g_high):
                    continue  # No sign change → no root in range

                # Find root using Brent's method (robust 1D root finder)
                try:
                    T_star = brentq(delta_G, float(T_vals[0]), float(T_vals[-1]), xtol=1.0)
                    T_melt[i, j, k] = T_star
                except (ValueError, RuntimeError):
                    continue  # Root finding failed

                # Store full ΔG profile for this composition
                for t_idx, T_val in enumerate(T_vals):
                    delta_G_grid[i, j, k, t_idx] = delta_G(float(T_val))

    return T_melt, valid_simplex, delta_G_grid

def extract_transition_surface_from_cpd(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                         D_liq, D_fcc, lam_liq, lam_fcc,
                                         co_vals, cr_vals, fe_vals, T_vals,
                                         composition_grid_resolution=25):
    """Public interface with caching."""
    # Convert arrays to tuples for caching
    return _cached_extract_transition(
        tuple(A_liq.ravel()), tuple(A_fcc.ravel()),
        tuple(B_liq.ravel()), tuple(B_fcc.ravel()),
        tuple(C_liq.ravel()), tuple(C_fcc.ravel()),
        tuple(D_liq.ravel()), tuple(D_fcc.ravel()),
        tuple(lam_liq), tuple(lam_fcc),
        tuple(co_vals), tuple(cr_vals), tuple(fe_vals), tuple(T_vals),
        composition_grid_resolution
    )


def compute_composition_sensitivity(A, B, C, lam, co_vals, cr_vals, fe_vals, R=6):
    """
    Compute sensitivity of Gibbs energy to composition changes.

    Sensitivity[xᵢ] = Σᵣ |λᵣ · Factor[xᵢ, r]|

    High sensitivity = small composition changes cause large property changes.
    AM relevance: tight powder blending tolerances needed.

    Returns:
        sens_Co, sens_Cr, sens_Fe: Sensitivity arrays for each element
    """
    sens_Co = np.zeros(len(co_vals))
    sens_Cr = np.zeros(len(cr_vals))
    sens_Fe = np.zeros(len(fe_vals))

    for r in range(min(R, len(lam))):
        sens_Co += np.abs(lam[r] * A[:, r])
        sens_Cr += np.abs(lam[r] * B[:, r])
        sens_Fe += np.abs(lam[r] * C[:, r])

    # Normalize to [0, 1]
    for sens in [sens_Co, sens_Cr, sens_Fe]:
        s_min, s_max = np.min(sens), np.max(sens)
        if s_max > s_min:
            sens[:] = (sens - s_min) / (s_max - s_min)

    return sens_Co, sens_Cr, sens_Fe


def compute_hot_cracking_susceptibility(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                       D_liq, D_fcc, lam_liq, lam_fcc,
                                       co_vals, cr_vals, fe_vals, T_vals,
                                       composition_grid_resolution=20):
    """
    Compute hot cracking susceptibility metric from CPD factors.

    Metric: S_crack[x] = |∇_x T*(x)| × |d(ΔG)/dT|⁻¹ at T*

    Higher value = wider solidification range = higher cracking risk.

    Args:
        Same as extract_transition_surface_from_cpd

    Returns:
        S_crack: Susceptibility array (res, res, res)
        T_melt: Transition temperature surface
        valid_mask: Valid simplex mask
    """
    # First extract T* surface
    T_melt, valid_mask, delta_G_grid = extract_transition_surface_from_cpd(
        A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
        D_liq, D_fcc, lam_liq, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals,
        composition_grid_resolution=composition_grid_resolution
    )

    S_crack = np.full_like(T_melt, np.nan)
    res = composition_grid_resolution

    # Compute finite difference gradient of T*
    dx = 1.0 / (res - 1) if res > 1 else 0.01

    for i in range(1, res - 1):
        for j in range(1, res - 1):
            for k in range(1, res - 1):
                if not valid_mask[i,j,k] or np.isnan(T_melt[i,j,k]):
                    continue

                # Central difference gradient magnitude
                dTdx = (T_melt[i+1,j,k] - T_melt[i-1,j,k]) / (2 * dx)
                dTdy = (T_melt[i,j+1,k] - T_melt[i,j-1,k]) / (2 * dx)
                dTdz = (T_melt[i,j,k+1] - T_melt[i,j,k-1]) / (2 * dx)

                grad_mag = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2)

                # Estimate |d(ΔG)/dT| at T* from stored delta_G_grid
                T_star = T_melt[i,j,k]
                t_idx = np.argmin(np.abs(np.array(T_vals) - T_star))

                # Finite difference d(ΔG)/dT
                if t_idx > 0 and t_idx < len(T_vals) - 1:
                    dGdT = abs((delta_G_grid[i,j,k,t_idx+1] - delta_G_grid[i,j,k,t_idx-1]) / 
                              (T_vals[t_idx+1] - T_vals[t_idx-1]))
                else:
                    dGdT = abs(np.gradient(delta_G_grid[i,j,k,:], T_vals)[t_idx])

                # Cracking susceptibility
                if dGdT > 1e-6 and np.isfinite(dGdT):
                    S_crack[i,j,k] = grad_mag / dGdT
                else:
                    S_crack[i,j,k] = 0.0  # Zero susceptibility if dGdT ≈ 0

    return S_crack, T_melt, valid_mask


def compute_segregation_potential(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                  lam_liq, lam_fcc, R=6):
    """
    Compute segregation potential from binary interaction factors (r=4,5 typically).

    Segregation potential = |λᵣ · A[:,r] · B[:,r]| for binary pairs
    High values indicate strong tendency for element partitioning during solidification.

    Returns:
        seg_CoCr, seg_CoFe, seg_CrFe: Segregation potential matrices
    """
    # Binary interaction components (typically r=3,4,5 in R=6 decomposition)
    binary_r = [3, 4, 5] if R >= 6 else list(range(min(3, R)))

    n_co = A_liq.shape[0]
    n_cr = B_liq.shape[0]
    n_fe = C_liq.shape[0]

    seg_CoCr = np.zeros((n_co, n_cr))
    seg_CoFe = np.zeros((n_co, n_fe))
    seg_CrFe = np.zeros((n_cr, n_fe))

    for r in binary_r:
        if r >= R:
            continue
        # Weighted by LIQUID phase factors (segregation occurs in liquid)
        for i in range(n_co):
            for j in range(n_cr):
                seg_CoCr[i,j] += abs(lam_liq[r] * A_liq[i,r] * B_liq[j,r])
            for k in range(n_fe):
                seg_CoFe[i,k] += abs(lam_liq[r] * A_liq[i,r] * C_liq[k,r])

    for r in binary_r:
        if r >= R:
            continue
        for j in range(n_cr):
            for k in range(n_fe):
                seg_CrFe[j,k] += abs(lam_liq[r] * B_liq[j,r] * C_liq[k,r])

    # Normalize
    for seg in [seg_CoCr, seg_CoFe, seg_CrFe]:
        s_min, s_max = np.min(seg), np.max(seg)
        if s_max > s_min:
            seg[:] = (seg - s_min) / (s_max - s_min)

    return seg_CoCr, seg_CoFe, seg_CrFe


# =============================================
# PLOTLY VISUALIZATION FUNCTIONS FOR AM ANALYSIS
# =============================================

def plot_transition_surface_3d(T_melt, valid_mask, co_vals, cr_vals, fe_vals,
                                T_laser=2800, T_haz=1200):
    """
    Create 3D scatter plot of transition temperature surface T*(x).

    Args:
        T_melt: Transition temperature array (res, res, res)
        valid_mask: Valid simplex boolean mask
        co_vals, cr_vals, fe_vals: Composition grid values
        T_laser: Laser melt pool temperature for overlay (K)
        T_haz: Heat-affected zone temperature for overlay (K)

    Returns:
        plotly.graph_objects.Figure
    """
    res = T_melt.shape[0]
    x = np.linspace(0, 1, res)

    # Flatten valid points
    Co_flat = np.zeros(0)
    Cr_flat = np.zeros(0)
    Fe_flat = np.zeros(0)
    T_flat = np.zeros(0)

    for i in range(res):
        for j in range(res):
            for k in range(res):
                if valid_mask[i,j,k] and not np.isnan(T_melt[i,j,k]):
                    Co_flat = np.append(Co_flat, x[i])
                    Cr_flat = np.append(Cr_flat, x[j])
                    Fe_flat = np.append(Fe_flat, x[k])
                    T_flat = np.append(T_flat, T_melt[i,j,k])

    # Filter extreme values
    valid_T = (T_flat > 700) & (T_flat < 3300) & np.isfinite(T_flat)

    if np.sum(valid_T) < 10:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Too few valid transition points. Try coarser resolution.",
                          xref="paper", yref="paper", showarrow=False, font_size=16)
        return fig

    fig = go.Figure()

    # Main T* surface
    fig.add_trace(go.Scatter3d(
        x=Co_flat[valid_T], y=Cr_flat[valid_T], z=Fe_flat[valid_T],
        mode='markers',
        marker=dict(
            size=4,
            color=T_flat[valid_T],
            colorscale='Magma',
            cmin=1000, cmax=3000,
            colorbar=dict(title="T* (K)", thickness=15, len=0.7),
            opacity=0.7
        ),
        name='T* Surface',
        hovertemplate="x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>T*=%{marker.color:.0f} K<extra></extra>"
    ))

    # Near melt pool overlay
    near_melt = np.abs(T_flat - T_laser) < 100
    if np.any(valid_T & near_melt):
        fig.add_trace(go.Scatter3d(
            x=Co_flat[valid_T & near_melt], 
            y=Cr_flat[valid_T & near_melt], 
            z=Fe_flat[valid_T & near_melt],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond', 
                       line=dict(width=2, color='white')),
            name=f'Near melt pool ({T_laser}K)',
            hovertemplate="⚠️ Near laser T<extra></extra>"
        ))

    # Near HAZ overlay
    near_haz = np.abs(T_flat - T_haz) < 100
    if np.any(valid_T & near_haz):
        fig.add_trace(go.Scatter3d(
            x=Co_flat[valid_T & near_haz], 
            y=Cr_flat[valid_T & near_haz], 
            z=Fe_flat[valid_T & near_haz],
            mode='markers',
            marker=dict(size=6, color='orange', symbol='square',
                       line=dict(width=1, color='white')),
            name=f'Near HAZ ({T_haz}K)',
            hovertemplate="⚠️ Phase transform in HAZ<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Composition-Dependent Transition Temperature T*(x)", font_size=14),
        scene=dict(
            xaxis=dict(title="x<sub>Co</sub>", range=[0, 1]),
            yaxis=dict(title="x<sub>Cr</sub>", range=[0, 1]),
            zaxis=dict(title="x<sub>Fe</sub>", range=[0, 1]),
            aspectmode='cube'
        ),
        height=650,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                   bgcolor="rgba(255,255,255,0.8)")
    )

    return fig


def plot_temperature_factors_am(D_liq, D_fcc, T_vals, lam_liq, lam_fcc, R=6):
    """
    Plot CPD temperature factors with AM thermal cycle overlay.

    Shows how each thermodynamic mode responds to temperature, with annotations
    for AM-relevant temperature regimes.

    Args:
        D_liq, D_fcc: Temperature factor matrices (n_T × R)
        T_vals: Temperature grid values
        lam_liq, lam_fcc: Component weights
        R: Number of components

    Returns:
        plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CPD Temperature Factors D[T,r] (LIQUID phase)', 
                       'Typical AM Thermal Cycle'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot weighted D factors for LIQUID
    for r in range(min(R, len(lam_liq))):
        weighted_D = lam_liq[r] * D_liq[:, r]
        fig.add_trace(
            go.Scatter(
                x=T_vals, y=weighted_D,
                mode='lines',
                name=f'r={r+1} (λ={lam_liq[r]:.3f})',
                line=dict(color=colors[r % len(colors)], width=2),
                legendgroup=f'liq_r{r+1}'
            ),
            row=1, col=1
        )

    # Add vertical lines for AM-relevant temperatures
    am_temps = {
        'Room T': 300,
        'Stress Relief': 800,
        'Fe Curie T': 1043,
        'HAZ Peak': 1400,
        'Solidus': 1600,
        'Liquidus': 2000,
        'Melt Pool': 2800,
    }

    for label, T_val in am_temps.items():
        if T_vals[0] <= T_val <= T_vals[-1]:
            fig.add_vline(x=T_val, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text=label, annotation_position="top left",
                         row=1, col=1)

    # AM thermal cycle (simplified)
    time_cycle = np.array([0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5])
    temp_cycle = np.array([300, 300, 2800, 2800, 1200, 1200, 800, 300])

    fig.add_trace(
        go.Scatter(
            x=time_cycle, y=temp_cycle,
            mode='lines+markers',
            name='AM Thermal Cycle',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black')
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=750,
        title_text="Temperature Factors + AM Thermal History",
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Weighted Factor λ·D[T,r]", row=1, col=1)
    fig.update_xaxes(title_text="Relative Time (a.u.)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)

    return fig


def plot_composition_sensitivity_am(A, B, C, lam, co_vals, cr_vals, fe_vals, R=6):
    """
    Plot composition sensitivity heatmaps for all three elements.

    Returns:
        plotly.graph_objects.Figure with subplots
    """
    from plotly.subplots import make_subplots

    sens_Co, sens_Cr, sens_Fe = compute_composition_sensitivity(
        A, B, C, lam, co_vals, cr_vals, fe_vals, R
    )

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Co Sensitivity', 'Cr Sensitivity', 'Fe Sensitivity'),
        horizontal_spacing=0.08
    )

    elements = [
        ('Co', co_vals, sens_Co, '#3498db'),
        ('Cr', cr_vals, sens_Cr, '#2ecc71'),
        ('Fe', fe_vals, sens_Fe, '#e74c3c')
    ]

    for idx, (elem, vals, sens, color) in enumerate(elements, 1):
        # Total sensitivity curve
        fig.add_trace(
            go.Scatter(
                x=vals, y=sens,
                mode='lines',
                name=f'{elem} Total',
                line=dict(color=color, width=3),
                showlegend=False
            ),
            row=1, col=idx
        )

        # Individual component contributions
        colors_r = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']
        for r in range(min(R, len(lam))):
            factor = A[:, r] if elem == 'Co' else (B[:, r] if elem == 'Cr' else C[:, r])
            contrib = np.abs(lam[r] * factor)
            c_min, c_max = np.min(contrib), np.max(contrib)
            if c_max > c_min:
                contrib = (contrib - c_min) / (c_max - c_min)

            fig.add_trace(
                go.Scatter(
                    x=vals, y=contrib,
                    mode='lines',
                    name=f'r={r+1}',
                    line=dict(color=colors_r[r], width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=(idx == 1)
                ),
                row=1, col=idx
            )

        fig.update_xaxes(title_text=f"x<sub>{elem}</sub>", row=1, col=idx)
        fig.update_yaxes(title_text="Normalized Sensitivity", row=1, col=idx)

    fig.update_layout(
        height=450,
        title_text="Composition Sensitivity Analysis",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    return fig


def plot_defect_susceptibility_3d(S_crack, valid_mask, co_vals, cr_vals, fe_vals,
                                   defect_type='hot_cracking'):
    """
    Plot defect susceptibility as 3D scatter with risk highlighting.

    Args:
        S_crack: Susceptibility array
        valid_mask: Valid simplex mask
        co_vals, cr_vals, fe_vals: Composition grid values
        defect_type: Type of defect for labeling

    Returns:
        plotly.graph_objects.Figure
    """
    res = S_crack.shape[0]
    x = np.linspace(0, 1, res)

    Co_flat, Cr_flat, Fe_flat, S_flat = [], [], [], []

    for i in range(res):
        for j in range(res):
            for k in range(res):
                if valid_mask[i,j,k] and np.isfinite(S_crack[i,j,k]):
                    Co_flat.append(x[i])
                    Cr_flat.append(x[j])
                    Fe_flat.append(x[k])
                    S_flat.append(S_crack[i,j,k])

    Co_flat = np.array(Co_flat)
    Cr_flat = np.array(Cr_flat)
    Fe_flat = np.array(Fe_flat)
    S_flat = np.array(S_flat)

    # Remove extreme outliers for visualization
    if len(S_flat) > 0:
        q99 = np.percentile(S_flat, 99)
        valid_S = S_flat < q99
    else:
        valid_S = np.array([], dtype=bool)

    if np.sum(valid_S) < 10:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Insufficient data for susceptibility map.",
                          xref="paper", yref="paper", showarrow=False, font_size=16)
        return fig

    colorscale = 'Reds' if defect_type == 'hot_cracking' else 'Viridis'
    cbar_title = "Cracking Susceptibility" if defect_type == 'hot_cracking' else "Susceptibility"
    threshold = np.percentile(S_flat[valid_S], 90) if np.sum(valid_S) > 0 else 1.0

    fig = go.Figure()

    # Main susceptibility scatter
    fig.add_trace(go.Scatter3d(
        x=Co_flat[valid_S], y=Cr_flat[valid_S], z=Fe_flat[valid_S],
        mode='markers',
        marker=dict(
            size=5,
            color=S_flat[valid_S],
            colorscale=colorscale,
            cmin=0, cmax=np.percentile(S_flat[valid_S], 95),
            colorbar=dict(title=cbar_title, thickness=15, len=0.7),
            opacity=0.7
        ),
        name='Susceptibility',
        hovertemplate=f"x_Co=%{{x:.3f}}<br>x_Cr=%{{y:.3f}}<br>x_Fe=%{{z:.3f}}<br>{cbar_title}=%{{marker.color:.3f}}<extra></extra>"
    ))

    # High risk overlay
    high_risk = S_flat > threshold
    if np.any(valid_S & high_risk):
        fig.add_trace(go.Scatter3d(
            x=Co_flat[valid_S & high_risk],
            y=Cr_flat[valid_S & high_risk],
            z=Fe_flat[valid_S & high_risk],
            mode='markers',
            marker=dict(size=8, color='red', symbol='x',
                       line=dict(width=2, color='white')),
            name='⚠️ High Risk',
            hovertemplate="HIGH RISK: Avoid for AM<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"AM Defect Susceptibility: {defect_type.replace('_', ' ').title()}", font_size=14),
        scene=dict(
            xaxis=dict(title="x<sub>Co</sub>", range=[0, 1]),
            yaxis=dict(title="x<sub>Cr</sub>", range=[0, 1]),
            zaxis=dict(title="x<sub>Fe</sub>", range=[0, 1]),
            aspectmode='cube'
        ),
        height=650,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def plot_segregation_heatmap(seg_matrix, x_vals, y_vals, x_label, y_label, title):
    """
    Plot segregation potential as 2D heatmap.

    Args:
        seg_matrix: 2D segregation potential array
        x_vals, y_vals: Axis values
        x_label, y_label: Axis labels
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=seg_matrix,
        x=x_vals,
        y=y_vals,
        colorscale='YlOrRd',
        colorbar=dict(title="Segregation Potential", thickness=15),
        hovertemplate=f"{x_label}=%{{x:.3f}}<br>{y_label}=%{{y:.3f}}<br>Potential=%{{z:.3f}}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, font_size=14),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        width=550
    )

    return fig


# =============================================
# STREAMLIT UI COMPONENTS FOR AM ANALYSIS
# =============================================



def validate_cpd_session_state():
    """
    Validate that CPD factors in session state have compatible dimensions.
    Returns (is_valid, error_message, adjusted_factors)
    """
    required_keys = ['A_liq', 'B_liq', 'C_liq', 'D_liq', 'lam_liq',
                     'A_fcc', 'B_fcc', 'C_fcc', 'D_fcc', 'lam_fcc']

    for key in required_keys:
        if key not in st.session_state:
            return False, f"Missing session state key: {key}", None

    A_liq = st.session_state['A_liq']
    A_fcc = st.session_state['A_fcc']
    B_liq = st.session_state['B_liq']
    B_fcc = st.session_state['B_fcc']
    C_liq = st.session_state['C_liq']
    C_fcc = st.session_state['C_fcc']
    D_liq = st.session_state['D_liq']
    D_fcc = st.session_state['D_fcc']
    lam_liq = st.session_state['lam_liq']
    lam_fcc = st.session_state['lam_fcc']

    # Check dimensions
    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)

    checks = [
        (A_liq.shape[1] == R_liq, f"A_liq cols ({A_liq.shape[1]}) != rank ({R_liq})"),
        (B_liq.shape[1] == R_liq, f"B_liq cols ({B_liq.shape[1]}) != rank ({R_liq})"),
        (C_liq.shape[1] == R_liq, f"C_liq cols ({C_liq.shape[1]}) != rank ({R_liq})"),
        (D_liq.shape[1] == R_liq, f"D_liq cols ({D_liq.shape[1]}) != rank ({R_liq})"),
        (A_fcc.shape[1] == R_fcc, f"A_fcc cols ({A_fcc.shape[1]}) != rank ({R_fcc})"),
        (B_fcc.shape[1] == R_fcc, f"B_fcc cols ({B_fcc.shape[1]}) != rank ({R_fcc})"),
        (C_fcc.shape[1] == R_fcc, f"C_fcc cols ({C_fcc.shape[1]}) != rank ({R_fcc})"),
        (D_fcc.shape[1] == R_fcc, f"D_fcc cols ({D_fcc.shape[1]}) != rank ({R_fcc})"),
    ]

    for check, msg in checks:
        if not check:
            return False, f"Dimension mismatch: {msg}", None

    # Check grid metadata
    if 'tdt_metadata' not in st.session_state:
        return False, "Missing tdt_metadata in session state", None

    meta = st.session_state['tdt_metadata']
    required_meta = ['co_vals', 'cr_vals', 'fe_vals', 'T_vals']
    for key in required_meta:
        if key not in meta:
            return False, f"Missing metadata key: {key}", None

    return True, "Valid", {
        'A_liq': A_liq, 'B_liq': B_liq, 'C_liq': C_liq, 'D_liq': D_liq, 'lam_liq': lam_liq,
        'A_fcc': A_fcc, 'B_fcc': B_fcc, 'C_fcc': C_fcc, 'D_fcc': D_fcc, 'lam_fcc': lam_fcc,
        'co_vals': meta['co_vals'], 'cr_vals': meta['cr_vals'], 
        'fe_vals': meta['fe_vals'], 'T_vals': meta['T_vals']
    }

def render_am_transition_surface_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                      D_liq, D_fcc, lam_liq, lam_fcc,
                                      co_vals, cr_vals, fe_vals, T_vals):
    """Render Streamlit UI for transition temperature surface analysis."""

    # Validate dimensions first
    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)
    R = min(R_liq, R_fcc, D_liq.shape[1], D_fcc.shape[1])

    if R < min(R_liq, R_fcc):
        st.warning(f"⚠️ Rank mismatch detected: LIQUID rank={R_liq}, FCC rank={R_fcc}. Using truncated rank R={R}.")

    st.subheader("🔥 Phase Transition Temperature Surface T*(x)")
    st.markdown(r"""
    **Physical meaning**: Temperature where $G_{LIQ} = G_{FCC}$ (melting/solidification point).  
    **AM relevance**: Predicts melt pool stability, solidification cracking susceptibility, 
    and optimal laser parameters for each composition.
    """)

    col1, col2 = st.columns(2)
    with col1:
        resolution = st.slider("Grid Resolution", 10, 35, 20, 
                              help="Higher = more accurate but slower (~seconds per point)")
    with col2:
        T_laser = st.slider("Laser Melt Pool T (K)", 2000, 3500, 2800)
        T_haz = st.slider("HAZ Temperature (K)", 800, 1800, 1200)

    if st.button("🔬 Compute T* Surface", use_container_width=True, type="primary"):
        with st.spinner(f"Solving for transition temperatures on {resolution}³ grid..."):
            try:
                T_melt, valid_mask, delta_G_grid = extract_transition_surface_from_cpd(
                    A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                    D_liq, D_fcc, lam_liq, lam_fcc,
                    co_vals, cr_vals, fe_vals, T_vals,
                    composition_grid_resolution=resolution
                )
            except Exception as e:
                st.error(f"❌ Error computing transition surface: {str(e)}")
                st.info("💡 Try re-running CPD with consistent rank for both phases.")
                return

            if T_melt is None:
                st.error("❌ Failed to compute transition surface. Check CPD factor dimensions.")
                return

            valid_count = np.sum(valid_mask & ~np.isnan(T_melt))
            if valid_count < 10:
                st.warning("⚠️ Too few valid transition points. Check CPD convergence or reduce resolution.")
                return

            # Flatten for stats
            T_valid = T_melt[valid_mask & ~np.isnan(T_melt)]

            st.success(f"✅ Computed {valid_count:,} valid T* points")

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean T*", f"{np.mean(T_valid):.0f} K")
            c2.metric("Std T*", f"{np.std(T_valid):.0f} K")
            c3.metric("Min T*", f"{np.min(T_valid):.0f} K")
            c4.metric("Max T*", f"{np.max(T_valid):.0f} K")

            # 3D plot
            fig = plot_transition_surface_3d(T_melt, valid_mask, co_vals, cr_vals, fe_vals,
                                              T_laser=T_laser, T_haz=T_haz)
            st.plotly_chart(fig, use_container_width=True)

            # Composition recommendations
            with st.expander("💡 AM Process Recommendations from T* Surface", expanded=True):
                st.markdown(f"""
                **Based on computed T* distribution:**

                | Metric | Value | AM Implication |
                |--------|-------|---------------|
                | Mean T* | {np.mean(T_valid):.0f} K | Typical laser power required |
                | T* range | {np.max(T_valid) - np.min(T_valid):.0f} K | Composition sensitivity of melting |
                | Near melt pool ({T_laser}K) | {np.sum(np.abs(T_valid - T_laser) < 100)} pts | Risk of incomplete melting |
                | Near HAZ ({T_haz}K) | {np.sum(np.abs(T_valid - T_haz) < 100)} pts | Risk of HAZ phase transforms |

                **Recommendations:**
                - Compositions with T* < {np.percentile(T_valid, 25):.0f} K: Use lower laser power, higher scan speed
                - Compositions with T* > {np.percentile(T_valid, 75):.0f} K: Use higher laser power, preheat substrate
                - Avoid compositions where |T* - {T_laser}| < 50 K: Unstable melt pool
                """)


def render_am_temperature_factors_tab(D_liq, D_fcc, T_vals, lam_liq, lam_fcc):
    """Render Streamlit UI for temperature factor analysis."""
    st.subheader("🌡️ Temperature Factor Analysis: AM Thermal Response")
    st.markdown(r"""
    **What this shows**: How each CPD component (r=1..R) responds to temperature.  
    **AM insight**: Components with strong gradients activate during rapid thermal cycling.

    | Factor Pattern | Thermodynamic Meaning | AM Process Stage |
    |---------------|----------------------|-----------------|
    | **Linear increase** (r=2) | Entropy term (-S·T) | Melt pool: liquid stabilization |
    | **Quadratic + kink** (r=3) | Cp + magnetic transition | HAZ: Fe Curie point effects (~1043K) |
    | **Oscillatory** (r=4-6) | Binary/ternary interactions | Solidification: segregation control |
    | **Constant offset** (r=1) | Baseline enthalpy | All stages: reference energy |
    """)

    phase_select = st.radio("Select Phase", ["LIQUID", "FCC", "Both"], index=0,
                           horizontal=True)

    if phase_select == "Both":
        fig = plot_temperature_factors_am(D_liq, D_fcc, T_vals, lam_liq, lam_fcc, R=len(lam_liq))
        st.plotly_chart(fig, use_container_width=True)
    else:
        D_use = D_liq if phase_select == "LIQUID" else D_fcc
        lam_use = lam_liq if phase_select == "LIQUID" else lam_fcc
        R = len(lam_use)

        fig = go.Figure()
        colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

        for r in range(R):
            weighted_D = lam_use[r] * D_use[:, r]
            fig.add_trace(go.Scatter(
                x=T_vals, y=weighted_D,
                mode='lines', name=f'r={r+1} (λ={lam_use[r]:.3f})',
                line=dict(color=colors[r % len(colors)], width=2)
            ))

        # Add AM temperature annotations
        am_temps = {
            'Fe Curie T': 1043,
            'Solidus': 1600,
            'Melt Pool': 2800,
        }
        for label, T_val in am_temps.items():
            if T_vals[0] <= T_val <= T_vals[-1]:
                fig.add_vline(x=T_val, line_dash="dash", line_color="gray", opacity=0.5,
                             annotation_text=label, annotation_position="top left")

        fig.update_layout(
            title=f"{phase_select} Phase Temperature Factors",
            xaxis_title="Temperature (K)",
            yaxis_title="Weighted Factor Value λ·D[T,r]",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 How to Interpret for AM Process Design"):
        st.markdown("""
        ### Practical AM Applications:

        1. **Laser parameter selection**: Compositions where r=2 (entropy) dominates at melt pool 
           temperatures need higher energy density to maintain liquid phase.

        2. **Cracking mitigation**: If r=3 (Cp/magnetic) has strong activation in the HAZ range 
           (800-1400K), avoid those compositions or use post-process stress relief at 800K.

        3. **Post-process heat treatment**: Target temperatures where unwanted factors (r=4-6) 
           deactivate to achieve homogeneous microstructure.

        4. **Multi-material AM**: Use temperature factor analysis to design composition gradients 
           that maintain phase stability across thermal gradients.
        """)


def render_am_sensitivity_tab(A, B, C, lam, co_vals, cr_vals, fe_vals):
    """Render Streamlit UI for composition sensitivity analysis."""
    st.subheader("🎯 Composition Sensitivity Analysis")
    st.markdown(r"""
    **Physical meaning**: How much does Gibbs energy change when you vary one element?  
    **AM relevance**:
    - 🔴 High sensitivity = tight powder blending tolerances needed
    - 🟢 Low sensitivity = robust to composition variations (recycled powder OK)
    - 📊 Peak locations = compositions where small changes cause phase transitions

    Sensitivity metric: $S(x_i) = \sum_r |\lambda_r \cdot F_r(x_i)|$
    """)

    R_select = st.slider("Number of CPD Components", 1, 6, 6)

    fig = plot_composition_sensitivity_am(A, B, C, lam, co_vals, cr_vals, fe_vals, R=R_select)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed element analysis
    st.subheader("Element-Specific Recommendations")

    sens_Co, sens_Cr, sens_Fe = compute_composition_sensitivity(
        A, B, C, lam, co_vals, cr_vals, fe_vals, R_select
    )

    cols = st.columns(3)
    elements_data = [
        ("Co", co_vals, sens_Co, "#3498db", 
         "Moderate, smooth sensitivity. Good for composition gradients. No sharp peaks = tolerant to powder mixing variations."),
        ("Cr", cr_vals, sens_Cr, "#2ecc71",
         "Peak sensitivity near x_Cr ≈ 0.15-0.25. Avoid for first builds. High at x_Cr > 0.35: requires precise blending."),
        ("Fe", fe_vals, sens_Fe, "#e74c3c",
         "Strong peak near x_Fe ≈ 0.20 from magnetic transition. Low at x_Fe < 0.10 for non-magnetic apps.")
    ]

    for col, (elem, vals, sens, color, advice) in zip(cols, elements_data):
        with col:
            st.markdown(f"**{elem} Sensitivity**")
            # Find peak
            peak_idx = np.argmax(sens)
            peak_val = vals[peak_idx]
            st.metric("Peak at", f"x_{elem} = {peak_val:.2f}")
            st.markdown(f"<span style='color:{color}'>{advice}</span>", unsafe_allow_html=True)


def render_am_defect_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                         D_liq, D_fcc, lam_liq, lam_fcc,
                         co_vals, cr_vals, fe_vals, T_vals):
    """Render Streamlit UI for defect susceptibility analysis."""
    st.subheader("⚠️ Defect Susceptibility Analysis")
    st.markdown(r"""
    **Theory**: Hot cracking susceptibility combines two CPD-derived metrics:

    $$S_{crack}[x] = |
abla_x T^*(x)| 	imes |d(\Delta G)/dT|^{-1}_{T=T^*}$$

    - **Large |∇T*|** = composition-sensitive solidification range
    - **Small |d(ΔG)/dT|** = shallow Gibbs energy curve (unstable phase boundary)

    **Higher S_crack = wider solidification range = higher cracking risk**
    """)

    defect_type = st.selectbox("Defect Type", 
                              ["hot_cracking", "segregation", "porosity"],
                              format_func=lambda x: x.replace('_', ' ').title())

    resolution = st.slider("Grid Resolution", 10, 25, 15,
                          help="Lower resolution recommended for susceptibility (faster)")

    if st.button("🔬 Compute Susceptibility Map", use_container_width=True, type="primary"):
        with st.spinner("Computing susceptibility metric..."):
            if defect_type == "hot_cracking":
                S_defect, T_melt, valid_mask = compute_hot_cracking_susceptibility(
                    A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                    D_liq, D_fcc, lam_liq, lam_fcc,
                    co_vals, cr_vals, fe_vals, T_vals,
                    composition_grid_resolution=resolution
                )

                fig = plot_defect_susceptibility_3d(S_defect, valid_mask, co_vals, cr_vals, fe_vals,
                                                      defect_type=defect_type)
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                S_valid = S_defect[valid_mask & np.isfinite(S_defect)]
                if len(S_valid) > 0:
                    st.markdown(f"""
                    **Hot Cracking Susceptibility Summary:**
                    - Valid points: {len(S_valid):,}
                    - Mean susceptibility: {np.mean(S_valid):.3f}
                    - High-risk threshold (90th percentile): {np.percentile(S_valid, 90):.3f}
                    - High-risk compositions: {np.sum(S_valid > np.percentile(S_valid, 90))} ({100*np.sum(S_valid > np.percentile(S_valid, 90))/len(S_valid):.1f}%)
                    """)

                    # Recommendations
                    st.info("""
                    **AM Design Recommendations:**
                    - 🟢 **Low susceptibility (S < 0.3)**: Safe for all AM processes
                    - 🟡 **Moderate (0.3 < S < 0.7)**: Use controlled cooling, consider preheat
                    - 🔴 **High (S > 0.7)**: Avoid for critical applications; use hybrid manufacturing
                    """)

            elif defect_type == "segregation":
                seg_CoCr, seg_CoFe, seg_CrFe = compute_segregation_potential(
                    A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                    lam_liq, lam_fcc, R=len(lam_liq)
                )

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = plot_segregation_heatmap(seg_CoCr, co_vals, cr_vals, 
                                                    "x_Co", "x_Cr", "Co-Cr Segregation")
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = plot_segregation_heatmap(seg_CrFe, cr_vals, fe_vals,
                                                    "x_Cr", "x_Fe", "Cr-Fe Segregation")
                    st.plotly_chart(fig2, use_container_width=True)

                st.info("""
                **Segregation Analysis:**
                - Red regions = strong binary interaction = tendency for element partitioning
                - During rapid solidification, high-segregation compositions form inhomogeneous microstructures
                - Recommendation: Avoid peak segregation regions or use ultra-fast cooling (>10⁶ K/s)
                """)



# =============================================
# STREAMLIT APP: HEADER & SIDEBAR
# =============================================
st.title("🔷 Co-Cr-Fe-Ni Phase Stability Explorer v2")
st.markdown(r"""
**Single-Temperature Phase Comparison with Temperature-Driven Shape Morphing.**  

🔹 **FCC surfaces** become **crystalline & faceted** at low T (enthalpy-dominated regime)  
🔹 **LIQUID surfaces** become **fluid & expanded** at high T (entropy-dominated regime)  
🔹 **ΔG = 0 boundary** (gold) marks the exact phase transition frontier  

*Data: 31 temperatures (700-3300K), ~170K compositions each, CALPHAD-computed Gibbs energies*
""")

with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # --- PRESET VIEWS ---
    st.subheader("⚡ Quick Presets")
    preset = st.selectbox("Load Preset", [
        "Custom", 
        "Low-T FCC Crystal (700-1000K)", 
        "High-T Liquid Melt (2200-3300K)", 
        "Transition Region (1400-1600K)", 
        "Maximum Contrast"
    ], index=0)
    
    # --- TEMPERATURE SELECTION ---
    st.subheader("🌡️ Temperature")
    if preset == "Low-T FCC Crystal (700-1000K)":
        default_T = min(T for T in T_list if T <= 1000) if any(T <= 1000 for T in T_list) else T_min
    elif preset == "High-T Liquid Melt (2200-3300K)":
        default_T = max(T for T in T_list if T >= 2200) if any(T >= 2200 for T in T_list) else T_max
    elif preset == "Transition Region (1400-1600K)":
        default_T = min(T_list, key=lambda T: abs(T - 1500))
    else:
        default_T = T_list[len(T_list)//2] if T_list else 1500
    
    T_val = st.select_slider("T (K)", options=T_list, value=default_T)
    T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5
    
    # Predict expected phase based on temperature regime
    if T_factor < 0.3:
        phase_expected = "FCC (enthalpy-dominated)"
    elif T_factor > 0.7:
        phase_expected = "LIQUID (entropy-dominated)"
    else:
        phase_expected = "Transition (composition-dependent)"
    
    st.info(f"T = {T_val}K | Expected regime: **{phase_expected}**")
    
    st.divider()
    
    # --- COMPOSITION QUERY ---
    st.subheader("📍 Query Composition")
    col1, col2 = st.columns(2)
    with col1:
        q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
        q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    with col2:
        q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    
    comp_sum = q_co + q_cr + q_fe
    q_ni = 1.0 - comp_sum
    
    if comp_sum > 1.0:
        st.error(f"⚠️ Sum = {comp_sum:.2f} > 1.0 (invalid composition)")
    elif q_ni < 0:
        st.error(f"⚠️ x_Ni = {q_ni:.2f} < 0 (invalid composition)")
    
    eval_query = st.button("🔍 Evaluate Phase Stability", use_container_width=True, type="primary")
    
    st.divider()
    
    # --- VISUALIZATION MODES ---
    st.subheader("🎨 Visualization Mode")
    mode_options = [
        "Phase Boundary (Scientific)",
        "Dual SH Surfaces (Temperature Morph)",
        "ΔG Difference Surface",
        "Ternary Flat Projection",
        "Markers (Distinct Shapes)",
        "Animated Temperature Sweep"
    ]
    
    if not SCIPY_AVAILABLE:
        mode_options = [m for m in mode_options if "SH" not in m and "Difference" not in m and "Animated" not in m]
        st.warning("⚠️ SciPy missing: Advanced visualization modes disabled")
    
    render_mode = st.radio("Mode", mode_options, index=1 if SCIPY_AVAILABLE else 0, 
                          help="Select visualization style for phase stability")
    
    # --- MODE-SPECIFIC CONTROLS ---
    if render_mode == "Phase Boundary (Scientific)":
        st.subheader("🔧 Scientific Settings")
        grid_res = st.slider("Grid Resolution", 15, 80, 35, step=5, 
                            help="Higher = more points, slower rendering")
        boundary_threshold = st.slider("Boundary Width (J/mol)", 10, 300, 60, 10,
                                      help="ΔG tolerance for phase boundary identification")
        show_phase_volume = st.toggle("Show Phase Volume", value=True)
        volume_opacity = st.slider("Volume Opacity", 0.05, 0.6, 0.12, 0.05)
        volume_size = st.slider("Volume Point Size", 1, 8, 2)
        show_uncertainty = st.toggle("Fade Uncertain Regions", value=True,
                                    help="Reduce opacity for points far from CALPHAD data")
        uncertainty_fade = st.slider("Fade Strength", 0.0, 1.0, 0.6, 0.1)
        show_simplex = st.toggle("Show Simplex Frame", value=True)
        show_slice = st.toggle("Show Cross-Section Plane", value=False)
        slice_ni = st.slider("Slice x_Ni", 0.0, 1.0, 0.25, 0.05) if show_slice else 0.25
        
    elif render_mode == "Dual SH Surfaces (Temperature Morph)" and SCIPY_AVAILABLE:
        st.subheader("🔧 SH Morph Settings")
        
        # Preset-based defaults
        if "Low-T" in preset:
            sh_R_fixed, sh_l_max, liq_opacity, fcc_opacity = 0.45, 5, 0.35, 0.85
        elif "High-T" in preset:
            sh_R_fixed, sh_l_max, liq_opacity, fcc_opacity = 0.65, 2, 0.85, 0.25
        elif "Transition" in preset:
            sh_R_fixed, sh_l_max, liq_opacity, fcc_opacity = 0.50, 4, 0.70, 0.70
        else:
            sh_R_fixed, sh_l_max, liq_opacity, fcc_opacity = 0.50, 3, 0.60, 0.45
        
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, sh_R_fixed, 0.05)
        
        # Auto l_max based on temperature (physical prior: liquid smooths at high T)
        l_max_liq = max(1, int(sh_l_max - 1.5 * T_factor))
        l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_factor)))
        
        st.markdown(f"**Auto l_max:** LIQUID l={l_max_liq} (smooth), FCC l={l_max_fcc} (faceted)")
        
        sh_l_max_override = st.slider("Override l_max (base)", 1, 8, sh_l_max)
        if sh_l_max_override != sh_l_max:
            l_max_liq = max(1, int(sh_l_max_override - 1.5 * T_factor))
            l_max_fcc = max(2, int(sh_l_max_override + 1.0 * (1.0 - T_factor)))
        
        sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
        sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)
        liq_opacity = st.slider("LIQUID Opacity", 0.1, 1.0, liq_opacity, 0.05)
        fcc_opacity = st.slider("FCC Opacity", 0.1, 1.0, fcc_opacity, 0.05)
        show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
        show_data_density = st.toggle("Show Data Coverage", value=False)
        
        st.markdown("""
        <small>
        <b>Physical interpretation:</b><br>
        🔹 Low T: FCC = rigid crystalline ripples; LIQUID = small, faint<br>
        🔹 High T: LIQUID = expanded fluid surface; FCC = shrunk, matte<br>
        🔹 Gold contour = exact phase boundary (ΔG = 0)
        </small>
        """, unsafe_allow_html=True)
        
    elif render_mode == "ΔG Difference Surface" and SCIPY_AVAILABLE:
        st.subheader("🔧 ΔG Surface Settings")
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
        sh_l_max = st.slider("Max Harmonic Degree", 1, 8, 4)
        sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
        sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)
        dg_scale = st.slider("ΔG Deformation Scale", 0.001, 0.15, 0.025, 0.001,
                            help="Amplitude of surface deformation by driving force")
        show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
        
    elif render_mode == "Ternary Flat Projection":
        st.subheader("🔧 Ternary Settings")
        flat_color_by = st.radio("Color By", 
                                ["Stable Phase", "ΔG (diverging)", "G_magnitude", "Data Proximity"], 
                                index=1)
        flat_marker_size = st.slider("Marker Size", 2, 20, 7)
        flat_opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
        show_ternary_grid = st.toggle("Grid Lines", value=True)
        show_uncertainty = st.toggle("Fade Distant Points", value=True)
        
    elif render_mode == "Markers (Distinct Shapes)":
        st.subheader("🔧 Marker Settings")
        grid_res = st.slider("Grid Resolution", 15, 100, 35, step=5)
        marker_size = st.slider("Marker Size", 1, 12, 4)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
        show_phase = st.radio("Display", ["Stable Phase Only", "Both Phases (Distinct)"], index=1)
        cmap = st.selectbox("Colormap", COLORMAPS, 
                           index=COLORMAPS.index("RdBu_r") if "RdBu_r" in COLORMAPS else 0)
        show_boundary = st.toggle("Show ΔG≈0 Boundary", value=True)
        show_uncertainty = st.toggle("Fade Distant Points", value=True)
        
    else:  # Animated Temperature Sweep
        st.subheader("🔧 Animation Settings")
        anim_start = st.select_slider("Start T", options=T_list, value=T_min)
        anim_end = st.select_slider("End T", options=T_list, value=T_max)
        anim_frames = st.slider("Frames", 3, min(20, len(T_list)), min(8, len(T_list)))
        anim_mode = st.radio("Animation Style", ["Dual SH Morph", "ΔG Surface Morph"], index=0)
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
        sh_l_max = st.slider("l_max", 1, 6, 3)
        sh_n_theta = st.slider("Resolution", 30, 100, 50, step=10)
    
    st.divider()
    
    # --- GLOBAL OVERLAYS ---
    st.subheader("🔷 Overlays")
    show_axes_frame = st.toggle("Coordinate Axes", value=True)
    show_query_probe = st.toggle("Query Probe Sphere", value=True)
    show_comp_path = st.toggle("Show Composition Path", value=False,
                              help="Connects multiple query points with temperature-colored line")
    
    st.divider()
    st.subheader("✏️ Layout")
    template = st.selectbox("Template", ["plotly_white", "plotly_dark", "seaborn", "simple_white"], index=0)
    bg_color = st.color_picker("Background", "#ffffff")
    title_font = st.slider("Title Font", 12, 24, 16)
    
    st.divider()
    st.caption(f"📊 Data: {len(T_list)} temperatures ({T_min}-{T_max}K) | {len(df):,} total measurements")

# =============================================
# SESSION STATE FOR QUERY HISTORY
# =============================================
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# =============================================
# QUERY EVALUATION LOGIC
# =============================================
query_result = None
if eval_query:
    if comp_sum > 1.0 or q_ni < 0:
        st.error("❌ Invalid composition: sum must equal 1.0 with all xᵢ ≥ 0")
    else:
        interp_liq_q, interp_fcc_q = build_interpolators_for_T(df, T_val)
        if interp_liq_q is None:
            st.error(f"❌ No data available for T={T_val}K")
        else:
            pt = np.array([[q_co, q_cr, q_fe]])
            g_liq_q = float(interp_liq_q(pt)[0])
            g_fcc_q = float(interp_fcc_q(pt)[0])
            
            if np.isnan(g_liq_q) or np.isnan(g_fcc_q):
                st.error("❌ Query point outside CALPHAD data convex hull")
            else:
                g_stable_q = min(g_liq_q, g_fcc_q)
                phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"
                dG_q = g_liq_q - g_fcc_q
                
                query_result = {
                    "T": T_val, 
                    "Co": q_co, "Cr": q_cr, "Fe": q_fe, "Ni": round(q_ni, 3),
                    "G_LIQ": g_liq_q, "G_FCC": g_fcc_q,
                    "G_stable": g_stable_q, 
                    "Phase": phase_q, 
                    "dG": dG_q
                }
                
                # Add to history (keep last 10)
                st.session_state.query_history.append(query_result)
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history.pop(0)
                
                # Display results
                st.success(f"✅ Query evaluated at T={T_val}K")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("G_LIQ", f"{g_liq_q:,.0f}", "J/mol")
                c2.metric("G_FCC", f"{g_fcc_q:,.0f}", "J/mol")
                
                # Color-code ΔG: red for LIQUID-favored, blue for FCC-favored
                delta_color = "inverse" if dG_q < 0 else "normal"
                c3.metric("ΔG", f"{dG_q:,.0f}", "J/mol", delta_color=delta_color)
                c4.metric("Stable Phase", phase_q)
                c5.metric("|ΔG|", f"{abs(dG_q):,.0f}", "J/mol", 
                         help="Magnitude of driving force for phase transformation")
                
                st.divider()

# =============================================
# MAIN VISUALIZATION TABS
# =============================================
tab_main, tab_tensor, tab_am = st.tabs(["🎨 Phase Visualization", "📊 Tensor Decomposition (CPD)", "🏭 AM Design Assistant"])

with tab_main:
    # Build interpolators for current temperature
    interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
    if interp_liq is None:
        st.error(f"❌ No interpolator available for T={T_val}K")
        st.stop()
    
    fig = go.Figure()
    
    # ------------------------------------------------------------------
    # MODE 1: PHASE BOUNDARY (SCIENTIFIC) - Most accurate for research
    # ------------------------------------------------------------------
    if render_mode == "Phase Boundary (Scientific)":
        pts = generate_tetrahedral_grid(grid_res)
        G_liq = interp_liq(pts)
        G_fcc = interp_fcc(pts)
        valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        
        pts = pts[valid]
        G_liq = G_liq[valid]
        G_fcc = G_fcc[valid]
        dG = G_liq - G_fcc
        stable = np.where(dG <= 0, "LIQUID", "FCC")
        
        # Uncertainty metric: proximity to CALPHAD data points
        if show_uncertainty and HULL_AVAILABLE:
            proximity = compute_data_proximity(pts, all_pts, max_dist=0.2)
        else:
            proximity = np.ones(len(pts))
        
        # Plot phase volumes with distinct symbols
        for phase in ["LIQUID", "FCC"]:
            mask = stable == phase
            if mask.sum() == 0:
                continue
            p_pts = pts[mask]
            p_prox = proximity[mask]
            p_dG = dG[mask]
            
            fig.add_trace(go.Scatter3d(
                x=p_pts[:, 0], y=p_pts[:, 1], z=p_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=volume_size,
                    color=PHASE_COLORS[phase],
                    symbol=PHASE_SYMBOLS[phase],
                    opacity=volume_opacity * p_prox,
                    line=dict(width=0.5, color="white")
                ),
                name=f"{phase} Region",
                hovertemplate=(f"<b>{phase}</b><br>" +
                               "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                               "ΔG=%{customdata:.0f} J/mol<extra></extra>"),
                customdata=p_dG
            ))
        
        # Highlight phase boundary
        boundary_pts, boundary_dG = find_phase_boundary_points(pts, dG, boundary_threshold)
        if len(boundary_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
                mode="markers",
                marker=dict(size=5, color=PHASE_COLORS["BOUNDARY"], symbol="x",
                            line=dict(width=2, color="#b7950b")),
                name="ΔG = 0 Boundary",
                hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
            ))
        
        # Optional cross-section plane at fixed Ni
        if show_slice:
            plane_res = 30
            p_co = np.linspace(0, 1-slice_ni, plane_res)
            p_cr = np.linspace(0, 1-slice_ni, plane_res)
            PCO, PCR = np.meshgrid(p_co, p_cr)
            PFE = (1 - slice_ni) - PCO - PCR
            valid_plane = PFE >= 0
            PCO, PCR, PFE = PCO[valid_plane], PCR[valid_plane], PFE[valid_plane]
            
            fig.add_trace(go.Scatter3d(
                x=PCO, y=PCR, z=PFE,
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.15, symbol="square"),
                name=f"Slice x_Ni={slice_ni:.2f}",
                hoverinfo="skip"
            ))
        
        # Simplex frame for orientation
        if show_simplex:
            edges = [
                [(1,0,0),(0,1,0)], [(1,0,0),(0,0,1)], [(1,0,0),(0,0,0)],
                [(0,1,0),(0,0,1)], [(0,1,0),(0,0,0)], [(0,0,1),(0,0,0)]
            ]
            for e in edges:
                fig.add_trace(go.Scatter3d(
                    x=[e[0][0], e[1][0]], y=[e[0][1], e[1][1]], z=[e[0][2], e[1][2]],
                    mode="lines", line=dict(color="black", width=3),
                    hoverinfo="skip", showlegend=False
                ))
            vertices = [(1,0,0,"Co"), (0,1,0,"Cr"), (0,0,1,"Fe"), (0,0,0,"Ni")]
            for vx, vy, vz, vl in vertices:
                fig.add_trace(go.Scatter3d(
                    x=[vx], y=[vy], z=[vz], mode="text", text=[vl],
                    textposition="top center", textfont=dict(size=14, color="black"),
                    hoverinfo="skip", showlegend=False
                ))
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # ------------------------------------------------------------------
    # MODE 2: DUAL SH SURFACES (TEMPERATURE MORPH) - Aesthetic + Physical
    # ------------------------------------------------------------------
    elif render_mode == "Dual SH Surfaces (Temperature Morph)" and SCIPY_AVAILABLE:
        TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
            interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
        )
        
        # Pre-check: ensure we have valid interpolated data before fitting
        g_liq_raw = interp_liq(sphere_pts).reshape(TH.shape)
        g_fcc_raw = interp_fcc(sphere_pts).reshape(TH.shape)

        n_valid_liq = np.sum(~np.isnan(g_liq_raw))
        n_valid_fcc = np.sum(~np.isnan(g_fcc_raw))
        n_needed_liq = (l_max_liq + 1) ** 2
        n_needed_fcc = (l_max_fcc + 1) ** 2

        if n_valid_liq < n_needed_liq or n_valid_fcc < n_needed_fcc:
            st.warning(f"⚠️ Insufficient valid interpolation points for SH fitting. LIQUID: {n_valid_liq} valid (need ≥{n_needed_liq}) | FCC: {n_valid_fcc} valid (need ≥{n_needed_fcc}). Reduce Base Radius or increase resolution.")
            l_max_liq = max(1, int(np.floor(np.sqrt(n_valid_liq)) - 1)) if n_valid_liq > 0 else 1
            l_max_fcc = max(1, int(np.floor(np.sqrt(n_valid_fcc)) - 1)) if n_valid_fcc > 0 else 1
            st.info(f"Auto-reduced l_max: LIQUID l={l_max_liq}, FCC l={l_max_fcc}")

        # Fit spherical harmonics for each phase
        coeffs_liq, l_max_liq = fit_sh_coeffs(TH, PH, g_liq_raw, l_max=l_max_liq)
        coeffs_fcc, l_max_fcc = fit_sh_coeffs(TH, PH, g_fcc_raw, l_max=l_max_fcc)
        
        if coeffs_liq is not None and coeffs_fcc is not None:
            G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
            G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)
            
            # === TEMPERATURE-DRIVEN SHAPE MORPHING ===
            R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_factor)
            X_liq = R_liq * np.sin(PH) * np.cos(TH)
            Y_liq = R_liq * np.sin(PH) * np.sin(TH)
            Z_liq = R_liq * np.cos(PH)
            
            # LIQUID: fluid, shiny, expanded at high T
            fig.add_trace(go.Surface(
                x=X_liq, y=Y_liq, z=Z_liq,
                surfacecolor=G_liq_sh,
                colorscale="Reds",
                cmin=G_global_min, cmax=G_global_max,
                opacity=liq_opacity,
                name=f"LIQUID (l={l_max_liq}, fluid)",
                showscale=False,
                hovertemplate=f"<b>LIQUID</b><br>G=%{{surfacecolor:,.0f}} J/mol<br>T={T_val}K<extra></extra>",
                lighting=dict(ambient=0.55, diffuse=0.6, roughness=0.12, specular=0.9),
                lightposition=dict(x=100, y=100, z=50)
            ))
            
            R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_factor)
            X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
            Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
            Z_fcc = R_fcc * np.cos(PH)
            
            # FCC: crystalline, matte, faceted at low T
            fig.add_trace(go.Surface(
                x=X_fcc, y=Y_fcc, z=Z_fcc,
                surfacecolor=G_fcc_sh,
                colorscale="Blues",
                cmin=G_global_min, cmax=G_global_max,
                opacity=fcc_opacity,
                name=f"FCC (l={l_max_fcc}, crystal)",
                showscale=False,
                hovertemplate=f"<b>FCC</b><br>G=%{{surfacecolor:,.0f}} J/mol<br>T={T_val}K<extra></extra>",
                contours=dict(
                    x=dict(show=True, color="#1a5276", width=1.2, highlight=False),
                    y=dict(show=True, color="#1a5276", width=1.2, highlight=False),
                    z=dict(show=True, color="#1a5276", width=1.2, highlight=False)
                ),
                lighting=dict(ambient=0.65, diffuse=0.4, roughness=0.78, specular=0.15)
            ))
            
            # ΔG = 0 contour (phase boundary)
            if show_dg_contour:
                cx, cy, cz = extract_dg_zero_contour(TH, PH, dG_grid, sh_R_fixed)
                if len(cx) > 10:
                    fig.add_trace(go.Scatter3d(
                        x=cx, y=cy, z=cz,
                        mode="lines+markers",
                        line=dict(color=PHASE_COLORS["BOUNDARY"], width=5),
                        marker=dict(size=3, color="#f39c12", symbol="diamond"),
                        name="ΔG = 0 Transition",
                        hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                    ))
            
            # Optional: show CALPHAD data density on sphere
            if show_data_density:
                df_T = df[df["T"] == T_val]
                if len(df_T) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=df_T["Co"], y=df_T["Cr"], z=df_T["Fe"],
                        mode="markers",
                        marker=dict(size=3, color="black", symbol="cross", opacity=0.4),
                        name="CALPHAD Data Points",
                        hovertemplate="Data: Co=%{x:.3f} Cr=%{y:.3f} Fe=%{z:.3f}<extra></extra>"
                    ))
        else:
            st.warning("⚠️ Spherical harmonic fitting failed. Try reducing Base Radius to ≤0.55, increasing resolution, or reducing l_max.")
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # ------------------------------------------------------------------
    # MODE 3: ΔG DIFFERENCE SURFACE - Driving force visualization
    # ------------------------------------------------------------------
    elif render_mode == "ΔG Difference Surface" and SCIPY_AVAILABLE:
        TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
            interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
        )
        
        # Pre-check: ensure sufficient valid data points
        n_valid_dg = np.sum(~np.isnan(dG_grid))
        n_needed = (sh_l_max + 1) ** 2

        if n_valid_dg < n_needed:
            st.warning(f"⚠️ Insufficient valid points for ΔG SH fitting: {n_valid_dg} valid (need ≥{n_needed}). Auto-reducing l_max.")
            sh_l_max = max(1, int(np.floor(np.sqrt(n_valid_dg)) - 1)) if n_valid_dg > 0 else 1

        coeffs_dG, l_max = fit_sh_coeffs(TH, PH, dG_grid, l_max=sh_l_max)
        if coeffs_dG is not None:
            dG_smooth = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max)
            
            # Temperature-modulated deformation amplitude
            T_deform = 1.0 + 0.2 * T_factor
            radius = sh_R_fixed * T_deform + dg_scale * dG_smooth
            radius = np.clip(radius, 0.1, 2.0)  # Prevent extreme deformations
            
            X = radius * np.sin(PH) * np.cos(TH)
            Y = radius * np.sin(PH) * np.sin(TH)
            Z = radius * np.cos(PH)
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=dG_smooth,
                colorscale="RdBu_r",
                cmin=-dG_global_abs_max, cmax=dG_global_abs_max,
                opacity=0.9,
                name="ΔG Surface",
                colorbar=dict(
                    title=dict(text="ΔG = G_LIQ - G_FCC (J/mol)", font=dict(size=12)),
                    thickness=20, len=0.7
                ),
                hovertemplate="<b>ΔG Surface</b><br>ΔG=%{surfacecolor:,.0f} J/mol<extra></extra>",
                contours=dict(
                    z=dict(show=True, highlightcolor=PHASE_COLORS["BOUNDARY"], highlightwidth=3,
                           project=dict(z=True), usecolormap=False, color=PHASE_COLORS["BOUNDARY"])
                )
            ))
            
            if show_dg_contour:
                cx, cy, cz = extract_dg_zero_contour(TH, PH, dG_grid, sh_R_fixed)
                if len(cx) > 10:
                    fig.add_trace(go.Scatter3d(
                        x=cx, y=cy, z=cz,
                        mode="lines+markers",
                        line=dict(color=PHASE_COLORS["BOUNDARY"], width=5),
                        marker=dict(size=4, color="#f39c12", symbol="diamond"),
                        name="ΔG = 0",
                        hovertemplate="<b>BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                    ))
            
            st.info("""
            **Reading the ΔG Surface:**  
            🔴 **Red / dented inward** → LIQUID stable (ΔG < 0)  
            🔵 **Blue / bulged outward** → FCC stable (ΔG > 0)  
            🟡 **Gold contour** → ΔG = 0 phase boundary  
            Deformation amplitude = magnitude of driving force for phase transformation
            """)
        else:
            st.warning("⚠️ SH fitting failed for ΔG. Try reducing l_max or increasing resolution.")
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # ------------------------------------------------------------------
    # MODE 4: TERNARY FLAT PROJECTION - Traditional materials view
    # ------------------------------------------------------------------
    elif render_mode == "Ternary Flat Projection":
        pts = generate_tetrahedral_grid(40)
        G_liq = interp_liq(pts)
        G_fcc = interp_fcc(pts)
        valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        pts = pts[valid]
        G_liq = G_liq[valid]
        G_fcc = G_fcc[valid]
        dG = G_liq - G_fcc
        stable = np.where(dG <= 0, "LIQUID", "FCC")
        z_data = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]  # Ni = 1 - (Co+Cr+Fe)
        
        if flat_color_by == "Stable Phase":
            colors = [PHASE_COLORS[p] for p in stable]
            show_cbar = False
        elif flat_color_by == "ΔG (diverging)":
            colors = dG
            show_cbar = True
            cbar_title = "ΔG (J/mol)"
            cmin, cmax = -dG_global_abs_max, dG_global_abs_max
        elif flat_color_by == "G_magnitude":
            colors = np.minimum(G_liq, G_fcc)
            show_cbar = True
            cbar_title = "G_stable (J/mol)"
            cmin, cmax = G_global_min, G_global_max
        else:  # Data Proximity
            if HULL_AVAILABLE:
                colors = compute_data_proximity(pts, all_pts, max_dist=0.2)
            else:
                colors = np.ones(len(pts))
            show_cbar = True
            cbar_title = "Data Proximity"
            cmin, cmax = 0, 1
        
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=z_data,
            mode="markers",
            marker=dict(
                size=flat_marker_size,
                color=colors,
                colorscale="RdBu_r" if flat_color_by == "ΔG (diverging)" else None,
                cmin=cmin if show_cbar else None,
                cmax=cmax if show_cbar else None,
                opacity=flat_opacity,
                symbol=[PHASE_SYMBOLS[p] for p in stable],
                line=dict(width=0.5, color="white")
            ),
            name="Ternary View",
            hovertemplate="x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Ni=%{z:.3f}<br>Phase=%{text}<extra></extra>",
            text=stable
        ))
        
        if show_ternary_grid:
            for ni in [0.0, 0.25, 0.5, 0.75]:
                mask = np.abs(z_data - ni) < 0.02
                if mask.sum() > 10:
                    fig.add_trace(go.Scatter3d(
                        x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                        mode="markers", marker=dict(size=1, color="gray", opacity=0.3),
                        hoverinfo="skip", showlegend=False
                    ))
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Ni</sub>"
    
    # ------------------------------------------------------------------
    # MODE 5: MARKERS (DISTINCT SHAPES) - Classic scatter plot
    # ------------------------------------------------------------------
    elif render_mode == "Markers (Distinct Shapes)":
        pts = generate_tetrahedral_grid(grid_res)
        G_liq = interp_liq(pts)
        G_fcc = interp_fcc(pts)
        valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        pts = pts[valid]
        G_liq = G_liq[valid]
        G_fcc = G_fcc[valid]
        G_stable = np.minimum(G_liq, G_fcc)
        stable = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
        dG = G_liq - G_fcc
        
        if show_uncertainty and HULL_AVAILABLE:
            proximity = compute_data_proximity(pts, all_pts, max_dist=0.2)
        else:
            proximity = np.ones(len(pts))
        
        if show_phase == "Stable Phase Only":
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=G_stable,
                    colorscale=cmap,
                    opacity=opacity * proximity,
                    line=dict(width=1, color="white")
                ),
                name="Stable Phase",
                hovertemplate="<b>%{text}</b><br>G=%{marker.color:,.0f} J/mol<extra></extra>",
                text=stable
            ))
        else:
            for phase in ["LIQUID", "FCC"]:
                if phase == "LIQUID":
                    mask = G_liq <= G_fcc
                    g_vals = G_liq[mask]
                else:
                    mask = G_fcc < G_liq
                    g_vals = G_fcc[mask]
                
                if mask.sum() == 0:
                    continue
                
                p_pts = pts[mask]
                p_prox = proximity[mask]
                
                fig.add_trace(go.Scatter3d(
                    x=p_pts[:, 0], y=p_pts[:, 1], z=p_pts[:, 2],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=PHASE_COLORS[phase],
                        symbol=PHASE_SYMBOLS[phase],
                        opacity=opacity * p_prox,
                        line=dict(width=1, color="white")
                    ),
                    name=f"{phase} Phase",
                    hovertemplate=(f"<b>{phase}</b><br>" +
                                   "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                                   f"G={phase}=%{{marker.color:,.0f}} J/mol<extra></extra>"),
                    customdata=g_vals
                ))
            
            if show_boundary:
                boundary_mask = np.abs(dG) < 100
                if boundary_mask.sum() > 0:
                    fig.add_trace(go.Scatter3d(
                        x=pts[boundary_mask, 0], y=pts[boundary_mask, 1], z=pts[boundary_mask, 2],
                        mode="markers",
                        marker=dict(size=6, color=PHASE_COLORS["BOUNDARY"], symbol="x",
                                    line=dict(width=2, color="#b7950b")),
                        name="ΔG ≈ 0 Boundary",
                        hovertemplate="<b>BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                    ))
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # ------------------------------------------------------------------
    # MODE 6: ANIMATED TEMPERATURE SWEEP - Dynamic phase evolution
    # ------------------------------------------------------------------
    elif render_mode == "Animated Temperature Sweep" and SCIPY_AVAILABLE:
        T_frames = np.linspace(anim_start, anim_end, anim_frames)
        T_frames = [T_list[np.argmin(np.abs(np.array(T_list) - t))] for t in T_frames]
        T_frames = sorted(list(set(T_frames)))
        
        if len(T_frames) < 2:
            st.warning("⚠️ Need at least 2 distinct temperatures for animation")
        else:
            frames = []
            for T_frame in T_frames:
                interp_liq_f, interp_fcc_f = build_interpolators_for_T(df, T_frame)
                if interp_liq_f is None:
                    continue
                T_f = (T_frame - T_min) / T_range if T_range > 0 else 0.5
                
                TH, PH, _, dG_grid, _, sphere_pts = sample_g_on_sphere(
                    interp_liq_f, interp_fcc_f, sh_R_fixed, sh_n_theta, sh_n_phi
                )
                
                l_max_liq = max(1, int(sh_l_max - 1.5 * T_f))
                l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_f)))
                
                g_liq_raw = interp_liq_f(sphere_pts).reshape(TH.shape)
                g_fcc_raw = interp_fcc_f(sphere_pts).reshape(TH.shape)

                n_valid_liq = np.sum(~np.isnan(g_liq_raw))
                n_valid_fcc = np.sum(~np.isnan(g_fcc_raw))
                n_needed_liq = (l_max_liq + 1) ** 2
                n_needed_fcc = (l_max_fcc + 1) ** 2

                if n_valid_liq < n_needed_liq:
                    l_max_liq = max(1, int(np.floor(np.sqrt(n_valid_liq)) - 1)) if n_valid_liq > 0 else 1
                if n_valid_fcc < n_needed_fcc:
                    l_max_fcc = max(1, int(np.floor(np.sqrt(n_valid_fcc)) - 1)) if n_valid_fcc > 0 else 1

                coeffs_liq, l_max_liq = fit_sh_coeffs(TH, PH, g_liq_raw, l_max=l_max_liq)
                coeffs_fcc, l_max_fcc = fit_sh_coeffs(TH, PH, g_fcc_raw, l_max=l_max_fcc)
                
                if coeffs_liq is None or coeffs_fcc is None:
                    continue
                
                G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
                G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)
                
                R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_f)
                X_liq = R_liq * np.sin(PH) * np.cos(TH)
                Y_liq = R_liq * np.sin(PH) * np.sin(TH)
                Z_liq = R_liq * np.cos(PH)
                
                R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_f)
                X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
                Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
                Z_fcc = R_fcc * np.cos(PH)
                
                frame_data = [
                    go.Surface(x=X_liq, y=Y_liq, z=Z_liq, surfacecolor=G_liq_sh,
                               colorscale="Reds", cmin=G_global_min, cmax=G_global_max,
                               opacity=0.6 + 0.3 * T_f, name="LIQUID", showscale=False),
                    go.Surface(x=X_fcc, y=Y_fcc, z=Z_fcc, surfacecolor=G_fcc_sh,
                               colorscale="Blues", cmin=G_global_min, cmax=G_global_max,
                               opacity=0.8 - 0.4 * T_f, name="FCC", showscale=False)
                ]
                
                frames.append(go.Frame(data=frame_data, name=f"T={T_frame}"))
            
            if len(frames) > 0:
                # Initial frame
                T_init = T_frames[0]
                interp_liq_i, interp_fcc_i = build_interpolators_for_T(df, T_init)
                TH, PH, _, _, _, sphere_pts = sample_g_on_sphere(
                    interp_liq_i, interp_fcc_i, sh_R_fixed, sh_n_theta, sh_n_phi
                )
                T_i = (T_init - T_min) / T_range if T_range > 0 else 0.5
                l_max_liq = max(1, int(sh_l_max - 1.5 * T_i))
                l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_i)))
                g_liq_raw = interp_liq_i(sphere_pts).reshape(TH.shape)
                g_fcc_raw = interp_fcc_i(sphere_pts).reshape(TH.shape)

                n_valid_liq = np.sum(~np.isnan(g_liq_raw))
                n_valid_fcc = np.sum(~np.isnan(g_fcc_raw))
                n_needed_liq = (l_max_liq + 1) ** 2
                n_needed_fcc = (l_max_fcc + 1) ** 2

                if n_valid_liq < n_needed_liq:
                    l_max_liq = max(1, int(np.floor(np.sqrt(n_valid_liq)) - 1)) if n_valid_liq > 0 else 1
                if n_valid_fcc < n_needed_fcc:
                    l_max_fcc = max(1, int(np.floor(np.sqrt(n_valid_fcc)) - 1)) if n_valid_fcc > 0 else 1

                coeffs_liq, l_max_liq = fit_sh_coeffs(TH, PH, g_liq_raw, l_max=l_max_liq)
                coeffs_fcc, l_max_fcc = fit_sh_coeffs(TH, PH, g_fcc_raw, l_max=l_max_fcc)
                G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
                G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)
                R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_i)
                R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_i)
                
                fig.add_trace(go.Surface(
                    x=R_liq*np.sin(PH)*np.cos(TH), y=R_liq*np.sin(PH)*np.sin(TH), z=R_liq*np.cos(PH),
                    surfacecolor=G_liq_sh, colorscale="Reds", cmin=G_global_min, cmax=G_global_max,
                    opacity=0.6+0.3*T_i, name="LIQUID", showscale=False
                ))
                fig.add_trace(go.Surface(
                    x=R_fcc*np.sin(PH)*np.cos(TH), y=R_fcc*np.sin(PH)*np.sin(TH), z=R_fcc*np.cos(PH),
                    surfacecolor=G_fcc_sh, colorscale="Blues", cmin=G_global_min, cmax=G_global_max,
                    opacity=0.8-0.4*T_i, name="FCC", showscale=False
                ))
                
                fig.frames = frames
                
                # Animation controls
                fig.update_layout(
                    updatemenus=[{
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "▶️ Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 800, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 300}}]
                            },
                            {
                                "label": "⏸️ Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                  "mode": "immediate", "transition": {"duration": 0}}]
                            }
                        ],
                        "x": 0.1, "y": 0.05
                    }],
                    sliders=[{
                        "active": 0,
                        "yanchor": "top", "xanchor": "left",
                        "currentvalue": {"prefix": "Temperature: ", "visible": True, "xanchor": "right"},
                        "transition": {"duration": 300},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9, "x": 0.1, "y": 0,
                        "steps": [
                            {"method": "animate", "args": [[f"T={T_f}"], {"frame": {"duration": 300, "redraw": True},
                                                                          "mode": "immediate", "transition": {"duration": 300}}],
                             "label": f"{T_f}K"} for T_f in T_frames
                        ]
                    }]
                )
                
                st.info(f"✅ Animation ready: {len(frames)} frames from {anim_start}K to {anim_end}K. Click **▶️ Play** or drag the slider.")
            else:
                st.error("❌ Could not generate animation frames.")
        
        scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # ------------------------------------------------------------------
    # COMMON OVERLAYS (applied to all modes)
    # ------------------------------------------------------------------
    
    # Composition path connecting query history
    if show_comp_path and len(st.session_state.query_history) > 1:
        hist = st.session_state.query_history
        path_x = [h["Co"] for h in hist]
        path_y = [h["Cr"] for h in hist]
        path_z = [h["Fe"] for h in hist]
        path_T = [h["T"] for h in hist]
        
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode="lines+markers",
            line=dict(color="gold", width=4, dash="dot"),
            marker=dict(size=6, color=path_T, colorscale="Thermal", cmin=T_min, cmax=T_max,
                        showscale=True, colorbar=dict(title="Path T (K)", thickness=15, len=0.5)),
            name="Composition Path",
            hovertemplate="T=%{marker.color:.0f}K<br>Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<extra></extra>"
        ))
    
    # Query point marker
    if query_result is not None:
        q_color = PHASE_COLORS[query_result["Phase"]]
        q_symbol = PHASE_SYMBOLS[query_result["Phase"]]
        
        fig.add_trace(go.Scatter3d(
            x=[query_result["Co"]], y=[query_result["Cr"]], z=[query_result["Fe"]],
            mode="markers+text",
            marker=dict(size=18, color=q_color, symbol=q_symbol,
                        line=dict(width=3, color="white")),
            text=["QUERY"],
            textposition="top center",
            textfont=dict(size=12, color=q_color, family="Arial Black"),
            name=f"Query ({query_result['Phase']})",
            hovertemplate=(f"<b>QUERY</b><br>T={query_result['T']}K<br>" +
                           f"Co={query_result['Co']:.3f}<br>Cr={query_result['Cr']:.3f}<br>" +
                           f"Fe={query_result['Fe']:.3f}<br>Ni={query_result['Ni']:.3f}<br>" +
                           f"G_stable={query_result['G_stable']:,.0f}<br>ΔG={query_result['dG']:,.0f}<br>" +
                           f"Phase={query_result['Phase']}<extra></extra>")
        ))
        
        if show_query_probe:
            u = np.linspace(0, 2*np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            r_probe = 0.08
            x_p = query_result["Co"] + r_probe * np.outer(np.cos(u), np.sin(v))
            y_p = query_result["Cr"] + r_probe * np.outer(np.sin(u), np.sin(v))
            z_p = query_result["Fe"] + r_probe * np.outer(np.ones(np.size(u)), np.cos(v))
            fig.add_trace(go.Surface(
                x=x_p, y=y_p, z=z_p,
                opacity=0.2,
                colorscale=[[0, q_color], [1, q_color]],
                showscale=False,
                name="Query Probe",
                hoverinfo="skip"
            ))
    
    # Coordinate axes for orientation
    if show_axes_frame:
        axis_len = 1.05
        for coord, color, label in [(0, "#c0392b", "Co"), (1, "#27ae60", "Cr"), (2, "#2980b9", "Fe")]:
            x_line = [0, axis_len if coord==0 else 0]
            y_line = [0, axis_len if coord==1 else 0]
            z_line = [0, axis_len if coord==2 else 0]
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode="lines+text",
                line=dict(color=color, width=5),
                text=["", label],
                textposition="top center",
                textfont=dict(size=14, color=color, family="Arial Black"),
                hoverinfo="skip", showlegend=False
            ))
    
    # ------------------------------------------------------------------
    # LAYOUT CONFIGURATION
    # ------------------------------------------------------------------
    def make_axis(title_text):
        return dict(
            title=dict(text=title_text, font=dict(size=14)),
            tickfont=dict(size=11),
            showbackground=True,
            backgroundcolor=bg_color,
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.3)",
            zerolinewidth=1
        )
    
    fig.update_layout(
        template=template,
        scene=dict(
            xaxis=make_axis(scene_x),
            yaxis=make_axis(scene_y),
            zaxis=make_axis(scene_z),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
        ),
        title=dict(
            text=f"Co-Cr-Fe-Ni at T = {T_val} K | {render_mode} | {phase_expected}",
            font=dict(size=title_font)
        ),
        margin=dict(l=0, r=0, b=60 if render_mode=="Animated Temperature Sweep" else 0, t=50),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
    )
    
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Render error: {e}")

# =============================================
# TAB 2: TENSOR DECOMPOSITION ANALYSIS (CPD)
# =============================================
with tab_tensor:
    st.header("📊 Thermodynamic Data Tensor (TDT) Analysis")
    st.markdown("""
    Based on **Coutinho et al., npj Computational Materials 6, 2 (2020)**  
    
    This tab analyzes the Gibbs energy data as a **4th-order incomplete tensor** and performs 
    **Canonical Polyadic Decomposition (CPD)** to quantify rank, compression, and separability.
    
    **Why tensor decomposition for CALPHAD data?**
    - 🔹 Breaks curse of dimensionality: CPD coefficients scale as R×(I+J+K+L) vs O(I×J×K×L)
    - 🔹 Handles incomplete tensors: Only simplex-valid entries used in fitting
    - 🔹 Enables rapid prediction: Any entry computed in O(R) operations after decomposition
    - 🔹 Reveals physics: Factor matrices correspond to thermodynamic contributions
    """)
    
    # Build tensor data (cached)
    tdt_data = build_tensor_data(df)
    n_co, n_cr, n_fe, n_T = tdt_data['dims']
    
    # --- TENSOR INSPECTION ---
    st.subheader("🔍 Tensor Inspection")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Co dimension", f"{n_co}", f"step={tdt_data['co_step']:.3f}")
    col2.metric("Cr dimension", f"{n_cr}", f"step={tdt_data['cr_step']:.3f}")
    col3.metric("Fe dimension", f"{n_fe}", f"step={tdt_data['fe_step']:.3f}")
    col4.metric("T dimension", f"{n_T}", f"step={tdt_data['T_step']:.0f}K")
    
    full_size = n_co * n_cr * n_fe * n_T
    valid_liq = int(np.sum(~np.isnan(tdt_data['G_LIQ'])))
    valid_fcc = int(np.sum(~np.isnan(tdt_data['G_FCC'])))
    
    st.markdown(f"""
    | Property | Value | Physical Meaning |
    |----------|-------|-----------------|
    | **TDT Order** | 4 | Co × Cr × Fe × T thermodynamic state space |
    | **Full hypercube** | {full_size:,} entries | All possible grid combinations |
    | **Valid entries (G_LIQ)** | {valid_liq:,} ({100*valid_liq/full_size:.1f}%) | Simplex constraint: Co+Cr+Fe ≤ 1 |
    | **Valid entries (G_FCC)** | {valid_fcc:,} ({100*valid_fcc/full_size:.1f}%) | Same constraint for FCC phase |
    | **Compression potential** | ~6× reduction | CPD with R=6 needs ~900 coeffs vs ~170K valid |
    """)
    
    # --- RANK ANALYSIS ---
    st.subheader("📈 Multilinear Rank Analysis (SVD of Unfoldings)")
    st.markdown("""
    Unfolding the tensor along each mode and analyzing singular value decay to estimate 
    effective rank. This reveals the intrinsic dimensionality of each thermodynamic variable.
    
    **Expected results for Co-Cr-Fe-Ni with 31 temperatures:**
    - Temperature mode (Mode-3): rank ≈ 3 (baseline + linear entropy + Cp curvature)
    - Composition modes (0-2): rank ≈ 5-7 (polynomial mixing + magnetic effects)
    """)
    
    phase_for_tensor = st.selectbox("Select Phase for Analysis", ["G_LIQUID", "G_FCC"], index=0)

    # Convenience: run both phases automatically
    auto_both = st.toggle("🔄 Auto-run both phases", value=False,
                         help="Automatically run CPD for both LIQUID and FCC phases sequentially")

    tensor_sel = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']
    
    threshold = st.slider("Singular Value Threshold (% of max)", 0.01, 5.0, 0.3, 0.1, 
                         help="For Co-Cr-Fe-Ni: 0.2-0.5% captures physical modes, excludes noise")
    
    if st.button("🔬 Run Rank Analysis", use_container_width=True):
        with st.spinner("Computing SVD on all mode unfoldings..."):
            mode_names = ['Co', 'Cr', 'Fe', 'T']
            ranks = []
            all_s = []
            
            for mode in range(4):
                unfolded = unfold_tensor(tensor_sel, mode)
                rank, s, s_norm = svd_rank_analysis(unfolded, threshold=threshold/100.0)
                ranks.append(rank)
                all_s.append(s_norm)
            
            st.success(f"✅ Analysis complete! Multilinear rank: ({', '.join(map(str, ranks))})")
            
            # Create plotly figure for singular values
            fig_svd = go.Figure()
            colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']
            
            for mode in range(4):
                s_norm = all_s[mode]
                fig_svd.add_trace(go.Scatter(
                    x=list(range(1, len(s_norm)+1)),
                    y=s_norm,
                    mode='lines+markers',
                    name=f'Mode-{mode} ({mode_names[mode]}): rank={ranks[mode]}',
                    line=dict(color=colors[mode], width=2),
                    marker=dict(size=6)
                ))
            
            fig_svd.add_hline(y=threshold/100.0, line_dash="dash", line_color="gray",
                             annotation_text=f"Threshold ({threshold}%)")
            
            fig_svd.update_layout(
                title="Singular Value Decay Across Tensor Modes",
                xaxis_title="Singular Value Index",
                yaxis_title="Normalized Singular Value",
                yaxis_type="log",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig_svd, use_container_width=True)
            
            # Rank interpretation with real data context
            max_cp_rank = max(ranks)
            st.info(f"""
            **Interpretation for Co-Cr-Fe-Ni:**
            - **CP rank should be ≥ {max_cp_rank}** to capture all modes accurately
            - Temperature mode (Mode-3) has lowest rank (~3) because G(T) ≈ H₀ - S₀·T + Cp·corrections
            - Composition modes have higher rank (5-7) due to polynomial Redlich-Kister mixing terms
            - With R=6, CPD achieves ~1000× compression: ~900 coefficients vs ~170K valid entries
            """)
    
    # --- CPD COMPRESSION ANALYSIS ---
    st.subheader("🗜️ CPD Compression Analysis")
    
    R_test = st.slider("Test CP Rank (R)", 1, 20, 6, 1)
    
    cpd_coeffs = R_test * (n_co + n_cr + n_fe + n_T)
    compression = valid_liq / cpd_coeffs if cpd_coeffs > 0 else 0
    reduction = (1 - cpd_coeffs / valid_liq) * 100 if valid_liq > 0 else 0
    
    st.markdown(f"""
    | Metric | Value | Interpretation |
    |--------|-------|---------------|
    | **CP Rank (R)** | {R_test} | Number of separable thermodynamic components |
    | **CPD coefficients** | {cpd_coeffs:,} | R × (I+J+K+L) = {R_test} × ({n_co}+{n_cr}+{n_fe}+{n_T}) |
    | **Original valid entries** | {valid_liq:,} | Simplex-constrained CALPHAD data points |
    | **Compression ratio** | **{compression:.1f}×** | Storage reduction factor |
    | **Storage reduction** | **{reduction:.1f}%** | Memory savings vs dense tensor |
    """)
    
    # Visualize compression tradeoff
    ranks_range = list(range(1, 21))
    cpd_sizes = [r * (n_co + n_cr + n_fe + n_T) for r in ranks_range]
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=[f"R={r}" for r in ranks_range],
        y=[valid_liq] * len(ranks_range),
        name='TDT valid entries',
        marker_color='lightcoral'
    ))
    fig_comp.add_trace(go.Bar(
        x=[f"R={r}" for r in ranks_range],
        y=cpd_sizes,
        name='CPD coefficients',
        marker_color='steelblue'
    ))
    
    fig_comp.update_layout(
        title="TDT Entries vs CPD Coefficients: Compression Tradeoff",
        yaxis_title="Count",
        barmode='group',
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # --- CPD RECONSTRUCTION ---
    st.subheader("🔧 CPD Reconstruction")
    st.markdown("Run ALS-based CPD to reconstruct the tensor and measure approximation error.")
    
    max_iter = st.slider("Max ALS Iterations", 20, 200, 100, 10)
    
    if st.button("⚙️ Run CPD-ALS (may take 1-2 min for large tensors)", use_container_width=True):
        phases_to_run = []
        if auto_both:
            phases_to_run = [
                ("G_LIQUID", "LIQ", tdt_data['G_LIQ']),
                ("G_FCC", "FCC", tdt_data['G_FCC'])
            ]
        else:
            phase_key = "LIQ" if phase_for_tensor == "G_LIQUID" else "FCC"
            tensor_sel_phase = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']
            phases_to_run = [(phase_for_tensor, phase_key, tensor_sel_phase)]

        for phase_name, phase_key, tensor_phase in phases_to_run:
            with st.spinner(f"Running CP-ALS for {phase_name} with R={R_test}..."):
                # Center and scale for numerical stability
                tensor_mean = np.nanmean(tensor_phase)
                tensor_std = np.nanstd(tensor_phase)
                tensor_norm = (tensor_phase - tensor_mean) / (tensor_std + 1e-12)

                A, B, C, D, lam, error = cpd_als_4d(tensor_norm, R_test, max_iter=max_iter, tol=1e-5, use_weighted=True, reg=1e-8)

                # Reconstruct
                I, J, K, L = tensor_norm.shape
                recon = np.zeros_like(tensor_norm)
                mask = ~np.isnan(tensor_norm)

                for r in range(R_test):
                    recon += lam[r] * np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)

                # Error metrics
                rel_error = np.sqrt(np.sum(mask * (tensor_norm - recon)**2) / np.sum(mask))
                abs_error = rel_error * tensor_std

                st.success(f"✅ CPD complete for {phase_key}! Relative error: {rel_error:.6f} | Absolute error: {abs_error:.2f} J/mol")

                # SAVE TO SESSION STATE for AM analysis
                st.session_state[f'cpd_completed_{phase_key}'] = True
                st.session_state[f'A_{phase_key.lower()}'] = A
                st.session_state[f'B_{phase_key.lower()}'] = B
                st.session_state[f'C_{phase_key.lower()}'] = C
                st.session_state[f'D_{phase_key.lower()}'] = D
                st.session_state[f'lam_{phase_key.lower()}'] = lam
                st.session_state[f'error_{phase_key.lower()}'] = error
                st.session_state[f'rel_error_{phase_key.lower()}'] = rel_error
                st.session_state[f'abs_error_{phase_key.lower()}'] = abs_error
                st.session_state['tdt_metadata'] = {
                    'co_vals': tdt_data['co_vals'],
                    'cr_vals': tdt_data['cr_vals'],
                    'fe_vals': tdt_data['fe_vals'],
                    'T_vals': tdt_data['T_vals'],
                    'dims': tdt_data['dims'],
                    'co_step': tdt_data['co_step'],
                    'cr_step': tdt_data['cr_step'],
                    'fe_step': tdt_data['fe_step'],
                    'T_step': tdt_data['T_step']
                }

        # Check if both phases are complete
        liq_done = st.session_state.get('cpd_completed_LIQ', False)
        fcc_done = st.session_state.get('cpd_completed_FCC', False)
        if liq_done and fcc_done:
            st.session_state['cpd_both_complete'] = True
            st.balloons()
            st.success("🎉 Both LIQUID and FCC phases decomposed! AM Design Assistant is now fully enabled.")
        elif auto_both and len(phases_to_run) == 2:
            st.success("🎉 Auto-run complete! Both phases saved.")
        else:
            missing = []
            if not liq_done: missing.append("LIQUID")
            if not fcc_done: missing.append("FCC")
            st.info(f"⏳ Still need: {', '.join(missing)}. {'Toggle auto-run or ' if not auto_both else ''}select the other phase and run CPD again.")

        # Display factor matrices for the LAST run phase
        st.subheader("Factor Matrices")
        tabs = st.tabs(["A (Co)", "B (Cr)", "C (Fe)", "D (T)", "lambda (Weights)"])

        with tabs[0]:
            df_A = pd.DataFrame(A, columns=[f"r={r+1}" for r in range(R_test)])
            df_A.index = [f"Co={v:.3f}" for v in tdt_data['co_vals']]
            st.dataframe(df_A.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)
            st.caption("Each column = composition dependence of one CPD component")

        with tabs[1]:
            df_B = pd.DataFrame(B, columns=[f"r={r+1}" for r in range(R_test)])
            df_B.index = [f"Cr={v:.3f}" for v in tdt_data['cr_vals']]
            st.dataframe(df_B.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)

        with tabs[2]:
            df_C = pd.DataFrame(C, columns=[f"r={r+1}" for r in range(R_test)])
            df_C.index = [f"Fe={v:.3f}" for v in tdt_data['fe_vals']]
            st.dataframe(df_C.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)

        with tabs[3]:
            df_D = pd.DataFrame(D, columns=[f"r={r+1}" for r in range(R_test)])
            df_D.index = [f"T={v}K" for v in tdt_data['T_vals']]
            st.dataframe(df_D.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)
            st.caption("Temperature factors: r=1≈constant, r=2≈linear in T, r=3≈curvature")

        with tabs[4]:
            df_lam = pd.DataFrame({'Component': [f"r={r+1}" for r in range(R_test)], 'Weight': lam})
            st.dataframe(df_lam, use_container_width=True)

            fig_weights = go.Figure(go.Bar(
                x=[f"r={r+1}" for r in range(R_test)],
                y=lam,
                marker_color='teal'
            ))
            fig_weights.update_layout(title="CPD Component Weights (lambda)", template="plotly_white")
            st.plotly_chart(fig_weights, use_container_width=True)

        # Reconstruction quality visualization
        st.subheader("Reconstruction Quality")
        T_slice = st.selectbox("Select Temperature Slice", tdt_data['T_vals'])
        t_idx = tdt_data['T_vals'].index(T_slice)

        orig_slice = tensor_sel[:,:,:,t_idx]
        recon_slice = recon[:,:,:,t_idx] * tensor_std + tensor_mean

        valid_mask_slice = ~np.isnan(orig_slice)
        orig_valid = orig_slice[valid_mask_slice]
        recon_valid = recon_slice[valid_mask_slice]

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=orig_valid, y=recon_valid,
            mode='markers',
            marker=dict(size=4, color='steelblue', opacity=0.5),
            name='Data points'
        ))

        min_val = min(np.min(orig_valid), np.min(recon_valid))
        max_val = max(np.max(orig_valid), np.max(recon_valid))
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect fit (y=x)'
        ))

        fig_scatter.update_layout(
            title=f"Original vs Reconstructed G at T={T_slice}K",
            xaxis_title="Original G (J/mol)",
            yaxis_title="Reconstructed G (J/mol)",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
    with st.expander("📖 Tensor Decomposition Theory", expanded=False):
        st.markdown(r"""
        ### Canonical Polyadic Decomposition (CPD)
        
        From **Coutinho et al. (2020)**, the TDT is decomposed as a sum of rank-1 terms:
        
        ```
        G(i,j,k,t) ≈ Σᵣ₌₁ᴿ λᵣ · A(i,r) · B(j,r) · C(k,r) · D(t,r)
        ```
        
        where:
        - R = rank (number of separable thermodynamic components)
        - A, B, C, D = factor matrices for Co, Cr, Fe, T dimensions
        - λᵣ = component weight (relative importance)
        
        **Key advantages for CALPHAD data:**
        1. 🔹 **Breaks curse of dimensionality**: CPD coefficients scale as R×(I+J+K+L) -- LINEAR in dimensions
        2. 🔹 **Incomplete tensor handling**: Only simplex-valid entries used in fitting
        3. 🔹 **Polynomial constraints**: Factor vectors can be constrained to polynomials matching CALPHAD form
        4. 🔹 **Efficient evaluation**: Any entry computed in O(R) operations after decomposition
        
        **For this Co-Cr-Fe-Ni system:**
        - With step size 0.01, the TDT has ~170K valid entries per phase
        - With R=6, CPD needs only ~900 coefficients → **~190× compression**
        - Paper reports up to 1,000,000× compression for similar quaternary systems with finer grids
        
        **Physical interpretation of components (R=6):**
        | Component | Thermodynamic Meaning | Factor Behavior |
        |-----------|----------------------|----------------|
        | r=1 | Baseline enthalpy offset | A,B,C: smooth; D: ~constant |
        | r=2 | Linear entropy term (-S·T) | A,B,C: composition-dependent S; D: ~linear in T |
        | r=3 | Cp curvature + magnetic transitions | A,B,C: magnetic element weighting; D: quadratic/tanh |
        | r=4 | Binary interaction effects | Higher-order composition polynomials |
        | r=5 | Ternary mixing contributions | Fine composition structure |
        | r=6 | Ordering/short-range effects | Localized features |
        """)




# =============================================
# APPLICATION 6: MULTI-MATERIAL / GRADED STRUCTURES DESIGN
# =============================================

def evaluate_composition_path(path_func, s_vals, T_vals_query, 
                               A_liq, B_liq, C_liq, D_liq, lam_liq,
                               A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                               co_vals, cr_vals, fe_vals, T_vals_grid):
    """
    Evaluate Gibbs energy along a parametric composition path.

    A composition path is a function x(s) = (x_Co(s), x_Cr(s), x_Fe(s)) 
    for s ∈ [0,1], with x_Ni(s) = 1 - x_Co(s) - x_Cr(s) - x_Fe(s).

    This enables design of graded structures where composition varies
    smoothly from alloy A (s=0) to alloy B (s=1).
    """
    from scipy.interpolate import interp1d

    # Rank validation (same fix as above)
    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)
    R = min(R_liq, R_fcc, D_liq.shape[1], D_fcc.shape[1])

    n_s = len(s_vals)
    n_T = len(T_vals_query)

    # Pre-build interpolation functions for factor matrices
    def build_factor_interp(vals, factor_matrix):
        """Build interpolation functions for each CPD rank."""
        interps = []
        for r in range(factor_matrix.shape[1]):
            interps.append(interp1d(vals, factor_matrix[:, r], 
                                     kind='cubic', fill_value='extrapolate'))
        return interps

    A_liq_interp = build_factor_interp(co_vals, A_liq)
    B_liq_interp = build_factor_interp(cr_vals, B_liq)
    C_liq_interp = build_factor_interp(fe_vals, C_liq)
    A_fcc_interp = build_factor_interp(co_vals, A_fcc)
    B_fcc_interp = build_factor_interp(cr_vals, B_fcc)
    C_fcc_interp = build_factor_interp(fe_vals, C_fcc)

    # Temperature factor interpolation
    D_liq_interp = [interp1d(T_vals_grid, D_liq[:, r], kind='cubic',
                              fill_value='extrapolate') for r in range(R_liq)]
    D_fcc_interp = [interp1d(T_vals_grid, D_fcc[:, r], kind='cubic',
                              fill_value='extrapolate') for r in range(R_fcc)]

    # Evaluate path
    path_pts = np.array([path_func(s) for s in s_vals])  # (n_s, 3)

    # Storage
    G_liq_path = np.zeros((n_s, n_T))
    G_fcc_path = np.zeros((n_s, n_T))
    dG_path = np.zeros((n_s, n_T))
    T_star_path = np.full(n_s, np.nan)
    phase_path = np.full((n_s, n_T), '', dtype=object)

    # Evaluate at each path point
    for i, (co, cr, fe) in enumerate(path_pts):
        # Check simplex validity
        ni = 1.0 - co - cr - fe
        if ni < -0.01 or co < -0.01 or cr < -0.01 or fe < -0.01:
            continue  # Outside valid composition space

        # Interpolate composition factors
        A_liq_q = np.array([f(co) for f in A_liq_interp])
        B_liq_q = np.array([f(cr) for f in B_liq_interp])
        C_liq_q = np.array([f(fe) for f in C_liq_interp])
        A_fcc_q = np.array([f(co) for f in A_fcc_interp])
        B_fcc_q = np.array([f(cr) for f in B_fcc_interp])
        C_fcc_q = np.array([f(fe) for f in C_fcc_interp])

        # Evaluate at each temperature
        for t_idx, T in enumerate(T_vals_query):
            # LIQUID Gibbs energy
            D_liq_T = np.array([f(T) for f in D_liq_interp])
            G_liq_path[i, t_idx] = np.sum(lam_liq * A_liq_q * B_liq_q * C_liq_q * D_liq_T)

            # FCC Gibbs energy
            D_fcc_T = np.array([f(T) for f in D_fcc_interp])
            G_fcc_path[i, t_idx] = np.sum(lam_fcc * A_fcc_q * B_fcc_q * C_fcc_q * D_fcc_T)

            # Driving force and phase
            dG_path[i, t_idx] = G_liq_path[i, t_idx] - G_fcc_path[i, t_idx]
            phase_path[i, t_idx] = 'LIQUID' if dG_path[i, t_idx] <= 0 else 'FCC'

        # Find T* where dG=0 (transition temperature)
        if np.any(dG_path[i, :] > 0) and np.any(dG_path[i, :] < 0):
            from scipy.optimize import brentq
            try:
                dG_interp = interp1d(T_vals_query, dG_path[i, :], kind='cubic')
                pos_mask = dG_path[i, :] > 0
                neg_mask = dG_path[i, :] < 0
                if np.any(pos_mask) and np.any(neg_mask):
                    T_low = T_vals_query[neg_mask][0] if np.any(neg_mask) else T_vals_query[0]
                    T_high = T_vals_query[pos_mask][-1] if np.any(pos_mask) else T_vals_query[-1]
                    if dG_interp(T_low) > 0:
                        T_low, T_high = T_high, T_low
                    T_star_path[i] = brentq(dG_interp, T_low, T_high, xtol=1.0)
            except (ValueError, RuntimeError):
                pass

    # Compute gradient of T* along path
    grad_T_star = np.gradient(T_star_path, s_vals) if len(s_vals) > 1 else np.zeros(n_s)

    # Segregation risk: high when binary interaction factors (r=4,5) are strong
    seg_risk = np.zeros(n_s)
    for i, (co, cr, fe) in enumerate(path_pts):
        if np.isnan(T_star_path[i]):
            continue
        binary_r = [3, 4, 5] if R_liq >= 6 else list(range(min(3, R_liq)))
        seg_strength = 0
        for r in binary_r:
            if r < R_liq and r < len(A_liq_interp):
                seg_strength += abs(lam_liq[r] * A_liq_interp[r](co) * 
                                   B_liq_interp[r](cr) * C_liq_interp[r](fe))
        seg_risk[i] = seg_strength * abs(grad_T_star[i]) if not np.isnan(grad_T_star[i]) else 0

    # Normalize segregation risk
    if np.max(seg_risk) > np.min(seg_risk):
        seg_risk = (seg_risk - np.min(seg_risk)) / (np.max(seg_risk) - np.min(seg_risk))

    return {
        'G_liq': G_liq_path,
        'G_fcc': G_fcc_path,
        'dG': dG_path,
        'T_star': T_star_path,
        'phase': phase_path,
        'grad_T_star': grad_T_star,
        'segregation_risk': seg_risk,
        'path_pts': path_pts,
        's_vals': s_vals,
        'T_vals': T_vals_query
    }


def design_optimal_gradient(start_comp, end_comp, n_points=50, 
                            penalty_weight=1.0, curvature_weight=0.5,
                            A_liq=None, B_liq=None, C_liq=None, D_liq=None, lam_liq=None,
                            A_fcc=None, B_fcc=None, C_fcc=None, D_fcc=None, lam_fcc=None,
                            co_vals=None, cr_vals=None, fe_vals=None, T_vals=None,
                            T_process=2800):
    """
    Design an optimal composition gradient path between two alloys.

    The "optimal" path minimizes:
        1. T* variation along the path (prevents interfacial cracking)
        2. Curvature of the path (prevents sharp composition changes)
        3. Segregation potential in the mushy zone
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize

    s_vals = np.linspace(0, 1, n_points)

    def path_from_params(params):
        """Build cubic spline path from optimization parameters."""
        p1 = np.array([params[0], params[1], params[2]])  # s=1/3
        p2 = np.array([params[3], params[4], params[5]])  # s=2/3

        # Ensure simplex constraint: Co+Cr+Fe <= 1
        for p in [p1, p2]:
            if np.sum(p) > 1.0:
                p[:] = p / np.sum(p) * 0.99

        # Build cubic spline through: start -> p1 -> p2 -> end
        control_pts = np.vstack([start_comp, p1, p2, end_comp])
        s_control = np.array([0, 1/3, 2/3, 1])

        cs_co = CubicSpline(s_control, control_pts[:, 0])
        cs_cr = CubicSpline(s_control, control_pts[:, 1])
        cs_fe = CubicSpline(s_control, control_pts[:, 2])

        path = np.column_stack([cs_co(s_vals), cs_cr(s_vals), cs_fe(s_vals)])

        # Enforce simplex constraint along path
        for i in range(len(path)):
            if np.sum(path[i, :3]) > 1.0:
                path[i, :3] = path[i, :3] / np.sum(path[i, :3]) * 0.99

        return path

    def objective(params):
        """Cost function to minimize."""
        path = path_from_params(params)

        # Evaluate T* along path at process temperature
        path_results = evaluate_composition_path(
            lambda s: path[int(s * (n_points - 1))] if int(s * (n_points - 1)) < n_points else path[-1],
            s_vals, [T_process],
            A_liq, B_liq, C_liq, D_liq, lam_liq,
            A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
            co_vals, cr_vals, fe_vals, T_vals
        )

        T_star = path_results['T_star']

        # Penalty 1: T* variation (want smooth, minimal changes)
        tstar_variation = np.nanvar(T_star)

        # Penalty 2: Path curvature (want smooth geometric path)
        curvature = 0
        for dim in range(3):
            second_deriv = np.gradient(np.gradient(path[:, dim], s_vals), s_vals)
            curvature += np.mean(second_deriv**2)

        # Penalty 3: Segregation risk
        seg_risk = np.nanmean(path_results['segregation_risk'])

        # Total cost
        cost = (penalty_weight * tstar_variation + 
                curvature_weight * curvature + 
                0.3 * seg_risk)

        return cost

    # Initial guess: linear interpolation control points
    p1_init = start_comp + (end_comp - start_comp) / 3
    p2_init = start_comp + 2 * (end_comp - start_comp) / 3
    x0 = np.concatenate([p1_init, p2_init])

    # Bounds: compositions must stay in [0, 1]
    bounds = [(0, 1)] * 6

    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 100, 'disp': False})

    optimal_path = path_from_params(result.x)

    # Evaluate final path metrics
    final_results = evaluate_composition_path(
        lambda s: optimal_path[int(s * (n_points - 1))],
        s_vals, T_vals,
        A_liq, B_liq, C_liq, D_liq, lam_liq,
        A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals
    )

    metrics = {
        'T_star_range': np.nanmax(final_results['T_star']) - np.nanmin(final_results['T_star']),
        'T_star_std': np.nanstd(final_results['T_star']),
        'max_segregation_risk': np.nanmax(final_results['segregation_risk']),
        'mean_segregation_risk': np.nanmean(final_results['segregation_risk']),
        'path_length': np.sum(np.linalg.norm(np.diff(optimal_path, axis=0), axis=1)),
        'linear_path_length': np.linalg.norm(end_comp - start_comp)
    }

    return {
        'path': optimal_path,
        's_vals': s_vals,
        'cost_history': [result.fun],
        'metrics': metrics,
        'T_star': final_results['T_star'],
        'segregation_risk': final_results['segregation_risk'],
        'phase_at_T': final_results['phase']
    }


def check_gradient_feasibility(path_pts, T_vals_query, A_liq, B_liq, C_liq, D_liq, lam_liq,
                                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                                co_vals, cr_vals, fe_vals, T_vals_grid,
                                T_melt_pool=2800, T_solidus=1600, T_haz=1200):
    """Check if a composition gradient is feasible for AM processing."""
    results = evaluate_composition_path(
        lambda s: path_pts[int(s * (len(path_pts) - 1))] if int(s * (len(path_pts) - 1)) < len(path_pts) else path_pts[-1],
        np.linspace(0, 1, len(path_pts)), T_vals_query,
        A_liq, B_liq, C_liq, D_liq, lam_liq,
        A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals_grid
    )

    issues = []
    safety_score = 1.0
    T_star = results['T_star']

    # Criterion 1: Meltability
    if np.any(T_star > T_melt_pool):
        issues.append(f"❌ Some compositions have T* > melt pool T ({T_melt_pool}K) — cannot fully melt")
        safety_score -= 0.3

    # Criterion 2: T* gradient smoothness
    grad_T = np.gradient(T_star)
    max_grad = np.nanmax(np.abs(grad_T))
    if max_grad > 200:  # K per unit path length
        issues.append(f"⚠️ Steep T* gradient (max {max_grad:.1f} K/path unit) — risk of interfacial cracking")
        safety_score -= 0.2

    # Criterion 3: Phase consistency in mushy zone
    solidus_idx = np.argmin(np.abs(np.array(T_vals_query) - T_solidus))
    phases_solidus = results['phase'][:, solidus_idx]
    if len(set(phases_solidus)) > 1:
        issues.append("⚠️ Phase changes in mushy zone — risk of mixed microstructure")
        safety_score -= 0.2

    # Criterion 4: HAZ stability
    haz_idx = np.argmin(np.abs(np.array(T_vals_query) - T_haz))
    phases_haz = results['phase'][:, haz_idx]
    if np.any(phases_haz == 'LIQUID'):
        issues.append("⚠️ Some compositions partially melt in HAZ — risk of liquation cracking")
        safety_score -= 0.2

    # Criterion 5: Segregation risk
    if np.nanmax(results['segregation_risk']) > 0.7:
        issues.append("⚠️ High segregation risk in gradient zone")
        safety_score -= 0.1

    return {
        'feasible': len([i for i in issues if i.startswith('❌')]) == 0,
        'issues': issues,
        'safety_score': max(0, safety_score),
        'metrics': {
            'T_star_min': np.nanmin(T_star),
            'T_star_max': np.nanmax(T_star),
            'T_star_range': np.nanmax(T_star) - np.nanmin(T_star),
            'max_T_gradient': max_grad,
            'segregation_max': np.nanmax(results['segregation_risk'])
        }
    }


def plot_gradient_path_3d(path_pts, T_star, s_vals, 
                          start_label="Alloy A", end_label="Alloy B",
                          color_by='T_star'):
    """Plot composition gradient path in 3D composition space."""
    fig = go.Figure()

    # Main path line
    fig.add_trace(go.Scatter3d(
        x=path_pts[:, 0], y=path_pts[:, 1], z=path_pts[:, 2],
        mode='lines+markers',
        line=dict(color='royalblue', width=6),
        marker=dict(
            size=8,
            color=T_star,
            colorscale='Magma',
            cmin=np.nanmin(T_star), cmax=np.nanmax(T_star),
            colorbar=dict(title="T* (K)", thickness=15, len=0.7),
            showscale=True
        ),
        name='Gradient Path',
        hovertemplate=("s=%{customdata:.3f}<br>" +
                      "Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<br>" +
                      "T*=%{marker.color:.0f} K<extra></extra>"),
        customdata=s_vals
    ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[path_pts[0, 0]], y=[path_pts[0, 1]], z=[path_pts[0, 2]],
        mode='markers+text',
        marker=dict(size=15, color='green', symbol='diamond',
                   line=dict(width=2, color='white')),
        text=[start_label],
        textposition='top center',
        textfont=dict(size=14, color='green'),
        name=start_label,
        hovertemplate=f"<b>{start_label}</b><br>Co=%{{x:.3f}}<br>Cr=%{{y:.3f}}<br>Fe=%{{z:.3f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter3d(
        x=[path_pts[-1, 0]], y=[path_pts[-1, 1]], z=[path_pts[-1, 2]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='diamond',
                   line=dict(width=2, color='white')),
        text=[end_label],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        name=end_label,
        hovertemplate=f"<b>{end_label}</b><br>Co=%{{x:.3f}}<br>Cr=%{{y:.3f}}<br>Fe=%{{z:.3f}}<extra></extra>"
    ))

    # Simplex frame for reference
    edges = [
        [(1,0,0),(0,1,0)], [(1,0,0),(0,0,1)], [(1,0,0),(0,0,0)],
        [(0,1,0),(0,0,1)], [(0,1,0),(0,0,0)], [(0,0,1),(0,0,0)]
    ]
    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[e[0][0], e[1][0]], y=[e[0][1], e[1][1]], z=[e[0][2], e[1][2]],
            mode="lines", line=dict(color="gray", width=2, dash='dot'),
            hoverinfo="skip", showlegend=False
        ))

    vertices = [(1,0,0,"Co"), (0,1,0,"Cr"), (0,0,1,"Fe"), (0,0,0,"Ni")]
    for vx, vy, vz, vl in vertices:
        fig.add_trace(go.Scatter3d(
            x=[vx], y=[vy], z=[vz], mode="text", text=[vl],
            textposition="top center", textfont=dict(size=12, color="gray"),
            hoverinfo="skip", showlegend=False
        ))

    fig.update_layout(
        title=dict(text="Optimal Composition Gradient Path", font_size=16),
        scene=dict(
            xaxis=dict(title="x<sub>Co</sub>", range=[0, 1]),
            yaxis=dict(title="x<sub>Cr</sub>", range=[0, 1]),
            zaxis=dict(title="x<sub>Fe</sub>", range=[0, 1]),
            aspectmode='cube'
        ),
        height=650,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig


def plot_gradient_analysis_dashboard(path_results, T_vals_query, s_vals):
    """Create a comprehensive dashboard for gradient analysis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('T* along Gradient Path', 'Phase Stability Map (s vs T)',
                       'Segregation Risk', 'Driving Force ΔG at Process T'),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. T* along path
    fig.add_trace(go.Scatter(
        x=s_vals, y=path_results['T_star'],
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=6),
        name='T*(s)',
        hovertemplate="s=%{x:.3f}<br>T*=%{y:.0f} K<extra></extra>"
    ), row=1, col=1)

    # Add melt pool and HAZ reference lines
    fig.add_hline(y=2800, line_dash="dash", line_color="red", opacity=0.5,
                 annotation_text="Melt Pool", row=1, col=1)
    fig.add_hline(y=1200, line_dash="dash", line_color="orange", opacity=0.5,
                 annotation_text="HAZ", row=1, col=1)

    # 2. Phase stability map
    phase_numeric = np.where(path_results['phase'] == 'LIQUID', 1, 0)
    fig.add_trace(go.Heatmap(
        z=phase_numeric,
        x=T_vals_query,
        y=s_vals,
        colorscale=[[0, '#2980b9'], [1, '#e74c3c']],  # Blue=FCC, Red=LIQUID
        showscale=True,
        colorbar=dict(title="Phase", tickvals=[0, 1], ticktext=['FCC', 'LIQUID'],
                     len=0.4, y=0.8),
        hovertemplate="T=%{x:.0f}K<br>s=%{y:.3f}<br>Phase=%{z}<extra></extra>"
    ), row=1, col=2)

    # 3. Segregation risk
    fig.add_trace(go.Scatter(
        x=s_vals, y=path_results['segregation_risk'],
        mode='lines',
        line=dict(color='darkred', width=3),
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.2)',
        name='Segregation Risk',
        hovertemplate="s=%{x:.3f}<br>Risk=%{y:.3f}<extra></extra>"
    ), row=2, col=1)

    # Add risk threshold
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                 annotation_text="High Risk Threshold", row=2, col=1)

    # 4. dG at process T (assumes T_vals_query has process T)
    if len(T_vals_query) > 0:
        # Use middle temperature as "process T" for visualization
        t_mid = len(T_vals_query) // 2
        fig.add_trace(go.Scatter(
            x=s_vals, y=path_results['dG'][:, t_mid],
            mode='lines',
            line=dict(color='purple', width=3),
            name=f'ΔG at {T_vals_query[t_mid]}K',
            hovertemplate="s=%{x:.3f}<br>ΔG=%{y:.0f} J/mol<extra></extra>"
        ), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gold", row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Gradient Path Analysis Dashboard",
        showlegend=False
    )

    fig.update_xaxes(title_text="Path parameter s", row=1, col=1)
    fig.update_yaxes(title_text="T* (K)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_yaxes(title_text="Path parameter s", row=1, col=2)
    fig.update_xaxes(title_text="Path parameter s", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Risk", row=2, col=1)
    fig.update_xaxes(title_text="Path parameter s", row=2, col=2)
    fig.update_yaxes(title_text="ΔG (J/mol)", row=2, col=2)

    return fig


def render_gradient_design_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                  D_liq, D_fcc, lam_liq, lam_fcc,
                                  co_vals, cr_vals, fe_vals, T_vals):
    """Render Streamlit UI for multi-material gradient design."""

    st.subheader("🔗 Multi-Material / Graded Structures Design")
    st.markdown(r"""
    **Physical Problem**: When joining two dissimilar Co-Cr-Fe-Ni alloys via AM, 
    a sharp interface causes thermal mismatch cracking. A smooth composition 
    gradient prevents this — but only if designed correctly.

    **CPD Advantage**: The separable tensor structure allows instant evaluation 
    of $G(x(s), T)$ for any parametric path $x(s)$, enabling optimization of:
    - **T* smoothness**: Minimize $|
abla_s T^*|$ to prevent interfacial cracking
    - **Phase consistency**: Ensure stable FCC throughout the HAZ
    - **Segregation control**: Avoid high binary-interaction regions

    **Theory**: For a path $x(s) = (x_{Co}(s), x_{Cr}(s), x_{Fe}(s))$:
    $$T^*(s) = 	ext{root of } \Delta G(x(s), T) = 0$$
    $$	ext{Cracking Risk} \propto \left|rac{dT^*}{ds}
ight| 	imes S_{seg}(x(s))$$
    """)

    # --- COMPOSITION INPUTS ---
    st.subheader("📐 Define Alloy Compositions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Alloy A (Start)**")
        a_co = st.number_input("A: x_Co", 0.0, 1.0, 0.10, 0.01, key="a_co")
        a_cr = st.number_input("A: x_Cr", 0.0, 1.0, 0.35, 0.01, key="a_cr")
        a_fe = st.number_input("A: x_Fe", 0.0, 1.0, 0.15, 0.01, key="a_fe")
        a_ni = 1.0 - a_co - a_cr - a_fe
        st.markdown(f"A: x_Ni = **{a_ni:.3f}** {'✅' if a_ni >= 0 else '❌'}")

    with col2:
        st.markdown("**Alloy B (End)**")
        b_co = st.number_input("B: x_Co", 0.0, 1.0, 0.30, 0.01, key="b_co")
        b_cr = st.number_input("B: x_Cr", 0.0, 1.0, 0.10, 0.01, key="b_cr")
        b_fe = st.number_input("B: x_Fe", 0.0, 1.0, 0.25, 0.01, key="b_fe")
        b_ni = 1.0 - b_co - b_cr - b_fe
        st.markdown(f"B: x_Ni = **{b_ni:.3f}** {'✅' if b_ni >= 0 else '❌'}")

    # Validate compositions
    valid_a = a_ni >= 0 and a_co >= 0 and a_cr >= 0 and a_fe >= 0
    valid_b = b_ni >= 0 and b_co >= 0 and b_cr >= 0 and b_fe >= 0

    if not (valid_a and valid_b):
        st.error("❌ Invalid composition: all mole fractions must be ≥ 0 and sum to ≤ 1")
        return

    start_comp = np.array([a_co, a_cr, a_fe])
    end_comp = np.array([b_co, b_cr, b_fe])

    # --- OPTIMIZATION SETTINGS ---
    st.subheader("⚙️ Gradient Optimization")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        n_points = st.slider("Path Points", 20, 100, 50)
    with col_opt2:
        penalty_weight = st.slider("T* Smoothness Weight", 0.1, 5.0, 1.0, 0.1)
    with col_opt3:
        curvature_weight = st.slider("Path Curvature Weight", 0.1, 2.0, 0.5, 0.1)

    T_process = st.slider("Process Temperature (K)", 1500, 3500, 2800, 50)

    # --- COMPUTE GRADIENT ---
    if st.button("🔬 Design Optimal Gradient", use_container_width=True, type="primary"):
        with st.spinner("Optimizing composition gradient path..."):

            # Run optimization
            result = design_optimal_gradient(
                start_comp, end_comp, n_points=n_points,
                penalty_weight=penalty_weight, curvature_weight=curvature_weight,
                A_liq=A_liq, B_liq=B_liq, C_liq=C_liq, D_liq=D_liq, lam_liq=lam_liq,
                A_fcc=A_fcc, B_fcc=B_fcc, C_fcc=C_fcc, D_fcc=D_fcc, lam_fcc=lam_fcc,
                co_vals=co_vals, cr_vals=cr_vals, fe_vals=fe_vals, T_vals=T_vals,
                T_process=T_process
            )

            # Display metrics
            st.success("✅ Optimal gradient designed!")

            metrics = result['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T* Range", f"{metrics['T_star_range']:.0f} K", 
                     delta="Lower is better" if metrics['T_star_range'] < 200 else "⚠️ High")
            c2.metric("T* Std Dev", f"{metrics['T_star_std']:.1f} K")
            c3.metric("Max Segregation", f"{metrics['max_segregation_risk']:.3f}")
            c4.metric("Path Efficiency", f"{metrics['path_length']/metrics['linear_path_length']:.2f}×",
                     help="Ratio of actual path length to straight-line distance")

            # Feasibility check
            feasibility = check_gradient_feasibility(
                result['path'], T_vals, A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals,
                T_melt_pool=T_process, T_solidus=1600, T_haz=1200
            )

            st.subheader("🛡️ Feasibility Assessment")

            # Safety score gauge
            safety = feasibility['safety_score']
            if safety >= 0.8:
                st.success(f"✅ **SAFE** for AM (Score: {safety:.2f}/1.0)")
            elif safety >= 0.5:
                st.warning(f"⚠️ **CAUTION** (Score: {safety:.2f}/1.0) — Review issues below")
            else:
                st.error(f"❌ **RISKY** (Score: {safety:.2f}/1.0) — Significant redesign needed")

            for issue in feasibility['issues']:
                st.markdown(issue)

            # 3D Path Visualization
            st.subheader("🗺️ Gradient Path in Composition Space")
            fig_3d = plot_gradient_path_3d(
                result['path'], result['T_star'], result['s_vals'],
                start_label=f"A ({a_co:.2f},{a_cr:.2f},{a_fe:.2f})",
                end_label=f"B ({b_co:.2f},{b_cr:.2f},{b_fe:.2f})"
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # Analysis Dashboard
            st.subheader("📊 Gradient Analysis Dashboard")

            # Re-evaluate for dashboard
            path_results = evaluate_composition_path(
                lambda s: result['path'][int(s * (n_points - 1))] if int(s * (n_points - 1)) < n_points else result['path'][-1],
                result['s_vals'], T_vals,
                A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals
            )

            fig_dash = plot_gradient_analysis_dashboard(path_results, T_vals, result['s_vals'])
            st.plotly_chart(fig_dash, use_container_width=True)

            # Composition table
            with st.expander("📋 Detailed Composition Table", expanded=False):
                df_grad = pd.DataFrame({
                    's': result['s_vals'],
                    'x_Co': result['path'][:, 0],
                    'x_Cr': result['path'][:, 1],
                    'x_Fe': result['path'][:, 2],
                    'x_Ni': 1.0 - result['path'][:, 0] - result['path'][:, 1] - result['path'][:, 2],
                    'T* (K)': result['T_star'],
                    'Segregation Risk': result['segregation_risk']
                })
                st.dataframe(df_grad.style.format({
                    's': '{:.3f}', 'x_Co': '{:.4f}', 'x_Cr': '{:.4f}', 
                    'x_Fe': '{:.4f}', 'x_Ni': '{:.4f}', 'T* (K)': '{:.0f}',
                    'Segregation Risk': '{:.3f}'
                }).background_gradient(subset=['Segregation Risk'], cmap='Reds'), 
                use_container_width=True)

                csv_grad = df_grad.to_csv(index=False)
                st.download_button("📥 Download Gradient CSV", csv_grad, 
                                  "optimal_gradient.csv", "text/csv")

            # Design recommendations
            with st.expander("💡 Design Recommendations", expanded=True):
                st.markdown(f"""
                **Based on computed gradient analysis:**

                | Metric | Value | Implication |
                |--------|-------|-------------|
                | T* range | {metrics['T_star_range']:.0f} K | {'✅ Low variation — stable melting' if metrics['T_star_range'] < 150 else '⚠️ High variation — adjust laser power'} |
                | Path efficiency | {metrics['path_length']/metrics['linear_path_length']:.2f}× | {'✅ Near-linear — efficient' if metrics['path_length']/metrics['linear_path_length'] < 1.3 else '⚠️ Curved path — longer print time'} |
                | Max segregation | {metrics['max_segregation_risk']:.3f} | {'✅ Low risk' if metrics['max_segregation_risk'] < 0.5 else '⚠️ High risk — consider intermediate alloy'} |

                **AM Process Recommendations:**
                - **Laser power**: {'Constant' if metrics['T_star_range'] < 100 else 'Variable — map to T*(s)'}
                - **Scan speed**: {'Constant' if metrics['T_star_std'] < 50 else 'Reduce in high-T* regions'}
                - **Powder feed**: Pre-blend {'2' if metrics['max_segregation_risk'] > 0.5 else '1'} hopper system for gradient control
                - **Post-process**: {'Standard stress relief' if safety > 0.7 else 'Custom heat treatment for gradient zone'}
                """)


# =============================================
# TAB 3: ADDITIVE MANUFACTURING DESIGN ASSISTANT
# =============================================
with tab_am:
    st.header("🏭 Additive Manufacturing Design Assistant")
    st.markdown(r"""
    This tab uses **Canonical Polyadic Decomposition (CPD)** factors to predict 
    AM-relevant properties: transition temperatures, thermal response, composition 
    sensitivity, and defect susceptibility.

    **Theory**: The CPD decomposes Gibbs energy as:
    $$G[x_{Co}, x_{Cr}, x_{Fe}, T] \approx \sum_{r=1}^{R} \lambda_r \cdot A_r(x_{Co}) \cdot B_r(x_{Cr}) \cdot C_r(x_{Fe}) \cdot D_r(T)$$

    Each component $r$ captures a distinct thermodynamic mode (enthalpy, entropy, 
    heat capacity, binary interactions, etc.).
    """)

    # Check CPD completion status
    liq_done = st.session_state.get('cpd_completed_LIQ', False)
    fcc_done = st.session_state.get('cpd_completed_FCC', False)
    both_done = st.session_state.get('cpd_both_complete', False)

    # Status indicator
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("LIQUID CPD", "✅ Complete" if liq_done else "⏳ Needed", 
                 delta="Done" if liq_done else "Run in Tensor tab")
    with status_col2:
        st.metric("FCC CPD", "✅ Complete" if fcc_done else "⏳ Needed",
                 delta="Done" if fcc_done else "Run in Tensor tab")
    with status_col3:
        st.metric("AM Ready", "✅ Yes" if both_done else "❌ No",
                 delta="Both phases required" if not both_done else "Ready")

    st.divider()

    # Determine data source
    if both_done:
        st.success("✅ Both LIQUID and FCC CPD factors loaded from session state")

        # Retrieve all factors from session state
        A_liq = st.session_state['A_liq']
        B_liq = st.session_state['B_liq']
        C_liq = st.session_state['C_liq']
        D_liq = st.session_state['D_liq']
        lam_liq = st.session_state['lam_liq']

        A_fcc = st.session_state['A_fcc']
        B_fcc = st.session_state['B_fcc']
        C_fcc = st.session_state['C_fcc']
        D_fcc = st.session_state['D_fcc']
        lam_fcc = st.session_state['lam_fcc']

        meta = st.session_state['tdt_metadata']
        co_vals_am = meta['co_vals']
        cr_vals_am = meta['cr_vals']
        fe_vals_am = meta['fe_vals']
        T_vals_am = meta['T_vals']

        use_real_data = True

    elif liq_done or fcc_done:
        st.warning("⚠️ Only one phase complete. AM analysis requires both LIQUID and FCC. Please run CPD for the missing phase in the Tensor Decomposition tab.")
        use_demo = st.toggle("Use demo data for preview", value=False)
        if not use_demo:
            st.stop()
        use_real_data = False
    else:
        st.info("💡 **To use AM analysis, run CPD for both phases in the Tensor Decomposition tab.**")
        use_demo = st.toggle("Use demo CPD factors for preview", value=True)
        if not use_demo:
            st.stop()
        use_real_data = False

    # Demo data generation (used when real CPD not available)
    if not use_real_data:
        st.warning("⚠️ Using synthetic demo factors. Results are illustrative only.")

        # Use actual grid dimensions from tensor data if available
        if 'tdt_metadata' in st.session_state:
            meta = st.session_state['tdt_metadata']
            co_vals_demo = np.array(meta['co_vals'])
            cr_vals_demo = np.array(meta['cr_vals'])
            fe_vals_demo = np.array(meta['fe_vals'])
            T_vals_demo = np.array(meta['T_vals'])
            n_co, n_cr, n_fe, n_T = meta['dims']
        else:
            # Fallback defaults
            n_co, n_cr, n_fe, n_T = 20, 20, 20, 31
            co_vals_demo = np.linspace(0, 0.4, n_co)
            cr_vals_demo = np.linspace(0, 0.4, n_cr)
            fe_vals_demo = np.linspace(0, 0.4, n_fe)
            T_vals_demo = np.array(T_list) if 'T_list' in dir() else np.arange(700, 3701, 100)

        R_demo = 6
        np.random.seed(42)  # Reproducible demo

        A_liq = np.random.randn(n_co, R_demo) * 0.1
        A_fcc = np.random.randn(n_co, R_demo) * 0.1
        B_liq = np.random.randn(n_cr, R_demo) * 0.1
        B_fcc = np.random.randn(n_cr, R_demo) * 0.1
        C_liq = np.random.randn(n_fe, R_demo) * 0.1
        C_fcc = np.random.randn(n_fe, R_demo) * 0.1
        D_liq = np.random.randn(n_T, R_demo) * 0.1
        D_fcc = np.random.randn(n_T, R_demo) * 0.1
        lam_liq = np.array([1.0, 0.8, 0.5, 0.3, 0.2, 0.1])
        lam_fcc = np.array([1.0, 0.7, 0.6, 0.3, 0.2, 0.1])

        # Make D factors physically plausible
        if len(T_vals_demo) > 1:
            T_norm = (T_vals_demo - np.mean(T_vals_demo)) / (np.std(T_vals_demo) + 1e-12)
        else:
            T_norm = np.zeros_like(T_vals_demo)
        D_liq[:, 0] = 1.0
        D_liq[:, 1] = T_norm
        D_liq[:, 2] = T_norm**2
        D_fcc[:, 0] = 1.0
        D_fcc[:, 1] = T_norm * 0.9
        D_fcc[:, 2] = T_norm**2 * 1.1

        co_vals_am = co_vals_demo
        cr_vals_am = cr_vals_demo
        fe_vals_am = fe_vals_demo
        T_vals_am = T_vals_demo

    # AM Analysis Sub-tabs
    am_subtab = st.radio("AM Analysis", 
                        ["🔥 Transition Temperature", "🌡️ Thermal Response", 
                         "🎯 Composition Sensitivity", "⚠️ Defect Susceptibility",
                         "🔗 Gradient Design"],
                        horizontal=True)

    if am_subtab == "🔥 Transition Temperature":
        render_am_transition_surface_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                          D_liq, D_fcc, lam_liq, lam_fcc,
                                          co_vals_am, cr_vals_am, fe_vals_am, T_vals_am)

    elif am_subtab == "🌡️ Thermal Response":
        render_am_temperature_factors_tab(D_liq, D_fcc, T_vals_am, lam_liq, lam_fcc)

    elif am_subtab == "🎯 Composition Sensitivity":
        render_am_sensitivity_tab(A_liq, B_liq, C_liq, lam_liq,
                                  co_vals_am, cr_vals_am, fe_vals_am)

    elif am_subtab == "⚠️ Defect Susceptibility":
        render_am_defect_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                            D_liq, D_fcc, lam_liq, lam_fcc,
                            co_vals_am, cr_vals_am, fe_vals_am, T_vals_am)

    elif am_subtab == "🔗 Gradient Design":
        # Validate session state first
        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.error(f"❌ Cannot load gradient design: {msg}")
            st.info("💡 Please run CPD for both LIQUID and FCC phases in the Tensor Decomposition tab first.")
        else:
            render_gradient_design_tab(
                factors['A_liq'], factors['A_fcc'], 
                factors['B_liq'], factors['B_fcc'],
                factors['C_liq'], factors['C_fcc'],
                factors['D_liq'], factors['D_fcc'],
                factors['lam_liq'], factors['lam_fcc'],
                factors['co_vals'], factors['cr_vals'], 
                factors['fe_vals'], factors['T_vals']
            )

# =============================================
# EXPORT & FOOTER
# =============================================
with st.expander("💾 Export & Data", expanded=False):
    col1, col2 = st.columns(2)
    
    # Export current view data
    if render_mode in ["Phase Boundary (Scientific)", "Markers (Distinct Shapes)", "Ternary Flat Projection"]:
        pts = generate_tetrahedral_grid(35)
        G_liq = interp_liq(pts)
        G_fcc = interp_fcc(pts)
        valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        export_df = pd.DataFrame({
            "Co": pts[valid, 0], "Cr": pts[valid, 1], "Fe": pts[valid, 2],
            "Ni": 1.0 - pts[valid, 0] - pts[valid, 1] - pts[valid, 2],
            "G_LIQ": G_liq[valid], "G_FCC": G_fcc[valid],
            "dG": G_liq[valid] - G_fcc[valid],
            "Stable_Phase": np.where(G_liq[valid] <= G_fcc[valid], "LIQUID", "FCC"),
            "T": T_val
        })
        csv = export_df.to_csv(index=False)
        col1.download_button("📥 Download CSV", csv, f"CoCrFeNi_T{T_val}K.csv", "text/csv")
    
    # Export figure as HTML
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    col2.download_button("🌐 Download HTML", html_str, f"CoCrFeNi_T{T_val}K.html", "text/html")
    
    # Query history
    if len(st.session_state.query_history) > 0:
        st.subheader("Query History")
        hist_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(hist_df.style.format({
            "Co": "{:.3f}", "Cr": "{:.3f}", "Fe": "{:.3f}", "Ni": "{:.3f}",
            "G_LIQ": "{:.0f}", "G_FCC": "{:.0f}", "G_stable": "{:.0f}", "dG": "{:.0f}"
        }), use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()

with st.expander("📖 How to Read Each Mode", expanded=True):
    st.markdown("""
    ### Phase Boundary (Scientific) — **Most Accurate for Research**
    True tetrahedral composition space. 🔴 circles = LIQUID, 🔵 diamonds = FCC, 🟡 X's = ΔG≈0 boundary.  
    **Uncertainty fading**: points far from CALPHAD data are translucent. **Cross-section plane**: slice at fixed Ni.
    
    ### Dual SH Surfaces (Temperature Morph) — **Aesthetic + Physical Insight**
    Two spherical harmonic surfaces with **temperature-driven shape morphing**:
    - 🔴 **LIQUID** (Red): Becomes **larger, smoother, shinier** at high T (fluid expansion, entropy dominance)
    - 🔵 **FCC** (Blue): Becomes **smaller, faceted, matte** at low T (crystalline order, enthalpy dominance)
    - Auto l_max: LIQUID uses lower l at high T (smooth); FCC uses higher l at low T (faceted)
    - 🟡 Gold line = ΔG = 0 intersection (exact phase boundary)
    
    ### ΔG Difference Surface — **Driving Force Visualization**
    Single sphere deformed by ΔG. 🔴 **Red/dented inward** = LIQUID stable (negative ΔG), 🔵 **Blue/bulged outward** = FCC stable (positive ΔG). Amplitude = magnitude of driving force for phase transformation.
    
    ### Ternary Flat Projection — **Traditional Materials View**
    Standard ternary diagram: x=Co, y=Cr, z=Ni (Fe implicit). Shape = phase, color = ΔG or proximity. Familiar to metallurgists.
    
    ### Markers (Distinct Shapes) — **Classic 3D Scatter**
    Classic 3D scatter with **circle vs diamond** per phase. Boundary points in gold. Simple, interpretable.
    
    ### Animated Temperature Sweep — **Dynamic Phase Evolution**
    Play button morphs between temperatures. Watch LIQUID grow and FCC shrink as T increases. Reveals composition-dependent transition temperatures T*(x).
    """)

# =============================================
# CRITICAL IMPROVEMENTS FOR PRODUCTION USE
# =============================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Production Improvements Needed")

with st.sidebar.expander("Priority 1: Weighted ALS for Incomplete Tensor", expanded=True):
    st.markdown("""
    **Problem**: Current CPD-ALS uses zero-imputation for NaN entries, which biases results for the ~83% sparse simplex-constrained tensor.
    
    **Solution**: Implement weighted ALS that fits ONLY observed entries:
    ```python
    def cpd_als_weighted(tensor, mask, rank, max_iter=100):
        # Use mask as weights in least squares
        # Reference: Tomasi & Bro (2005), "PARAFAC and missing values"
        # Critical for accurate phase boundary prediction
    ```
    
    **Impact**: Reduces phase boundary prediction error by 30-50% near transition region.
    """)

with st.sidebar.expander("Priority 2: Transition-Aware Rank Selection", expanded=False):
    st.markdown("""
    **Problem**: Fixed SVD threshold may miss the transition-sign-change component (r=3) if threshold > 0.02.
    
    **Solution**: Weight reconstruction error by proximity to ΔG=0:
    ```python
    def select_rank_transition_optimal(tensors_by_T, rank_range):
        # Weight errors near ΔG=0 more heavily
        # Ensures CPD captures composition-dependent T*(x) surface
    ```
    
    **Impact**: Guarantees accurate melting point prediction across composition space.
    """)

with st.sidebar.expander("Priority 3: Sparse Tensor Storage", expanded=False):
    st.markdown("""
    **Problem**: Dense 4D arrays waste memory on ~83% NaN entries.
    
    **Solution**: Use scipy.sparse COO format for valid simplex entries only:
    ```python
    from scipy.sparse import coo_array
    # Store only (i,j,k,t,value) tuples for valid points
    # Reduces memory from ~34 MB to ~5.7 MB for typical grid
    ```
    
    **Impact**: Enables larger composition grids (step=0.005) without memory issues.
    """)

with st.sidebar.expander("Priority 4: Transition Surface Extraction", expanded=False):
    st.markdown("""
    **Problem**: Users must manually find T* where ΔG=0 for each composition.
    
    **Solution**: Add module to extract T*(x_Co,x_Cr,x_Fe) surface from CPD factors:
    ```python
    def extract_melting_surface(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, R):
        # For each composition, find T where Σ λᵣ·A·B·C·D = 0
        # Returns 3D array T_melt[Co_idx,Cr_idx,Fe_idx]
    ```
    
    **Impact**: Enables instant "What's the melting point of Co₀.₃Cr₀.₃Fe₀.₃Ni₀.₁?" queries.
    """)

with st.sidebar.expander("Priority 5: Uncertainty Quantification", expanded=False):
    st.markdown("""
    **Problem**: CALPHAD parameters have uncertainty (~0.5-2% of |G|), but predictions are deterministic.
    
    **Solution**: Bootstrap uncertainty propagation:
    ```python
    def bootstrap_gibbs_uncertainty(df, n_bootstrap=50, rel_error=0.01):
        # Perturb G values, re-run CPD, extract T* distribution
        # Returns mean ± std for melting temperature predictions
    ```
    
    **Impact**: Provides confidence intervals: "T* = 1480 ± 25 K" vs just "1480 K".
    """)

# Footer
st.markdown("---")
st.caption("""
🔷 Co-Cr-Fe-Ni Phase Stability Explorer v2 | Thermodynamic Data Tensor Analysis  
Based on CALPHAD computations | Canonical Polyadic Decomposition per Coutinho et al. (2020)  
*For research use. Validate predictions with experimental data before materials selection.*
""")
