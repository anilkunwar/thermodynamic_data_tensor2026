# 🔷 Co-Cr-Fe-Ni Phase Stability Explorer v3: Complete Integrated Implementation

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Co-Cr-Fe-Ni Phase Stability Explorer v3
========================================
Thermodynamic Data Tensor Analysis with Canonical Polyadic Decomposition (CPD)

Based on: Coutinho et al., npj Computational Materials 6, 2 (2020)
System: Quaternary Co-Cr-Fe-Ni high-entropy alloy
Data: Gibbs energies for LIQUID and FCC phases at 31 temperatures (700-3700K, ΔT=100K)
Grid: Composition step ≈ 0.01 mole fraction, simplex constraint: Co+Cr+Fe+Ni=1

Key Features:
- 4D Thermodynamic Data Tensor (Co × Cr × Fe × T) construction
- Weighted ALS for incomplete tensor (simplex-constrained) handling
- Physics-informed factor initialization for faster convergence
- Transition surface extraction: T_melt(x_Co, x_Cr, x_Fe) prediction
- Bootstrap uncertainty quantification for CALPHAD parameter propagation
- Interactive 3D visualization with spherical harmonic surface morphing

Author: Materials Informatics Team
Date: 2026
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, cKDTree
from scipy import linalg
from scipy.special import sph_harm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & GLOBAL CONSTANTS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="CoCrFeNi Phase Stability Explorer v3",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color and symbol libraries for visualization
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

PHASE_SYMBOLS = {"LIQUID": "circle", "FCC": "diamond", "BOUNDARY": "x", "COEXISTENCE": "star"}
PHASE_COLORS = {"LIQUID": "#e74c3c", "FCC": "#2980b9", "BOUNDARY": "#f1c40f", "COEXISTENCE": "#9b59b6"}
PHASE_COLORS_RGBA = {
    "LIQUID": "rgba(231, 76, 60, 0.25)", 
    "FCC": "rgba(41, 128, 185, 0.25)",
    "BOUNDARY": "rgba(241, 196, 15, 0.4)",
    "COEXISTENCE": "rgba(155, 89, 182, 0.3)"
}

# Thermodynamic constants and defaults
DEFAULT_CP_RANK = 6  # Recommended CP rank for Co-Cr-Fe-Ni system
DEFAULT_SVD_THRESHOLD = 0.003  # 0.3% threshold for effective rank estimation
BOOTSTRAP_SAMPLES = 50  # Number of bootstrap samples for uncertainty quantification
CALPHAD_REL_ERROR = 0.01  # 1% relative error model for CALPHAD Gibbs energies
TRANSITION_WEIGHT_SIGMA = 300  # K: Gaussian width for transition-weighted fitting

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading Gibbs energy data...")
def load_all_data(csv_dir=CSV_FILES_DIR):
    """
    Load all Gibbs_*.csv files and concatenate into single DataFrame.
    
    Expected file format: Gibbs_<T>K.csv with columns:
        Co, Cr, Fe, Ni, G_LIQ, G_FCC (all in mole fraction and J/mol)
    
    Returns:
        pd.DataFrame with columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC, T
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    
    if not files:
        st.error(f"❌ No CSV files found in `{csv_dir}`.")
        st.info("Expected files: Gibbs_700K.csv, Gibbs_800K.csv, ..., Gibbs_3700K.csv")
        st.stop()
    
    dfs = []
    for f in files:
        basename = os.path.basename(f)
        try:
            # Parse temperature from filename: Gibbs_<T>K.csv → T (int)
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            
            # Read required columns only for memory efficiency
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            
            # Validate composition constraint
            comp_sum = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
            invalid = ~np.isclose(comp_sum, 1.0, atol=1e-6)
            if invalid.any():
                n_invalid = invalid.sum()
                st.warning(f"⚠️ {n_invalid} rows in {basename} violate Co+Cr+Fe+Ni=1; removing")
                df = df[~invalid]
            
            # Add temperature column
            df["T"] = T
            dfs.append(df)
            
        except Exception as e:
            st.warning(f"⚠️ Skipping {f}: {type(e).__name__}: {e}")
    
    if not dfs:
        st.error("❌ No valid data loaded from any files.")
        st.stop()
    
    # Concatenate all temperature slices
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Compute derived quantity: ΔG = G_LIQ - G_FCC
    df_combined["dG"] = df_combined["G_LIQ"] - df_combined["G_FCC"]
    
    # Determine stable phase at each point
    df_combined["Stable_Phase"] = np.where(
        df_combined["dG"] < -50, "LIQUID",  # Threshold for numerical stability
        np.where(df_combined["dG"] > 50, "FCC", "COEXISTENCE")
    )
    
    return df_combined


# =============================================================================
# TENSOR CONSTRUCTION: 4D THERMODYNAMIC DATA TENSOR
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="Building 4D thermodynamic tensor...")
def build_tensor_data(df):
    """
    Build 4D Thermodynamic Data Tensor (TDT) from DataFrame.
    
    Tensor structure:
        G_LIQ[i, j, k, t] = Gibbs energy of LIQUID phase at:
            - Co = co_vals[i] ∈ [0, 1], step ≈ 0.01
            - Cr = cr_vals[j] ∈ [0, 1], step ≈ 0.01
            - Fe = fe_vals[k] ∈ [0, 1], step ≈ 0.01
            - T  = T_vals[t]  ∈ {700, 800, ..., 3700} K (31 values)
    
    Constraint: Only entries with Co+Cr+Fe ≤ 1 are physically valid (Ni = 1-Σ)
    → ~16.7% of full hypercube contains data; rest filled with NaN
    
    Returns:
        dict with keys:
            'G_LIQ', 'G_FCC': 4D numpy arrays (n_co, n_cr, n_fe, n_T)
            'dims': tuple (n_co, n_cr, n_fe, n_T)
            'co_vals', 'cr_vals', 'fe_vals', 'T_vals': 1D arrays of grid values
            'co_step', 'cr_step', 'fe_step', 'T_step': grid spacing
            'valid_fraction': fraction of tensor entries that are physically valid
    """
    # Extract unique grid values for each dimension
    co_vals = np.sort(df["Co"].unique())
    cr_vals = np.sort(df["Cr"].unique())
    fe_vals = np.sort(df["Fe"].unique())
    T_vals = np.sort(df["T"].unique())
    
    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)
    
    # Create O(1) lookup dictionaries with rounding for floating-point tolerance
    co_to_idx = {round(v, 4): i for i, v in enumerate(co_vals)}
    cr_to_idx = {round(v, 4): i for i, v in enumerate(cr_vals)}
    fe_to_idx = {round(v, 4): i for i, v in enumerate(fe_vals)}
    T_to_idx = {int(T): i for i, T in enumerate(T_vals)}
    
    # Initialize 4D arrays with NaN (invalid entries remain NaN)
    G_LIQ_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    G_FCC_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    
    # Populate tensor with valid simplex points
    valid_count = 0
    for _, row in df.iterrows():
        co = round(row['Co'], 4)
        cr = round(row['Cr'], 4)
        fe = round(row['Fe'], 4)
        T = int(row['T'])
        
        # Check all indices exist in grid
        if (co in co_to_idx and cr in cr_to_idx and 
            fe in fe_to_idx and T in T_to_idx):
            
            i = co_to_idx[co]
            j = cr_to_idx[cr]
            k = fe_to_idx[fe]
            t = T_to_idx[T]
            
            # Verify simplex constraint (should always be true for valid data)
            if co + cr + fe <= 1.0 + 1e-8:
                G_LIQ_tdt[i, j, k, t] = row['G_LIQ']
                G_FCC_tdt[i, j, k, t] = row['G_FCC']
                valid_count += 1
    
    # Compute tensor statistics
    full_size = n_co * n_cr * n_fe * n_T
    valid_fraction = valid_count / full_size if full_size > 0 else 0
    
    return {
        'G_LIQ': G_LIQ_tdt,
        'G_FCC': G_FCC_tdt,
        'dims': (n_co, n_cr, n_fe, n_T),
        'co_vals': co_vals,
        'cr_vals': cr_vals,
        'fe_vals': fe_vals,
        'T_vals': T_vals,
        'co_step': np.min(np.diff(co_vals)) if len(co_vals) > 1 else 0,
        'cr_step': np.min(np.diff(cr_vals)) if len(cr_vals) > 1 else 0,
        'fe_step': np.min(np.diff(fe_vals)) if len(fe_vals) > 1 else 0,
        'T_step': np.min(np.diff(T_vals)) if len(T_vals) > 1 else 0,
        'valid_fraction': valid_fraction,
        'full_size': full_size,
        'valid_count': valid_count
    }


def unfold_tensor(tensor, mode):
    """
    Unfold (matricize) 4D tensor along specified mode.
    
    Mode mapping:
        0 → Co dimension:   shape (n_co, n_cr×n_fe×n_T)
        1 → Cr dimension:   shape (n_cr, n_co×n_fe×n_T)
        2 → Fe dimension:   shape (n_fe, n_co×n_cr×n_T)
        3 → T dimension:    shape (n_T, n_co×n_cr×n_fe)
    
    Used for: SVD-based rank analysis, CPD initialization
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
        raise ValueError(f"Invalid mode {mode}; must be 0, 1, 2, or 3")


# =============================================================================
# RANK ANALYSIS: SVD-BASED EFFECTIVE RANK ESTIMATION
# =============================================================================

def svd_rank_analysis(matrix, threshold=DEFAULT_SVD_THRESHOLD):
    """
    Estimate effective rank via SVD with robust NaN handling.
    
    For Co-Cr-Fe-Ni with 31 temperatures:
        - Temperature mode: Expected rank 3-4 (baseline + entropy + Cp + transition)
        - Composition modes: Expected rank 5-7 (polynomial mixing + magnetic effects)
    
    Args:
        matrix: 2D numpy array (may contain NaN)
        threshold: Fraction of max singular value for rank cutoff (default: 0.003 = 0.3%)
    
    Returns:
        rank: Estimated effective rank (int)
        s: Full array of singular values
        s_norm: Normalized singular values (s / s_max)
    """
    matrix_filled = matrix.copy().astype(np.float64)
    
    # Column-wise NaN imputation using column mean (preserves column structure)
    for col in range(matrix_filled.shape[1]):
        col_data = matrix_filled[:, col]
        valid = ~np.isnan(col_data)
        if np.sum(valid) > 0:
            col_mean = np.mean(col_data[valid])
            matrix_filled[:, col] = np.where(np.isnan(col_data), col_mean, col_data)
        else:
            matrix_filled[:, col] = 0.0
    
    # Check for zero matrix
    if np.linalg.norm(matrix_filled) < 1e-12:
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))
    
    # Compute SVD
    try:
        U, s, Vh = linalg.svd(matrix_filled, full_matrices=False)
    except Exception as e:
        st.warning(f"⚠️ SVD failed: {type(e).__name__}: {e}")
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))
    
    # Normalize singular values
    s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
    s_norm = s / s_max
    
    # Count singular values above threshold
    rank = int(np.sum(s_norm > threshold))
    
    return rank, s, s_norm


# =============================================================================
# CANONICAL POLYADIC DECOMPOSITION: WEIGHTED ALS FOR INCOMPLETE TENSORS
# =============================================================================

def cpd_als_weighted_incomplete(tensor, mask, rank, max_iter=100, tol=1e-6, 
                                T_vals=None, co_vals=None, cr_vals=None, fe_vals=None):
    """
    4-way CP decomposition via Alternating Least Squares with weighted fitting
    for incomplete (simplex-constrained) tensors.
    
    Only observed entries (mask=True) contribute to the least-squares solution.
    Essential for Co-Cr-Fe-Ni where ~83% of hypercube entries are NaN.
    
    Physics-informed initialization (optional):
        - Temperature factor D: [constant, linear, quadratic, tanh-transition]
        - Composition factors: Legendre polynomial basis for smoothness
    
    Args:
        tensor: 4D numpy array with NaN for invalid entries
        mask: Boolean 4D array, True where tensor has valid data
        rank: Target CP rank R
        max_iter: Maximum ALS iterations
        tol: Convergence tolerance on relative error
        T_vals, co_vals, cr_vals, fe_vals: Optional grid values for physics-informed init
    
    Returns:
        A, B, C, D: Factor matrices (n_co×R), (n_cr×R), (n_fe×R), (n_T×R)
        lam: Component weights array (R,)
        error: Final RMSE on observed entries
        history: Dict with convergence history for diagnostics
    """
    I, J, K, L = tensor.shape  # L = n_T = 31 for Co-Cr-Fe-Ni
    
    # Initialize factor matrices
    if T_vals is not None and rank >= 3:
        # Physics-informed initialization for temperature factor D
        T_arr = np.array(T_vals)
        T_mean, T_std = np.mean(T_arr), np.std(T_arr) + 1e-12
        T_norm = (T_arr - T_mean) / T_std
        
        D = np.zeros((L, rank))
        D[:, 0] = 1.0  # r=1: Constant baseline (enthalpy offset)
        if rank >= 2:
            D[:, 1] = T_norm  # r=2: Linear entropy term (-S·T)
        if rank >= 3:
            D[:, 2] = (T_norm**2 - 1) * 0.5  # r=3: Orthogonalized quadratic (Cp)
        if rank >= 4:
            # r=4: tanh-like transition function centered near median T
            T_trans = np.tanh(2 * (T_norm - np.median(T_norm)))
            D[:, 3] = T_trans - np.mean(T_trans)  # Zero-mean
        if rank > 4:
            D[:, 4:] = np.random.rand(L, rank-4) * 0.01  # Higher-order: random
    else:
        # Default: SVD initialization on mode-0 unfolding
        X_unfolded = unfold_tensor(np.where(mask, tensor, 0), mode=0)
        try:
            U, s, Vh = linalg.svd(X_unfolded, full_matrices=False)
            A = U[:, :rank] * np.sqrt(s[:rank])
        except:
            A = np.random.rand(I, rank) * 0.1
        D = np.random.rand(L, rank) * 0.1
    
    # Initialize composition factors with Legendre polynomial basis for smoothness
    def legendre_basis(x, degree):
        """Normalized Legendre polynomials on [0, 1]"""
        x_norm = 2*x - 1  # Map [0,1] → [-1,1]
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return x_norm
        elif degree == 2:
            return 0.5 * (3*x_norm**2 - 1)
        elif degree == 3:
            return 0.5 * (5*x_norm**3 - 3*x_norm)
        else:
            return np.random.rand(len(x)) * 0.1
    
    if co_vals is not None:
        B = np.column_stack([legendre_basis(cr_vals, d) for d in range(min(rank, 4))])
        if rank > 4:
            B = np.hstack([B, np.random.rand(J, rank-4) * 0.01])
    else:
        B = np.random.rand(J, rank) * 0.1
    
    if cr_vals is not None:
        C = np.column_stack([legendre_basis(fe_vals, d) for d in range(min(rank, 4))])
        if rank > 4:
            C = np.hstack([C, np.random.rand(K, rank-4) * 0.01])
    else:
        C = np.random.rand(K, rank) * 0.1
    
    # ALS iterations
    prev_error = np.inf
    history = {'errors': [], 'A_norms': [], 'D_norms': []}
    
    for iteration in range(max_iter):
        # === Update A (Co factor) ===
        for i in range(I):
            # Extract observed entries for this Co value
            valid = mask[i, :, :, :].ravel()
            if np.sum(valid) <= rank:
                continue  # Skip if insufficient data
            
            # Build Khatri-Rao product BCD for observed columns only
            BCD_valid = np.zeros((np.sum(valid), rank))
            for r in range(rank):
                # Khatri-Rao: column-wise Kronecker product
                BCD_valid[:, r] = np.kron(
                    np.kron(D[:, r], C[:, r]), 
                    B[:, r]
                )[valid]
            
            # Weighted least squares: fit only observed entries
            X_i_valid = tensor[i, :, :, :].ravel()[valid]
            try:
                A[i, :], _, _, _ = linalg.lstsq(BCD_valid, X_i_valid, cond=1e-12)
            except:
                pass  # Keep previous value if LS fails
        
        # Normalize A columns
        norms_A = np.linalg.norm(A, axis=0) + 1e-12
        A = A / norms_A
        
        # === Update B (Cr factor) ===
        for j in range(J):
            valid = mask[:, j, :, :].transpose(1, 0, 2).ravel()
            if np.sum(valid) <= rank:
                continue
            
            ACD_valid = np.zeros((np.sum(valid), rank))
            for r in range(rank):
                ACD_valid[:, r] = np.kron(
                    np.kron(D[:, r], C[:, r]),
                    A[:, r]
                )[mask[:, j, :, :].transpose(1, 0, 2).ravel()[valid]]
            
            X_j_valid = tensor[:, j, :, :].transpose(1, 0, 2).ravel()[valid]
            try:
                B[j, :], _, _, _ = linalg.lstsq(ACD_valid, X_j_valid, cond=1e-12)
            except:
                pass
        
        norms_B = np.linalg.norm(B, axis=0) + 1e-12
        B = B / norms_B
        
        # === Update C (Fe factor) ===
        for k in range(K):
            valid = mask[:, :, k, :].transpose(2, 0, 1).ravel()
            if np.sum(valid) <= rank:
                continue
            
            ABD_valid = np.zeros((np.sum(valid), rank))
            for r in range(rank):
                ABD_valid[:, r] = np.kron(
                    np.kron(D[:, r], B[:, r]),
                    A[:, r]
                )[mask[:, :, k, :].transpose(2, 0, 1).ravel()[valid]]
            
            X_k_valid = tensor[:, :, k, :].transpose(2, 0, 1).ravel()[valid]
            try:
                C[k, :], _, _, _ = linalg.lstsq(ABD_valid, X_k_valid, cond=1e-12)
            except:
                pass
        
        norms_C = np.linalg.norm(C, axis=0) + 1e-12
        C = C / norms_C
        
        # === Update D (Temperature factor) ===
        for t in range(L):
            valid = mask[:, :, :, t].ravel()
            if np.sum(valid) <= rank:
                continue
            
            ABC_valid = np.zeros((np.sum(valid), rank))
            for r in range(rank):
                ABC_valid[:, r] = np.kron(
                    np.kron(C[:, r], B[:, r]),
                    A[:, r]
                )[valid]
            
            X_t_valid = tensor[:, :, :, t].ravel()[valid]
            try:
                D[t, :], _, _, _ = linalg.lstsq(ABC_valid, X_t_valid, cond=1e-12)
            except:
                pass
        
        norms_D = np.linalg.norm(D, axis=0) + 1e-12
        D = D / norms_D
        
        # === Compute reconstruction error on OBSERVED entries only ===
        recon = np.zeros_like(tensor)
        for r in range(rank):
            # Outer product of factor columns
            recon += np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)
        
        # RMSE on observed entries
        observed_residuals = (tensor - recon)[mask]
        if len(observed_residuals) > 0:
            error = np.sqrt(np.mean(observed_residuals**2))
        else:
            error = np.inf
        
        # Store convergence history
        history['errors'].append(error)
        history['A_norms'].append(norms_A.copy())
        history['D_norms'].append(norms_D.copy())
        
        # Check convergence
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    # === Compute component weights lambda ===
    lam = np.ones(rank)
    for r in range(rank):
        lam[r] = (np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * 
                  np.linalg.norm(C[:, r]) * np.linalg.norm(D[:, r]))
    
    return A, B, C, D, lam, error, history


# =============================================================================
# TRANSITION SURFACE EXTRACTION: MELTING POINT PREDICTION
# =============================================================================

def extract_transition_surface_cpd(A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ,
                                  A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC,
                                  co_vals, cr_vals, fe_vals, T_vals, rank):
    """
    Compute 3D array T_melt[Co_idx, Cr_idx, Fe_idx] = transition temperature (K)
    where ΔG = G_LIQ - G_FCC = 0, using CPD factors for both phases.
    
    For each composition point:
        1. Evaluate G_LIQ(T) and G_FCC(T) at all 31 temperatures via CPD
        2. Compute ΔG(T) = G_LIQ(T) - G_FCC(T)
        3. Find T* where ΔG(T*) = 0 via linear interpolation
        4. Store T* or np.nan if no sign change in temperature range
    
    Args:
        A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ: CPD factors for LIQUID phase
        A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC: CPD factors for FCC phase
        co_vals, cr_vals, fe_vals, T_vals: Grid values for each dimension
        rank: CP rank used for decomposition
    
    Returns:
        T_melt: 3D numpy array (n_co, n_cr, n_fe) with transition temperatures
        dG_min, dG_max: Min/max ΔG values for diagnostics
    """
    n_co, n_cr, n_fe = len(co_vals), len(cr_vals), len(fe_vals)
    n_T = len(T_vals)
    
    T_melt = np.full((n_co, n_cr, n_fe), np.nan)
    dG_all = []
    
    for i in range(n_co):
        for j in range(n_cr):
            for k in range(n_fe):
                # Skip invalid simplex points
                if co_vals[i] + cr_vals[j] + fe_vals[k] > 1.0 + 1e-8:
                    continue
                
                # Evaluate G_LIQ(T) and G_FCC(T) at all temperatures via CPD
                G_LIQ_T = np.zeros(n_T)
                G_FCC_T = np.zeros(n_T)
                
                for t in range(n_T):
                    # LIQUID phase
                    g_liq = 0.0
                    for r in range(rank):
                        g_liq += lam_LIQ[r] * A_LIQ[i,r] * B_LIQ[j,r] * C_LIQ[k,r] * D_LIQ[t,r]
                    G_LIQ_T[t] = g_liq
                    
                    # FCC phase
                    g_fcc = 0.0
                    for r in range(rank):
                        g_fcc += lam_FCC[r] * A_FCC[i,r] * B_FCC[j,r] * C_FCC[k,r] * D_FCC[t,r]
                    G_FCC_T[t] = g_fcc
                
                # Compute ΔG(T) = G_LIQ - G_FCC
                dG_T = G_LIQ_T - G_FCC_T
                dG_all.extend(dG_T)
                
                # Find T* where ΔG=0 via interpolation
                if np.sign(dG_T[0]) != np.sign(dG_T[-1]):  # Sign change exists
                    # Find bracketing indices
                    for t in range(n_T - 1):
                        if dG_T[t] * dG_T[t+1] < 0:
                            # Linear interpolation
                            frac = abs(dG_T[t]) / (abs(dG_T[t]) + abs(dG_T[t+1]) + 1e-12)
                            T_melt[i,j,k] = T_vals[t] + frac * (T_vals[t+1] - T_vals[t])
                            break
    
    dG_min = np.min(dG_all) if dG_all else 0
    dG_max = np.max(dG_all) if dG_all else 0
    
    return T_melt, dG_min, dG_max


# =============================================================================
# UNCERTAINTY QUANTIFICATION: BOOTSTRAP PROPAGATION
# =============================================================================

def bootstrap_gibbs_uncertainty(df, n_bootstrap=BOOTSTRAP_SAMPLES, 
                               rel_error=CALPHAD_REL_ERROR, rank=DEFAULT_CP_RANK):
    """
    Propagate CALPHAD parameter uncertainty through tensor and CPD via bootstrap.
    
    Assumes relative error model: σ_G ≈ rel_error × |G| (typical: 0.5-2% for CALPHAD)
    
    For each bootstrap sample:
        1. Perturb G_LIQ and G_FCC with Gaussian noise
        2. Rebuild tensor and run CPD
        3. Extract transition surface T_melt
        4. Store result
    
    Returns mean and std of T_melt across bootstrap samples for confidence intervals.
    
    Args:
        df: Original DataFrame with Gibbs energies
        n_bootstrap: Number of bootstrap samples (default: 50)
        rel_error: Relative error for Gibbs energy perturbation (default: 0.01 = 1%)
        rank: CP rank for decomposition
    
    Returns:
        T_melt_mean: Mean transition temperature array (n_co, n_cr, n_fe)
        T_melt_std: Standard deviation array for confidence intervals
        convergence_stats: Dict with bootstrap convergence statistics
    """
    T_melt_samples = []
    error_samples = []
    
    # Get grid values from original data
    co_vals = np.sort(df["Co"].unique())
    cr_vals = np.sort(df["Cr"].unique())
    fe_vals = np.sort(df["Fe"].unique())
    T_vals = np.sort(df["T"].unique())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for b in range(n_bootstrap):
        status_text.text(f"Bootstrap sample {b+1}/{n_bootstrap}...")
        
        # Perturb Gibbs energies with Gaussian noise
        df_perturbed = df.copy()
        
        # Relative error model: σ = rel_error × |G|
        noise_liq = np.random.normal(0, rel_error * np.abs(df['G_LIQ']))
        noise_fcc = np.random.normal(0, rel_error * np.abs(df['G_FCC']))
        
        df_perturbed['G_LIQ'] = df['G_LIQ'] + noise_liq
        df_perturbed['G_FCC'] = df['G_FCC'] + noise_fcc
        
        # Rebuild tensor
        tdt_b = build_tensor_data(df_perturbed)
        
        # Run CPD for both phases
        mask_liq = ~np.isnan(tdt_b['G_LIQ'])
        mask_fcc = ~np.isnan(tdt_b['G_FCC'])
        
        try:
            A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ, err_LIQ, _ = cpd_als_weighted_incomplete(
                tdt_b['G_LIQ'], mask_liq, rank, max_iter=50, tol=1e-5,
                T_vals=T_vals, co_vals=co_vals, cr_vals=cr_vals, fe_vals=fe_vals
            )
            
            A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC, err_FCC, _ = cpd_als_weighted_incomplete(
                tdt_b['G_FCC'], mask_fcc, rank, max_iter=50, tol=1e-5,
                T_vals=T_vals, co_vals=co_vals, cr_vals=cr_vals, fe_vals=fe_vals
            )
            
            # Extract transition surface
            T_melt_b, _, _ = extract_transition_surface_cpd(
                A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ,
                A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC,
                co_vals, cr_vals, fe_vals, T_vals, rank
            )
            
            T_melt_samples.append(T_melt_b)
            error_samples.append((err_LIQ, err_FCC))
            
        except Exception as e:
            st.warning(f"⚠️ Bootstrap {b+1} failed: {type(e).__name__}: {e}")
            continue
        
        progress_bar.progress((b + 1) / n_bootstrap)
    
    status_text.empty()
    progress_bar.empty()
    
    if not T_melt_samples:
        st.error("❌ No successful bootstrap samples; cannot compute uncertainty.")
        return None, None, {}
    
    # Compute mean and std across successful samples
    T_melt_array = np.array(T_melt_samples)  # Shape: (n_success, n_co, n_cr, n_fe)
    T_melt_mean = np.mean(T_melt_array, axis=0)
    T_melt_std = np.std(T_melt_array, axis=0)
    
    convergence_stats = {
        'n_successful': len(T_melt_samples),
        'n_total': n_bootstrap,
        'mean_err_LIQ': np.mean([e[0] for e in error_samples]),
        'mean_err_FCC': np.mean([e[1] for e in error_samples]),
        'std_T_melt': np.mean(T_melt_std[T_melt_std > 0]) if np.any(T_melt_std > 0) else 0
    }
    
    return T_melt_mean, T_melt_std, convergence_stats


# =============================================================================
# INTERPOLATION & QUERY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    """
    Build LinearNDInterpolator for G_LIQ and G_FCC at specified temperature.
    
    Uses only data points at temperature T for interpolation in composition space.
    """
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    
    pts = df_T[["Co", "Cr", "Fe"]].values
    
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values, fill_value=np.nan)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values, fill_value=np.nan)
    
    return interp_liq, interp_fcc


def query_phase_stability(df, T_query, x_Co, x_Cr, x_Fe):
    """
    Query phase stability for given composition and temperature.
    
    Returns dict with Gibbs energies, ΔG, stable phase, and interpolation status.
    """
    x_Ni = 1.0 - (x_Co + x_Cr + x_Fe)
    
    if x_Ni < -1e-8:
        return {
            'error': f"Invalid composition: Co+Cr+Fe = {x_Co+x_Cr+x_Fe:.3f} > 1.0",
            'valid': False
        }
    
    # Build interpolators for query temperature
    interp_liq, interp_fcc = build_interpolators_for_T(df, T_query)
    
    if interp_liq is None:
        return {
            'error': f"No data available at T = {T_query} K",
            'valid': False
        }
    
    # Query interpolators
    pt = np.array([[x_Co, x_Cr, x_Fe]])
    G_LIQ = float(interp_liq(pt)[0])
    G_FCC = float(interp_fcc(pt)[0])
    
    if np.isnan(G_LIQ) or np.isnan(G_FCC):
        return {
            'error': "Query point outside data convex hull; interpolation failed",
            'valid': False,
            'x_Ni': x_Ni
        }
    
    # Determine stable phase
    dG = G_LIQ - G_FCC
    if dG < -50:
        phase = "LIQUID"
    elif dG > 50:
        phase = "FCC"
    else:
        phase = "COEXISTENCE"
    
    return {
        'valid': True,
        'T': T_query,
        'Co': x_Co, 'Cr': x_Cr, 'Fe': x_Fe, 'Ni': x_Ni,
        'G_LIQ': G_LIQ, 'G_FCC': G_FCC, 'dG': dG,
        'Phase': phase,
        'driving_force': abs(dG)
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def generate_tetrahedral_grid(resolution=25):
    """
    Generate uniform grid points in Co-Cr-Fe composition simplex.
    
    Constraint: Co ≥ 0, Cr ≥ 0, Fe ≥ 0, Co+Cr+Fe ≤ 1
    Ni is dependent: Ni = 1 - (Co+Cr+Fe)
    
    Returns:
        pts: N×3 array of valid composition points [Co, Cr, Fe]
    """
    x = np.linspace(0, 1, resolution)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    return grid_pts[valid_mask]


def compute_data_proximity(pts, data_pts, max_dist=0.15):
    """
    Compute normalized proximity to nearest data point.
    
    proximity = 1.0 at data points, decays linearly to 0 at max_dist.
    Used for uncertainty visualization: fade points far from data.
    """
    tree = cKDTree(data_pts)
    dists, _ = tree.query(pts, k=1)
    proximity = np.clip(1.0 - dists / max_dist, 0.0, 1.0)
    return proximity


def find_phase_boundary_points(pts, dG_values, threshold=50.0):
    """
    Find points near phase boundary (|ΔG| < threshold).
    
    Returns boundary points and their ΔG values for visualization.
    """
    boundary_mask = np.abs(dG_values) < threshold
    return pts[boundary_mask], dG_values[boundary_mask]


# =============================================================================
# SPHERICAL HARMONICS FOR SURFACE VISUALIZATION
# =============================================================================

def get_real_sph_harm(l, m, theta, phi):
    """
    Compute real spherical harmonics for surface visualization.
    
    Converts complex scipy.special.sph_harm to real-valued functions:
        m > 0: √2 × Re[Y_l^m]
        m < 0: √2 × Im[Y_l^|m|]
        m = 0: Y_l^0 (already real)
    """
    # Note: scipy uses (m, l, theta, phi) order, with theta=azimuth, phi=polar
    Y_complex = sph_harm(m, l, theta, phi)
    
    if m > 0:
        return np.sqrt(2.0) * Y_complex.real
    elif m < 0:
        Y_pos = sph_harm(abs(m), l, theta, phi)
        return np.sqrt(2.0) * Y_pos.imag
    else:
        return Y_complex.real


def sample_g_on_sphere(interp_liq, interp_fcc, R_fixed, n_theta=60, n_phi=60):
    """
    Sample Gibbs energies on spherical composition grid for visualization.
    
    Maps spherical coordinates (θ, φ) to composition space:
        Co = R·sin(φ)·cos(θ)
        Cr = R·sin(φ)·sin(θ)
        Fe = R·cos(φ)
    
    Only samples points satisfying Co+Cr+Fe ≤ 1 (simplex constraint).
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    
    # Map to composition space
    x = R_fixed * np.sin(PH) * np.cos(TH)
    y = R_fixed * np.sin(PH) * np.sin(TH)
    z = R_fixed * np.cos(PH)
    
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    valid = (pts[:, 0] + pts[:, 1] + pts[:, 2]) <= 1.0
    
    # Query interpolators
    G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
    G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)
    
    # Compute stable phase and ΔG
    G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
    dG = G_liq - G_fcc
    
    # Mask invalid points
    valid = valid & ~np.isnan(G_stable)
    
    return (TH, PH, 
            G_stable.reshape(TH.shape), 
            dG.reshape(TH.shape), 
            valid.reshape(TH.shape), 
            pts)


def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=3):
    """
    Fit spherical harmonic coefficients to Gibbs energy data on sphere.
    
    Solves linear least squares: g ≈ Σ_{l=0}^{l_max} Σ_{m=-l}^{l} c_{lm} Y_{lm}(θ,φ)
    """
    theta_flat = theta_vals.ravel()
    phi_flat = phi_vals.ravel()
    g_flat = g_vals.ravel()
    
    # Mask invalid entries
    valid = ~np.isnan(g_flat)
    theta_flat = theta_flat[valid]
    phi_flat = phi_flat[valid]
    g_flat = g_flat[valid]
    
    if len(theta_flat) == 0:
        return None, l_max
    
    # Build design matrix with real spherical harmonics
    A = []
    for t, p in zip(theta_flat, phi_flat):
        row = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                y = get_real_sph_harm(l, m, t, p)
                row.append(y)
        A.append(row)
    
    A = np.array(A)
    
    # Solve least squares
    try:
        coeffs, _, _, _ = linalg.lstsq(A, g_flat, cond=1e-12)
        return coeffs, l_max
    except:
        return None, l_max


def reconstruct_sh_surface(theta_grid, phi_grid, coeffs, l_max):
    """Reconstruct Gibbs energy surface from spherical harmonic coefficients."""
    recon = np.zeros_like(theta_grid, dtype=float)
    idx = 0
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
            recon += coeffs[idx] * Y
            idx += 1
    
    return recon


def extract_dg_zero_contour(TH, PH, dG_grid, R_fixed):
    """
    Extract ΔG=0 contour via edge-walking and linear interpolation.
    
    Returns x, y, z coordinates of contour points in composition space.
    """
    contours_x, contours_y, contours_z = [], [], []
    
    # Horizontal edges (constant phi)
    for i in range(dG_grid.shape[0]):
        for j in range(dG_grid.shape[1] - 1):
            if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i,j+1])):
                continue
            if dG_grid[i,j] * dG_grid[i,j+1] < 0:  # Sign change
                t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i,j+1]) + 1e-12)
                th_mid = TH[i,j] + t * (TH[i,j+1] - TH[i,j])
                ph_mid = PH[i,j] + t * (PH[i,j+1] - PH[i,j])
                contours_x.append(R_fixed * np.sin(ph_mid) * np.cos(th_mid))
                contours_y.append(R_fixed * np.sin(ph_mid) * np.sin(th_mid))
                contours_z.append(R_fixed * np.cos(ph_mid))
    
    # Vertical edges (constant theta)
    for i in range(dG_grid.shape[0] - 1):
        for j in range(dG_grid.shape[1]):
            if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i+1,j])):
                continue
            if dG_grid[i,j] * dG_grid[i+1,j] < 0:
                t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i+1,j]) + 1e-12)
                th_mid = TH[i,j] + t * (TH[i+1,j] - TH[i,j])
                ph_mid = PH[i,j] + t * (PH[i+1,j] - PH[i,j])
                contours_x.append(R_fixed * np.sin(ph_mid) * np.cos(th_mid))
                contours_y.append(R_fixed * np.sin(ph_mid) * np.sin(th_mid))
                contours_z.append(R_fixed * np.cos(ph_mid))
    
    return (np.array(contours_x), np.array(contours_y), np.array(contours_z))


# =============================================================================
# TEMPERATURE-DRIVEN SHAPE MORPHING FUNCTIONS
# =============================================================================

def get_liquid_radius(G_sh, sh_R_fixed, T_factor):
    """
    Compute LIQUID phase radius with temperature-driven morphing.
    
    Fluid, expanded, smooth behavior at high T:
        - Thermal expansion: radius increases with T
        - Fluid undulations: sinusoidal modulation based on G
        - Shininess: high specular reflection at high T
    """
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    
    thermal_exp = 1.0 + 0.35 * T_factor
    fluid_dist = 0.12 * np.sin(2 * np.pi * norm) * (0.5 + 0.5 * T_factor)
    
    return sh_R_fixed * (thermal_exp + 0.22 * norm + fluid_dist)


def get_fcc_radius(G_sh, sh_R_fixed, T_factor):
    """
    Compute FCC phase radius with temperature-driven morphing.
    
    Crystalline, faceted, rigid behavior at low T:
        - Thermal contraction: radius decreases with T
        - Crystalline ripples: multiple harmonic modulation
        - Matte appearance: low specular reflection at low T
    """
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    
    rigidity = 1.0 - 0.20 * T_factor
    crystal_factor = 0.28 * (1.0 - T_factor)
    
    # Multiple harmonics for crystalline faceting
    crystal_ripples = crystal_factor * (
        0.6 * np.sin(6 * np.pi * norm) +
        0.3 * np.sin(10 * np.pi * norm) +
        0.1 * np.sin(14 * np.pi * norm)
    )
    
    return sh_R_fixed * (rigidity + 0.20 * norm + crystal_ripples)


# =============================================================================
# STREAMLIT APP: MAIN INTERFACE
# =============================================================================

def main():
    # =============================================================================
    # HEADER & INTRODUCTION
    # =============================================================================
    st.title("🔷 Co-Cr-Fe-Ni Phase Stability Explorer v3")
    st.markdown(r"""
    **Thermodynamic Data Tensor Analysis with Canonical Polyadic Decomposition**
    
    This application analyzes Gibbs energy data for the quaternary Co-Cr-Fe-Ni high-entropy alloy system
    using a 4D thermodynamic data tensor and tensor decomposition techniques.
    
    **Key capabilities:**
    - 🎨 Interactive 3D visualization of phase stability in composition-temperature space
    - 📊 Tensor decomposition (CPD) for compression and interpretation
    - 🔍 Instant phase prediction for arbitrary compositions and temperatures
    - 🌡️ Melting point surface extraction: T_melt(x_Co, x_Cr, x_Fe)
    - 📈 Uncertainty quantification via bootstrap propagation
    
    *Data: 31 temperatures (700-3700 K, ΔT=100 K) | Composition grid: ~0.01 mole fraction step*
    """)
    
    # =============================================================================
    # LOAD DATA
    # =============================================================================
    with st.spinner("📥 Loading Gibbs energy data..."):
        df = load_all_data()
    
    if df is None or len(df) == 0:
        st.error("❌ Failed to load data. Please check CSV files in `csv_files/` directory.")
        return
    
    # =============================================================================
    # GLOBAL STATISTICS
    # =============================================================================
    T_list = sorted(df["T"].unique())
    T_min, T_max = min(T_list), max(T_list)
    T_range = T_max - T_min if T_max > T_min else 1.0
    
    # Global Gibbs energy ranges for consistent visualization scaling
    G_LIQ_global_min = df["G_LIQ"].min()
    G_LIQ_global_max = df["G_LIQ"].max()
    G_FCC_global_min = df["G_FCC"].min()
    G_FCC_global_max = df["G_FCC"].max()
    G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
    G_global_max = max(G_LIQ_global_max, G_FCC_global_max)
    
    dG_global_min = df["dG"].min()
    dG_global_max = df["dG"].max()
    dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))
    
    # Convex hull for uncertainty/distance calculations
    all_pts = df[["Co", "Cr", "Fe"]].values
    try:
        data_hull = ConvexHull(all_pts)
        HULL_AVAILABLE = True
    except:
        HULL_AVAILABLE = False
        data_hull = None
    
    # =============================================================================
    # SIDEBAR: CONTROL PANEL
    # =============================================================================
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # --- PRESET VIEWS ---
        st.subheader("⚡ Quick Presets")
        preset = st.selectbox("Load Preset", [
            "Custom", "Low-T FCC Crystal", "High-T Liquid Melt", 
            "Transition Region", "Maximum Contrast"
        ], index=0)
        
        # --- TEMPERATURE SELECTION ---
        st.subheader("🌡️ Temperature")
        if preset == "Low-T FCC Crystal":
            default_T = T_min
        elif preset == "High-T Liquid Melt":
            default_T = T_max
        elif preset == "Transition Region":
            default_T = T_list[len(T_list)//2] if T_list else 1400
        else:
            default_T = T_list[len(T_list)//2] if T_list else 1400
        
        T_val = st.select_slider("T (K)", options=T_list, value=default_T)
        T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5
        
        # Expected phase based on temperature
        if T_factor < 0.3:
            phase_expected = "FCC (low-T stable)"
        elif T_factor > 0.7:
            phase_expected = "LIQUID (high-T stable)"
        else:
            phase_expected = "Transition region"
        
        st.info(f"**T = {T_val} K** | Expected: {phase_expected}")
        
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
        if comp_sum > 1.0:
            st.warning(f"⚠️ Sum = {comp_sum:.3f} > 1.0 (invalid)")
        elif comp_sum < 0.99:
            st.caption(f"x_Ni = {1.0 - comp_sum:.3f}")
        
        eval_query = st.button("🔍 Evaluate Query", use_container_width=True, type="primary")
        
        st.divider()
        
        # --- VISUALIZATION MODE ---
        st.subheader("🎨 Visualization Mode")
        mode_options = [
            "Phase Boundary (Scientific)",
            "Dual SH Surfaces (Temperature Morph)",
            "ΔG Difference Surface",
            "Ternary Flat Projection",
            "Markers (Distinct Shapes)",
            "Animated Temperature Sweep"
        ]
        render_mode = st.radio("Mode", mode_options, index=1)
        
        # --- MODE-SPECIFIC CONTROLS ---
        if render_mode == "Phase Boundary (Scientific)":
            st.subheader("🔧 Scientific Settings")
            grid_res = st.slider("Grid Resolution", 15, 80, 35, step=5)
            boundary_threshold = st.slider("Boundary Width (J/mol)", 10, 300, 60, 10)
            show_phase_volume = st.toggle("Show Phase Volume", value=True)
            volume_opacity = st.slider("Volume Opacity", 0.05, 0.6, 0.12, 0.05)
            volume_size = st.slider("Volume Point Size", 1, 8, 2)
            show_uncertainty = st.toggle("Fade Uncertain Regions", value=True)
            uncertainty_fade = st.slider("Fade Strength", 0.0, 1.0, 0.6, 0.1)
            show_simplex = st.toggle("Show Simplex Frame", value=True)
            show_slice = st.toggle("Show Cross-Section Plane", value=False)
            slice_ni = st.slider("Slice x_Ni", 0.0, 1.0, 0.25, 0.05) if show_slice else 0.25
            
        elif render_mode == "Dual SH Surfaces (Temperature Morph)":
            st.subheader("🔧 SH Morph Settings")
            sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
            sh_l_max = st.slider("Max Harmonic Degree", 1, 8, 3)
            sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
            sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)
            liq_opacity = st.slider("LIQUID Opacity", 0.1, 1.0, 0.60, 0.05)
            fcc_opacity = st.slider("FCC Opacity", 0.1, 1.0, 0.45, 0.05)
            show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
            show_data_density = st.toggle("Show Data Coverage", value=False)
            
        elif render_mode == "ΔG Difference Surface":
            st.subheader("🔧 ΔG Surface Settings")
            sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
            sh_l_max = st.slider("Max Harmonic Degree", 1, 8, 4)
            sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
            sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)
            dg_scale = st.slider("ΔG Deformation Scale", 0.001, 0.15, 0.025, 0.001)
            show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
            
        elif render_mode == "Ternary Flat Projection":
            st.subheader("🔧 Ternary Settings")
            flat_color_by = st.radio("Color By", 
                ["Stable Phase", "ΔG (diverging)", "G_magnitude", "Data Proximity"], index=1)
            flat_marker_size = st.slider("Marker Size", 2, 20, 7)
            flat_opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
            show_ternary_grid = st.toggle("Grid Lines", value=True)
            show_uncertainty = st.toggle("Fade Distant Points", value=True)
            
        elif render_mode == "Markers (Distinct Shapes)":
            st.subheader("🔧 Marker Settings")
            grid_res = st.slider("Grid Resolution", 15, 100, 35, step=5)
            marker_size = st.slider("Marker Size", 1, 12, 4)
            opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
            show_phase = st.radio("Display", 
                ["Stable Phase Only", "Both Phases (Distinct)"], index=1)
            cmap = st.selectbox("Colormap", COLORMAPS, 
                index=COLORMAPS.index("RdBu_r") if "RdBu_r" in COLORMAPS else 0)
            show_boundary = st.toggle("Show ΔG≈0 Boundary", value=True)
            show_uncertainty = st.toggle("Fade Distant Points", value=True)
            
        else:  # Animated Temperature Sweep
            st.subheader("🔧 Animation Settings")
            anim_start = st.select_slider("Start T", options=T_list, value=T_min)
            anim_end = st.select_slider("End T", options=T_list, value=T_max)
            anim_frames = st.slider("Frames", 3, min(20, len(T_list)), min(8, len(T_list)))
            anim_mode = st.radio("Animation Style", 
                ["Dual SH Morph", "ΔG Surface Morph"], index=0)
            sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
            sh_l_max = st.slider("l_max", 1, 6, 3)
            sh_n_theta = st.slider("Resolution", 30, 100, 50, step=10)
        
        st.divider()
        
        # --- GLOBAL OVERLAYS ---
        st.subheader("🔷 Overlays")
        show_axes_frame = st.toggle("Coordinate Axes", value=True)
        show_query_probe = st.toggle("Query Probe Sphere", value=True)
        show_comp_path = st.toggle("Show Composition Path", value=False)
        
        st.divider()
        
        # --- LAYOUT ---
        st.subheader("✏️ Layout")
        template = st.selectbox("Template", 
            ["plotly_white", "plotly_dark", "seaborn", "simple_white"], index=0)
        bg_color = st.color_picker("Background", "#ffffff")
        title_font = st.slider("Title Font", 12, 24, 16)
        
        st.divider()
        st.caption(f"📊 Data: {len(T_list)} temperatures | {len(df):,} compositions")
    
    # =============================================================================
    # SESSION STATE FOR QUERY HISTORY
    # =============================================================================
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # =============================================================================
    # QUERY EVALUATION
    # =============================================================================
    query_result = None
    if eval_query:
        if comp_sum > 1.0:
            st.error("❌ Composition sum exceeds 1.0")
        else:
            result = query_phase_stability(df, T_val, q_co, q_cr, q_fe)
            
            if result['valid']:
                query_result = result
                st.session_state.query_history.append(query_result)
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history.pop(0)
            else:
                st.error(f"❌ {result['error']}")
    
    if query_result:
        st.success(f"✅ Query at T={query_result['T']}K, x_Ni={query_result['Ni']:.3f}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
        c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
        c3.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol",
                 delta_color="inverse" if query_result['dG'] < 0 else "normal")
        c4.metric("Stable", query_result['Phase'])
        c5.metric("|ΔG|", f"{abs(query_result['dG']):,.0f}", "J/mol")
        st.divider()
    
    # =============================================================================
    # MAIN TABS: VISUALIZATION & TENSOR ANALYSIS
    # =============================================================================
    tab_main, tab_tensor, tab_help = st.tabs([
        "🎨 Phase Visualization", 
        "📊 Tensor Decomposition (CPD)",
        "📖 Documentation"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: PHASE VISUALIZATION
    # -------------------------------------------------------------------------
    with tab_main:
        # Build interpolators for current temperature
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
        if interp_liq is None:
            st.error(f"❌ No interpolator available at T={T_val}K")
            st.stop()
        
        fig = go.Figure()
        
        # ---------------------------------------------------------------------
        # MODE 1: PHASE BOUNDARY (SCIENTIFIC)
        # ---------------------------------------------------------------------
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
            
            # Uncertainty via data proximity
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
            
            # Phase boundary
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
            
            # Cross-section plane
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
            
            # Simplex frame
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
        
        # ---------------------------------------------------------------------
        # MODE 2: DUAL SH SURFACES (TEMPERATURE MORPH)
        # ---------------------------------------------------------------------
        elif render_mode == "Dual SH Surfaces (Temperature Morph)":
            TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
                interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
            )
            
            # Fit spherical harmonics for each phase
            coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
            coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
            
            if coeffs_liq is not None and coeffs_fcc is not None:
                G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, sh_l_max)
                G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, sh_l_max)
                
                # Temperature-driven shape morphing
                R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_factor)
                X_liq = R_liq * np.sin(PH) * np.cos(TH)
                Y_liq = R_liq * np.sin(PH) * np.sin(TH)
                Z_liq = R_liq * np.cos(PH)
                
                # LIQUID surface: fluid, expanded, shiny at high T
                fig.add_trace(go.Surface(
                    x=X_liq, y=Y_liq, z=Z_liq,
                    surfacecolor=G_liq_sh,
                    colorscale="Reds",
                    cmin=G_global_min, cmax=G_global_max,
                    opacity=liq_opacity,
                    name=f"LIQUID (l={sh_l_max}, fluid)",
                    showscale=False,
                    hovertemplate=f"<b>LIQUID</b><br>G=%{{surfacecolor:,.0f}} J/mol<br>T={T_val}K<extra></extra>",
                    lighting=dict(ambient=0.55, diffuse=0.6, roughness=0.12, specular=0.9),
                    lightposition=dict(x=100, y=100, z=50)
                ))
                
                R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_factor)
                X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
                Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
                Z_fcc = R_fcc * np.cos(PH)
                
                # FCC surface: crystalline, faceted, matte at low T
                fig.add_trace(go.Surface(
                    x=X_fcc, y=Y_fcc, z=Z_fcc,
                    surfacecolor=G_fcc_sh,
                    colorscale="Blues",
                    cmin=G_global_min, cmax=G_global_max,
                    opacity=fcc_opacity,
                    name=f"FCC (l={sh_l_max}, crystal)",
                    showscale=False,
                    hovertemplate=f"<b>FCC</b><br>G=%{{surfacecolor:,.0f}} J/mol<br>T={T_val}K<extra></extra>",
                    contours=dict(
                        x=dict(show=True, color="#1a5276", width=1.2, highlight=False),
                        y=dict(show=True, color="#1a5276", width=1.2, highlight=False),
                        z=dict(show=True, color="#1a5276", width=1.2, highlight=False)
                    ),
                    lighting=dict(ambient=0.65, diffuse=0.4, roughness=0.78, specular=0.15)
                ))
                
                # ΔG=0 contour
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
                
                # Data coverage overlay
                if show_data_density:
                    df_T = df[df["T"] == T_val]
                    if len(df_T) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=df_T["Co"], y=df_T["Cr"], z=df_T["Fe"],
                            mode="markers",
                            marker=dict(size=3, color="black", symbol="cross", opacity=0.4),
                            name="Data Points",
                            hovertemplate="Data: Co=%{x:.3f} Cr=%{y:.3f} Fe=%{z:.3f}<extra></extra>"
                        ))
            else:
                st.warning("⚠️ SH fitting failed. Try adjusting l_max or resolution.")
            
            scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
        
        # ---------------------------------------------------------------------
        # MODE 3: ΔG DIFFERENCE SURFACE
        # ---------------------------------------------------------------------
        elif render_mode == "ΔG Difference Surface":
            TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
                interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
            )
            
            coeffs_dG, l_max = fit_sh_coeffs(TH, PH, dG_grid, l_max=sh_l_max)
            if coeffs_dG is not None:
                dG_smooth = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max)
                
                # Temperature-modulated deformation
                T_deform = 1.0 + 0.2 * T_factor
                radius = sh_R_fixed * T_deform + dg_scale * dG_smooth
                radius = np.clip(radius, 0.1, 2.0)
                
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
                🔴 **Red / dented inward** → LIQUID stable (negative ΔG)  
                🔵 **Blue / bulged outward** → FCC stable (positive ΔG)  
                🟡 **Gold contour** → ΔG = 0 phase boundary  
                Deformation amplitude = driving force magnitude.
                """)
            else:
                st.warning("⚠️ SH fitting failed for ΔG")
            
            scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
        
        # ---------------------------------------------------------------------
        # MODE 4: TERNARY FLAT PROJECTION
        # ---------------------------------------------------------------------
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
            z_data = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]  # Ni = 1 - Co - Cr - Fe
            
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
        
        # ---------------------------------------------------------------------
        # MODE 5: MARKERS (DISTINCT SHAPES)
        # ---------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------
        # MODE 6: ANIMATED TEMPERATURE SWEEP
        # ---------------------------------------------------------------------
        elif render_mode == "Animated Temperature Sweep":
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
                    
                    coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq_f(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
                    coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc_f(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
                    
                    if coeffs_liq is None or coeffs_fcc is None:
                        continue
                    
                    G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, sh_l_max)
                    G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, sh_l_max)
                    
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
                    
                    coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq_i(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
                    coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc_i(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
                    
                    if coeffs_liq is not None and coeffs_fcc is not None:
                        G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, sh_l_max)
                        G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, sh_l_max)
                        
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
                        st.error("❌ Could not generate animation frames: SH fitting failed")
                else:
                    st.error("❌ Could not generate animation frames: no valid temperatures")
            
            scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
        
        # ---------------------------------------------------------------------
        # COMMON OVERLAYS: Query point, path, axes
        # ---------------------------------------------------------------------
        
        # Composition path (connects query history)
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
        if query_result is not None and query_result['valid']:
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
        
        # Coordinate axes
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
        
        # ---------------------------------------------------------------------
        # LAYOUT & RENDER
        # ---------------------------------------------------------------------
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
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"❌ Render error: {type(e).__name__}: {e}")
    
    # -------------------------------------------------------------------------
    # TAB 2: TENSOR DECOMPOSITION ANALYSIS
    # -------------------------------------------------------------------------
    with tab_tensor:
        st.header("📊 Thermodynamic Data Tensor (TDT) Analysis")
        st.markdown("""
        Based on **Coutinho et al., npj Computational Materials 6, 2 (2020)**  
        
        This tab analyzes the Gibbs energy data as a **4th-order incomplete tensor** and performs 
        **Canonical Polyadic Decomposition (CPD)** to quantify rank, compression, and separability.
        
        **Tensor structure**: G[Co, Cr, Fe, T] with simplex constraint Co+Cr+Fe+Ni=1
        **CPD model**: G ≈ Σᵣ λᵣ · A(Co) · B(Cr) · C(Fe) · D(T)
        """)
        
        # Build tensor data
        with st.spinner("🔧 Building 4D tensor..."):
            tdt_data = build_tensor_data(df)
        
        n_co, n_cr, n_fe, n_T = tdt_data['dims']
        
        # ---------------------------------------------------------------------
        # TENSOR INSPECTION
        # ---------------------------------------------------------------------
        st.subheader("🔍 Tensor Inspection")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Co dim", n_co, f"step={tdt_data['co_step']:.3f}")
        col2.metric("Cr dim", n_cr, f"step={tdt_data['cr_step']:.3f}")
        col3.metric("Fe dim", n_fe, f"step={tdt_data['fe_step']:.3f}")
        col4.metric("T dim", n_T, f"step={tdt_data['T_step']:.0f}K")
        
        full_size = tdt_data['full_size']
        valid_liq = tdt_data['valid_count']
        valid_fraction = tdt_data['valid_fraction']
        
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **TDT Order** | 4 (Co × Cr × Fe × T) |
        | **Full hypercube** | {full_size:,} entries |
        | **Valid simplex entries** | {valid_liq:,} ({100*valid_fraction:.1f}%) |
        | **Sparsity** | {100*(1-valid_fraction):.1f}% NaN (simplex constraint) |
        | **Constraint** | Co + Cr + Fe + Ni = 1 (incomplete tensor) |
        """)
        
        # ---------------------------------------------------------------------
        # RANK ANALYSIS
        # ---------------------------------------------------------------------
        st.subheader("📈 Multilinear Rank Analysis (SVD of Unfoldings)")
        st.markdown("Unfolding the tensor along each mode and analyzing singular value decay to estimate effective rank.")
        
        phase_for_tensor = st.selectbox("Select Phase for Analysis", ["G_LIQUID", "G_FCC"], index=0)
        tensor_sel = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']
        
        threshold_pct = st.slider("Singular Value Threshold (% of max)", 0.01, 5.0, 0.3, 0.1, 
                                  help="For Co-Cr-Fe-Ni: 0.2-0.5% captures physical modes; lower includes fine structure")
        
        if st.button("🔬 Run Rank Analysis", use_container_width=True):
            with st.spinner("Computing SVD on all mode unfoldings..."):
                mode_names = ['Co', 'Cr', 'Fe', 'T']
                ranks = []
                all_s = []
                
                for mode in range(4):
                    unfolded = unfold_tensor(tensor_sel, mode)
                    rank, s, s_norm = svd_rank_analysis(unfolded, threshold=threshold_pct/100.0)
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
                
                fig_svd.add_hline(y=threshold_pct/100.0, line_dash="dash", line_color="gray",
                                 annotation_text=f"Threshold ({threshold_pct}%)")
                
                fig_svd.update_layout(
                    title="Singular Value Decay Across Tensor Modes",
                    xaxis_title="Singular Value Index",
                    yaxis_title="Normalized Singular Value",
                    yaxis_type="log",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_svd, use_container_width=True)
                
                # Rank interpretation
                max_cp_rank = max(ranks)
                st.info(f"""
                **Interpretation for Co-Cr-Fe-Ni:**
                - **CP rank** should be at least {max_cp_rank} to capture all modes accurately
                - Temperature mode (Mode-3) typically has lowest rank (~3-4): baseline + entropy + Cp + transition
                - Composition modes have higher rank (~5-7): polynomial mixing + magnetic contributions
                - Recommended CP rank R: **6-8** for <1% reconstruction error on phase boundary
                - Paper reports R=6 for similar quaternary systems with step size 0.01
                """)
        
        # ---------------------------------------------------------------------
        # CPD COMPRESSION ANALYSIS
        # ---------------------------------------------------------------------
        st.subheader("🗜️ CPD Compression Analysis")
        
        R_test = st.slider("Test CP Rank (R)", 1, 20, DEFAULT_CP_RANK, 1)
        
        cpd_coeffs = R_test * (n_co + n_cr + n_fe + n_T)
        compression = valid_liq / cpd_coeffs if cpd_coeffs > 0 else 0
        reduction = (1 - cpd_coeffs / valid_liq) * 100 if valid_liq > 0 else 0
        
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **CP Rank (R)** | {R_test} |
        | **CPD coefficients** | R × (I+J+K+L) = {R_test} × ({n_co}+{n_cr}+{n_fe}+{n_T}) = **{cpd_coeffs:,}** |
        | **Original valid entries** | {valid_liq:,} |
        | **Compression ratio** | **{compression:.1f}×** |
        | **Storage reduction** | **{reduction:.1f}%** |
        """)
        
        # Visualize compression
        ranks_range = list(range(1, 21))
        cpd_sizes = [r * (n_co + n_cr + n_fe + n_T) for r in ranks_range]
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=[f"R={r}" for r in ranks_range],
            y=[valid_liq] * len(ranks_range),
            name='TDT entries',
            marker_color='lightcoral'
        ))
        fig_comp.add_trace(go.Bar(
            x=[f"R={r}" for r in ranks_range],
            y=cpd_sizes,
            name='CPD coefficients',
            marker_color='steelblue'
        ))
        
        fig_comp.update_layout(
            title="TDT Entries vs CPD Coefficients",
            yaxis_title="Count",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # ---------------------------------------------------------------------
        # CPD RECONSTRUCTION
        # ---------------------------------------------------------------------
        st.subheader("🔧 CPD Reconstruction")
        st.markdown("Run weighted ALS-based CPD to reconstruct the tensor and measure approximation error.")
        
        max_iter = st.slider("Max ALS Iterations", 20, 200, 100, 10)
        
        if st.button("⚙️ Run CPD-ALS (may take 1-2 min)", use_container_width=True):
            with st.spinner(f"Running weighted CP-ALS with R={R_test}..."):
                # Center and scale for numerical stability
                tensor_mean = np.nanmean(tensor_sel)
                tensor_std = np.nanstd(tensor_sel)
                tensor_norm = (tensor_sel - tensor_mean) / (tensor_std + 1e-12)
                
                mask = ~np.isnan(tensor_norm)
                
                # Run weighted ALS with physics-informed initialization
                A, B, C, D, lam, error, history = cpd_als_weighted_incomplete(
                    tensor_norm, mask, R_test, max_iter=max_iter, tol=1e-5,
                    T_vals=tdt_data['T_vals'],
                    co_vals=tdt_data['co_vals'],
                    cr_vals=tdt_data['cr_vals'],
                    fe_vals=tdt_data['fe_vals']
                )
                
                # Reconstruct
                I, J, K, L = tensor_norm.shape
                recon = np.zeros_like(tensor_norm)
                for r in range(R_test):
                    recon += lam[r] * np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)
                
                # Error metrics
                rel_error = np.sqrt(np.sum(mask * (tensor_norm - recon)**2) / np.sum(mask))
                abs_error = rel_error * tensor_std
                
                st.success(f"✅ CPD complete! Relative error: {rel_error:.6f} | Absolute error: {abs_error:.2f} J/mol")
                
                # Display factor matrices
                st.subheader("Factor Matrices")
                
                tabs = st.tabs(["A (Co)", "B (Cr)", "C (Fe)", "D (T)", "lambda (Weights)"])
                
                with tabs[0]:
                    df_A = pd.DataFrame(A, columns=[f"r={r+1}" for r in range(R_test)])
                    df_A.index = [f"Co={v:.3f}" for v in tdt_data['co_vals']]
                    st.dataframe(df_A.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)
                
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
                
                # Convergence history
                if 'errors' in history and len(history['errors']) > 0:
                    st.subheader("Convergence History")
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        y=history['errors'],
                        mode='lines+markers',
                        name='RMSE (observed entries)',
                        line=dict(color='#2980b9', width=2)
                    ))
                    fig_conv.update_layout(
                        title="ALS Convergence",
                        xaxis_title="Iteration",
                        yaxis_title="RMSE",
                        yaxis_type="log",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
        
        # ---------------------------------------------------------------------
        # TRANSITION SURFACE EXTRACTION
        # ---------------------------------------------------------------------
        st.subheader("🌡️ Melting Point Surface Extraction")
        st.markdown("""
        Compute T_melt(x_Co, x_Cr, x_Fe) where ΔG = G_LIQ - G_FCC = 0.
        This enables instant prediction of melting/solidification temperatures for arbitrary compositions.
        """)
        
        if st.button("🔍 Extract T_melt Surface (CPD R=6)", use_container_width=True):
            with st.spinner("Extracting transition surface from CPD factors..."):
                # Run CPD for both phases if not already done
                # (In production, cache these results)
                mask_liq = ~np.isnan(tdt_data['G_LIQ'])
                mask_fcc = ~np.isnan(tdt_data['G_FCC'])
                
                A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ, _, _ = cpd_als_weighted_incomplete(
                    tdt_data['G_LIQ'], mask_liq, DEFAULT_CP_RANK, max_iter=50,
                    T_vals=tdt_data['T_vals'], co_vals=tdt_data['co_vals'],
                    cr_vals=tdt_data['cr_vals'], fe_vals=tdt_data['fe_vals']
                )
                
                A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC, _, _ = cpd_als_weighted_incomplete(
                    tdt_data['G_FCC'], mask_fcc, DEFAULT_CP_RANK, max_iter=50,
                    T_vals=tdt_data['T_vals'], co_vals=tdt_data['co_vals'],
                    cr_vals=tdt_data['cr_vals'], fe_vals=tdt_data['fe_vals']
                )
                
                # Extract transition surface
                T_melt, dG_min, dG_max = extract_transition_surface_cpd(
                    A_LIQ, B_LIQ, C_LIQ, D_LIQ, lam_LIQ,
                    A_FCC, B_FCC, C_FCC, D_FCC, lam_FCC,
                    tdt_data['co_vals'], tdt_data['cr_vals'], 
                    tdt_data['fe_vals'], tdt_data['T_vals'], DEFAULT_CP_RANK
                )
                
                st.success(f"✅ Transition surface extracted! ΔG range: [{dG_min:.0f}, {dG_max:.0f}] J/mol")
                
                # Query interface for T_melt
                col1, col2, col3 = st.columns(3)
                with col1:
                    q_co_melt = st.number_input("x_Co for T_melt", 0.0, 1.0, 0.25, 0.01, key="melt_co")
                with col2:
                    q_cr_melt = st.number_input("x_Cr for T_melt", 0.0, 1.0, 0.25, 0.01, key="melt_cr")
                with col3:
                    q_fe_melt = st.number_input("x_Fe for T_melt", 0.0, 1.0, 0.25, 0.01, key="melt_fe")
                
                if q_co_melt + q_cr_melt + q_fe_melt <= 1.0:
                    # Find nearest grid indices
                    i = np.argmin(np.abs(np.array(tdt_data['co_vals']) - q_co_melt))
                    j = np.argmin(np.abs(np.array(tdt_data['cr_vals']) - q_cr_melt))
                    k = np.argmin(np.abs(np.array(tdt_data['fe_vals']) - q_fe_melt))
                    
                    T_pred = T_melt[i, j, k]
                    
                    if np.isnan(T_pred):
                        st.info(f"ℹ️ No phase transition in 700-3700K range for this composition")
                    else:
                        st.metric("Predicted T_melt", f"{T_pred:.0f} K")
                        
                        # Plot ΔG(T) at this composition
                        dG_vs_T = np.zeros(len(tdt_data['T_vals']))
                        for t, T in enumerate(tdt_data['T_vals']):
                            g_liq = sum(lam_LIQ[r] * A_LIQ[i,r] * B_LIQ[j,r] * C_LIQ[k,r] * D_LIQ[t,r] for r in range(DEFAULT_CP_RANK))
                            g_fcc = sum(lam_FCC[r] * A_FCC[i,r] * B_FCC[j,r] * C_FCC[k,r] * D_FCC[t,r] for r in range(DEFAULT_CP_RANK))
                            dG_vs_T[t] = g_liq - g_fcc
                        
                        fig_dg = go.Figure()
                        fig_dg.add_trace(go.Scatter(
                            x=tdt_data['T_vals'], y=dG_vs_T,
                            mode='markers+lines', name='ΔG(T)',
                            marker=dict(size=4)
                        ))
                        fig_dg.add_hline(y=0, line_dash='dash', annotation_text='ΔG=0')
                        fig_dg.add_vline(x=T_pred, line_dash='dot', annotation_text=f'T*={T_pred:.0f}K',
                                       line_color='gold')
                        fig_dg.update_layout(
                            title=f"ΔG(T) at Co={q_co_melt:.2f}, Cr={q_cr_melt:.2f}, Fe={q_fe_melt:.2f}",
                            xaxis_title="Temperature (K)", yaxis_title="ΔG (J/mol)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_dg, use_container_width=True)
                else:
                    st.error("❌ Composition sum exceeds 1.0")
        
        # ---------------------------------------------------------------------
        # UNCERTAINTY QUANTIFICATION
        # ---------------------------------------------------------------------
        st.subheader("📊 Uncertainty Quantification (Bootstrap)")
        st.markdown("""
        Propagate CALPHAD parameter uncertainty through CPD via bootstrap sampling.
        Assumes relative error model: σ_G ≈ 1% × |G| (typical for CALPHAD assessments).
        """)
        
        n_boot = st.slider("Bootstrap samples", 10, 100, BOOTSTRAP_SAMPLES, 10)
        
        if st.button("🔄 Run Bootstrap Uncertainty", use_container_width=True):
            with st.spinner(f"Running {n_boot} bootstrap samples..."):
                T_melt_mean, T_melt_std, conv_stats = bootstrap_gibbs_uncertainty(
                    df, n_bootstrap=n_boot, rel_error=CALPHAD_REL_ERROR, rank=DEFAULT_CP_RANK
                )
                
                if T_melt_mean is not None:
                    st.success(f"✅ Bootstrap complete! {conv_stats['n_successful']}/{n_boot} successful samples")
                    st.info(f"Mean T_melt uncertainty: ±{conv_stats['std_T_melt']:.1f} K")
                    
                    # Show example composition
                    i, j, k = n_co//2, n_cr//2, n_fe//2  # Near-equ
</think>
