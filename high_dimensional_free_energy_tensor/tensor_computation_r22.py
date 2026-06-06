"""
================================================================================
Gibbs Energy CPD & Error Minimization Tutorial
================================================================================
A focused educational app demonstrating:
1. Loading Gibbs energy data from CSV files
2. Building 4D thermodynamic data tensors
3. Canonical Polyadic Decomposition (CPD) with proper normalization
4. Reconstruction and denormalization to physical units
5. Quadratic expansion for phase-field modeling
6. Visual comparison showing how normalization minimizes reconstruction error

Based on: Coutinho et al., npj Computational Materials 6, 2 (2020)
================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy import linalg

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(
    page_title="Gibbs Energy CPD & Error Minimization Tutorial",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# DATA LOADING
# =============================================
@st.cache_data(ttl=3600)
def load_all_data(csv_dir=CSV_FILES_DIR):
    """Load Gibbs energy data from CSV files (Gibbs_700K.csv to Gibbs_3300K.csv)."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))

    if not files:
        st.error(f"No CSV files found in `{csv_dir}`.\\n\\nExpected: Gibbs_700K.csv, Gibbs_800K.csv, ..., Gibbs_3300K.csv")
        st.stop()

    dfs = []
    found_temps = []

    for f in files:
        basename = os.path.basename(f)
        try:
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            df["T"] = T
            dfs.append(df)
            found_temps.append(T)
        except Exception as e:
            st.warning(f"Skipping {f}: {e}")

    if not dfs:
        st.error("No valid data loaded from any files.")
        st.stop()

    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined["dG"] = df_combined["G_LIQ"] - df_combined["G_FCC"]

    return df_combined

# =============================================
# TENSOR CONSTRUCTION
# =============================================
@st.cache_data(ttl=7200)
def build_tensor_data(df):
    """Build 4D Thermodynamic Data Tensor from DataFrame (Vectorized)."""
    co_vals = sorted(df["Co"].unique())
    cr_vals = sorted(df["Cr"].unique())
    fe_vals = sorted(df["Fe"].unique())
    T_vals = sorted(df["T"].unique())

    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)

    co_to_idx = {round(v, 4): i for i, v in enumerate(co_vals)}
    cr_to_idx = {round(v, 4): i for i, v in enumerate(cr_vals)}
    fe_to_idx = {round(v, 4): i for i, v in enumerate(fe_vals)}
    T_to_idx = {T: i for i, T in enumerate(T_vals)}

    # Vectorized mapping instead of slow iterrows()
    df_copy = df.copy()
    df_copy['i'] = df_copy['Co'].round(4).map(co_to_idx)
    df_copy['j'] = df_copy['Cr'].round(4).map(cr_to_idx)
    df_copy['k'] = df_copy['Fe'].round(4).map(fe_to_idx)
    df_copy['t'] = df_copy['T'].map(T_to_idx)

    df_valid = df_copy.dropna(subset=['i', 'j', 'k', 't'])
    df_valid = df_valid.astype({'i': int, 'j': int, 'k': int, 't': int})

    G_LIQ_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    G_FCC_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)

    # Vectorized assignment using advanced indexing
    G_LIQ_tdt[df_valid['i'].values, df_valid['j'].values, df_valid['k'].values, df_valid['t'].values] = df_valid['G_LIQ'].values
    G_FCC_tdt[df_valid['i'].values, df_valid['j'].values, df_valid['k'].values, df_valid['t'].values] = df_valid['G_FCC'].values

    valid_count = len(df_valid)
    full_size = n_co * n_cr * n_fe * n_T
    sparsity = 1.0 - (valid_count / full_size)

    return {
        'G_LIQ': G_LIQ_tdt,
        'G_FCC': G_FCC_tdt,
        'dims': (n_co, n_cr, n_fe, n_T),
        'co_vals': co_vals,
        'cr_vals': cr_vals,
        'fe_vals': fe_vals,
        'T_vals': T_vals,
        'valid_count': valid_count,
        'sparsity': sparsity,
        'full_size': full_size
    }

# =============================================
# TENSOR UNFOLDING FOR SVD ANALYSIS
# =============================================
def unfold_tensor(tensor, mode):
    """Unfold (matricize) 4D tensor along specified mode."""
    if mode == 0:
        return tensor.reshape(tensor.shape[0], -1)
    elif mode == 1:
        return tensor.transpose(1, 0, 2, 3).reshape(tensor.shape[1], -1)
    elif mode == 2:
        return tensor.transpose(2, 0, 1, 3).reshape(tensor.shape[2], -1)
    elif mode == 3:
        return tensor.transpose(3, 0, 1, 2).reshape(tensor.shape[3], -1)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def svd_rank_analysis(matrix, threshold=0.01):
    """Estimate effective rank via SVD with NaN handling."""
    matrix_filled = matrix.copy().astype(np.float64)

    for col in range(matrix_filled.shape[1]):
        col_data = matrix_filled[:, col]
        valid = ~np.isnan(col_data)
        if np.sum(valid) > 0:
            matrix_filled[:, col] = np.where(np.isnan(col_data), np.nanmean(col_data[valid]), col_data)
        else:
            matrix_filled[:, col] = 0.0

    if np.linalg.norm(matrix_filled) < 1e-12:
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))

    try:
        U, s, Vh = linalg.svd(matrix_filled, full_matrices=False)
    except Exception as e:
        st.warning(f"SVD failed: {e}")
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))

    s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
    s_norm = s / s_max
    rank = int(np.sum(s_norm > threshold))

    return rank, s, s_norm

# =============================================
# CPD-ALS WITH PROPER NORMALIZATION
# =============================================
def cpd_als_4d(tensor, rank, max_iter=100, tol=1e-6, reg=1e-8):
    """
    4-way Canonical Polyadic Decomposition via Weighted Alternating Least Squares.

    OPTIMIZED & CORRECTED:
    - Kronecker products precomputed once per mode update (outside inner loops)
    - Correct Kronecker order matching NumPy C-order flattening (last index varies fastest)

    Decomposition: G[i,j,k,t] ~ sum_r lambda_r * A[i,r] * B[j,r] * C[k,r] * D[t,r]
    """
    # STEP 1: MANUAL Z-SCORE NORMALIZATION
    mask = ~np.isnan(tensor)
    valid_entries = tensor[mask]

    if len(valid_entries) > 0:
        mu = float(np.mean(valid_entries))
        sigma = float(np.std(valid_entries))
    else:
        mu = 0.0
        sigma = 1.0

    if sigma < 1e-12:
        sigma = 1.0

    X_norm = (tensor - mu) / sigma
    X = np.where(mask, X_norm, 0.0)

    I, J, K, L = tensor.shape

    # STEP 2: INITIALIZATION
    if L == 31:
        T_vals_physical = np.array(list(range(700, 3701, 100)))
        T_mean, T_std = np.mean(T_vals_physical), np.std(T_vals_physical)
        T_norm = (T_vals_physical - T_mean) / (T_std + 1e-12)

        D = np.zeros((L, rank))
        D[:, 0] = 1.0
        if rank >= 2:
            D[:, 1] = T_norm
        if rank >= 3:
            D[:, 2] = (T_norm**2 - 1) * 0.5
        if rank >= 4:
            D[:, 3] = np.tanh(2 * T_norm) - np.mean(np.tanh(2 * T_norm))
        if rank > 4:
            D[:, 4:] = np.random.randn(L, rank-4) * 0.01
    else:
        D = np.random.randn(L, rank) * 0.1

    X_unfolded = unfold_tensor(X, mode=0)
    try:
        U, s, Vh = linalg.svd(X_unfolded, full_matrices=False)
        A = U[:, :rank] * np.sqrt(np.maximum(s[:rank], 0))
    except:
        A = np.random.randn(I, rank) * 0.1

    B = np.random.randn(J, rank) * 0.1
    C = np.random.randn(K, rank) * 0.1

    prev_error = np.inf

    # STEP 3: WEIGHTED ALS (OPTIMIZED & CORRECTED)
    for iteration in range(max_iter):

        # --- Update A (Mode 0) ---
        # Flattened mask[i, :, :, :] is (J, K, L) -> L varies fastest (C-order)
        # Correct Kronecker order: B, C, D (D varies fastest)
        BCD_full = np.zeros((J * K * L, rank))
        for r in range(rank):
            BCD_full[:, r] = np.kron(np.kron(B[:, r], C[:, r]), D[:, r])

        for i in range(I):
            valid = mask[i, :, :, :].ravel()
            if np.sum(valid) > rank:
                BCD = BCD_full[valid, :]
                y = X[i, :, :, :].ravel()[valid]
                AtA = BCD.T @ BCD + reg * np.eye(rank)
                Aty = BCD.T @ y
                try:
                    A[i, :] = linalg.solve(AtA, Aty, assume_a='pos')
                except linalg.LinAlgError:
                    A[i, :] = linalg.lstsq(BCD, y, rcond=None)[0]

        norms = np.linalg.norm(A, axis=0) + 1e-12
        A = A / norms

        # --- Update B (Mode 1) ---
        X_flat = X.transpose(1, 0, 2, 3).reshape(J, -1)
        mask_flat = mask.transpose(1, 0, 2, 3).reshape(J, -1)
        # Flattened dims: (I, K, L) -> L varies fastest -> Order: A, C, D
        ACD_full = np.zeros((I * K * L, rank))
        for r in range(rank):
            ACD_full[:, r] = np.kron(np.kron(A[:, r], C[:, r]), D[:, r])

        for j in range(J):
            valid = mask_flat[j, :]
            if np.sum(valid) > rank:
                ACD = ACD_full[valid, :]
                y = X_flat[j, valid]
                AtA = ACD.T @ ACD + reg * np.eye(rank)
                Aty = ACD.T @ y
                try:
                    B[j, :] = linalg.solve(AtA, Aty, assume_a='pos')
                except linalg.LinAlgError:
                    B[j, :] = linalg.lstsq(ACD, y, rcond=None)[0]

        norms = np.linalg.norm(B, axis=0) + 1e-12
        B = B / norms

        # --- Update C (Mode 2) ---
        X_flat = X.transpose(2, 0, 1, 3).reshape(K, -1)
        mask_flat = mask.transpose(2, 0, 1, 3).reshape(K, -1)
        # Flattened dims: (I, J, L) -> L varies fastest -> Order: A, B, D
        ABD_full = np.zeros((I * J * L, rank))
        for r in range(rank):
            ABD_full[:, r] = np.kron(np.kron(A[:, r], B[:, r]), D[:, r])

        for k in range(K):
            valid = mask_flat[k, :]
            if np.sum(valid) > rank:
                ABD = ABD_full[valid, :]
                y = X_flat[k, valid]
                AtA = ABD.T @ ABD + reg * np.eye(rank)
                Aty = ABD.T @ y
                try:
                    C[k, :] = linalg.solve(AtA, Aty, assume_a='pos')
                except linalg.LinAlgError:
                    C[k, :] = linalg.lstsq(ABD, y, rcond=None)[0]

        norms = np.linalg.norm(C, axis=0) + 1e-12
        C = C / norms

        # --- Update D (Mode 3) ---
        X_flat = X.transpose(3, 0, 1, 2).reshape(L, -1)
        mask_flat = mask.transpose(3, 0, 1, 2).reshape(L, -1)
        # Flattened dims: (I, J, K) -> K varies fastest -> Order: A, B, C
        ABC_full = np.zeros((I * J * K, rank))
        for r in range(rank):
            ABC_full[:, r] = np.kron(np.kron(A[:, r], B[:, r]), C[:, r])

        for t in range(L):
            valid = mask_flat[t, :]
            if np.sum(valid) > rank:
                ABC = ABC_full[valid, :]
                y = X_flat[t, valid]
                AtA = ABC.T @ ABC + reg * np.eye(rank)
                Aty = ABC.T @ y
                try:
                    D[t, :] = linalg.solve(AtA, Aty, assume_a='pos')
                except linalg.LinAlgError:
                    D[t, :] = linalg.lstsq(ABC, y, rcond=None)[0]

        norms = np.linalg.norm(D, axis=0) + 1e-12
        D = D / norms

        # --- Compute Reconstruction & Error ---
        recon_norm = np.zeros_like(X)
        for r in range(rank):
            # Flattened dims: (J, K, L) -> L varies fastest -> Order: B, C, D
            recon_norm += np.outer(A[:, r], np.kron(np.kron(B[:, r], C[:, r]), D[:, r])).reshape(I, J, K, L)

        observed_residuals = (X_norm - recon_norm)[mask]
        error_norm = np.sqrt(np.mean(observed_residuals**2)) if len(observed_residuals) > 0 else np.inf
        if abs(prev_error - error_norm) < tol:
            break
        prev_error = error_norm

    # STEP 4 & 5: COMPONENT WEIGHTS & DENORMALIZATION
    lam = np.ones(rank)
    for r in range(rank):
        lam[r] = (np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * 
                  np.linalg.norm(C[:, r]) * np.linalg.norm(D[:, r]))

    recon_physical = recon_norm * sigma + mu
    physical_residuals = (tensor - recon_physical)[mask]
    final_error_physical = np.sqrt(np.mean(physical_residuals**2)) if len(physical_residuals) > 0 else np.inf

    meta = {
        'mu': mu,
        'sigma': sigma,
        'error_physical': final_error_physical,
        'error_norm': error_norm
    }

    return A, B, C, D, lam, meta


# =============================================
# QUADRATIC EXPANSION FROM CPD FACTORS
# =============================================
def compute_quadratic_coefficients_from_cpd(
    A, B, C, D, lam, 
    co_vals, cr_vals, fe_vals, T_vals,
    c_eq_co, c_eq_cr, c_eq_fe, T_m
):
    """
    Compute quadratic expansion coefficients from CPD factor matrices.
    Returns coefficients in normalized space - caller must denormalize.
    """
    R = len(lam)

    def get_factor_function(x_vals, F_matrix, r):
        return UnivariateSpline(x_vals, F_matrix[:, r], s=0, ext=3)

    A_funcs = [get_factor_function(co_vals, A, r) for r in range(R)]
    B_funcs = [get_factor_function(cr_vals, B, r) for r in range(R)]
    C_funcs = [get_factor_function(fe_vals, C, r) for r in range(R)]
    D_funcs = [get_factor_function(T_vals, D, r) for r in range(R)]

    def second_derivative(func, x):
        h = 1e-5
        return (func(x + h) - 2*func(x) + func(x - h)) / (h**2)

    A_Co_sum = 0.0
    for r in range(R):
        A_pp = second_derivative(A_funcs[r], c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Co_sum += lam[r] * A_pp * B_val * C_val * D_val
    A_Co = 0.5 * A_Co_sum

    A_Cr_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_pp = second_derivative(B_funcs[r], c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Cr_sum += lam[r] * A_val * B_pp * C_val * D_val
    A_Cr = 0.5 * A_Cr_sum

    A_Fe_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_pp = second_derivative(C_funcs[r], c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Fe_sum += lam[r] * A_val * B_val * C_pp * D_val
    A_Fe = 0.5 * A_Fe_sum

    A_T_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_pp = second_derivative(D_funcs[r], T_m)
        A_T_sum += lam[r] * A_val * B_val * C_val * D_pp
    A_T = 0.5 * A_T_sum

    G_eq = 0.0
    for r in range(R):
        G_eq += lam[r] * A_funcs[r](c_eq_co) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_m)

    return {
        'A_Co': A_Co,
        'A_Cr': A_Cr,
        'A_Fe': A_Fe,
        'A_T': A_T,
        'G_eq': G_eq,
        'c_eq': [c_eq_co, c_eq_cr, c_eq_fe],
        'T_m': T_m
    }


def verify_quadratic_approximation(coeffs, A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, 
                                    test_compositions, test_temperatures, sigma=1.0, mu=0.0):
    """Verify quadratic approximation against full CPD reconstruction."""
    R = len(lam)
    A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
    B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
    C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
    D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]

    results = []
    for c_co, c_cr, c_fe, T in zip(test_compositions[:, 0], test_compositions[:, 1], 
                                    test_compositions[:, 2], test_temperatures):
        G_norm = sum(lam[r] * A_funcs[r](c_co) * B_funcs[r](c_cr) * 
                     C_funcs[r](c_fe) * D_funcs[r](T) for r in range(R))
        G_full = G_norm * sigma + mu

        dc_co = c_co - coeffs['c_eq'][0]
        dc_cr = c_cr - coeffs['c_eq'][1]
        dc_fe = c_fe - coeffs['c_eq'][2]
        dT = T - coeffs['T_m']

        G_quad = (coeffs['G_eq'] + 
                  coeffs['A_Co'] * dc_co**2 + 
                  coeffs['A_Cr'] * dc_cr**2 + 
                  coeffs['A_Fe'] * dc_fe**2 + 
                  coeffs['A_T'] * dT**2)

        results.append({
            'c_Co': c_co, 'c_Cr': c_cr, 'c_Fe': c_fe, 'T': T,
            'G_full': G_full,
            'G_quadratic': G_quad,
            'absolute_error': abs(G_full - G_quad),
            'relative_error': abs(G_full - G_quad) / (abs(G_full) + 1e-10)
        })

    return pd.DataFrame(results)


# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def plot_parity_comparison(original, reconstructed, mask, title, x_label, y_label):
    """Create parity plot: Original vs Reconstructed values (WebGL + downsampled)."""
    orig_valid = original[mask]
    recon_valid = reconstructed[mask]

    # --- FIX 1: Downsample to prevent browser freeze on large datasets ---
    MAX_POINTS = 10000
    if len(orig_valid) > MAX_POINTS:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(orig_valid), MAX_POINTS, replace=False)
        orig_valid = orig_valid[indices]
        recon_valid = recon_valid[indices]

    fig = go.Figure()

    # --- FIX 2: Use Scattergl for WebGL rendering performance ---
    fig.add_trace(go.Scattergl(
        x=orig_valid,
        y=recon_valid,
        mode='markers',
        marker=dict(size=4, color='steelblue', opacity=0.4),
        name='Data points',
        hovertemplate=f"Original: %{{x:,.0f}} J/mol<br>Reconstructed: %{{y:,.0f}} J/mol<extra></extra>"
    ))

    min_val = min(np.min(orig_valid), np.min(recon_valid))
    max_val = max(np.max(orig_valid), np.max(recon_valid))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect fit (y=x)'
    ))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val * 1.01, max_val * 1.01],
        mode='lines',
        line=dict(color='orange', dash='dot', width=1),
        name='+1% error',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val * 0.99, max_val * 0.99],
        mode='lines',
        line=dict(color='orange', dash='dot', width=1),
        name='-1% error',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def plot_1d_comparison(co_slice, G_full_slice, G_quad_slice, c_eq_co, T_val, phase_name):
    """Plot 1D slice comparison between Full CPD and Quadratic."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=co_slice, y=G_full_slice,
        mode='lines',
        name='Full CPD',
        line=dict(color='blue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=co_slice, y=G_quad_slice,
        mode='lines',
        name='Quadratic Approx.',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.add_vline(x=c_eq_co, line_dash="dot", line_color="green",
                  annotation_text="Equilibrium", annotation_position="top right")

    fig.update_layout(
        title=f"{phase_name} Gibbs Energy: Full CPD vs Quadratic at T={T_val}K",
        xaxis_title="x_Co",
        yaxis_title="Gibbs Energy (J/mol)",
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def plot_error_distribution(verify_df):
    """Plot error distribution histogram and scatter."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Absolute Error Distribution', 'Relative Error vs Distance from Equilibrium'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )

    fig.add_trace(go.Histogram(
        x=verify_df['absolute_error'],
        nbinsx=30,
        name='Abs Error',
        marker_color='steelblue'
    ), row=1, col=1)

    dist_from_eq = np.sqrt(
        (verify_df['c_Co'] - verify_df['c_Co'].mean())**2 +
        (verify_df['c_Cr'] - verify_df['c_Cr'].mean())**2 +
        (verify_df['c_Fe'] - verify_df['c_Fe'].mean())**2
    )

    fig.add_trace(go.Scatter(
        x=dist_from_eq,
        y=verify_df['relative_error'] * 100,
        mode='markers',
        marker=dict(size=6, color=verify_df['absolute_error'], 
                   colorscale='Reds', showscale=True,
                   colorbar=dict(title="Abs Error<br>(J/mol)", len=0.5)),
        name='Rel Error',
        hovertemplate="Dist=%{x:.4f}<br>Rel Error=%{y:.2f}%<extra></extra>"
    ), row=1, col=2)

    fig.update_layout(
        height=450,
        template='plotly_white',
        showlegend=False
    )

    fig.update_xaxes(title_text="Absolute Error (J/mol)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Distance from Equilibrium", row=1, col=2)
    fig.update_yaxes(title_text="Relative Error (%)", row=1, col=2)

    return fig


def plot_factor_profiles(A, B, C, lam, co_vals, cr_vals, fe_vals, R=6):
    """Plot CPD factor matrix profiles."""
    fig = make_subplots(
        rows=3, cols=R,
        subplot_titles=[f'r={r+1} (lambda={lam[r]:.3f})' for r in range(R)],
        vertical_spacing=0.12, horizontal_spacing=0.08
    )
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    for r in range(R):
        fig.add_trace(go.Scatter(
            x=co_vals, y=lam[r]*A[:,r], mode='lines+markers',
            marker=dict(color=colors[r%len(colors)]),
            line=dict(width=2, color=colors[r%len(colors)]),
            name=f'r={r+1} Co', showlegend=(r==0)
        ), row=1, col=r+1)

        fig.add_trace(go.Scatter(
            x=cr_vals, y=lam[r]*B[:,r], mode='lines+markers',
            marker=dict(color=colors[r%len(colors)]),
            line=dict(width=2, color=colors[r%len(colors)]),
            showlegend=False
        ), row=2, col=r+1)

        fig.add_trace(go.Scatter(
            x=fe_vals, y=lam[r]*C[:,r], mode='lines+markers',
            marker=dict(color=colors[r%len(colors)]),
            line=dict(width=2, color=colors[r%len(colors)]),
            showlegend=False
        ), row=3, col=r+1)

    for r in range(R):
        fig.update_xaxes(title_text="x_Co", row=1, col=r+1)
        fig.update_xaxes(title_text="x_Cr", row=2, col=r+1)
        fig.update_xaxes(title_text="x_Fe", row=3, col=r+1)
        fig.update_yaxes(title_text="lambda*A", row=1, col=r+1)
        fig.update_yaxes(title_text="lambda*B", row=2, col=r+1)
        fig.update_yaxes(title_text="lambda*C", row=3, col=r+1)

    fig.update_layout(
        height=800,
        title_text="CPD Factor Matrix Profiles (Weighted by lambda)",
        template='plotly_white'
    )
    return fig


# =============================================
# STREAMLIT UI
# =============================================

st.title("📚 Gibbs Energy CPD & Error Minimization Tutorial")
st.markdown(r"""
**Learn how proper normalization and denormalization minimizes reconstruction errors**
in Canonical Polyadic Decomposition (CPD) of thermodynamic data tensors.

**Key Concepts:**
- Z-Score Normalization: Scale Gibbs energies to N(0,1) for numerical stability
- CPD-ALS: Decompose 4D tensor into separable rank-1 components
- Denormalization: Convert reconstructed values back to physical units (J/mol)
- Quadratic Expansion: Local Taylor approximation for phase-field modeling
""")

# Load data
df = load_all_data()
T_list = sorted(df["T"].unique())

# Sidebar controls
with st.sidebar:
    st.header("Tutorial Controls")

    st.subheader("Phase Selection")
    phase_for_tensor = st.selectbox("Select Phase for CPD", ["G_LIQUID", "G_FCC"], index=0)

    st.subheader("CPD Parameters")
    R_test = st.slider("CP Rank (R)", 1, 12, 6, 1,
                       help="Number of separable thermodynamic components")
    max_iter = st.slider("Max ALS Iterations", 20, 200, 100, 10)

    st.subheader("Quadratic Expansion")
    c_eq_co = st.number_input("Equilibrium x_Co", 0.0, 1.0, 0.25, 0.01)
    c_eq_cr = st.number_input("Equilibrium x_Cr", 0.0, 1.0, 0.25, 0.01)
    c_eq_fe = st.number_input("Equilibrium x_Fe", 0.0, 1.0, 0.25, 0.01)
    T_m = st.number_input("Equilibrium T_m (K)", 700, 3300, 1500, 50)

    c_eq_ni = 1.0 - c_eq_co - c_eq_cr - c_eq_fe
    if c_eq_ni < 0:
        st.error(f"Invalid: x_Ni = {c_eq_ni:.3f} < 0")
    else:
        st.success(f"x_Ni = {c_eq_ni:.3f}")

    st.divider()
    st.caption(f"Data: {len(T_list)} temperatures | {len(df):,} total measurements")

# =============================================
# MAIN TABS
# =============================================
tab_data, tab_tensor, tab_cpd, tab_recon, tab_quadratic, tab_theory = st.tabs([
    "Data Loading", "Tensor Construction", "CPD Decomposition", 
    "Reconstruction & Error", "Quadratic Expansion", "Theory"
])

# ---------------------------------------------
# TAB 1: DATA LOADING
# ---------------------------------------------
with tab_data:
    st.header("Step 1: Data Loading")
    st.markdown(r"""
    Gibbs energy data is loaded from CSV files in the csv_files/ directory.
    Each file contains CALPHAD-computed values for one temperature.

    File format: Gibbs_TK.csv with columns: Co, Cr, Fe, Ni, G_LIQ, G_FCC
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperatures", len(T_list))
    col2.metric("Total Points", f"{len(df):,}")
    col3.metric("Temperature Range", f"{min(T_list)}-{max(T_list)} K")

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Gibbs Energy Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["G_LIQ"], name="G_LIQ", opacity=0.7, nbinsx=50,
        marker_color='#e74c3c'
    ))
    fig_dist.add_trace(go.Histogram(
        x=df["G_FCC"], name="G_FCC", opacity=0.7, nbinsx=50,
        marker_color='#2980b9'
    ))
    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title="Gibbs Energy (J/mol)",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="plotly_dist")

    st.info(r"""
    Observation: Raw Gibbs energies range from approximately -300,000 to 0 J/mol.
    These large magnitudes can cause numerical instability in ALS algorithms.
    This is why normalization is critical before CPD decomposition.
    """)

# ---------------------------------------------
# TAB 2: TENSOR CONSTRUCTION
# ---------------------------------------------
with tab_tensor:
    st.header("Step 2: Tensor Construction")
    st.markdown(r"""
    The data is organized into a 4th-order tensor:
    G[i, j, k, t] = Gibbs energy at (x_Co^(i), x_Cr^(j), x_Fe^(k), T^(t))

    Due to the simplex constraint (x_Co + x_Cr + x_Fe + x_Ni = 1),
    only ~16.7% of the hypercube contains valid entries.
    """)

    tdt_data = build_tensor_data(df)
    n_co, n_cr, n_fe, n_T = tdt_data['dims']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Co dimension", n_co)
    col2.metric("Cr dimension", n_cr)
    col3.metric("Fe dimension", n_fe)
    col4.metric("T dimension", n_T)

    st.markdown(f"""
    | Property | Value | Physical Meaning |
    |----------|-------|-----------------|
    | Tensor Order | 4 | Co x Cr x Fe x T |
    | Full hypercube | {tdt_data['full_size']:,} entries | All grid combinations |
    | Valid entries | {tdt_data['valid_count']:,} ({100*(1-tdt_data['sparsity']):.1f}%) | Simplex-constrained |
    | Sparsity | {tdt_data['sparsity']*100:.1f}% | Invalid (non-simplex) entries |
    """)

    st.subheader("Temperature Slice Visualization")
    T_slice = st.selectbox("Select Temperature", tdt_data['T_vals'], index=len(tdt_data['T_vals'])//2)
    t_idx = tdt_data['T_vals'].index(T_slice)

    tensor_sel = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']
    slice_2d = tensor_sel[:, :, n_fe//2, t_idx]

    fig_slice = go.Figure(data=go.Heatmap(
        z=slice_2d,
        x=tdt_data['co_vals'],
        y=tdt_data['cr_vals'],
        colorscale='RdBu_r',
        colorbar=dict(title="G (J/mol)")
    ))
    fig_slice.update_layout(
        title=f"{phase_for_tensor} at T={T_slice}K, Fe={tdt_data['fe_vals'][n_fe//2]:.3f}",
        xaxis_title="x_Co",
        yaxis_title="x_Cr",
        height=500
    )
    st.plotly_chart(fig_slice, use_container_width=True, key="plotly_tensor_slice")

    st.info(r"""
    Why sparse tensors matter: Standard matrix factorization fails on sparse data.
    CPD-ALS with masked weighting ensures we only fit observed (simplex-valid) entries,
    preventing bias from the ~83% invalid entries.
    """)

# ---------------------------------------------
# TAB 3: CPD DECOMPOSITION
# ---------------------------------------------
with tab_cpd:
    st.header("Step 3: CPD Decomposition")
    st.markdown(r"""
    Canonical Polyadic Decomposition factorizes the tensor into rank-1 components:
    G[i,j,k,t] ~ sum_r lambda_r * A_r(i) * B_r(j) * C_r(k) * D_r(t)

    CRITICAL: Normalization before decomposition

    Raw Gibbs energies (~10^5 J/mol) cause numerical overflow. We apply Z-score normalization:
    G_norm = (G_raw - mu) / sigma

    where mu and sigma are computed ONLY on valid (non-NaN) entries.
    """)

    tensor_sel = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']

    valid_mask = ~np.isnan(tensor_sel)
    valid_entries = tensor_sel[valid_mask]
    mu_raw = np.mean(valid_entries)
    sigma_raw = np.std(valid_entries)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean (mu)", f"{mu_raw:,.0f} J/mol")
    col2.metric("Std Dev (sigma)", f"{sigma_raw:,.0f} J/mol")
    col3.metric("Normalized Range", f"[{(valid_entries.min()-mu_raw)/sigma_raw:.2f}, {(valid_entries.max()-mu_raw)/sigma_raw:.2f}]")

    st.subheader("Normalization Effect Visualization")

    fig_norm = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Raw Gibbs Energy Distribution', 'Normalized Distribution'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}]]
    )

    fig_norm.add_trace(go.Histogram(
        x=valid_entries, nbinsx=50, marker_color='coral',
        name='Raw'
    ), row=1, col=1)

    normalized = (valid_entries - mu_raw) / sigma_raw
    fig_norm.add_trace(go.Histogram(
        x=normalized, nbinsx=50, marker_color='steelblue',
        name='Normalized'
    ), row=1, col=2)

    fig_norm.update_xaxes(title_text="G (J/mol)", row=1, col=1)
    fig_norm.update_xaxes(title_text="G_norm (sigma units)", row=1, col=2)
    fig_norm.update_yaxes(title_text="Count", row=1, col=1)
    fig_norm.update_yaxes(title_text="Count", row=1, col=2)
    fig_norm.update_layout(height=400, template='plotly_white', showlegend=False)

    st.plotly_chart(fig_norm, use_container_width=True, key="plotly_norm_effect")

    if st.button("Run CPD-ALS Decomposition", use_container_width=True, type="primary"):
        with st.spinner(f"Running CP-ALS for {phase_for_tensor} with R={R_test}..."):
            tensor_mean = float(np.nanmean(tensor_sel))
            tensor_std = float(np.nanstd(tensor_sel))
            tensor_norm = (tensor_sel - tensor_mean) / (tensor_std + 1e-12)

            A, B, C, D, lam, meta = cpd_als_4d(
                tensor_norm, R_test, max_iter=max_iter, tol=1e-5, reg=1e-8
            )

            I, J, K, L = tensor_norm.shape
            recon = np.zeros_like(tensor_norm)
            mask = ~np.isnan(tensor_norm)

            for r in range(R_test):
                recon += lam[r] * np.outer(A[:, r], np.kron(np.kron(B[:, r], C[:, r]), D[:, r])).reshape(I, J, K, L)

            # --- FIX: Cache the reconstructed tensor to avoid recomputing in Tab 4 ---
            st.session_state[f'recon_norm_{phase_key.lower()}'] = recon
            # -----------------------------------------------------------------------

            rel_error = np.sqrt(np.sum(mask * (tensor_norm - recon)**2) / np.sum(mask))
            abs_error = rel_error * tensor_std

            phase_key = "LIQ" if phase_for_tensor == "G_LIQUID" else "FCC"
            st.session_state[f'A_{phase_key.lower()}'] = A
            st.session_state[f'B_{phase_key.lower()}'] = B
            st.session_state[f'C_{phase_key.lower()}'] = C
            st.session_state[f'D_{phase_key.lower()}'] = D
            st.session_state[f'lam_{phase_key.lower()}'] = lam
            st.session_state[f'cpd_mu_{phase_key.lower()}'] = tensor_mean
            st.session_state[f'cpd_sigma_{phase_key.lower()}'] = tensor_std
            st.session_state[f'cpd_completed_{phase_key}'] = True
            st.session_state[f'rel_error_{phase_key.lower()}'] = rel_error
            st.session_state[f'abs_error_{phase_key.lower()}'] = abs_error
            st.session_state['tdt_metadata'] = {
                'co_vals': tdt_data['co_vals'],
                'cr_vals': tdt_data['cr_vals'],
                'fe_vals': tdt_data['fe_vals'],
                'T_vals': tdt_data['T_vals']
            }

            st.success(f"CPD complete! Normalized error: {rel_error:.6f} | Physical RMSE: {abs_error:.2f} J/mol")

            st.subheader("Factor Matrix Profiles")
            fig_factors = plot_factor_profiles(A, B, C, lam, 
                                               tdt_data['co_vals'], tdt_data['cr_vals'], 
                                               tdt_data['fe_vals'], R=R_test)
            st.plotly_chart(fig_factors, use_container_width=True, key="plotly_factors")

    liq_done = st.session_state.get('cpd_completed_LIQ', False)
    fcc_done = st.session_state.get('cpd_completed_FCC', False)

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("LIQUID CPD", "Complete" if liq_done else "Not run")
    with status_col2:
        st.metric("FCC CPD", "Complete" if fcc_done else "Not run")

# ---------------------------------------------
# TAB 4: RECONSTRUCTION & ERROR
# ---------------------------------------------
with tab_recon:
    st.header("Step 4: Reconstruction & Error Minimization")
    st.markdown(r"""
    The Critical Step: Denormalization

    After CPD decomposition in normalized space, we must convert back to physical units:
    G_physical = G_normalized * sigma + mu

    Without denormalization, reconstructed values cluster near 0 (the normalized mean),
    giving errors of ~10^5 J/mol. With proper denormalization, errors drop to <1%.
    """)

    phase_key = "LIQ" if phase_for_tensor == "G_LIQUID" else "FCC"

    if not st.session_state.get(f'cpd_completed_{phase_key}', False):
        st.warning(f"Please run CPD for {phase_for_tensor} in the CPD Decomposition tab first.")
    else:
        A = st.session_state[f'A_{phase_key.lower()}']
        B = st.session_state[f'B_{phase_key.lower()}']
        C = st.session_state[f'C_{phase_key.lower()}']
        D = st.session_state[f'D_{phase_key.lower()}']
        lam = st.session_state[f'lam_{phase_key.lower()}']
        mu = st.session_state[f'cpd_mu_{phase_key.lower()}']
        sigma = st.session_state[f'cpd_sigma_{phase_key.lower()}']

        tensor_sel = tdt_data['G_LIQ'] if phase_for_tensor == "G_LIQUID" else tdt_data['G_FCC']
        I, J, K, L = tensor_sel.shape
        mask = ~np.isnan(tensor_sel)

        # --- FIX: Load cached reconstruction instead of recomputing heavy Kronecker products ---
        if f'recon_norm_{phase_key.lower()}' in st.session_state:
            recon_norm = st.session_state[f'recon_norm_{phase_key.lower()}']
        else:
            # Fallback if somehow not in session state
            recon_norm = np.zeros_like(tensor_sel)
            for r in range(len(lam)):
                recon_norm += lam[r] * np.outer(A[:, r], np.kron(np.kron(B[:, r], C[:, r]), D[:, r])).reshape(I, J, K, L)
        # ------------------------------------------------------------------------------------

        recon_physical = recon_norm * sigma + mu
        recon_buggy = recon_norm

        physical_residuals = (tensor_sel - recon_physical)[mask]
        buggy_residuals = (tensor_sel - recon_buggy)[mask]

        rmse_physical = np.sqrt(np.mean(physical_residuals**2))
        rmse_buggy = np.sqrt(np.mean(buggy_residuals**2))
        mae_physical = np.mean(np.abs(physical_residuals))
        mae_buggy = np.mean(np.abs(buggy_residuals))

        st.subheader("Error Comparison: With vs Without Denormalization")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE (Correct)", f"{rmse_physical:.2f} J/mol")
        col2.metric("RMSE (Buggy)", f"{rmse_buggy:,.0f} J/mol")
        col3.metric("MAE (Correct)", f"{mae_physical:.2f} J/mol")
        col4.metric("MAE (Buggy)", f"{mae_buggy:,.0f} J/mol")

        st.subheader("Parity Plot: Correct Denormalization")
        fig_parity_correct = plot_parity_comparison(
            tensor_sel, recon_physical, mask,
            f"{phase_for_tensor}: Original vs Correctly Denormalized",
            "Original G (J/mol)", "Reconstructed G (J/mol)"
        )
        st.plotly_chart(fig_parity_correct, use_container_width=True, key="plotly_parity_correct")

        st.subheader("Parity Plot: BUGGY (No Denormalization)")
        fig_parity_buggy = plot_parity_comparison(
            tensor_sel, recon_buggy, mask,
            f"{phase_for_tensor}: Original vs BUGGY (No Denormalization)",
            "Original G (J/mol)", "Reconstructed G (J/mol)"
        )
        st.plotly_chart(fig_parity_buggy, use_container_width=True, key="plotly_parity_buggy")

        st.error(r"""
        The Bug Explained:

        When we normalize: G_norm = (G_raw - mu) / sigma, the CPD factors 
        reconstruct G_norm (values near +/-3). 

        Forgetting to denormalize means we plot G_norm against G_raw (~10^5 J/mol),
        creating a massive error. The fix is simple:

        # CORRECT: Denormalize back to physical units
        G_physical = G_reconstructed_norm * sigma + mu

        # BUGGY: Forgetting denormalization
        G_buggy = G_reconstructed_norm  # Values near 0, not ~-10^5 J/mol!
        """)

        st.subheader("Temperature Slice: Original vs Reconstructed")
        T_slice_recon = st.selectbox("Select Temperature", tdt_data['T_vals'], 
                                      index=len(tdt_data['T_vals'])//2, key="recon_T")
        t_idx = tdt_data['T_vals'].index(T_slice_recon)

        orig_slice = tensor_sel[:, :, n_fe//2, t_idx]
        recon_slice = recon_physical[:, :, n_fe//2, t_idx]

        fig_recon_slice = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original', 'Reconstructed'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )

        zmin = min(np.nanmin(orig_slice), np.nanmin(recon_slice))
        zmax = max(np.nanmax(orig_slice), np.nanmax(recon_slice))

        fig_recon_slice.add_trace(go.Heatmap(
            z=orig_slice, x=tdt_data['co_vals'], y=tdt_data['cr_vals'],
            colorscale='RdBu_r', zmin=zmin, zmax=zmax,
            colorbar=dict(title="G (J/mol)", x=0.45)
        ), row=1, col=1)

        fig_recon_slice.add_trace(go.Heatmap(
            z=recon_slice, x=tdt_data['co_vals'], y=tdt_data['cr_vals'],
            colorscale='RdBu_r', zmin=zmin, zmax=zmax,
            colorbar=dict(title="G (J/mol)", x=1.0)
        ), row=1, col=2)

        fig_recon_slice.update_layout(
            title=f"{phase_for_tensor} at T={T_slice_recon}K",
            height=450, template='plotly_white'
        )
        st.plotly_chart(fig_recon_slice, use_container_width=True, key="plotly_recon_slice")

# ---------------------------------------------
# TAB 5: QUADRATIC EXPANSION
# ---------------------------------------------
with tab_quadratic:
    st.header("Step 5: Quadratic Expansion for Phase-Field Modeling")
    st.markdown(r"""
    The phase-field model uses a physics-preserving simplification of the full CPD tensor.
    Instead of querying the full 4D tensor, we use a quadratic Taylor expansion around equilibrium:

    G(c, T) ~ G_eq + A_Co*(c_Co - c_eq_Co)^2 + A_Cr*(c_Cr - c_eq_Cr)^2 + A_Fe*(c_Fe - c_eq_Fe)^2 + A_T*(T - T_m)^2

    CRITICAL: Quadratic coefficients from CPD factors are in normalized space.
    They must be denormalized:
    - A_alpha_physical = A_alpha_normalized * sigma
    - G_eq_physical = G_eq_normalized * sigma + mu
    """)

    if not st.session_state.get(f'cpd_completed_{phase_key}', False):
        st.warning(f"Please run CPD for {phase_for_tensor} first.")
    else:
        meta = st.session_state['tdt_metadata']
        A = st.session_state[f'A_{phase_key.lower()}']
        B = st.session_state[f'B_{phase_key.lower()}']
        C = st.session_state[f'C_{phase_key.lower()}']
        D = st.session_state[f'D_{phase_key.lower()}']
        lam = st.session_state[f'lam_{phase_key.lower()}']
        sigma = st.session_state[f'cpd_sigma_{phase_key.lower()}']
        mu = st.session_state[f'cpd_mu_{phase_key.lower()}']

        if st.button("Compute Quadratic Coefficients", use_container_width=True, type="primary"):
            with st.spinner("Computing Hessian from CPD factors..."):
                coeffs_norm = compute_quadratic_coefficients_from_cpd(
                    A, B, C, D, lam,
                    meta['co_vals'], meta['cr_vals'], meta['fe_vals'], meta['T_vals'],
                    c_eq_co, c_eq_cr, c_eq_fe, T_m
                )

                coeffs = {
                    'A_Co': coeffs_norm['A_Co'] * sigma,
                    'A_Cr': coeffs_norm['A_Cr'] * sigma,
                    'A_Fe': coeffs_norm['A_Fe'] * sigma,
                    'A_T':  coeffs_norm['A_T']  * sigma,
                    'G_eq': coeffs_norm['G_eq'] * sigma + mu,
                    'c_eq': coeffs_norm['c_eq'],
                    'T_m':  coeffs_norm['T_m']
                }

                st.session_state[f'quadratic_coeffs_{phase_key.lower()}'] = coeffs
                st.success("Quadratic coefficients computed and denormalized!")

        if f'quadratic_coeffs_{phase_key.lower()}' in st.session_state:
            coeffs = st.session_state[f'quadratic_coeffs_{phase_key.lower()}']

            st.subheader("Computed Quadratic Coefficients (Physical Units)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Coefficients:**")
                st.latex(rf"A_{{Co}} = {coeffs['A_Co']:.3e} \text{{ J/mol}}")
                st.latex(rf"A_{{Cr}} = {coeffs['A_Cr']:.3e} \text{{ J/mol}}")
                st.latex(rf"A_{{Fe}} = {coeffs['A_Fe']:.3e} \text{{ J/mol}}")
                st.latex(rf"A_{{T}} = {coeffs['A_T']:.3e} \text{{ J/(mol}}\cdot\text{{K}}^2)")
                st.latex(rf"G_{{eq}} = {coeffs['G_eq']:.3e} \text{{ J/mol}}")

            with col2:
                st.markdown("**Physical Interpretation:**")
                st.info(f"""
                - A_Co = {coeffs['A_Co']:.3e}: Curvature along Co direction
                  Positive = concave up (stable mixing)
                  Negative = concave down (tendency to phase separate)

                - A_T = {coeffs['A_T']:.3e}: Temperature curvature
                  Related to heat capacity: Cp ~ -2T*A_T

                - G_eq = {coeffs['G_eq']:.3e}: Gibbs energy at equilibrium
                  Reference point for phase-field driving force
                """)

            st.subheader("Verification: Full CPD vs Quadratic Approximation")

            np.random.seed(42)
            n_test = 200
            test_compositions = np.random.uniform(
                [max(0, c_eq_co-0.05), max(0, c_eq_cr-0.05), max(0, c_eq_fe-0.05)],
                [min(1, c_eq_co+0.05), min(1, c_eq_cr+0.05), min(1, c_eq_fe+0.05)],
                (n_test, 3)
            )
            valid_mask_test = np.sum(test_compositions, axis=1) <= 1.0
            test_compositions = test_compositions[valid_mask_test]
            test_temperatures = np.random.uniform(max(700, T_m-200), min(3300, T_m+200), len(test_compositions))

            verify_df = verify_quadratic_approximation(
                coeffs, A, B, C, D, lam,
                meta['co_vals'], meta['cr_vals'], meta['fe_vals'], meta['T_vals'],
                test_compositions, test_temperatures,
                sigma=sigma, mu=mu
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Abs Error", f"{verify_df['absolute_error'].mean():.2f} J/mol")
            c2.metric("Max Abs Error", f"{verify_df['absolute_error'].max():.2f} J/mol")
            c3.metric("Mean Rel Error", f"{verify_df['relative_error'].mean()*100:.2f}%")

            st.subheader("1D Slice: Varying x_Co at Fixed Composition")
            co_slice = np.linspace(max(0, c_eq_co-0.1), min(1, c_eq_co+0.1), 50)

            R = len(lam)
            A_funcs = [UnivariateSpline(meta['co_vals'], A[:, r], s=0, ext=3) for r in range(R)]
            B_funcs = [UnivariateSpline(meta['cr_vals'], B[:, r], s=0, ext=3) for r in range(R)]
            C_funcs = [UnivariateSpline(meta['fe_vals'], C[:, r], s=0, ext=3) for r in range(R)]
            D_funcs = [UnivariateSpline(meta['T_vals'], D[:, r], s=0, ext=3) for r in range(R)]

            G_full_slice, G_quad_slice = [], []
            for c_co in co_slice:
                g_norm = sum(lam[r] * A_funcs[r](c_co) * B_funcs[r](c_eq_cr) * 
                             C_funcs[r](c_eq_fe) * D_funcs[r](T_m) for r in range(R))
                G_full_slice.append(g_norm * sigma + mu)

                g_quad = coeffs['G_eq'] + coeffs['A_Co']*(c_co - c_eq_co)**2
                G_quad_slice.append(g_quad)

            fig_1d = plot_1d_comparison(co_slice, G_full_slice, G_quad_slice, c_eq_co, T_m, phase_for_tensor)
            st.plotly_chart(fig_1d, use_container_width=True, key="plotly_1d_quad")

            st.subheader("Error Analysis")
            fig_err = plot_error_distribution(verify_df)
            st.plotly_chart(fig_err, use_container_width=True, key="plotly_err_dist")

            st.info(r"""
            Key Insight: The quadratic approximation is most accurate near the equilibrium point
            (within +/-5% composition, +/-200K temperature). This is expected -- it's a local Taylor expansion.

            The divergence at the edges represents the truncation error of neglecting higher-order terms,
            which is acceptable for localized phase-field simulations where the system stays near equilibrium.
            """)

# ---------------------------------------------
# TAB 6: THEORY
# ---------------------------------------------
with tab_theory:
    st.header("Theory: Error Minimization in CPD")
    st.markdown(r"""
    ### The Complete Mathematical Procedure

    #### Step 1: Normalization (Before CPD)

    Raw Gibbs energies G_raw span ~[-300000, 0] J/mol. ALS algorithms struggle with:
    - Overflow: Large values cause numerical instability in matrix inversions
    - Poor convergence: Gradient updates are dominated by magnitude, not structure

    Solution -- Z-Score Normalization:
    G_norm = (G_raw - mu) / sigma

    where:
    mu = (1/N_valid) * sum of G_raw over valid simplex entries
    sigma = sqrt((1/N_valid) * sum of (G_raw - mu)^2 over valid entries)

    Important: mu and sigma are computed ONLY on valid (non-NaN) entries.
    Including invalid entries would bias the statistics.

    ---

    #### Step 2: CPD Decomposition (in Normalized Space)

    G_norm[i,j,k,t] ~ sum_r lambda_r * A_r(i) * B_r(j) * C_r(k) * D_r(t)

    The ALS algorithm iteratively updates each factor matrix while holding others fixed.
    Because G_norm ~ N(0,1), all factors are well-conditioned and converge reliably.

    ---

    #### Step 3: Reconstruction & Denormalization (The Fix)

    The Bug: Reconstructing G_rec_norm and forgetting to convert back:
    G_buggy = G_rec_norm    (WRONG! Values near 0)

    The Fix: Apply the inverse transform:
    G_physical = G_rec_norm * sigma + mu

    This restores the correct physical scale (~10^5 J/mol) and gives RMSE < 1%.

    ---

    #### Step 4: Quadratic Expansion (Denormalized Coefficients)

    The quadratic coefficients are second derivatives of the CPD reconstruction:
    A_alpha = 0.5 * sum_r lambda_r * d2F_alpha/dc_alpha^2|eq * prod_{beta!=alpha} F_beta(c_eq_beta) * D(T_m)

    Since these are computed from normalized factors, they inherit the normalization:
    A_alpha_physical = A_alpha_normalized * sigma
    G_eq_physical = G_eq_normalized * sigma + mu

    Forgetting this scaling gives coefficients that are off by ~5 orders of magnitude!

    ---

    #### Summary Table

    | Quantity | Normalized Space | Physical Space | Conversion |
    |----------|-------------------|----------------|------------|
    | Gibbs Energy G | ~ N(0,1) | ~ [-300000, 0] J/mol | G_phys = G_norm * sigma + mu |
    | Quadratic coeff A_alpha | ~ O(1) | ~ O(10^5) J/mol | A_alpha_phys = A_alpha_norm * sigma |
    | G_eq | ~ O(1) | ~ O(10^5) J/mol | G_eq_phys = G_eq_norm * sigma + mu |
    | RMSE | ~ 10^-2 | ~ 10^2 J/mol | Multiply by sigma |

    ---

    ### Why This Matters for Phase-Field Modeling

    In phase-field simulations, the driving force for phase transformation is:
    Delta G = G_LIQ - G_FCC

    If either G_LIQ or G_FCC is incorrectly denormalized, the driving force is wrong by ~10^5 J/mol,
    causing:
    - Incorrect interface velocity
    - Wrong nucleation barrier
    - Spurious phase transformations

    Proper normalization/denormalization ensures thermodynamic consistency between:
    1. The full CPD tensor (global accuracy)
    2. The quadratic approximation (local efficiency)
    3. The phase-field equations (physical correctness)
    """)
