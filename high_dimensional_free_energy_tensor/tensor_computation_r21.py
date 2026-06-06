import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import glob
from scipy.interpolate import UnivariateSpline

# =============================================
# 1. PATH CONFIGURATION & DATA LOADING
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(page_title="CPD Tutorial: Error Minimization", layout="wide")
st.title(" Tutorial: CPD Reconstruction & Error Minimization")
st.markdown("This app demonstrates how to correctly scale CPD factors to match physical units (J/mol).")

@st.cache_data(ttl=3600)
def load_and_build_tensor(csv_dir):
    """Load CSVs and build a 4D sparse tensor."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        st.error(f"No CSV files found in `{csv_dir}`.")
        st.stop()
    
    # For tutorial speed, we'll use a subset of temperatures if there are many
    # In production, use all 31 files. Here we just take the first 5 for fast demo.
    files = files[:5] 
    
    dfs = []
    for f in files:
        T = int(os.path.basename(f).replace("Gibbs_", "").replace("K.csv", ""))
        df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ"])
        df["T"] = T
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Define grid
    co_vals = np.unique(df_all["Co"].round(3))
    cr_vals = np.unique(df_all["Cr"].round(3))
    fe_vals = np.unique(df_all["Fe"].round(3))
    T_vals = np.unique(df_all["T"])
    
    # Build 4D Tensor
    G_tensor = np.full((len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)), np.nan)
    
    co_idx = {v: i for i, v in enumerate(co_vals)}
    cr_idx = {v: i for i, v in enumerate(cr_vals)}
    fe_idx = {v: i for i, v in enumerate(fe_vals)}
    T_idx = {v: i for i, v in enumerate(T_vals)}
    
    for _, row in df_all.iterrows():
        i = co_idx.get(round(row["Co"], 3))
        j = cr_idx.get(round(row["Cr"], 3))
        k = fe_idx.get(round(row["Fe"], 3))
        t = T_idx.get(row["T"])
        if i is not None and j is not None and k is not None and t is not None:
            G_tensor[i, j, k, t] = row["G_LIQ"]
            
    return G_tensor, co_vals, cr_vals, fe_vals, T_vals

# Load Data
with st.spinner("Loading data and building tensor..."):
    G_tensor, co_vals, cr_vals, fe_vals, T_vals = load_and_build_tensor(CSV_FILES_DIR)

st.success(f"✅ Tensor built: Shape {G_tensor.shape}. Valid entries: {np.sum(~np.isnan(G_tensor)):,}")

# =============================================
# 2. NORMALIZATION (The Key to Stability)
# =============================================
st.header("Step 2: Normalization")
st.info("CPD algorithms (ALS) are numerically unstable with large magnitudes (~10⁵). We must normalize to mean=0, std=1.")

mask = ~np.isnan(G_tensor)
mu = np.nanmean(G_tensor)
sigma = np.nanstd(G_tensor)

st.metric("Physical Mean (μ)", f"{mu:,.0f} J/mol")
st.metric("Physical Std Dev (σ)", f"{sigma:,.0f} J/mol")

G_norm = (G_tensor - mu) / sigma
st.success(f"Normalized tensor range: [{np.nanmin(G_norm):.2f}, {np.nanmax(G_norm):.2f}]")

# =============================================
# 3. CPD DECOMPOSITION (Masked ALS)
# =============================================
st.header("Step 3: CPD Decomposition")
rank = st.slider("CPD Rank (R)", 1, 10, 3)
max_iter = st.slider("Max Iterations", 10, 100, 30)

#
#

def run_masked_als(X, mask, rank, max_iter=50, reg=1e-4):
    """
    Mathematically rigorous Masked ALS for 4D CPD.
    Uses explicit loops for np.linalg.solve to prevent NumPy broadcasting ValueErrors.
    """
    I, J, K, L = X.shape
    
    # Initialize factors randomly
    A = np.random.rand(I, rank)
    B = np.random.rand(J, rank)
    C = np.random.rand(K, rank)
    D = np.random.rand(L, rank)
    
    X_filled = np.where(mask, X, 0.0)
    mask_float = mask.astype(float)
    
    # Tikhonov regularization matrix
    I_reg = reg * np.eye(rank)
    
    for it in range(max_iter):
        # --- UPDATE A (Mode-1) ---
        T_A = np.einsum('ijkl,jr,kr,lr->ir', X_filled, B, C, D)
        V_A = np.einsum('ijkl,jr,js,kr,ks,lr,ls->irs', mask_float, B, B, C, C, D, D)
        for i in range(I):
            try:
                A[i, :] = np.linalg.solve(V_A[i] + I_reg, T_A[i])
            except np.linalg.LinAlgError:
                A[i, :] = np.linalg.lstsq(V_A[i] + I_reg, T_A[i], rcond=None)[0]
                
        # --- UPDATE B (Mode-2) ---
        T_B = np.einsum('ijkl,ir,kr,lr->jr', X_filled, A, C, D)
        V_B = np.einsum('ijkl,ir,is,kr,ks,lr,ls->jrs', mask_float, A, A, C, C, D, D)
        for j in range(J):
            try:
                B[j, :] = np.linalg.solve(V_B[j] + I_reg, T_B[j])
            except np.linalg.LinAlgError:
                B[j, :] = np.linalg.lstsq(V_B[j] + I_reg, T_B[j], rcond=None)[0]
                
        # --- UPDATE C (Mode-3) ---
        T_C = np.einsum('ijkl,ir,jr,lr->kr', X_filled, A, B, D)
        V_C = np.einsum('ijkl,ir,is,jr,js,lr,ls->krs', mask_float, A, A, B, B, D, D)
        for k in range(K):
            try:
                C[k, :] = np.linalg.solve(V_C[k] + I_reg, T_C[k])
            except np.linalg.LinAlgError:
                C[k, :] = np.linalg.lstsq(V_C[k] + I_reg, T_C[k], rcond=None)[0]
                
        # --- UPDATE D (Mode-4) ---
        T_D = np.einsum('ijkl,ir,jr,kr->lr', X_filled, A, B, C)
        V_D = np.einsum('ijkl,ir,is,jr,js,kr,ks->lrs', mask_float, A, A, B, B, C, C)
        for l_idx in range(L):
            try:
                D[l_idx, :] = np.linalg.solve(V_D[l_idx] + I_reg, T_D[l_idx])
            except np.linalg.LinAlgError:
                D[l_idx, :] = np.linalg.lstsq(V_D[l_idx] + I_reg, T_D[l_idx], rcond=None)[0]
                
        # --- NORMALIZE FACTORS ---
        # Prevents magnitude drift and keeps factors bounded
        for r in range(rank):
            nA = np.linalg.norm(A[:, r]) or 1.0
            nB = np.linalg.norm(B[:, r]) or 1.0
            nC = np.linalg.norm(C[:, r]) or 1.0
            nD = np.linalg.norm(D[:, r]) or 1.0
            
            factor = (nA * nB * nC * nD) ** 0.25
            if factor > 1e-12:
                A[:, r] *= (nA / factor)
                B[:, r] *= (nB / factor)
                C[:, r] *= (nC / factor)
                D[:, r] *= (nD / factor)

    # --- EXTRACT WEIGHTS (lambda) ---
    lam = np.ones(rank)
    for r in range(rank):
        nA = np.linalg.norm(A[:, r])
        nB = np.linalg.norm(B[:, r])
        nC = np.linalg.norm(C[:, r])
        nD = np.linalg.norm(D[:, r])
        lam[r] = nA * nB * nC * nD
        
        if lam[r] > 1e-12:
            A[:, r] /= nA
            B[:, r] /= nB
            C[:, r] /= nC
            D[:, r] /= nD
            
    return A, B, C, D, lam


if st.button("Run CPD-ALS"):
    with st.spinner("Decomposing..."):
        A, B, C, D, lam = run_masked_als(G_norm, mask, rank, max_iter)
        st.session_state['cpd_factors'] = (A, B, C, D, lam)
        st.session_state['scaler'] = {'mu': mu, 'sigma': sigma}
        st.success("✅ CPD Complete!")

# =============================================
# 4. RECONSTRUCTION & THE "FIX"
# =============================================
st.header("Step 4: Reconstruction & Error Minimization")
if 'cpd_factors' in st.session_state:
    A, B, C, D, lam = st.session_state['cpd_factors']
    meta = st.session_state['scaler']
    
    st.subheader("The Parity Plot: Original vs Reconstructed")
    st.markdown("This plot proves whether our reconstruction is correct.")
    
    # Pick a temperature slice for visualization
    t_idx = st.selectbox("Select Temperature Index", range(len(T_vals)), format_func=lambda x: f"T={T_vals[x]}K")
    
    orig_slice = G_tensor[:, :, :, t_idx]
    
    # Reconstruct in normalized space
    G_rec_norm = np.zeros_like(orig_slice)
    for r in range(rank):
        G_rec_norm += lam[r] * np.outer(A[:, r], np.kron(np.kron(D[t_idx, r], C[:, r]), B[:, r])).reshape(orig_slice.shape)
        
    # ❌ WRONG WAY (What causes the 10^-2 magnitude error)
    G_rec_wrong = G_rec_norm 
    
    # ✅ RIGHT WAY (Denormalization)
    G_rec_right = G_rec_norm * meta['sigma'] + meta['mu']
    
    valid = ~np.isnan(orig_slice)
    
    col1, col2 = st.columns(2)
    with col1:
        st.warning("❌ Without Denormalization (Normalized Space)")
        fig_wrong = go.Figure()
        fig_wrong.add_trace(go.Scatter(x=orig_slice[valid], y=G_rec_wrong[valid], mode='markers', name='Data'))
        fig_wrong.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='y=x', line=dict(dash='dash', color='red')))
        fig_wrong.update_layout(title="Magnitude ~1.0 (WRONG)", xaxis_title="Original", yaxis_title="Reconstructed")
        st.plotly_chart(fig_wrong, use_container_width=True)
        
    with col2:
        st.success("✅ With Denormalization (Physical Space)")
        fig_right = go.Figure()
        fig_right.add_trace(go.Scatter(x=orig_slice[valid], y=G_rec_right[valid], mode='markers', name='Data', marker_color='green'))
        min_val = min(np.min(orig_slice[valid]), np.min(G_rec_right[valid]))
        max_val = max(np.max(orig_slice[valid]), np.max(G_rec_right[valid]))
        fig_right.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='y=x', line=dict(dash='dash', color='red')))
        fig_right.update_layout(title=f"Magnitude ~{meta['mu']:,.0f} J/mol (CORRECT)", xaxis_title="Original", yaxis_title="Reconstructed")
        st.plotly_chart(fig_right, use_container_width=True)

    # =============================================
    # 5. QUADRATIC EXPANSION (Taylor Series)
    # =============================================
    st.header("Step 5: Quadratic Expansion (Phase-Field)")
    st.markdown("Quadratic coefficients must ALSO be scaled by σ to match physical units.")
    
    c_eq_co = st.slider("Equilibrium x_Co", 0.0, 1.0, 0.33)
    c_eq_cr = st.slider("Equilibrium x_Cr", 0.0, 1.0, 0.33)
    c_eq_fe = st.slider("Equilibrium x_Fe", 0.0, 1.0, 0.33)
    
    if st.button("Compute Quadratic"):
        # Build splines for factors
        A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(rank)]
        B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(rank)]
        C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(rank)]
        D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(rank)]
        
        # 1. Baseline G_eq (Normalized)
        G_eq_norm = sum(lam[r] * A_funcs[r](c_eq_co) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_vals[t_idx]) for r in range(rank))
        
        # 2. Curvature A_Co (Normalized) - Second derivative
        A_Co_norm = 0.5 * sum(lam[r] * A_funcs[r].derivative(2)(c_eq_co) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_vals[t_idx]) for r in range(rank))
        
        # ✅ THE FIX: Scale coefficients back to physical units
        G_eq_phys = G_eq_norm * meta['sigma'] + meta['mu']
        A_Co_phys = A_Co_norm * meta['sigma']
        
        st.metric("Physical G_eq", f"{G_eq_phys:,.0f} J/mol")
        st.metric("Physical A_Co (Curvature)", f"{A_Co_phys:,.2f} J/mol")
        
        # Plot 1D slice
        co_slice = np.linspace(max(0, c_eq_co-0.1), min(1, c_eq_co+0.1), 50)
        G_cpd_slice = []
        G_quad_slice = []
        
        for c in co_slice:
            # Full CPD (Physical)
            g_cpd_norm = sum(lam[r] * A_funcs[r](c) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_vals[t_idx]) for r in range(rank))
            G_cpd_slice.append(g_cpd_norm * meta['sigma'] + meta['mu'])
            
            # Quadratic (Physical)
            G_quad_slice.append(G_eq_phys + A_Co_phys * (c - c_eq_co)**2)
            
        fig_quad = go.Figure()
        fig_quad.add_trace(go.Scatter(x=co_slice, y=G_cpd_slice, mode='lines', name='Full CPD (Physical)', line=dict(color='blue')))
        fig_quad.add_trace(go.Scatter(x=co_slice, y=G_quad_slice, mode='lines', name='Quadratic (Physical)', line=dict(color='red', dash='dash')))
        fig_quad.add_vline(x=c_eq_co, line_dash="dot", color="green", annotation_text="Equilibrium")
        fig_quad.update_layout(title="Quadratic vs Full CPD (Aligned Magnitudes)", xaxis_title="x_Co", yaxis_title="Gibbs Energy (J/mol)")
        st.plotly_chart(fig_quad, use_container_width=True)

else:
    st.info(" Please run the CPD-ALS in Step 3 first.")
