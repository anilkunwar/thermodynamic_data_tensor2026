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

st.set_page_config(page_title="Co-Cr-Fe-Ni Phase Stability Explorer v3", layout="wide")
st.title("🔬 Co-Cr-Fe-Ni Phase Stability Explorer v3")
st.markdown("Upgraded with robust Masked ALS, correct physical scaling, and crash-free Plotly rendering.")

@st.cache_data(ttl=3600)
def load_and_build_tensor(csv_dir):
    """Load CSVs and build a 4D sparse tensor."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        st.error(f"No CSV files found in `{csv_dir}`.")
        st.stop()
    
    dfs = []
    for f in files:
        T = int(os.path.basename(f).replace("Gibbs_", "").replace("K.csv", ""))
        df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
        df["T"] = T
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dG"] = df_all["G_LIQ"] - df_all["G_FCC"]
    
    co_vals = np.unique(df_all["Co"].round(3))
    cr_vals = np.unique(df_all["Cr"].round(3))
    fe_vals = np.unique(df_all["Fe"].round(3))
    T_vals = np.unique(df_all["T"])
    
    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)
    G_LIQ = np.full((n_co, n_cr, n_fe, n_T), np.nan)
    G_FCC = np.full((n_co, n_cr, n_fe, n_T), np.nan)
    
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
            G_LIQ[i, j, k, t] = row["G_LIQ"]
            G_FCC[i, j, k, t] = row["G_FCC"]
            
    return G_LIQ, G_FCC, co_vals, cr_vals, fe_vals, T_vals, df_all

# Load Data
with st.spinner("Loading data and building tensor..."):
    G_LIQ, G_FCC, co_vals, cr_vals, fe_vals, T_vals, df_all = load_and_build_tensor(CSV_FILES_DIR)

st.success(f"✅ Tensor built: Shape {G_LIQ.shape}. Valid entries: {np.sum(~np.isnan(G_LIQ)):,}")

# =============================================
# 2. ROBUST MASKED ALS DECOMPOSITION
# =============================================
def run_masked_als(X, mask, rank, max_iter=50, reg=1e-4):
    """
    Mathematically rigorous Masked ALS for 4D CPD.
    Uses explicit loops for np.linalg.solve to prevent NumPy broadcasting ValueErrors.
    Uses Tikhonov regularization to prevent factor explosion in sparse tensors.
    """
    I, J, K, L = X.shape
    A = np.random.rand(I, rank)
    B = np.random.rand(J, rank)
    C = np.random.rand(K, rank)
    D = np.random.rand(L, rank)
    
    X_filled = np.where(mask, X, 0.0)
    mask_float = mask.astype(float)
    I_reg = reg * np.eye(rank)
    
    for it in range(max_iter):
        # UPDATE A (Mode-1)
        T_A = np.einsum('ijkl,jr,kr,lr->ir', X_filled, B, C, D)
        V_A = np.einsum('ijkl,jr,js,kr,ks,lr,ls->irs', mask_float, B, B, C, C, D, D)
        for i in range(I):
            try: A[i, :] = np.linalg.solve(V_A[i] + I_reg, T_A[i])
            except: A[i, :] = np.linalg.lstsq(V_A[i] + I_reg, T_A[i], rcond=None)[0]
                
        # UPDATE B (Mode-2)
        T_B = np.einsum('ijkl,ir,kr,lr->jr', X_filled, A, C, D)
        V_B = np.einsum('ijkl,ir,is,kr,ks,lr,ls->jrs', mask_float, A, A, C, C, D, D)
        for j in range(J):
            try: B[j, :] = np.linalg.solve(V_B[j] + I_reg, T_B[j])
            except: B[j, :] = np.linalg.lstsq(V_B[j] + I_reg, T_B[j], rcond=None)[0]
                
        # UPDATE C (Mode-3)
        T_C = np.einsum('ijkl,ir,jr,lr->kr', X_filled, A, B, D)
        V_C = np.einsum('ijkl,ir,is,jr,js,lr,ls->krs', mask_float, A, A, B, B, D, D)
        for k in range(K):
            try: C[k, :] = np.linalg.solve(V_C[k] + I_reg, T_C[k])
            except: C[k, :] = np.linalg.lstsq(V_C[k] + I_reg, T_C[k], rcond=None)[0]
                
        # UPDATE D (Mode-4)
        T_D = np.einsum('ijkl,ir,jr,kr->lr', X_filled, A, B, C)
        V_D = np.einsum('ijkl,ir,is,jr,js,kr,ks->lrs', mask_float, A, A, B, B, C, C)
        for l_idx in range(L):
            try: D[l_idx, :] = np.linalg.solve(V_D[l_idx] + I_reg, T_D[l_idx])
            except: D[l_idx, :] = np.linalg.lstsq(V_D[l_idx] + I_reg, T_D[l_idx], rcond=None)[0]
                
        # NORMALIZE FACTORS
        for r in range(rank):
            nA = np.linalg.norm(A[:, r]) or 1.0; nB = np.linalg.norm(B[:, r]) or 1.0
            nC = np.linalg.norm(C[:, r]) or 1.0; nD = np.linalg.norm(D[:, r]) or 1.0
            factor = (nA * nB * nC * nD) ** 0.25
            if factor > 1e-12:
                A[:, r] *= (nA / factor); B[:, r] *= (nB / factor)
                C[:, r] *= (nC / factor); D[:, r] *= (nD / factor)

    # EXTRACT WEIGHTS (lambda)
    lam = np.ones(rank)
    for r in range(rank):
        nA = np.linalg.norm(A[:, r]); nB = np.linalg.norm(B[:, r])
        nC = np.linalg.norm(C[:, r]); nD = np.linalg.norm(D[:, r])
        lam[r] = nA * nB * nC * nD
        if lam[r] > 1e-12:
            A[:, r] /= nA; B[:, r] /= nB; C[:, r] /= nC; D[:, r] /= nD
            
    return A, B, C, D, lam

# =============================================
# 3. QUADRATIC EXPANSION (PHASE-FIELD)
# =============================================
def compute_quadratic_coefficients(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, c_eq, T_m, mu, sigma):
    """Compute quadratic coefficients and scale them back to physical units."""
    R = len(lam)
    A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
    B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
    C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
    D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]
    
    # Compute in normalized space
    G_eq_norm = sum(lam[r] * A_funcs[r](c_eq[0]) * B_funcs[r](c_eq[1]) * C_funcs[r](c_eq[2]) * D_funcs[r](T_m) for r in range(R))
    A_Co_norm = 0.5 * sum(lam[r] * A_funcs[r].derivative(2)(c_eq[0]) * B_funcs[r](c_eq[1]) * C_funcs[r](c_eq[2]) * D_funcs[r](T_m) for r in range(R))
    A_Cr_norm = 0.5 * sum(lam[r] * A_funcs[r](c_eq[0]) * B_funcs[r].derivative(2)(c_eq[1]) * C_funcs[r](c_eq[2]) * D_funcs[r](T_m) for r in range(R))
    A_Fe_norm = 0.5 * sum(lam[r] * A_funcs[r](c_eq[0]) * B_funcs[r](c_eq[1]) * C_funcs[r].derivative(2)(c_eq[2]) * D_funcs[r](T_m) for r in range(R))
    A_T_norm = 0.5 * sum(lam[r] * A_funcs[r](c_eq[0]) * B_funcs[r](c_eq[1]) * C_funcs[r](c_eq[2]) * D_funcs[r].derivative(2)(T_m) for r in range(R))
    
    # SCALE BACK TO PHYSICAL UNITS
    return {
        'G_eq': G_eq_norm * sigma + mu,
        'A_Co': A_Co_norm * sigma, 'A_Cr': A_Cr_norm * sigma,
        'A_Fe': A_Fe_norm * sigma, 'A_T': A_T_norm * sigma,
        'c_eq': c_eq, 'T_m': T_m
    }

# =============================================
# UI TABS
# =============================================
tab1, tab2, tab3 = st.tabs([" Phase Visualization", "🔧 Tensor Decomposition (CPD)", "📐 Quadratic Expansion"])

with tab1:
    st.header("Phase Visualization (Raw Data)")
    T_val = st.slider("Temperature (K)", int(T_vals.min()), int(T_vals.max()), 1500, step=100)
    
    pts = df_all[df_all["T"]==T_val][["Co","Cr","Fe","G_LIQ","G_FCC"]].values
    if len(pts) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers',
                                   marker=dict(color=pts[:,3], colorscale='Reds', size=3, opacity=0.6), name='G_LIQ'))
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers',
                                   marker=dict(color=pts[:,4], colorscale='Blues', size=3, opacity=0.6), name='G_FCC'))
        fig.update_layout(scene=dict(xaxis_title='Co', yaxis_title='Cr', zaxis_title='Fe'), height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Tensor Decomposition (CPD-ALS)")
    phase = st.radio("Select Phase", ["G_LIQ", "G_FCC"], horizontal=True)
    tensor = G_LIQ if phase == "G_LIQ" else G_FCC
    rank = st.slider("Rank (R)", 1, 10, 4)
    
    if st.button("Run CPD-ALS", type="primary"):
        mask = ~np.isnan(tensor)
        mu = np.nanmean(tensor)
        sigma = np.nanstd(tensor)
        tensor_norm = (tensor - mu) / sigma
        
        with st.spinner("Decomposing..."):
            A, B, C, D, lam = run_masked_als(tensor_norm, mask, rank, max_iter=30)
            
        st.session_state[f'cpd_{phase}'] = {'A':A, 'B':B, 'C':C, 'D':D, 'lam':lam, 'mu':mu, 'sigma':sigma}
        st.success("✅ CPD Complete!")
        
        # Parity Plot (Denormalized)
        G_rec_norm = np.zeros_like(tensor)
        for r in range(rank):
            G_rec_norm += lam[r] * np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(tensor.shape)
        G_rec_phys = G_rec_norm * sigma + mu
        
        valid = ~np.isnan(tensor)
        fig_parity = go.Figure()
        fig_parity.add_trace(go.Scatter(x=tensor[valid], y=G_rec_phys[valid], mode='markers', name='Data', marker_color='green'))
        min_v, max_v = min(np.min(tensor[valid]), np.min(G_rec_phys[valid])), max(np.max(tensor[valid]), np.max(G_rec_phys[valid]))
        fig_parity.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='y=x', line=dict(dash='dash', color='red')))
        fig_parity.update_layout(title=f"Parity Plot ({phase}) - Physical Units", xaxis_title="Original", yaxis_title="Reconstructed")
        st.plotly_chart(fig_parity, use_container_width=True)

with tab3:
    st.header("Quadratic Expansion (Phase-Field)")
    if 'cpd_G_LIQ' not in st.session_state:
        st.warning("⚠️ Please run CPD for G_LIQ in the Tensor Decomposition tab first.")
    else:
        cpd = st.session_state['cpd_G_LIQ']
        A, B, C, D, lam = cpd['A'], cpd['B'], cpd['C'], cpd['D'], cpd['lam']
        mu, sigma = cpd['mu'], cpd['sigma']
        
        c_eq_co = st.slider("x_Co", 0.0, 1.0, 0.33)
        c_eq_cr = st.slider("x_Cr", 0.0, 1.0, 0.33)
        c_eq_fe = st.slider("x_Fe", 0.0, 1.0, 0.33)
        T_m = st.slider("T_m (K)", int(T_vals.min()), int(T_vals.max()), 1500)
        
        if st.button("Compute Quadratic", type="primary"):
            coeffs = compute_quadratic_coefficients(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, [c_eq_co, c_eq_cr, c_eq_fe], T_m, mu, sigma)
            
            st.metric("Physical G_eq", f"{coeffs['G_eq']:,.0f} J/mol")
            st.metric("Physical A_Co", f"{coeffs['A_Co']:,.2f} J/mol")
            
            # 1D Slice Comparison
            co_slice = np.linspace(max(0, c_eq_co-0.1), min(1, c_eq_co+0.1), 50)
            R = len(lam)
            A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
            B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
            C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
            D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]
            
            G_cpd_slice, G_quad_slice = [], []
            for c in co_slice:
                g_cpd_norm = sum(lam[r] * A_funcs[r](c) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_m) for r in range(R))
                G_cpd_slice.append(g_cpd_norm * sigma + mu)
                G_quad_slice.append(coeffs['G_eq'] + coeffs['A_Co']*(c - c_eq_co)**2)
                
            # ROBUST PLOTTING (Bypasses add_vline crashes)
            fig_quad = go.Figure()
            fig_quad.add_trace(go.Scatter(x=co_slice, y=G_cpd_slice, mode='lines', name='Full CPD', line=dict(color='blue', width=3)))
            fig_quad.add_trace(go.Scatter(x=co_slice, y=G_quad_slice, mode='lines', name='Quadratic', line=dict(color='red', dash='dash', width=2)))
            
            y_min = float(np.nanmin([G_cpd_slice, G_quad_slice]))
            y_max = float(np.nanmax([G_cpd_slice, G_quad_slice]))
            y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 1000.0
            
            # Draw vertical line as a Scatter trace
            fig_quad.add_trace(go.Scatter(x=[c_eq_co, c_eq_co], y=[y_min - y_pad, y_max + y_pad], 
                                          mode='lines', line=dict(color='green', dash='dot', width=2), 
                                          name='Equilibrium', showlegend=False, hoverinfo='skip'))
            
            # Explicitly set axis ranges to prevent auto-scaling crashes
            fig_quad.update_xaxes(range=[float(co_slice.min()), float(co_slice.max())], title_text="x_Co")
            fig_quad.update_yaxes(range=[y_min - y_pad, y_max + y_pad], title_text="Gibbs Energy (J/mol)")
            fig_quad.update_layout(title="Quadratic vs Full CPD (Aligned Magnitudes)", template="plotly_white", height=500)
            st.plotly_chart(fig_quad, use_container_width=True)
