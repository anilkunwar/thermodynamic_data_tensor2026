"""
================================================================================
Co-Cr-Fe-Ni Phase Stability Explorer v2
Thermodynamic Data Tensor Analysis with Canonical Polyadic Decomposition (CPD)
================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline
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
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        st.error(f"❌ No CSV files found in `{csv_dir}`.")
        st.stop()

    expected_temps = list(range(300, 3301, 100))
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
    df_combined["dG"] = df_combined["G_LIQ"] - df_combined["G_FCC"]
    st.caption(f"✅ Loaded {len(df_combined):,} measurements across {len(found_temps)} temperatures")
    return df_combined

df = load_all_data()

T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))

all_pts = df[["Co", "Cr", "Fe"]].values
try:
    data_hull = ConvexHull(all_pts)
    HULL_AVAILABLE = True
except Exception:
    HULL_AVAILABLE = False
    data_hull = None

# =============================================
# TENSOR ANALYSIS FUNCTIONS
# =============================================
@st.cache_data(ttl=7200)
def build_tensor_data(df):
    co_vals = sorted(df["Co"].unique())
    cr_vals = sorted(df["Cr"].unique())
    fe_vals = sorted(df["Fe"].unique())
    T_vals = sorted(df["T"].unique())

    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)

    co_to_idx = {round(v, 4): i for i, v in enumerate(co_vals)}
    cr_to_idx = {round(v, 4): i for i, v in enumerate(cr_vals)}
    fe_to_idx = {round(v, 4): i for i, v in enumerate(fe_vals)}
    T_to_idx = {T: i for i, T in enumerate(T_vals)}

    G_LIQ_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    G_FCC_tdt = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)

    valid_count = 0
    for _, row in df.iterrows():
        co = round(row['Co'], 4)
        cr = round(row['Cr'], 4)
        fe = round(row['Fe'], 4)
        T = row['T']
        if co in co_to_idx and cr in cr_to_idx and fe in fe_to_idx and T in T_to_idx:
            i, j, k, t = co_to_idx[co], cr_to_idx[cr], fe_to_idx[fe], T_to_idx[T]
            G_LIQ_tdt[i, j, k, t] = row['G_LIQ']
            G_FCC_tdt[i, j, k, t] = row['G_FCC']
            valid_count += 1

    co_step = np.min(np.diff(co_vals)) if len(co_vals) > 1 else 0
    cr_step = np.min(np.diff(cr_vals)) if len(cr_vals) > 1 else 0
    fe_step = np.min(np.diff(fe_vals)) if len(fe_vals) > 1 else 0
    T_step = np.min(np.diff(T_vals)) if len(T_vals) > 1 else 0

    st.caption(f"Tensor built: {valid_count:,} valid entries ({100*valid_count/(n_co*n_cr*n_fe*n_T):.1f}% of hypercube)")

    return {
        'G_LIQ': G_LIQ_tdt, 'G_FCC': G_FCC_tdt,
        'dims': (n_co, n_cr, n_fe, n_T),
        'co_vals': co_vals, 'cr_vals': cr_vals, 'fe_vals': fe_vals, 'T_vals': T_vals,
        'co_step': co_step, 'cr_step': cr_step, 'fe_step': fe_step, 'T_step': T_step,
        'valid_count': valid_count
    }

def unfold_tensor(tensor, mode):
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
        st.warning(f"⚠️ SVD failed: {e}")
        return 0, np.zeros(min(matrix_filled.shape)), np.zeros(min(matrix_filled.shape))

    s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
    s_norm = s / s_max
    rank = int(np.sum(s_norm > threshold))
    return rank, s, s_norm

def cpd_als_4d(tensor, rank, max_iter=100, tol=1e-6):
    I, J, K, L = tensor.shape
    mask = ~np.isnan(tensor)
    X = np.where(mask, tensor, 0)

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
            D[:, 4:] = np.random.rand(L, rank-4) * 0.01
    else:
        D = np.random.rand(L, rank) * 0.1

    X_unfolded = unfold_tensor(X, mode=0)
    try:
        U, s, Vh = linalg.svd(X_unfolded, full_matrices=False)
        A = U[:, :rank] * np.sqrt(s[:rank])
    except:
        A = np.random.rand(I, rank) * 0.1

    B = np.random.rand(J, rank) * 0.1
    C = np.random.rand(K, rank) * 0.1

    prev_error = np.inf

    for iteration in range(max_iter):
        BCD = np.zeros((J*K*L, rank))
        for r in range(rank):
            BCD[:, r] = np.kron(np.kron(D[:, r], C[:, r]), B[:, r])
        X_flat = X.reshape(I, -1)
        mask_flat = mask.reshape(I, -1)
        for i in range(I):
            valid = mask_flat[i, :]
            if np.sum(valid) > rank:
                A[i, :] = linalg.lstsq(BCD[valid, :], X_flat[i, valid])[0]
        norms = np.linalg.norm(A, axis=0) + 1e-12
        A = A / norms

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

        recon = np.zeros_like(X)
        for r in range(rank):
            recon += np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)

        observed_residuals = (tensor - recon)[mask]
        if len(observed_residuals) > 0:
            error = np.sqrt(np.mean(observed_residuals**2))
        else:
            error = np.inf

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    lam = np.ones(rank)
    for r in range(rank):
        lam[r] = (np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * 
                  np.linalg.norm(C[:, r]) * np.linalg.norm(D[:, r]))

    return A, B, C, D, lam, error

# =============================================
# INTERPOLATION
# =============================================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values, fill_value=np.nan)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values, fill_value=np.nan)
    return interp_liq, interp_fcc

# =============================================
# TETRAHEDRAL GRID
# =============================================
def generate_tetrahedral_grid(resolution=25):
    x = np.linspace(0, 1, resolution)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    return grid_pts[valid_mask]

def compute_data_proximity(pts, data_pts, max_dist=0.15):
    tree = cKDTree(data_pts)
    dists, _ = tree.query(pts, k=1)
    return np.clip(1.0 - dists / max_dist, 0.0, 1.0)

def find_phase_boundary_points(pts, dG_values, threshold=50.0):
    boundary_mask = np.abs(dG_values) < threshold
    return pts[boundary_mask], dG_values[boundary_mask]

# =============================================
# SPHERICAL HARMONICS
# =============================================
if SCIPY_AVAILABLE:
    def get_real_sph_harm(l, m, theta, phi):
        if hasattr(special, 'sph_harm_y'):
            Y_complex = special.sph_harm_y(l, m, phi, theta)
        else:
            Y_complex = special.sph_harm(m, l, theta, phi)
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
        R_max_safe = 1.0 / np.sqrt(3.0)
        if R_fixed > R_max_safe:
            R_fixed = R_max_safe
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)
        x = R_fixed * np.sin(PH) * np.cos(TH)
        y = R_fixed * np.sin(PH) * np.sin(TH)
        z = R_fixed * np.cos(PH)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        valid = (pts[:,0] + pts[:,1] + pts[:,2]) <= 1.0
        valid = valid & (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 2] >= 0)
        G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
        G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)
        G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
        dG = G_liq - G_fcc
        valid = valid & ~np.isnan(G_stable)
        return (TH, PH, G_stable.reshape(TH.shape), dG.reshape(TH.shape), 
                valid.reshape(TH.shape), pts)

    @st.cache_data(ttl=3600)
    def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=3):
        theta_flat = theta_vals.ravel()
        phi_flat = phi_vals.ravel()
        g_flat = g_vals.ravel()
        valid = ~np.isnan(g_flat)
        theta_flat = theta_flat[valid]
        phi_flat = phi_flat[valid]
        g_flat = g_flat[valid]
        if len(theta_flat) == 0:
            return None, l_max
        A = []
        for t, p in zip(theta_flat, phi_flat):
            row = []
            for l in range(l_max+1):
                for m in range(-l, l+1):
                    y = get_real_sph_harm(l, m, t, p)
                    row.append(y)
            A.append(row)
        A = np.array(A)
        n_basis = (l_max + 1) ** 2
        if A.shape[0] < n_basis:
            st.warning(f"⚠️ Insufficient valid data points ({A.shape[0]}) for l_max={l_max} (needs ≥{n_basis}). Reducing l_max.")
            while l_max > 0 and A.shape[0] < (l_max + 1) ** 2:
                l_max -= 1
            if l_max < 0:
                return None, 0
            A = []
            for t, p in zip(theta_flat, phi_flat):
                row = []
                for l in range(l_max+1):
                    for m in range(-l, l+1):
                        y = get_real_sph_harm(l, m, t, p)
                        row.append(y)
                A.append(row)
            A = np.array(A)
        if A.size == 0 or A.shape[0] == 0 or A.shape[1] == 0:
            return None, l_max
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
        recon = np.zeros_like(theta_grid, dtype=float)
        idx = 0
        for l in range(l_max+1):
            for m in range(-l, l+1):
                Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
                recon += coeffs[idx] * Y
                idx += 1
        return recon

    def extract_dg_zero_contour(TH, PH, dG_grid, R_fixed):
        contours_x, contours_y, contours_z = [], [], []
        for i in range(dG_grid.shape[0]):
            for j in range(dG_grid.shape[1]-1):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i,j+1])):
                    continue
                if dG_grid[i,j] * dG_grid[i,j+1] < 0:
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i,j+1]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i,j+1] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i,j+1] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))
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
# TEMPERATURE-DRIVEN SHAPE MORPHING
# =============================================
def get_liquid_radius(G_sh, sh_R_fixed, T_factor):
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    thermal_exp = 1.0 + 0.35 * T_factor
    fluid_dist = 0.12 * np.sin(2 * np.pi * norm) * (0.5 + 0.5 * T_factor)
    return sh_R_fixed * (thermal_exp + 0.22 * norm + fluid_dist)

def get_fcc_radius(G_sh, sh_R_fixed, T_factor):
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    rigidity = 1.0 - 0.20 * T_factor
    crystal_factor = 0.28 * (1.0 - T_factor)
    crystal_ripples = crystal_factor * (
        0.6 * np.sin(6 * np.pi * norm) + 0.3 * np.sin(10 * np.pi * norm) + 0.1 * np.sin(14 * np.pi * norm))
    return sh_R_fixed * (rigidity + 0.20 * norm + crystal_ripples)

# =============================================
# ADDITIVE MANUFACTURING CPD ANALYSIS MODULE
# =============================================

# CRITICAL FIX: Helper function for denormalized reconstruction
# All CPD factors stored in session state are normalized.
# To get physical J/mol values, multiply by sigma and add mu.
def reconstruct_cpd_physical(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals,
                               co_query, cr_query, fe_query, T_query,
                               tensor_mean, tensor_std):
    """
    Reconstruct Gibbs energy in PHYSICAL units (J/mol) from normalized CPD factors.

    This is the CRITICAL FIX for the scaling discrepancy:
    - CPD is trained on normalized tensor: G_norm = (G - mu) / sigma
    - Factors A, B, C, D, lam are in normalized space
    - To get physical G: G_phys = G_norm * sigma + mu

    Args:
        A, B, C, D, lam: Normalized CPD factor matrices and weights
        co_vals, cr_vals, fe_vals, T_vals: Grid values
        co_query, cr_query, fe_query, T_query: Query point
        tensor_mean, tensor_std: Normalization parameters from training

    Returns:
        G_physical: Gibbs energy in J/mol
    """
    R = len(lam)

    # Interpolate factors to query point
    def interp_factor(vals, factor_matrix, query):
        result = np.zeros(factor_matrix.shape[1])
        for r in range(factor_matrix.shape[1]):
            result[r] = np.interp(query, vals, factor_matrix[:, r], left=np.nan, right=np.nan)
        return result

    A_q = interp_factor(co_vals, A, co_query)
    B_q = interp_factor(cr_vals, B, cr_query)
    C_q = interp_factor(fe_vals, C, fe_query)
    D_q = interp_factor(T_vals, D, T_query)

    # Compute normalized Gibbs energy
    G_norm = np.sum(lam * A_q * B_q * C_q * D_q)

    # DENORMALIZE to physical units
    G_physical = G_norm * tensor_std + tensor_mean

    return G_physical


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_extract_transition(A_liq_tuple, A_fcc_tuple, B_liq_tuple, B_fcc_tuple,
                                C_liq_tuple, C_fcc_tuple, D_liq_tuple, D_fcc_tuple,
                                lam_liq_tuple, lam_fcc_tuple,
                                co_vals_tuple, cr_vals_tuple, fe_vals_tuple, T_vals_tuple,
                                mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                                composition_grid_resolution=25):
    """Cached wrapper for transition surface extraction WITH denormalization."""
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
                                     mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                                     composition_grid_resolution)

def _extract_transition_impl(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                             D_liq, D_fcc, lam_liq, lam_fcc,
                             co_vals, cr_vals, fe_vals, T_vals,
                             mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                             composition_grid_resolution=25):
    """
    Extract T*(x_Co, x_Cr, x_Fe) surface where G_LIQ = G_FCC using CPD factors.
    CRITICAL FIX: Now includes denormalization parameters for physical units.
    """
    from scipy.optimize import brentq

    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)
    R = min(R_liq, R_fcc)

    if D_liq.shape[1] < R_liq or D_fcc.shape[1] < R_fcc:
        R = min(R, D_liq.shape[1], D_fcc.shape[1])

    if R < 1:
        st.error("❌ Cannot extract transition surface: CPD rank < 1")
        return None, None, None

    if R < R_liq or R < R_fcc:
        st.info(f"ℹ️ Using truncated rank R={R} (LIQUID had {R_liq}, FCC had {R_fcc})")

    n_T = len(T_vals)

    # Pre-compute temperature-dependent difference in NORMALIZED space
    D_diff = np.zeros((n_T, R))
    for r in range(R):
        D_diff[:, r] = lam_liq[r] * D_liq[:, r] - lam_fcc[r] * D_fcc[:, r]

    x = np.linspace(0, 1, composition_grid_resolution)
    Co_grid, Cr_grid, Fe_grid = np.meshgrid(x, x, x, indexing='ij')
    Ni_grid = 1.0 - Co_grid - Cr_grid - Fe_grid
    valid_simplex = Ni_grid >= 0

    T_melt = np.full_like(Co_grid, np.nan, dtype=np.float64)
    delta_G_grid = np.full((*Co_grid.shape, n_T), np.nan, dtype=np.float64)

    def interp_factor(vals, factor_matrix, query):
        result = np.zeros(factor_matrix.shape[1])
        for r in range(factor_matrix.shape[1]):
            result[r] = np.interp(query, vals, factor_matrix[:, r], left=np.nan, right=np.nan)
        return result

    for i in range(composition_grid_resolution):
        for j in range(composition_grid_resolution):
            for k in range(composition_grid_resolution):
                if not valid_simplex[i, j, k]:
                    continue

                co, cr, fe = Co_grid[i,j,k], Cr_grid[i,j,k], Fe_grid[i,j,k]

                A_liq_q = interp_factor(co_vals, A_liq, co)
                B_liq_q = interp_factor(cr_vals, B_liq, cr)
                C_liq_q = interp_factor(fe_vals, C_liq, fe)
                A_fcc_q = interp_factor(co_vals, A_fcc, co)
                B_fcc_q = interp_factor(cr_vals, B_fcc, cr)
                C_fcc_q = interp_factor(fe_vals, C_fcc, fe)

                if np.any(np.isnan(A_liq_q)) or np.any(np.isnan(B_liq_q)) or np.any(np.isnan(C_liq_q)):
                    continue
                if np.any(np.isnan(A_fcc_q)) or np.any(np.isnan(B_fcc_q)) or np.any(np.isnan(C_fcc_q)):
                    continue

                comp_coeff_liq = lam_liq[:R] * A_liq_q[:R] * B_liq_q[:R] * C_liq_q[:R]
                comp_coeff_fcc = lam_fcc[:R] * A_fcc_q[:R] * B_fcc_q[:R] * C_fcc_q[:R]

                def delta_G(T_query):
                    D_q = np.zeros(R)
                    for r in range(R):
                        D_q[r] = np.interp(T_query, T_vals, D_diff[:, r])
                    # DENORMALIZE: G_phys = G_norm * sigma + mu
                    G_liq_norm = np.sum(comp_coeff_liq * D_q)
                    G_fcc_norm = np.sum(comp_coeff_fcc * D_q)
                    G_liq_phys = G_liq_norm * sigma_liq + mu_liq
                    G_fcc_phys = G_fcc_norm * sigma_fcc + mu_fcc
                    return float(G_liq_phys - G_fcc_phys)

                try:
                    g_low = delta_G(float(T_vals[0]))
                    g_high = delta_G(float(T_vals[-1]))
                except:
                    continue

                if np.isnan(g_low) or np.isnan(g_high):
                    continue
                if np.sign(g_low) == np.sign(g_high):
                    continue

                try:
                    T_star = brentq(delta_G, float(T_vals[0]), float(T_vals[-1]), xtol=1.0)
                    T_melt[i, j, k] = T_star
                except (ValueError, RuntimeError):
                    continue

                for t_idx, T_val in enumerate(T_vals):
                    delta_G_grid[i, j, k, t_idx] = delta_G(float(T_val))

    return T_melt, valid_simplex, delta_G_grid


def extract_transition_surface_from_cpd(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                         D_liq, D_fcc, lam_liq, lam_fcc,
                                         co_vals, cr_vals, fe_vals, T_vals,
                                         mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                                         composition_grid_resolution=25):
    """Public interface with caching - NOW INCLUDES DENORMALIZATION PARAMS."""
    return _cached_extract_transition(
        tuple(A_liq.ravel()), tuple(A_fcc.ravel()),
        tuple(B_liq.ravel()), tuple(B_fcc.ravel()),
        tuple(C_liq.ravel()), tuple(C_fcc.ravel()),
        tuple(D_liq.ravel()), tuple(D_fcc.ravel()),
        tuple(lam_liq), tuple(lam_fcc),
        tuple(co_vals), tuple(cr_vals), tuple(fe_vals), tuple(T_vals),
        float(mu_liq), float(sigma_liq), float(mu_fcc), float(sigma_fcc),
        composition_grid_resolution
    )


def compute_composition_sensitivity(A, B, C, lam, co_vals, cr_vals, fe_vals, R=6):
    sens_Co = np.zeros(len(co_vals))
    sens_Cr = np.zeros(len(cr_vals))
    sens_Fe = np.zeros(len(fe_vals))
    for r in range(min(R, len(lam))):
        sens_Co += np.abs(lam[r] * A[:, r])
        sens_Cr += np.abs(lam[r] * B[:, r])
        sens_Fe += np.abs(lam[r] * C[:, r])
    for sens in [sens_Co, sens_Cr, sens_Fe]:
        s_min, s_max = np.min(sens), np.max(sens)
        if s_max > s_min:
            sens[:] = (sens - s_min) / (s_max - s_min)
    return sens_Co, sens_Cr, sens_Fe


def compute_hot_cracking_susceptibility(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                       D_liq, D_fcc, lam_liq, lam_fcc,
                                       co_vals, cr_vals, fe_vals, T_vals,
                                       mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                                       composition_grid_resolution=20):
    T_melt, valid_mask, delta_G_grid = extract_transition_surface_from_cpd(
        A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
        D_liq, D_fcc, lam_liq, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals,
        mu_liq, sigma_liq, mu_fcc, sigma_fcc,
        composition_grid_resolution=composition_grid_resolution
    )

    S_crack = np.full_like(T_melt, np.nan)
    res = composition_grid_resolution
    dx = 1.0 / (res - 1) if res > 1 else 0.01

    for i in range(1, res - 1):
        for j in range(1, res - 1):
            for k in range(1, res - 1):
                if not valid_mask[i,j,k] or np.isnan(T_melt[i,j,k]):
                    continue
                dTdx = (T_melt[i+1,j,k] - T_melt[i-1,j,k]) / (2 * dx)
                dTdy = (T_melt[i,j+1,k] - T_melt[i,j-1,k]) / (2 * dx)
                dTdz = (T_melt[i,j,k+1] - T_melt[i,j,k-1]) / (2 * dx)
                grad_mag = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2)
                T_star = T_melt[i,j,k]
                t_idx = np.argmin(np.abs(np.array(T_vals) - T_star))
                if t_idx > 0 and t_idx < len(T_vals) - 1:
                    dGdT = abs((delta_G_grid[i,j,k,t_idx+1] - delta_G_grid[i,j,k,t_idx-1]) / 
                              (T_vals[t_idx+1] - T_vals[t_idx-1]))
                else:
                    dGdT = abs(np.gradient(delta_G_grid[i,j,k,:], T_vals)[t_idx])
                if dGdT > 1e-6 and np.isfinite(dGdT):
                    S_crack[i,j,k] = grad_mag / dGdT
                else:
                    S_crack[i,j,k] = 0.0
    return S_crack, T_melt, valid_mask


def compute_segregation_potential(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                  lam_liq, lam_fcc, R=6):
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
    for seg in [seg_CoCr, seg_CoFe, seg_CrFe]:
        s_min, s_max = np.min(seg), np.max(seg)
        if s_max > s_min:
            seg[:] = (seg - s_min) / (s_max - s_min)
    return seg_CoCr, seg_CoFe, seg_CrFe

# =============================================
# QUADRATIC EXPANSION (PHASE-FIELD) FUNCTIONS
# =============================================

def compute_quadratic_coefficients_from_cpd(
    A, B, C, D, lam, 
    co_vals, cr_vals, fe_vals, T_vals,
    c_eq_co, c_eq_cr, c_eq_fe, T_m,
    tensor_mean, tensor_std
):
    """
    Compute quadratic expansion coefficients from CPD factor matrices.
    CRITICAL FIX: Now includes tensor_mean and tensor_std for denormalization.

    Returns coefficients in J/mol (same units as input Gibbs energy) for direct
    comparison with CPD values. The quadratic form is:

        G ≈ G_eq + A_Co*(c_Co - c_eq_Co)^2 + A_Cr*(c_Cr - c_eq_Cr)^2 
               + A_Fe*(c_Fe - c_eq_Fe)^2 + A_T*(T - T_m)^2

    where A_α = ½ * ∂²G/∂c_α² and A_T = ½ * ∂²G/∂T² evaluated at equilibrium.
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

    # Compute A_Co: ½ * second derivative w.r.t. Co (J/mol)
    A_Co_sum = 0.0
    for r in range(R):
        A_pp = second_derivative(A_funcs[r], c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Co_sum += lam[r] * A_pp * B_val * C_val * D_val
    # DENORMALIZE: multiply by sigma
    A_Co = 0.5 * A_Co_sum * tensor_std

    # Compute A_Cr: ½ * second derivative w.r.t. Cr (J/mol)
    A_Cr_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_pp = second_derivative(B_funcs[r], c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Cr_sum += lam[r] * A_val * B_pp * C_val * D_val
    A_Cr = 0.5 * A_Cr_sum * tensor_std

    # Compute A_Fe: ½ * second derivative w.r.t. Fe (J/mol)
    A_Fe_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_pp = second_derivative(C_funcs[r], c_eq_fe)
        D_val = D_funcs[r](T_m)
        A_Fe_sum += lam[r] * A_val * B_val * C_pp * D_val
    A_Fe = 0.5 * A_Fe_sum * tensor_std

    # Compute A_T: ½ * second derivative w.r.t. Temperature (J/(mol·K²))
    A_T_sum = 0.0
    for r in range(R):
        A_val = A_funcs[r](c_eq_co)
        B_val = B_funcs[r](c_eq_cr)
        C_val = C_funcs[r](c_eq_fe)
        D_pp = second_derivative(D_funcs[r], T_m)
        A_T_sum += lam[r] * A_val * B_val * C_val * D_pp
    A_T = 0.5 * A_T_sum * tensor_std

    # Compute G at equilibrium (constant term, J/mol)
    G_eq_norm = 0.0
    for r in range(R):
        G_eq_norm += lam[r] * A_funcs[r](c_eq_co) * B_funcs[r](c_eq_cr) * C_funcs[r](c_eq_fe) * D_funcs[r](T_m)
    # DENORMALIZE
    G_eq = G_eq_norm * tensor_std + tensor_mean

    return {
        'A_Co': A_Co,
        'A_Cr': A_Cr,
        'A_Fe': A_Fe,
        'A_T': A_T,
        'G_eq': G_eq,
        'c_eq': [c_eq_co, c_eq_cr, c_eq_fe],
        'T_m': T_m,
        'tensor_mean': tensor_mean,
        'tensor_std': tensor_std
    }


def verify_quadratic_approximation(coeffs, A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals,
                                   test_compositions, test_temperatures):
    """
    Verify quadratic approximation against full CPD reconstruction.
    CRITICAL FIX: Now properly denormalizes both full CPD and quadratic results.
    """
    R = len(lam)
    tensor_mean = coeffs['tensor_mean']
    tensor_std = coeffs['tensor_std']

    A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
    B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
    C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
    D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]

    results = []
    for c_co, c_cr, c_fe, T in zip(test_compositions[:, 0], test_compositions[:, 1], test_compositions[:, 2], test_temperatures):
        # Full CPD evaluation (normalized) then denormalize
        G_full_norm = sum(lam[r] * A_funcs[r](c_co) * B_funcs[r](c_cr) * C_funcs[r](c_fe) * D_funcs[r](T) for r in range(R))
        G_full = G_full_norm * tensor_std + tensor_mean

        # Quadratic approximation (already in physical units)
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


def plot_cpd_vs_quadratic_comparison(coeffs, A, B, C, D, lam, 
                                      co_vals, cr_vals, fe_vals, T_vals,
                                      c_eq, T_m):
    """
    Create comprehensive comparison between Full CPD and Quadratic Approximation.
    CRITICAL FIX: Properly denormalizes CPD results for physical units.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '1D Slice: Full CPD vs Quadratic (varying Co)',
            'Error Distribution (Full - Quad)',
            '2D Composition Slice at T_m',
            'Temperature Dependence at Fixed Composition'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    R = len(lam)
    tensor_mean = coeffs['tensor_mean']
    tensor_std = coeffs['tensor_std']

    # 1. 1D slice comparison
    co_slice = np.linspace(max(0, c_eq[0]-0.1), min(1, c_eq[0]+0.1), 50)
    G_full_slice, G_quad_slice = [], []

    A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
    B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
    C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
    D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]

    for c_co in co_slice:
        g_full_norm = sum(lam[r] * A_funcs[r](c_co) * B_funcs[r](c_eq[1]) * 
                     C_funcs[r](c_eq[2]) * D_funcs[r](T_m) for r in range(R))
        g_full = g_full_norm * tensor_std + tensor_mean
        G_full_slice.append(g_full)

        g_quad = coeffs['G_eq'] + coeffs['A_Co']*(c_co - c_eq[0])**2
        G_quad_slice.append(g_quad)

    fig.add_trace(go.Scatter(x=co_slice, y=G_full_slice, mode='lines', 
                            name='Full CPD', line=dict(color='blue', width=3)),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=co_slice, y=G_quad_slice, mode='lines', 
                            name='Quadratic', line=dict(color='red', width=2, dash='dash')),
                 row=1, col=1)
    fig.add_vline(x=c_eq[0], line_dash="dot", line_color="green", 
                 annotation_text="Equilibrium", row=1, col=1)

    # 2. Error distribution
    error = np.array(G_full_slice) - np.array(G_quad_slice)
    fig.add_trace(go.Scatter(x=co_slice, y=error, mode='lines', 
                            name='Error', line=dict(color='purple', width=2),
                            fill='tozeroy'),
                 row=1, col=2)

    # 3. 2D composition slice
    co_2d = np.linspace(max(0, c_eq[0]-0.15), min(1, c_eq[0]+0.15), 30)
    cr_2d = np.linspace(max(0, c_eq[1]-0.15), min(1, c_eq[1]+0.15), 30)
    CO_2D, CR_2D = np.meshgrid(co_2d, cr_2d)

    G_full_2d = np.zeros_like(CO_2D)
    G_quad_2d = np.zeros_like(CO_2D)

    for i in range(len(co_2d)):
        for j in range(len(cr_2d)):
            g_full_norm = sum(lam[r] * A_funcs[r](co_2d[i]) * B_funcs[r](cr_2d[j]) * 
                        C_funcs[r](c_eq[2]) * D_funcs[r](T_m) for r in range(R))
            G_full_2d[j, i] = g_full_norm * tensor_std + tensor_mean

            g_quad = (coeffs['G_eq'] + 
                     coeffs['A_Co']*(co_2d[i] - c_eq[0])**2 +
                     coeffs['A_Cr']*(cr_2d[j] - c_eq[1])**2)
            G_quad_2d[j, i] = g_quad

    error_2d = G_full_2d - G_quad_2d
    fig.add_trace(go.Heatmap(x=co_2d, y=cr_2d, z=error_2d,
                            colorscale='RdBu', zmid=0,
                            colorbar=dict(title="Error<br>(J/mol)", len=0.4)),
                 row=2, col=1)

    # 4. Temperature dependence
    T_slice = np.linspace(max(700, T_m-300), min(3300, T_m+300), 50)
    G_full_T, G_quad_T = [], []

    for T in T_slice:
        g_full_norm = sum(lam[r] * A_funcs[r](c_eq[0]) * B_funcs[r](c_eq[1]) * 
                     C_funcs[r](c_eq[2]) * D_funcs[r](T) for r in range(R))
        G_full_T.append(g_full_norm * tensor_std + tensor_mean)

        g_quad = coeffs['G_eq'] + coeffs['A_T']*(T - T_m)**2
        G_quad_T.append(g_quad)

    fig.add_trace(go.Scatter(x=T_slice, y=G_full_T, mode='lines', 
                            name='Full CPD (T)', line=dict(color='blue', width=3)),
                 row=2, col=2)
    fig.add_trace(go.Scatter(x=T_slice, y=G_quad_T, mode='lines', 
                            name='Quadratic (T)', line=dict(color='red', width=2, dash='dash')),
                 row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Full CPD vs Quadratic Approximation: Comprehensive Comparison",
        showlegend=True
    )

    fig.update_xaxes(title_text="x_Co", row=1, col=1)
    fig.update_yaxes(title_text="Gibbs Energy (J/mol)", row=1, col=1)
    fig.update_xaxes(title_text="x_Co", row=1, col=2)
    fig.update_yaxes(title_text="Error (J/mol)", row=1, col=2)
    fig.update_xaxes(title_text="x_Co", row=2, col=1)
    fig.update_yaxes(title_text="x_Cr", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
    fig.update_yaxes(title_text="Gibbs Energy (J/mol)", row=2, col=2)

    return fig


def plot_3d_comparison_surface(coeffs, A, B, C, D, lam,
                                co_vals, cr_vals, fe_vals, T_vals,
                                c_eq, T_m, sh_R_fixed=0.5):
    """
    3D spherical harmonic comparison of Full CPD vs Quadratic.
    CRITICAL FIX: Properly denormalizes CPD results.
    """
    n_theta, n_phi = 60, 60
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)

    x = sh_R_fixed * np.sin(PH) * np.cos(TH)
    y = sh_R_fixed * np.sin(PH) * np.sin(TH)
    z = sh_R_fixed * np.cos(PH)

    valid = (x + y + z) <= 1.0
    valid = valid & (x >= 0) & (y >= 0) & (z >= 0)

    R = len(lam)
    tensor_mean = coeffs['tensor_mean']
    tensor_std = coeffs['tensor_std']

    A_funcs = [UnivariateSpline(co_vals, A[:, r], s=0, ext=3) for r in range(R)]
    B_funcs = [UnivariateSpline(cr_vals, B[:, r], s=0, ext=3) for r in range(R)]
    C_funcs = [UnivariateSpline(fe_vals, C[:, r], s=0, ext=3) for r in range(R)]
    D_funcs = [UnivariateSpline(T_vals, D[:, r], s=0, ext=3) for r in range(R)]

    G_full = np.full_like(x, np.nan)
    G_quad = np.full_like(x, np.nan)

    for i in range(n_phi):
        for j in range(n_theta):
            if valid[i, j]:
                g_full_norm = sum(lam[r] * A_funcs[r](x[i,j]) * B_funcs[r](y[i,j]) * 
                            C_funcs[r](z[i,j]) * D_funcs[r](T_m) for r in range(R))
                G_full[i, j] = g_full_norm * tensor_std + tensor_mean

                g_quad = (coeffs['G_eq'] + 
                         coeffs['A_Co']*(x[i,j] - c_eq[0])**2 +
                         coeffs['A_Cr']*(y[i,j] - c_eq[1])**2 +
                         coeffs['A_Fe']*(z[i,j] - c_eq[2])**2)
                G_quad[i, j] = g_quad

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Full CPD Gibbs Energy', 'Quadratic Approximation')
    )

    fig.add_trace(go.Surface(
        x=x, y=y, z=z, surfacecolor=G_full,
        colorscale='Viridis',
        name='Full CPD',
        showscale=True,
        colorbar=dict(title="G (J/mol)", len=0.8, x=0.0)
    ), row=1, col=1)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z, surfacecolor=G_quad,
        colorscale='Viridis',
        name='Quadratic',
        showscale=True,
        colorbar=dict(title="G (J/mol)", len=0.8, x=1.0)
    ), row=1, col=2)

    fig.update_layout(
        title=f"3D Comparison: Full CPD vs Quadratic at T={T_m}K",
        height=600,
        scene=dict(
            xaxis=dict(title="x_Co", range=[0, sh_R_fixed]),
            yaxis=dict(title="x_Cr", range=[0, sh_R_fixed]),
            zaxis=dict(title="x_Fe", range=[0, sh_R_fixed]),
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis=dict(title="x_Co", range=[0, sh_R_fixed]),
            yaxis=dict(title="x_Cr", range=[0, sh_R_fixed]),
            zaxis=dict(title="x_Fe", range=[0, sh_R_fixed]),
            aspectmode='cube'
        )
    )

    return fig


def plot_error_metrics_dashboard(verify_df):
    """Create dashboard showing error metrics and statistics."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Absolute Error Distribution',
            'Relative Error vs Composition Distance',
            'Error Histogram',
            'Error Statistics by Temperature'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "box"}]]
    )

    dist_from_eq = np.sqrt(
        (verify_df['c_Co'] - verify_df['c_Co'].mean())**2 +
        (verify_df['c_Cr'] - verify_df['c_Cr'].mean())**2 +
        (verify_df['c_Fe'] - verify_df['c_Fe'].mean())**2
    )

    fig.add_trace(go.Scatter(
        x=verify_df['T'], y=verify_df['absolute_error'],
        mode='markers',
        marker=dict(color=verify_df['absolute_error'], 
                   colorscale='Reds', showscale=True),
        name='Absolute Error',
        hovertemplate="T=%{x}K<br>Error=%{marker.color:.2f} J/mol<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dist_from_eq, y=verify_df['relative_error']*100,
        mode='markers',
        marker=dict(color=verify_df['T'], 
                   colorscale='Viridis', showscale=True),
        name='Relative Error',
        hovertemplate="Distance=%{x:.3f}<br>Rel Error=%{y:.2f}%<br>T=%{marker.color:.0f}K<extra></extra>"
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=verify_df['absolute_error'],
        nbinsx=30,
        name='Error Distribution',
        marker_color='steelblue'
    ), row=2, col=1)

    T_bins = pd.cut(verify_df['T'], bins=5, labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
    fig.add_trace(go.Box(
        y=verify_df['absolute_error'],
        x=T_bins,
        name='Error by T',
        marker_color='coral'
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Quadratic Approximation Error Analysis Dashboard",
        showlegend=False
    )

    fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Error (J/mol)", row=1, col=1)
    fig.update_xaxes(title_text="Distance from Equilibrium", row=1, col=2)
    fig.update_yaxes(title_text="Relative Error (%)", row=1, col=2)
    fig.update_xaxes(title_text="Absolute Error (J/mol)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Temperature Range", row=2, col=2)
    fig.update_yaxes(title_text="Absolute Error (J/mol)", row=2, col=2)

    return fig

# =============================================
# PLOTLY VISUALIZATION FUNCTIONS FOR AM ANALYSIS
# =============================================

def plot_transition_surface_3d(T_melt, valid_mask, co_vals, cr_vals, fe_vals,
                                T_laser=2800, T_haz=1200):
    """Create 3D scatter plot of transition temperature surface T*(x)."""
    res = T_melt.shape[0]
    x = np.linspace(0, 1, res)

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

    valid_T = (T_flat > 700) & (T_flat < 3300) & np.isfinite(T_flat)

    if np.sum(valid_T) < 10:
        fig = go.Figure()
        fig.add_annotation(text="⚠️ Too few valid transition points. Try coarser resolution.",
                          xref="paper", yref="paper", showarrow=False, font_size=16)
        return fig

    fig = go.Figure()

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
    """Plot CPD temperature factors with AM thermal cycle overlay."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CPD Temperature Factors D[T,r] (LIQUID phase)', 
                       'Typical AM Thermal Cycle'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

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

    am_temps = {
        'Room T': 300,
        'Stress Relief': 800,
        'Fe Curie T': 1043,
        'HAZ Peak': 1400,
        'Solidus': 1600,
        'Melt Pool': 2800,
    }
    for label, T_val in am_temps.items():
        if T_vals[0] <= T_val <= T_vals[-1]:
            fig.add_vline(x=T_val, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text=label, annotation_position="top left",
                         row=1, col=1)

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
    """Plot composition sensitivity heatmaps for all three elements."""
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
        ("Co", co_vals, sens_Co, '#3498db'),
        ("Cr", cr_vals, sens_Cr, '#2ecc71'),
        ("Fe", fe_vals, sens_Fe, '#e74c3c')
    ]

    for idx, (elem, vals, sens, color) in enumerate(elements, 1):
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
    """Plot defect susceptibility as 3D scatter with risk highlighting."""
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
    """Plot segregation potential as 2D heatmap."""
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
# GRADIENT DESIGN FUNCTIONS (Application 6)
# =============================================

def evaluate_composition_path(path_func, s_vals, T_vals_query, 
                               A_liq, B_liq, C_liq, D_liq, lam_liq,
                               A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                               co_vals, cr_vals, fe_vals, T_vals_grid,
                               mu_liq, sigma_liq, mu_fcc, sigma_fcc):
    """
    Evaluate Gibbs energy along a parametric composition path.
    CRITICAL FIX: Includes denormalization parameters for physical units.
    """
    from scipy.interpolate import interp1d

    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)
    R = min(R_liq, R_fcc, D_liq.shape[1], D_fcc.shape[1])

    n_s = len(s_vals)
    n_T = len(T_vals_query)

    def build_factor_interp(vals, factor_matrix):
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

    D_liq_interp = [interp1d(T_vals_grid, D_liq[:, r], kind='cubic',
                              fill_value='extrapolate') for r in range(R_liq)]
    D_fcc_interp = [interp1d(T_vals_grid, D_fcc[:, r], kind='cubic',
                              fill_value='extrapolate') for r in range(R_fcc)]

    path_pts = np.array([path_func(s) for s in s_vals])

    G_liq_path = np.zeros((n_s, n_T))
    G_fcc_path = np.zeros((n_s, n_T))
    dG_path = np.zeros((n_s, n_T))
    T_star_path = np.full(n_s, np.nan)
    phase_path = np.full((n_s, n_T), '', dtype=object)

    for i, (co, cr, fe) in enumerate(path_pts):
        ni = 1.0 - co - cr - fe
        if ni < -0.01 or co < -0.01 or cr < -0.01 or fe < -0.01:
            continue

        A_liq_q = np.array([f(co) for f in A_liq_interp])
        B_liq_q = np.array([f(cr) for f in B_liq_interp])
        C_liq_q = np.array([f(fe) for f in C_liq_interp])
        A_fcc_q = np.array([f(co) for f in A_fcc_interp])
        B_fcc_q = np.array([f(cr) for f in B_fcc_interp])
        C_fcc_q = np.array([f(fe) for f in C_fcc_interp])

        for t_idx, T in enumerate(T_vals_query):
            D_liq_T = np.array([f(T) for f in D_liq_interp])
            G_liq_norm = np.sum(lam_liq * A_liq_q * B_liq_q * C_liq_q * D_liq_T)
            G_liq_path[i, t_idx] = G_liq_norm * sigma_liq + mu_liq

            D_fcc_T = np.array([f(T) for f in D_fcc_interp])
            G_fcc_norm = np.sum(lam_fcc * A_fcc_q * B_fcc_q * C_fcc_q * D_fcc_T)
            G_fcc_path[i, t_idx] = G_fcc_norm * sigma_fcc + mu_fcc

            dG_path[i, t_idx] = G_liq_path[i, t_idx] - G_fcc_path[i, t_idx]
            phase_path[i, t_idx] = 'LIQUID' if dG_path[i, t_idx] <= 0 else 'FCC'

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

    grad_T_star = np.gradient(T_star_path, s_vals) if len(s_vals) > 1 else np.zeros(n_s)

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
                            mu_liq=0, sigma_liq=1, mu_fcc=0, sigma_fcc=1,
                            T_process=2800):
    """
    Design optimal composition gradient path between two alloys.
    CRITICAL FIX: Includes denormalization parameters.
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize

    s_vals = np.linspace(0, 1, n_points)

    def path_from_params(params):
        p1 = np.array([params[0], params[1], params[2]])
        p2 = np.array([params[3], params[4], params[5]])
        for p in [p1, p2]:
            if np.sum(p) > 1.0:
                p[:] = p / np.sum(p) * 0.99
        control_pts = np.vstack([start_comp, p1, p2, end_comp])
        s_control = np.array([0, 1/3, 2/3, 1])
        cs_co = CubicSpline(s_control, control_pts[:, 0])
        cs_cr = CubicSpline(s_control, control_pts[:, 1])
        cs_fe = CubicSpline(s_control, control_pts[:, 2])
        path = np.column_stack([cs_co(s_vals), cs_cr(s_vals), cs_fe(s_vals)])
        for i in range(len(path)):
            if np.sum(path[i, :3]) > 1.0:
                path[i, :3] = path[i, :3] / np.sum(path[i, :3]) * 0.99
        return path

    def objective(params):
        path = path_from_params(params)
        path_results = evaluate_composition_path(
            lambda s: path[int(s * (n_points - 1))] if int(s * (n_points - 1)) < n_points else path[-1],
            s_vals, [T_process],
            A_liq, B_liq, C_liq, D_liq, lam_liq,
            A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
            co_vals, cr_vals, fe_vals, T_vals,
            mu_liq, sigma_liq, mu_fcc, sigma_fcc
        )

        T_star = path_results['T_star']
        tstar_variation = np.nanvar(T_star)
        curvature = 0
        for dim in range(3):
            second_deriv = np.gradient(np.gradient(path[:, dim], s_vals), s_vals)
            curvature += np.mean(second_deriv**2)
        seg_risk = np.nanmean(path_results['segregation_risk'])
        cost = (penalty_weight * tstar_variation + 
                curvature_weight * curvature + 
                0.3 * seg_risk)
        return cost

    p1_init = start_comp + (end_comp - start_comp) / 3
    p2_init = start_comp + 2 * (end_comp - start_comp) / 3
    x0 = np.concatenate([p1_init, p2_init])
    bounds = [(0, 1)] * 6

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 100, 'disp': False})

    optimal_path = path_from_params(result.x)

    final_results = evaluate_composition_path(
        lambda s: optimal_path[int(s * (n_points - 1))],
        s_vals, T_vals,
        A_liq, B_liq, C_liq, D_liq, lam_liq,
        A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals,
        mu_liq, sigma_liq, mu_fcc, sigma_fcc
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
                                mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                                T_melt_pool=2800, T_solidus=1600, T_haz=1200):
    """Check if a composition gradient is feasible for AM processing."""
    results = evaluate_composition_path(
        lambda s: path_pts[int(s * (len(path_pts) - 1))] if int(s * (len(path_pts) - 1)) < len(path_pts) else path_pts[-1],
        np.linspace(0, 1, len(path_pts)), T_vals_query,
        A_liq, B_liq, C_liq, D_liq, lam_liq,
        A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals_grid,
        mu_liq, sigma_liq, mu_fcc, sigma_fcc
    )

    issues = []
    safety_score = 1.0
    T_star = results['T_star']

    if np.any(T_star > T_melt_pool):
        issues.append(f"❌ Some compositions have T* > melt pool T ({T_melt_pool}K) — cannot fully melt")
        safety_score -= 0.3

    grad_T = np.gradient(T_star)
    max_grad = np.nanmax(np.abs(grad_T))
    if max_grad > 200:
        issues.append(f"⚠️ Steep T* gradient (max {max_grad:.1f} K/path unit) — risk of interfacial cracking")
        safety_score -= 0.2

    solidus_idx = np.argmin(np.abs(np.array(T_vals_query) - T_solidus))
    phases_solidus = results['phase'][:, solidus_idx]
    if len(set(phases_solidus)) > 1:
        issues.append("⚠️ Phase changes in mushy zone — risk of mixed microstructure")
        safety_score -= 0.2

    haz_idx = np.argmin(np.abs(np.array(T_vals_query) - T_haz))
    phases_haz = results['phase'][:, haz_idx]
    if np.any(phases_haz == 'LIQUID'):
        issues.append("⚠️ Some compositions partially melt in HAZ — risk of liquation cracking")
        safety_score -= 0.2

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
    """Create comprehensive dashboard for gradient analysis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('T* along Gradient Path', 'Phase Stability Map (s vs T)',
                       'Segregation Risk', 'Driving Force ΔG at Process T'),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Scatter(
        x=s_vals, y=path_results['T_star'],
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=6),
        name='T*(s)',
        hovertemplate="s=%{x:.3f}<br>T*=%{y:.0f} K<extra></extra>"
    ), row=1, col=1)

    fig.add_hline(y=2800, line_dash="dash", line_color="red", opacity=0.5,
                 annotation_text="Melt Pool", row=1, col=1)
    fig.add_hline(y=1200, line_dash="dash", line_color="orange", opacity=0.5,
                 annotation_text="HAZ", row=1, col=1)

    phase_numeric = np.where(path_results['phase'] == 'LIQUID', 1, 0)
    fig.add_trace(go.Heatmap(
        z=phase_numeric,
        x=T_vals_query,
        y=s_vals,
        colorscale=[[0, '#2980b9'], [1, '#e74c3c']],
        showscale=True,
        colorbar=dict(title="Phase", tickvals=[0, 1], ticktext=['FCC', 'LIQUID'],
                     len=0.4, y=0.8),
        hovertemplate="T=%{x:.0f}K<br>s=%{y:.3f}<br>Phase=%{z}<extra></extra>"
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=s_vals, y=path_results['segregation_risk'],
        mode='lines',
        line=dict(color='darkred', width=3),
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.2)',
        name='Segregation Risk',
        hovertemplate="s=%{x:.3f}<br>Risk=%{y:.3f}<extra></extra>"
    ), row=2, col=1)

    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                 annotation_text="High Risk Threshold", row=2, col=1)

    if len(T_vals_query) > 0:
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

# =============================================
# STREAMLIT UI RENDERING FUNCTIONS
# =============================================

def render_factor_matrix_visualisation(A_liq, B_liq, C_liq, D_liq, lam_liq,
                                       A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                                       co_vals, cr_vals, fe_vals, T_vals):
    """Render Streamlit UI for Factor Matrix Visualisation."""
    co_vals = np.asarray(co_vals, dtype=float)
    cr_vals = np.asarray(cr_vals, dtype=float)
    fe_vals = np.asarray(fe_vals, dtype=float)
    T_vals = np.asarray(T_vals, dtype=float)

    st.header("🔢 Factor Matrix Visualisation for AM Process Design")
    st.markdown(r"""
    The CPD factorises Gibbs energy: $G \approx \sum_{r=1}^{R} \lambda_r \, A_r(x_{Co}) \, B_r(x_{Cr}) \, C_r(x_{Fe}) \, D_r(T)$

    **Temperature** from filenames (`Gibbs_700K.csv`) is encoded in the **D matrix**.
    """)

    st.subheader("📊 Unified Factor Matrix View (All Four Matrices)")
    st.caption("A (Co), B (Cr), C (Fe) as profiles | D (Temperature) as heatmap")

    phase_choice = st.radio("Select phase", ["LIQUID", "FCC"], index=0, horizontal=True, key="unified_phase")

    fig_unified = plot_unified_factor_matrices(
        A_liq, B_liq, C_liq, D_liq, lam_liq,
        A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
        co_vals, cr_vals, fe_vals, T_vals,
        phase=phase_choice, R=min(len(lam_liq), len(lam_fcc))
    )
    st.plotly_chart(fig_unified, use_container_width=True, key="plotly_unified_main")

    st.info("""
    **Reading the unified figure:**
    - **Top-left (A)**: Co composition dependence. Each curve = component r, weighted by λ.
    - **Top-right (B)**: Cr composition dependence. Same color scheme.
    - **Bottom-left (C)**: Fe composition dependence. Same color scheme.
    - **Bottom-right (D)**: Temperature heatmap. Rows = temperatures from filenames, columns = components.
      • r=1: ~constant (enthalpy baseline) | r=2: linear in T (entropy −S·T) | r=3: curvature (Cp + magnetic)
    """)

    st.divider()

    with st.expander("🔍 Side-by-Side LIQUID vs FCC Comparison", expanded=False):
        R = min(len(lam_liq), len(lam_fcc))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LIQUID**")
            fig_liq = plot_unified_factor_matrices(
                A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals, phase='LIQUID', R=R
            )
            st.plotly_chart(fig_liq, use_container_width=True, key="plotly_liq_compare")
        with col2:
            st.markdown("**FCC**")
            fig_fcc = plot_unified_factor_matrices(
                A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals, phase='FCC', R=R
            )
            st.plotly_chart(fig_fcc, use_container_width=True, key="plotly_fcc_compare")

    st.subheader("🔥 Temperature Factors + AM Thermal Cycle")
    fig_temp = plot_temperature_factors_am(D_liq, D_fcc, T_vals, lam_liq, lam_fcc, R=min(len(lam_liq), len(lam_fcc)))
    st.plotly_chart(fig_temp, use_container_width=True, key="plotly_temp_am_cycle")

    st.subheader("🗺️ Single-Component Spatial Heatmap (2D Slice)")
    phase_heat = st.radio("Phase", ["LIQUID", "FCC"], index=0, horizontal=True, key="heat_phase")
    if phase_heat == "LIQUID":
        A, B, C, D, lam = A_liq, B_liq, C_liq, D_liq, lam_liq
    else:
        A, B, C, D, lam = A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc

    R = len(lam)
    col1, col2, col3 = st.columns(3)
    with col1:
        r_select = st.selectbox("Component r", list(range(1, R+1)), index=min(2, R-1), key="comp_r")
    with col2:
        fe_step = float(fe_vals[1]-fe_vals[0]) if len(fe_vals)>1 else 0.01
        fixed_Fe = st.slider("Fixed Fe", float(fe_vals.min()), float(fe_vals.max()), 
                            float(np.median(fe_vals)), fe_step, key="fixed_fe")
    with col3:
        T_step = float(T_vals[1]-T_vals[0]) if len(T_vals)>1 else 100.0
        fixed_T = st.slider("Fixed T (K)", float(T_vals.min()), float(T_vals.max()),
                           float(T_vals[len(T_vals)//2]), T_step, key="fixed_T_heat")

    fig_heat = plot_component_heatmap(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals,
                                      r_select-1, fixed_Fe, fixed_T)
    st.plotly_chart(fig_heat, use_container_width=True, key="plotly_single_component")

    st.subheader("✅ Reconstruction Quality Check")
    if st.button("Evaluate reconstruction error (LIQUID only)", key="eval_recon"):
        interp_liq_T, _ = build_interpolators_for_T(df, fixed_T)
        if interp_liq_T is not None:
            fig_err = plot_reconstruction_surface(interp_liq_T, A_liq, B_liq, C_liq, D_liq, lam_liq,
                                                  co_vals, cr_vals, fe_vals, T_vals, fixed_Fe, fixed_T)
            st.plotly_chart(fig_err, use_container_width=True, key="plotly_recon_error")
        else:
            st.warning(f"No interpolator for T={fixed_T}K.")


def render_am_transition_surface_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                      D_liq, D_fcc, lam_liq, lam_fcc,
                                      co_vals, cr_vals, fe_vals, T_vals,
                                      mu_liq, sigma_liq, mu_fcc, sigma_fcc):
    """Render Streamlit UI for transition temperature surface analysis."""
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
                    mu_liq, sigma_liq, mu_fcc, sigma_fcc,
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

            T_valid = T_melt[valid_mask & ~np.isnan(T_melt)]

            st.success(f"✅ Computed {valid_count:,} valid T* points")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean T*", f"{np.mean(T_valid):.0f} K")
            c2.metric("Std T*", f"{np.std(T_valid):.0f} K")
            c3.metric("Min T*", f"{np.min(T_valid):.0f} K")
            c4.metric("Max T*", f"{np.max(T_valid):.0f} K")

            fig = plot_transition_surface_3d(T_melt, valid_mask, co_vals, cr_vals, fe_vals,
                                              T_laser=T_laser, T_haz=T_haz)
            st.plotly_chart(fig, use_container_width=True, key="plotly_007")

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
        st.plotly_chart(fig, use_container_width=True, key="plotly_008")
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

        st.plotly_chart(fig, use_container_width=True, key="plotly_009")

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
    st.plotly_chart(fig, use_container_width=True, key="plotly_010")

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
            peak_idx = np.argmax(sens)
            peak_val = vals[peak_idx]
            st.metric("Peak at", f"x_{elem} = {peak_val:.2f}")
            st.markdown(f"<span style='color:{color}'>{advice}</span>", unsafe_allow_html=True)


def render_am_defect_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                         D_liq, D_fcc, lam_liq, lam_fcc,
                         co_vals, cr_vals, fe_vals, T_vals,
                         mu_liq, sigma_liq, mu_fcc, sigma_fcc):
    """Render Streamlit UI for defect susceptibility analysis."""
    st.subheader("⚠️ Defect Susceptibility Analysis")
    st.markdown(r"""
    **Theory**: Hot cracking susceptibility combines two CPD-derived metrics:

    $$S_{crack}[x] = |\nabla_x T^*(x)| \times |d(\Delta G)/dT|^{-1}_{T=T^*}$$

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
                    mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                    composition_grid_resolution=resolution
                )

                fig = plot_defect_susceptibility_3d(S_defect, valid_mask, co_vals, cr_vals, fe_vals,
                                                      defect_type=defect_type)
                st.plotly_chart(fig, use_container_width=True, key="plotly_011")

                S_valid = S_defect[valid_mask & np.isfinite(S_defect)]
                if len(S_valid) > 0:
                    st.markdown(f"""
                    **Hot Cracking Susceptibility Summary:**
                    - Valid points: {len(S_valid):,}
                    - Mean susceptibility: {np.mean(S_valid):.3f}
                    - High-risk threshold (90th percentile): {np.percentile(S_valid, 90):.3f}
                    - High-risk compositions: {np.sum(S_valid > np.percentile(S_valid, 90))} ({100*np.sum(S_valid > np.percentile(S_valid, 90))/len(S_valid):.1f}%)
                    """)

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
                    st.plotly_chart(fig1, use_container_width=True, key="plotly_012")
                with col2:
                    fig2 = plot_segregation_heatmap(seg_CrFe, cr_vals, fe_vals,
                                                    "x_Cr", "x_Fe", "Cr-Fe Segregation")
                    st.plotly_chart(fig2, use_container_width=True, key="plotly_013")

                st.info("""
                **Segregation Analysis:**
                - Red regions = strong binary interaction = tendency for element partitioning
                - During rapid solidification, high-segregation compositions form inhomogeneous microstructures
                - Recommendation: Avoid peak segregation regions or use ultra-fast cooling (>10⁶ K/s)
                """)


def render_gradient_design_tab(A_liq, A_fcc, B_liq, B_fcc, C_liq, C_fcc,
                                  D_liq, D_fcc, lam_liq, lam_fcc,
                                  co_vals, cr_vals, fe_vals, T_vals,
                                  mu_liq, sigma_liq, mu_fcc, sigma_fcc):
    """Render Streamlit UI for multi-material gradient design."""

    st.subheader("🔗 Multi-Material / Graded Structures Design")
    st.markdown(r"""
    **Physical Problem**: When joining two dissimilar Co-Cr-Fe-Ni alloys via AM, 
    a sharp interface causes thermal mismatch cracking. A smooth composition 
    gradient prevents this — but only if designed correctly.

    **CPD Advantage**: The separable tensor structure allows instant evaluation 
    of $G(x(s), T)$ for any parametric path $x(s)$, enabling optimization of:
    - **T* smoothness**: Minimize $|\nabla_s T^*|$ to prevent interfacial cracking
    - **Phase consistency**: Ensure stable FCC throughout the HAZ
    - **Segregation control**: Avoid high binary-interaction regions

    **Theory**: For a path $x(s) = (x_{Co}(s), x_{Cr}(s), x_{Fe}(s))$:
    $$T^*(s) = \text{root of } \Delta G(x(s), T) = 0$$
    $$\text{Cracking Risk} \propto \left|\frac{dT^*}{ds}\right| \times S_{seg}(x(s))$$
    """)

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

    valid_a = a_ni >= 0 and a_co >= 0 and a_cr >= 0 and a_fe >= 0
    valid_b = b_ni >= 0 and b_co >= 0 and b_cr >= 0 and b_fe >= 0

    if not (valid_a and valid_b):
        st.error("❌ Invalid composition: all mole fractions must be ≥ 0 and sum to ≤ 1")
        return

    start_comp = np.array([a_co, a_cr, a_fe])
    end_comp = np.array([b_co, b_cr, b_fe])

    st.subheader("⚙️ Gradient Optimization")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        n_points = st.slider("Path Points", 20, 100, 50)
    with col_opt2:
        penalty_weight = st.slider("T* Smoothness Weight", 0.1, 5.0, 1.0, 0.1)
    with col_opt3:
        curvature_weight = st.slider("Path Curvature Weight", 0.1, 2.0, 0.5, 0.1)

    T_process = st.slider("Process Temperature (K)", 1500, 3500, 2800, 50)

    if st.button("🔬 Design Optimal Gradient", use_container_width=True, type="primary"):
        with st.spinner("Optimizing composition gradient path..."):

            result = design_optimal_gradient(
                start_comp, end_comp, n_points=n_points,
                penalty_weight=penalty_weight, curvature_weight=curvature_weight,
                A_liq=A_liq, B_liq=B_liq, C_liq=C_liq, D_liq=D_liq, lam_liq=lam_liq,
                A_fcc=A_fcc, B_fcc=B_fcc, C_fcc=C_fcc, D_fcc=D_fcc, lam_fcc=lam_fcc,
                co_vals=co_vals, cr_vals=cr_vals, fe_vals=fe_vals, T_vals=T_vals,
                mu_liq=mu_liq, sigma_liq=sigma_liq, mu_fcc=mu_fcc, sigma_fcc=sigma_fcc,
                T_process=T_process
            )

            st.success("✅ Optimal gradient designed!")

            metrics = result['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T* Range", f"{metrics['T_star_range']:.0f} K", 
                     delta="Lower is better" if metrics['T_star_range'] < 200 else "⚠️ High")
            c2.metric("T* Std Dev", f"{metrics['T_star_std']:.1f} K")
            c3.metric("Max Segregation", f"{metrics['max_segregation_risk']:.3f}")
            c4.metric("Path Efficiency", f"{metrics['path_length']/metrics['linear_path_length']:.2f}×",
                     help="Ratio of actual path length to straight-line distance")

            feasibility = check_gradient_feasibility(
                result['path'], T_vals, A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals,
                mu_liq, sigma_liq, mu_fcc, sigma_fcc,
                T_melt_pool=T_process, T_solidus=1600, T_haz=1200
            )

            st.subheader("🛡️ Feasibility Assessment")

            safety = feasibility['safety_score']
            if safety >= 0.8:
                st.success(f"✅ **SAFE** for AM (Score: {safety:.2f}/1.0)")
            elif safety >= 0.5:
                st.warning(f"⚠️ **CAUTION** (Score: {safety:.2f}/1.0) — Review issues below")
            else:
                st.error(f"❌ **RISKY** (Score: {safety:.2f}/1.0) — Significant redesign needed")

            for issue in feasibility['issues']:
                st.markdown(issue)

            st.subheader("🗺️ Gradient Path in Composition Space")
            fig_3d = plot_gradient_path_3d(
                result['path'], result['T_star'], result['s_vals'],
                start_label=f"A ({a_co:.2f},{a_cr:.2f},{a_fe:.2f})",
                end_label=f"B ({b_co:.2f},{b_cr:.2f},{b_fe:.2f})"
            )
            st.plotly_chart(fig_3d, use_container_width=True, key="plotly_019")

            st.subheader("📊 Gradient Analysis Dashboard")

            path_results = evaluate_composition_path(
                lambda s: result['path'][int(s * (n_points - 1))] if int(s * (n_points - 1)) < n_points else result['path'][-1],
                result['s_vals'], T_vals,
                A_liq, B_liq, C_liq, D_liq, lam_liq,
                A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                co_vals, cr_vals, fe_vals, T_vals,
                mu_liq, sigma_liq, mu_fcc, sigma_fcc
            )

            fig_dash = plot_gradient_analysis_dashboard(path_results, T_vals, result['s_vals'])
            st.plotly_chart(fig_dash, use_container_width=True, key="plotly_020")

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
# FACTOR MATRIX VISUALISATION HELPER FUNCTIONS
# =============================================

def plot_factor_profiles(A, B, C, lam, co_vals, cr_vals, fe_vals, R=6):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=R, subplot_titles=[f'r={r+1} (λ={lam[r]:.3f})' for r in range(R)],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    colors = ['#e74c3c','#2980b9','#27ae60','#f39c12','#9b59b6','#1abc9c']
    for r in range(R):
        fig.add_trace(go.Scatter(x=co_vals, y=lam[r]*A[:,r], mode='lines+markers',
                                 marker=dict(color=colors[r%len(colors)]),
                                 line=dict(width=2, color=colors[r%len(colors)]),
                                 name=f'r={r+1} Co'), row=1, col=r+1)
        fig.add_trace(go.Scatter(x=cr_vals, y=lam[r]*B[:,r], mode='lines+markers',
                                 marker=dict(color=colors[r%len(colors)]),
                                 line=dict(width=2, color=colors[r%len(colors)]),
                                 showlegend=False), row=2, col=r+1)
        fig.add_trace(go.Scatter(x=fe_vals, y=lam[r]*C[:,r], mode='lines+markers',
                                 marker=dict(color=colors[r%len(colors)]),
                                 line=dict(width=2, color=colors[r%len(colors)]),
                                 showlegend=False), row=3, col=r+1)
    for r in range(R):
        fig.update_xaxes(title_text="x_Co", row=1, col=r+1)
        fig.update_xaxes(title_text="x_Cr", row=2, col=r+1)
        fig.update_xaxes(title_text="x_Fe", row=3, col=r+1)
        fig.update_yaxes(title_text="λ·A", row=1, col=r+1)
        fig.update_yaxes(title_text="λ·B", row=2, col=r+1)
        fig.update_yaxes(title_text="λ·C", row=3, col=r+1)
    fig.update_layout(height=800, title_text="Factor Matrix Profiles (Weighted by λ)", showlegend=False)
    return fig


def plot_component_heatmap(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, r_idx, fixed_fe, fixed_T):
    fe_idx = np.argmin(np.abs(np.asarray(fe_vals, dtype=float) - fixed_fe))
    T_idx = np.argmin(np.abs(np.asarray(T_vals, dtype=float) - fixed_T))
    Co_mesh, Cr_mesh = np.meshgrid(np.asarray(co_vals, dtype=float), np.asarray(cr_vals, dtype=float), indexing='ij')
    n_pts = Co_mesh.ravel().shape[0]
    comp_value = lam[r_idx] * A[:,r_idx][:,None] * B[:,r_idx][None,:] * C[fe_idx, r_idx] * D[T_idx, r_idx]
    fig = go.Figure(data=go.Heatmap(z=comp_value, x=np.asarray(co_vals, dtype=float), y=np.asarray(cr_vals, dtype=float), 
                                    colorscale='RdBu_r',
                                    colorbar=dict(title=f"Component r={r_idx+1}")))
    fig.update_layout(title=f"Component r={r_idx+1} (λ={lam[r_idx]:.3f}) at Fe={fixed_fe:.3f}, T={fixed_T}K",
                      xaxis_title="x_Co", yaxis_title="x_Cr", height=500)
    return fig


def plot_reconstruction_surface(interp_liq, A_liq, B_liq, C_liq, D_liq, lam_liq,
                                co_vals, cr_vals, fe_vals, T_vals, fixed_Fe, fixed_T):
    fe_idx = np.argmin(np.abs(np.asarray(fe_vals, dtype=float) - fixed_Fe))
    T_idx = np.argmin(np.abs(np.asarray(T_vals, dtype=float) - fixed_T))
    Co_mesh, Cr_mesh = np.meshgrid(np.asarray(co_vals, dtype=float), np.asarray(cr_vals, dtype=float), indexing='ij')
    n_pts = Co_mesh.ravel().shape[0]
    pts_grid = np.column_stack([Co_mesh.ravel(), Cr_mesh.ravel(), np.full(n_pts, fixed_Fe, dtype=float)])
    G_orig = interp_liq(pts_grid).reshape(Co_mesh.shape)
    R = len(lam_liq)
    G_recon = np.zeros_like(Co_mesh)
    co_arr = np.asarray(co_vals, dtype=float)
    cr_arr = np.asarray(cr_vals, dtype=float)
    for i, co in enumerate(co_arr):
        A_vals = np.array([np.interp(co, co_arr, A_liq[:,r]) for r in range(R)])
        for j, cr in enumerate(cr_arr):
            B_vals = np.array([np.interp(cr, cr_arr, B_liq[:,r]) for r in range(R)])
            C_vals = C_liq[fe_idx, :]
            D_vals = D_liq[T_idx, :]
            G_recon[i,j] = np.sum(lam_liq * A_vals * B_vals * C_vals * D_vals)
    error = np.abs(G_orig - G_recon)
    fig = go.Figure(data=go.Heatmap(z=error, x=co_arr, y=cr_arr, colorscale='Viridis',
                                    colorbar=dict(title="|Error| (J/mol)")))
    fig.update_layout(title=f"Reconstruction Error (LIQUID) at Fe={fixed_Fe:.3f}, T={fixed_T}K",
                      xaxis_title="x_Co", yaxis_title="x_Cr", height=500)
    return fig


def plot_unified_factor_matrices(A_liq, B_liq, C_liq, D_liq, lam_liq,
                                  A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc,
                                  co_vals, cr_vals, fe_vals, T_vals,
                                  phase='LIQUID', R=6):
    """
    Unified visualization of ALL FOUR CPD factor matrices (A, B, C, D) in one figure.
    Layout: 2x2 grid
      [A: Co profiles]  [B: Cr profiles]
      [C: Fe profiles]  [D: Temperature heatmap]
    """
    from plotly.subplots import make_subplots

    if phase == 'LIQUID':
        A, B, C, D, lam = A_liq, B_liq, C_liq, D_liq, lam_liq
    else:
        A, B, C, D, lam = A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc

    co_arr = np.asarray(co_vals, dtype=float)
    cr_arr = np.asarray(cr_vals, dtype=float)
    fe_arr = np.asarray(fe_vals, dtype=float)
    T_arr = np.asarray(T_vals, dtype=float)

    R_eff = min(R, len(lam), A.shape[1], B.shape[1], C.shape[1], D.shape[1])
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'A: Co factor (λ·A) — {phase}',
            f'B: Cr factor (λ·B) — {phase}',
            f'C: Fe factor (λ·C) — {phase}',
            f'D: Temperature factor (λ·D) — {phase}'
        ),
        vertical_spacing=0.12, horizontal_spacing=0.10,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )

    for r in range(R_eff):
        fig.add_trace(go.Scatter(
            x=co_arr, y=lam[r] * A[:, r], mode='lines+markers',
            name=f'r={r+1}', line=dict(color=colors[r % len(colors)], width=2),
            marker=dict(size=5, color=colors[r % len(colors)]),
            showlegend=True, legendgroup=f'r{r+1}'
        ), row=1, col=1)

    for r in range(R_eff):
        fig.add_trace(go.Scatter(
            x=cr_arr, y=lam[r] * B[:, r], mode='lines+markers',
            name=f'r={r+1}', line=dict(color=colors[r % len(colors)], width=2),
            marker=dict(size=5, color=colors[r % len(colors)]),
            showlegend=False, legendgroup=f'r{r+1}'
        ), row=1, col=2)

    for r in range(R_eff):
        fig.add_trace(go.Scatter(
            x=fe_arr, y=lam[r] * C[:, r], mode='lines+markers',
            name=f'r={r+1}', line=dict(color=colors[r % len(colors)], width=2),
            marker=dict(size=5, color=colors[r % len(colors)]),
            showlegend=False, legendgroup=f'r{r+1}'
        ), row=2, col=1)

    D_weighted = D[:, :R_eff] * lam[:R_eff][None, :]
    T_labels = [f"{int(t)}K" for t in T_arr]
    r_labels = [f"r={r+1}" for r in range(R_eff)]

    fig.add_trace(go.Heatmap(
        z=D_weighted, x=r_labels, y=T_labels, colorscale='RdBu_r', zmid=0,
        colorbar=dict(title="λ·D(T,r)", thickness=15, len=0.5, y=0.25, x=1.02),
        hovertemplate="T=%{y}<br>r=%{x}<br>λ·D=%{z:.4f}<extra></extra>"
    ), row=2, col=2)

    fig.update_xaxes(title_text="x_Co", row=1, col=1)
    fig.update_xaxes(title_text="x_Cr", row=1, col=2)
    fig.update_xaxes(title_text="x_Fe", row=2, col=1)
    fig.update_xaxes(title_text="Component r", row=2, col=2)
    fig.update_yaxes(title_text="λ·A", row=1, col=1)
    fig.update_yaxes(title_text="λ·B", row=1, col=2)
    fig.update_yaxes(title_text="λ·C", row=2, col=1)
    fig.update_yaxes(title_text="Temperature", row=2, col=2)

    fig.update_layout(
        height=900,
        title_text=f"Unified CPD Factor Matrices — {phase} Phase (R={R_eff})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title="Component"),
        template="plotly_white"
    )
    return fig


def validate_cpd_session_state():
    """
    Validate that CPD factors in session state have compatible dimensions.
    Returns (is_valid, error_message, adjusted_factors)
    """
    required_keys = ['A_liq','B_liq','C_liq','D_liq','lam_liq',
                     'A_fcc','B_fcc','C_fcc','D_fcc','lam_fcc']

    for key in required_keys:
        if key not in st.session_state:
            return False, f"Missing session state key: {key}", None

    A_liq = st.session_state['A_liq']; A_fcc = st.session_state['A_fcc']
    B_liq = st.session_state['B_liq']; B_fcc = st.session_state['B_fcc']
    C_liq = st.session_state['C_liq']; C_fcc = st.session_state['C_fcc']
    D_liq = st.session_state['D_liq']; D_fcc = st.session_state['D_fcc']
    lam_liq = st.session_state['lam_liq']; lam_fcc = st.session_state['lam_fcc']

    R_liq = len(lam_liq)
    R_fcc = len(lam_fcc)

    checks = [
        (A_liq.shape[1]==R_liq, f"A_liq cols ({A_liq.shape[1]}) != rank ({R_liq})"),
        (B_liq.shape[1]==R_liq, f"B_liq cols ({B_liq.shape[1]}) != rank ({R_liq})"),
        (C_liq.shape[1]==R_liq, f"C_liq cols ({C_liq.shape[1]}) != rank ({R_liq})"),
        (D_liq.shape[1]==R_liq, f"D_liq cols ({D_liq.shape[1]}) != rank ({R_liq})"),
        (A_fcc.shape[1]==R_fcc, f"A_fcc cols ({A_fcc.shape[1]}) != rank ({R_fcc})"),
        (B_fcc.shape[1]==R_fcc, f"B_fcc cols ({B_fcc.shape[1]}) != rank ({R_fcc})"),
        (C_fcc.shape[1]==R_fcc, f"C_fcc cols ({C_fcc.shape[1]}) != rank ({R_fcc})"),
        (D_fcc.shape[1]==R_fcc, f"D_fcc cols ({D_fcc.shape[1]}) != rank ({R_fcc})"),
    ]

    for check, msg in checks:
        if not check:
            return False, f"Dimension mismatch: {msg}", None

    if 'tdt_metadata' not in st.session_state:
        return False, "Missing tdt_metadata in session state", None

    meta = st.session_state['tdt_metadata']
    required_meta = ['co_vals','cr_vals','fe_vals','T_vals']
    for key in required_meta:
        if key not in meta:
            return False, f"Missing metadata key: {key}", None

    # CRITICAL FIX: Also check for denormalization parameters
    has_denorm = all(k in st.session_state for k in ['mu_liq', 'sigma_liq', 'mu_fcc', 'sigma_fcc'])

    result = {
        'A_liq':A_liq, 'B_liq':B_liq, 'C_liq':C_liq, 'D_liq':D_liq, 'lam_liq':lam_liq,
        'A_fcc':A_fcc, 'B_fcc':B_fcc, 'C_fcc':C_fcc, 'D_fcc':D_fcc, 'lam_fcc':lam_fcc,
        'co_vals': np.asarray(meta['co_vals'], dtype=float),
        'cr_vals': np.asarray(meta['cr_vals'], dtype=float),
        'fe_vals': np.asarray(meta['fe_vals'], dtype=float),
        'T_vals': np.asarray(meta['T_vals'], dtype=float)
    }

    # Add denormalization parameters if available
    if has_denorm:
        result['mu_liq'] = st.session_state['mu_liq']
        result['sigma_liq'] = st.session_state['sigma_liq']
        result['mu_fcc'] = st.session_state['mu_fcc']
        result['sigma_fcc'] = st.session_state['sigma_fcc']
    else:
        # Default to no denormalization (backward compat)
        result['mu_liq'] = 0.0
        result['sigma_liq'] = 1.0
        result['mu_fcc'] = 0.0
        result['sigma_fcc'] = 1.0

    return True, "Valid", result

# =============================================
# MAIN STREAMLIT APP
# =============================================

def main():
    st.title("🔷 CoCrFeNi Phase Stability Explorer v2")
    st.caption("Thermodynamic Data Tensor Analysis with CPD | Coutinho et al. 2020 Framework")

    st.sidebar.header("📁 Data & Tensor Settings")
    csv_dir = st.sidebar.text_input("CSV directory", CSV_FILES_DIR)
    if csv_dir != CSV_FILES_DIR:
        global df, T_list, T_min, T_max, T_range
        df = load_all_data(csv_dir)
        T_list = sorted(df["T"].unique())
        T_min, T_max = min(T_list), max(T_list)
        T_range = T_max - T_min if T_max > T_min else 1.0

    st.sidebar.markdown(f"**Loaded:** {len(df):,} rows | {len(T_list)} temps ({T_min}–{T_max} K)")

    # --- Tensor Analysis ---
    st.sidebar.divider()
    st.sidebar.header("🔧 Tensor Analysis")

    if "tdt_metadata" not in st.session_state:
        with st.spinner("Building 4D Thermodynamic Data Tensor..."):
            tdt_data = build_tensor_data(df)
            st.session_state['tdt_metadata'] = {
                'dims': tdt_data['dims'],
                'co_vals': tdt_data['co_vals'],
                'cr_vals': tdt_data['cr_vals'],
                'fe_vals': tdt_data['fe_vals'],
                'T_vals': tdt_data['T_vals'],
                'co_step': tdt_data['co_step'],
                'cr_step': tdt_data['cr_step'],
                'fe_step': tdt_data['fe_step'],
                'T_step': tdt_data['T_step']
            }
            st.session_state['G_LIQ_tensor'] = tdt_data['G_LIQ']
            st.session_state['G_FCC_tensor'] = tdt_data['G_FCC']

    tdt_meta = st.session_state['tdt_metadata']
    n_co, n_cr, n_fe, n_T = tdt_meta['dims']
    co_vals = np.asarray(tdt_meta['co_vals'], dtype=float)
    cr_vals = np.asarray(tdt_meta['cr_vals'], dtype=float)
    fe_vals = np.asarray(tdt_meta['fe_vals'], dtype=float)
    T_vals = np.asarray(tdt_meta['T_vals'], dtype=float)

    st.sidebar.markdown(f"**Tensor:** {n_co}×{n_cr}×{n_fe}×{n_T} = {n_co*n_cr*n_fe*n_T:,} entries")

    # --- Rank Analysis ---
    if st.sidebar.button("🔍 Run SVD Rank Analysis", use_container_width=True):
        with st.spinner("Running SVD on all four modes..."):
            G_LIQ = st.session_state['G_LIQ_tensor']
            G_FCC = st.session_state['G_FCC_tensor']

            for phase_name, tensor in [("LIQUID", G_LIQ), ("FCC", G_FCC)]:
                st.subheader(f"📊 SVD Rank Analysis — {phase_name}")
                cols = st.columns(4)
                modes = [("Co (Mode-0)", 0), ("Cr (Mode-1)", 1),
                        ("Fe (Mode-2)", 2), ("T (Mode-3)", 3)]
                for (name, mode), col in zip(modes, cols):
                    with col:
                        matrix = unfold_tensor(tensor, mode)
                        rank, s, s_norm = svd_rank_analysis(matrix, threshold=0.01)
                        st.metric(name, f"Rank ≈ {rank}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(range(1, min(len(s_norm), 15)+1)),
                                                  y=s_norm[:15], mode='lines+markers',
                                                  marker=dict(size=8, color='steelblue'),
                                                  line=dict(color='steelblue', width=2)))
                        fig.add_hline(y=0.01, line_dash="dash", line_color="red",
                                     annotation_text="Threshold")
                        fig.update_layout(title=f"Singular Values — {name}",
                                         xaxis_title="Index", yaxis_title="Normalized SV",
                                         height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_svd_{phase_name}_{mode}")

    # --- CPD Analysis ---
    st.sidebar.divider()
    st.sidebar.header("🧮 CPD Analysis")

    R_liq = st.sidebar.slider("CPD Rank (LIQUID)", 1, 12, 6)
    R_fcc = st.sidebar.slider("CPD Rank (FCC)", 1, 12, 6)
    max_iter = st.sidebar.slider("Max ALS Iterations", 10, 300, 100)

    if st.sidebar.button("🚀 Run CPD-ALS", use_container_width=True, type="primary"):
        with st.spinner(f"Running CPD-ALS (LIQUID R={R_liq}, FCC R={R_fcc})..."):
            G_LIQ = st.session_state['G_LIQ_tensor']
            G_FCC = st.session_state['G_FCC_tensor']

            # === CRITICAL FIX: Store normalization parameters for denormalization ===
            # Compute mean and std for each phase tensor (for denormalization later)
            mu_liq = np.nanmean(G_LIQ)
            sigma_liq = np.nanstd(G_LIQ)
            mu_fcc = np.nanmean(G_FCC)
            sigma_fcc = np.nanstd(G_FCC)

            # Normalize tensors for CPD training
            G_LIQ_norm = (G_LIQ - mu_liq) / (sigma_liq + 1e-12)
            G_FCC_norm = (G_FCC - mu_fcc) / (sigma_fcc + 1e-12)

            # Run CPD on normalized tensors
            A_liq, B_liq, C_liq, D_liq, lam_liq, err_liq = cpd_als_4d(G_LIQ_norm, R_liq, max_iter=max_iter, tol=1e-5)
            A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc, err_fcc = cpd_als_4d(G_FCC_norm, R_fcc, max_iter=max_iter, tol=1e-5)

            # === CRITICAL FIX: Store denormalization parameters ===
            st.session_state['mu_liq'] = float(mu_liq)
            st.session_state['sigma_liq'] = float(sigma_liq)
            st.session_state['mu_fcc'] = float(mu_fcc)
            st.session_state['sigma_fcc'] = float(sigma_fcc)

            st.session_state['A_liq'] = A_liq
            st.session_state['B_liq'] = B_liq
            st.session_state['C_liq'] = C_liq
            st.session_state['D_liq'] = D_liq
            st.session_state['lam_liq'] = lam_liq
            st.session_state['A_fcc'] = A_fcc
            st.session_state['B_fcc'] = B_fcc
            st.session_state['C_fcc'] = C_fcc
            st.session_state['D_fcc'] = D_fcc
            st.session_state['lam_fcc'] = lam_fcc
            st.session_state['cpd_error_liq'] = err_liq
            st.session_state['cpd_error_fcc'] = err_fcc
            st.session_state['cpd_converged'] = True

            st.success(f"✅ CPD converged! LIQUID RMSE={err_liq:.2f} J/mol, FCC RMSE={err_fcc:.2f} J/mol")
            st.info(f"📊 Denormalization params stored: μ_LIQ={mu_liq:.1f}, σ_LIQ={sigma_liq:.1f}; μ_FCC={mu_fcc:.1f}, σ_FCC={sigma_fcc:.1f}")

    if 'cpd_converged' in st.session_state and st.session_state['cpd_converged']:
        st.sidebar.success("✅ CPD factors ready")

    # --- MAIN TABS ---
    tabs = st.tabs([
        "📊 Raw Data Explorer",
        "🧊 3D Tetrahedral",
        "🌐 Spherical Harmonics",
        "🔄 Temperature Morph",
        "🔢 Factor Visualisation",
        "🔥 AM: Transition Surface",
        "🌡️ AM: Temperature Factors",
        "🎯 AM: Sensitivity",
        "⚠️ AM: Defect Analysis",
        "🔗 AM: Gradient Design",
        "📐 Quadratic Expansion"
    ])

    # =============================================
    # TAB 1: Raw Data Explorer
    # =============================================
    with tabs[0]:
        st.header("📊 Raw CALPHAD Data Explorer")
        T_sel = st.select_slider("Temperature", options=T_list, value=T_list[len(T_list)//2])
        df_T = df[df["T"] == T_sel]

        col1, col2 = st.columns(2)
        with col1:
            fig_liq = go.Figure(data=go.Scatter3d(
                x=df_T["Co"], y=df_T["Cr"], z=df_T["Fe"],
                mode='markers', marker=dict(size=3, color=df_T["G_LIQ"],
                    colorscale='Viridis', cmin=G_global_min, cmax=G_global_max,
                    colorbar=dict(title="G_LIQ (J/mol)", thickness=15)),
                hovertemplate="Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<br>G_LIQ=%{marker.color:.1f}<extra></extra>"))
            fig_liq.update_layout(title=f"LIQUID Gibbs Energy at {T_sel}K", scene=dict(aspectmode='cube'),
                                 height=600, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig_liq, use_container_width=True, key="plotly_001")
        with col2:
            fig_fcc = go.Figure(data=go.Scatter3d(
                x=df_T["Co"], y=df_T["Cr"], z=df_T["Fe"],
                mode='markers', marker=dict(size=3, color=df_T["G_FCC"],
                    colorscale='Plasma', cmin=G_global_min, cmax=G_global_max,
                    colorbar=dict(title="G_FCC (J/mol)", thickness=15)),
                hovertemplate="Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<br>G_FCC=%{marker.color:.1f}<extra></extra>"))
            fig_fcc.update_layout(title=f"FCC Gibbs Energy at {T_sel}K", scene=dict(aspectmode='cube'),
                                 height=600, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig_fcc, use_container_width=True, key="plotly_002")

        st.subheader("Driving Force ΔG = G_LIQ - G_FCC")
        fig_dG = go.Figure(data=go.Scatter3d(
            x=df_T["Co"], y=df_T["Cr"], z=df_T["Fe"],
            mode='markers', marker=dict(size=3, color=df_T["dG"],
                colorscale='RdBu_r', cmin=-dG_global_abs_max, cmax=dG_global_abs_max,
                colorbar=dict(title="ΔG (J/mol)", thickness=15)),
            hovertemplate="Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<br>ΔG=%{marker.color:.1f}<extra></extra>"))
        fig_dG.update_layout(title=f"Phase Driving Force at {T_sel}K", scene=dict(aspectmode='cube'),
                            height=600, margin=dict(l=0,r=0,b=0,t=40))
        st.plotly_chart(fig_dG, use_container_width=True, key="plotly_003")

        c1, c2, c3 = st.columns(3)
        c1.metric("G_LIQ mean", f"{df_T['G_LIQ'].mean():.0f} J/mol")
        c2.metric("G_FCC mean", f"{df_T['G_FCC'].mean():.0f} J/mol")
        c3.metric("ΔG mean", f"{df_T['dG'].mean():.0f} J/mol")

    # =============================================
    # TAB 2: 3D Tetrahedral
    # =============================================
    with tabs[1]:
        st.header("🧊 3D Tetrahedral Composition Space")
        T_sel = st.select_slider("Temperature", options=T_list, value=T_list[len(T_list)//2], key="tet_T")
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_sel)

        if interp_liq is None:
            st.error(f"No data for T={T_sel}K")
        else:
            resolution = st.slider("Grid resolution", 10, 50, 25, key="tet_res")
            with st.spinner("Generating tetrahedral grid..."):
                grid_pts = generate_tetrahedral_grid(resolution)
                G_liq = interp_liq(grid_pts)
                G_fcc = interp_fcc(grid_pts)
                dG = G_liq - G_fcc
                valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)

                if np.sum(valid) < 10:
                    st.warning("Too few valid points. Try lower resolution.")
                else:
                    proximity = compute_data_proximity(grid_pts[valid], all_pts)
                    boundary_pts, boundary_dG = find_phase_boundary_points(grid_pts[valid], dG[valid])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=grid_pts[valid,0], y=grid_pts[valid,1], z=grid_pts[valid,2],
                        mode='markers', marker=dict(size=3, color=dG[valid],
                            colorscale='RdBu_r', cmin=-dG_global_abs_max, cmax=dG_global_abs_max,
                            colorbar=dict(title="ΔG (J/mol)", thickness=15),
                            opacity=0.6),
                        name='ΔG field'))
                    if len(boundary_pts) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=boundary_pts[:,0], y=boundary_pts[:,1], z=boundary_pts[:,2],
                            mode='markers', marker=dict(size=4, color='gold', symbol='x'),
                            name='Phase boundary'))
                    fig.update_layout(title=f"Tetrahedral Space at {T_sel}K", scene=dict(aspectmode='cube'),
                                     height=650, margin=dict(l=0,r=0,b=0,t=40))
                    st.plotly_chart(fig, use_container_width=True, key="plotly_004")

                    st.markdown(f"**Valid points:** {np.sum(valid):,} | **Boundary points:** {len(boundary_pts)}")

    # =============================================
    # TAB 3: Spherical Harmonics
    # =============================================
    with tabs[2]:
        st.header("🌐 Spherical Harmonic Surface Reconstruction")
        if not SCIPY_AVAILABLE:
            st.error("scipy.special required for spherical harmonics.")
        else:
            T_sel = st.select_slider("Temperature", options=T_list, value=T_list[len(T_list)//2], key="sh_T")
            interp_liq, interp_fcc = build_interpolators_for_T(df, T_sel)

            if interp_liq is None:
                st.error(f"No data for T={T_sel}K")
            else:
                sh_R = st.slider("Sphere radius", 0.1, 0.6, 0.35, 0.05, key="sh_R")
                l_max = st.slider("SH degree l_max", 1, 6, 3, key="sh_lmax")

                with st.spinner("Sampling and fitting spherical harmonics..."):
                    TH, PH, G_stable, dG, valid, pts = sample_g_on_sphere(interp_liq, interp_fcc, sh_R)

                    if np.sum(valid) < 10:
                        st.warning("Too few valid spherical points. Reduce radius.")
                    else:
                        coeffs, l_max_actual = fit_sh_coeffs(TH, PH, G_stable, l_max)
                        if coeffs is None:
                            st.error("SH fitting failed. Try lower l_max or larger radius.")
                        else:
                            G_recon = reconstruct_sh_surface(TH, PH, coeffs, l_max_actual)
                            error = np.abs(G_stable - G_recon)

                            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'surface'},{'type':'surface'}]],
                                               subplot_titles=("Original Data","SH Reconstruction"))
                            fig.add_trace(go.Surface(x=pts[:,0].reshape(TH.shape), y=pts[:,1].reshape(TH.shape),
                                                       z=pts[:,2].reshape(TH.shape), surfacecolor=G_stable,
                                                       colorscale='Viridis', name='Original'), row=1, col=1)
                            fig.add_trace(go.Surface(x=pts[:,0].reshape(TH.shape), y=pts[:,1].reshape(TH.shape),
                                                       z=pts[:,2].reshape(TH.shape), surfacecolor=G_recon,
                                                       colorscale='Viridis', name='SH Recon'), row=1, col=2)
                            fig.update_layout(height=500, margin=dict(l=0,r=0,b=0,t=40))
                            st.plotly_chart(fig, use_container_width=True, key="plotly_005")

                            c1, c2, c3 = st.columns(3)
                            c1.metric("SH degree", f"l_max={l_max_actual}")
                            c2.metric("Coefficients", f"{len(coeffs)}")
                            c3.metric("Max error", f"{np.nanmax(error):.1f} J/mol")

    # =============================================
    # TAB 4: Temperature Morph
    # =============================================
    with tabs[3]:
        st.header("🔄 Temperature-Driven Shape Morphing")
        if not SCIPY_AVAILABLE:
            st.error("scipy.special required.")
        else:
            T_morph = st.slider("Temperature", T_min, T_max, (T_min+T_max)//2, 
                                 int(T_vals[1]-T_vals[0]) if len(T_vals)>1 else 100, key="morph_T")
            interp_liq, interp_fcc = build_interpolators_for_T(df, T_morph)

            if interp_liq is None:
                st.error(f"No data for T={T_morph}K")
            else:
                sh_R = st.slider("Sphere radius", 0.1, 0.6, 0.35, 0.05, key="morph_R")
                l_max = st.slider("SH degree", 1, 6, 3, key="morph_lmax")
                T_factor = (T_morph - T_min) / T_range

                with st.spinner("Morphing..."):
                    TH, PH, G_stable, dG, valid, pts = sample_g_on_sphere(interp_liq, interp_fcc, sh_R)
                    if np.sum(valid) < 10:
                        st.warning("Too few valid points.")
                    else:
                        coeffs, l_max_actual = fit_sh_coeffs(TH, PH, G_stable, l_max)
                        if coeffs is None:
                            st.error("SH fitting failed.")
                        else:
                            G_recon = reconstruct_sh_surface(TH, PH, coeffs, l_max_actual)
                            r_liq = get_liquid_radius(G_recon, sh_R, T_factor)
                            r_fcc = get_fcc_radius(G_recon, sh_R, T_factor)

                            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'surface'},{'type':'surface'}]],
                                               subplot_titles=("LIQUID Phase Morph","FCC Phase Morph"))
                            fig.add_trace(go.Surface(
                                x=r_liq*np.sin(PH)*np.cos(TH), y=r_liq*np.sin(PH)*np.sin(TH), z=r_liq*np.cos(PH),
                                surfacecolor=G_recon, colorscale='Reds', name='LIQUID'), row=1, col=1)
                            fig.add_trace(go.Surface(
                                x=r_fcc*np.sin(PH)*np.cos(TH), y=r_fcc*np.sin(PH)*np.sin(TH), z=r_fcc*np.cos(PH),
                                surfacecolor=G_recon, colorscale='Blues', name='FCC'), row=1, col=2)
                            fig.update_layout(height=500, margin=dict(l=0,r=0,b=0,t=40))
                            st.plotly_chart(fig, use_container_width=True, key="plotly_006")

    # =============================================
    # TAB 5: Factor Visualisation
    # =============================================
    with tabs[4]:
        st.header("🔢 CPD Factor Matrix Visualisation")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            render_factor_matrix_visualisation(
                factors['A_liq'], factors['B_liq'], factors['C_liq'], factors['D_liq'], factors['lam_liq'],
                factors['A_fcc'], factors['B_fcc'], factors['C_fcc'], factors['D_fcc'], factors['lam_fcc'],
                factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals']
            )

    # =============================================
    # TAB 6: AM Transition Surface
    # =============================================
    with tabs[5]:
        st.header("🔥 AM: Composition-Dependent Transition Temperature")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            render_am_transition_surface_tab(
                factors['A_liq'], factors['A_fcc'], factors['B_liq'], factors['B_fcc'],
                factors['C_liq'], factors['C_fcc'], factors['D_liq'], factors['D_fcc'],
                factors['lam_liq'], factors['lam_fcc'],
                factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                factors['mu_liq'], factors['sigma_liq'], factors['mu_fcc'], factors['sigma_fcc']
            )

    # =============================================
    # TAB 7: AM Temperature Factors
    # =============================================
    with tabs[6]:
        st.header("🌡️ AM: Temperature Factor Analysis")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            render_am_temperature_factors_tab(
                factors['D_liq'], factors['D_fcc'], factors['T_vals'],
                factors['lam_liq'], factors['lam_fcc']
            )

    # =============================================
    # TAB 8: AM Sensitivity
    # =============================================
    with tabs[7]:
        st.header("🎯 AM: Composition Sensitivity Analysis")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            phase_sens = st.radio("Phase for sensitivity", ["LIQUID", "FCC"], index=0, horizontal=True)
            if phase_sens == "LIQUID":
                A, B, C, lam = factors['A_liq'], factors['B_liq'], factors['C_liq'], factors['lam_liq']
            else:
                A, B, C, lam = factors['A_fcc'], factors['B_fcc'], factors['C_fcc'], factors['lam_fcc']

            render_am_sensitivity_tab(A, B, C, lam, factors['co_vals'], factors['cr_vals'], factors['fe_vals'])

    # =============================================
    # TAB 9: AM Defect Analysis
    # =============================================
    with tabs[8]:
        st.header("⚠️ AM: Defect Susceptibility Analysis")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            render_am_defect_tab(
                factors['A_liq'], factors['A_fcc'], factors['B_liq'], factors['B_fcc'],
                factors['C_liq'], factors['C_fcc'], factors['D_liq'], factors['D_fcc'],
                factors['lam_liq'], factors['lam_fcc'],
                factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                factors['mu_liq'], factors['sigma_liq'], factors['mu_fcc'], factors['sigma_fcc']
            )

    # =============================================
    # TAB 10: AM Gradient Design
    # =============================================
    with tabs[9]:
        st.header("🔗 AM: Multi-Material Gradient Design")

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            render_gradient_design_tab(
                factors['A_liq'], factors['A_fcc'], factors['B_liq'], factors['B_fcc'],
                factors['C_liq'], factors['C_fcc'], factors['D_liq'], factors['D_fcc'],
                factors['lam_liq'], factors['lam_fcc'],
                factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                factors['mu_liq'], factors['sigma_liq'], factors['mu_fcc'], factors['sigma_fcc']
            )

    # =============================================
    # TAB 11: Quadratic Expansion
    # =============================================
    with tabs[10]:
        st.header("📐 Quadratic Expansion from CPD Factors")
        st.markdown(r"""
        **Theory**: Near equilibrium, Gibbs energy can be approximated quadratically:

        $$G \approx G_{eq} + A_{Co}(c_{Co}-c_{eq}^{Co})^2 + A_{Cr}(c_{Cr}-c_{eq}^{Cr})^2 + A_{Fe}(c_{Fe}-c_{eq}^{Fe})^2 + A_T(T-T_m)^2$$

        where $A_\alpha = \frac{1}{2}\frac{\partial^2 G}{\partial c_\alpha^2}$ and $A_T = \frac{1}{2}\frac{\partial^2 G}{\partial T^2}$.

        **CRITICAL FIX**: Coefficients are now computed in **physical J/mol** units (not normalized),
        enabling direct comparison with full CPD reconstruction values.
        """)

        is_valid, msg, factors = validate_cpd_session_state()
        if not is_valid:
            st.warning(f"⚠️ {msg}\n\nRun CPD-ALS first (sidebar).")
        else:
            phase_quad = st.radio("Phase", ["LIQUID", "FCC"], index=0, horizontal=True, key="quad_phase")

            if phase_quad == "LIQUID":
                A, B, C, D, lam = (factors['A_liq'], factors['B_liq'], factors['C_liq'], 
                                  factors['D_liq'], factors['lam_liq'])
                mu = factors['mu_liq']
                sigma = factors['sigma_liq']
            else:
                A, B, C, D, lam = (factors['A_fcc'], factors['B_fcc'], factors['C_fcc'],
                                  factors['D_fcc'], factors['lam_fcc'])
                mu = factors['mu_fcc']
                sigma = factors['sigma_fcc']

            st.subheader("📍 Equilibrium Composition")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                c_eq_co = st.number_input("c_eq (Co)", 0.0, 1.0, 0.25, 0.01, key="c_eq_co")
            with col2:
                c_eq_cr = st.number_input("c_eq (Cr)", 0.0, 1.0, 0.25, 0.01, key="c_eq_cr")
            with col3:
                c_eq_fe = st.number_input("c_eq (Fe)", 0.0, 1.0, 0.25, 0.01, key="c_eq_fe")
            with col4:
                T_m = st.number_input("T_m (K)", float(T_vals.min()), float(T_vals.max()), 
                                     float(T_vals[len(T_vals)//2]), 50.0, key="T_m")

            if st.button("📐 Compute Quadratic Coefficients", use_container_width=True, type="primary"):
                with st.spinner("Computing quadratic expansion coefficients..."):
                    coeffs = compute_quadratic_coefficients_from_cpd(
                        A, B, C, D, lam,
                        factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                        c_eq_co, c_eq_cr, c_eq_fe, T_m,
                        mu, sigma
                    )

                    st.success("✅ Quadratic coefficients computed (PHYSICAL units)")

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("A_Co", f"{coeffs['A_Co']:.1f}", "J/mol")
                    c2.metric("A_Cr", f"{coeffs['A_Cr']:.1f}", "J/mol")
                    c3.metric("A_Fe", f"{coeffs['A_Fe']:.1f}", "J/mol")
                    c4.metric("A_T", f"{coeffs['A_T']:.4f}", "J/(mol·K²)")
                    c5.metric("G_eq", f"{coeffs['G_eq']:.0f}", "J/mol")

                    st.markdown(f"""
                    **Physical Interpretation:**
                    - **A_Co = {coeffs['A_Co']:.1f} J/mol**: Gibbs energy curvature w.r.t. Co. 
                      {'High' if abs(coeffs['A_Co']) > 50000 else 'Moderate' if abs(coeffs['A_Co']) > 10000 else 'Low'} sensitivity to Co variations.
                    - **A_Cr = {coeffs['A_Cr']:.1f} J/mol**: Gibbs energy curvature w.r.t. Cr.
                      {'High' if abs(coeffs['A_Cr']) > 50000 else 'Moderate' if abs(coeffs['A_Cr']) > 10000 else 'Low'} sensitivity to Cr variations.
                    - **A_Fe = {coeffs['A_Fe']:.1f} J/mol**: Gibbs energy curvature w.r.t. Fe.
                      {'High' if abs(coeffs['A_Fe']) > 50000 else 'Moderate' if abs(coeffs['A_Fe']) > 10000 else 'Low'} sensitivity to Fe variations.
                    - **A_T = {coeffs['A_T']:.4f} J/(mol·K²)**: Temperature curvature. 
                      {'Strong' if abs(coeffs['A_T']) > 0.1 else 'Moderate' if abs(coeffs['A_T']) > 0.01 else 'Weak'} temperature dependence.
                    """)

                    st.subheader("🔍 Verification: Full CPD vs Quadratic")

                    n_test = st.slider("Number of test points", 10, 200, 50, key="n_test")
                    np.random.seed(42)
                    test_comps = np.random.rand(n_test, 3) * 0.1
                    test_comps[:, 0] += c_eq_co - 0.05
                    test_comps[:, 1] += c_eq_cr - 0.05
                    test_comps[:, 2] += c_eq_fe - 0.05
                    test_T = T_m + np.random.randn(n_test) * 100

                    verify_df = verify_quadratic_approximation(
                        coeffs, A, B, C, D, lam,
                        factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                        test_comps, test_T
                    )

                    fig_err = plot_error_metrics_dashboard(verify_df)
                    st.plotly_chart(fig_err, use_container_width=True, key="plotly_014")

                    mean_abs_err = verify_df['absolute_error'].mean()
                    mean_rel_err = verify_df['relative_error'].mean() * 100
                    st.metric("Mean Absolute Error", f"{mean_abs_err:.1f} J/mol")
                    st.metric("Mean Relative Error", f"{mean_rel_err:.2f}%")

                    if mean_abs_err < 500:
                        st.success("✅ Excellent approximation — quadratic form is valid near equilibrium")
                    elif mean_abs_err < 2000:
                        st.info("ℹ️ Good approximation — valid in local neighborhood")
                    elif mean_abs_err < 5000:
                        st.warning("⚠️ Moderate approximation — use smaller perturbation range")
                    else:
                        st.error("❌ Poor approximation — equilibrium point may be at phase boundary or outside valid range")

                    st.subheader("📊 Full CPD vs Quadratic Comparison")
                    fig_comp = plot_cpd_vs_quadratic_comparison(
                        coeffs, A, B, C, D, lam,
                        factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                        [c_eq_co, c_eq_cr, c_eq_fe], T_m
                    )
                    st.plotly_chart(fig_comp, use_container_width=True, key="plotly_015")

                    st.subheader("🌐 3D Surface Comparison")
                    fig_3d = plot_3d_comparison_surface(
                        coeffs, A, B, C, D, lam,
                        factors['co_vals'], factors['cr_vals'], factors['fe_vals'], factors['T_vals'],
                        [c_eq_co, c_eq_cr, c_eq_fe], T_m, sh_R_fixed=0.35
                    )
                    st.plotly_chart(fig_3d, use_container_width=True, key="plotly_016")

                    st.subheader("📥 Download Results")
                    csv_verify = verify_df.to_csv(index=False)
                    st.download_button("Download Verification Data", csv_verify, 
                                      "quadratic_verification.csv", "text/csv")

                    with st.expander("📖 Theoretical Background", expanded=False):
                        st.markdown(r"""
                        ### Quadratic Expansion Theory

                        The quadratic approximation is derived from a second-order Taylor expansion:

                        $$G(c, T) \approx G(c_{eq}, T_m) + \frac{1}{2}\sum_{\alpha} A_\alpha (c_\alpha - c_\alpha^{eq})^2 + \frac{1}{2}A_T(T-T_m)^2$$

                        where the cross-terms vanish at equilibrium (minimum condition). The coefficients are:

                        $$A_\alpha = \frac{1}{2}\frac{\partial^2 G}{\partial c_\alpha^2}\bigg|_{eq}, \quad A_T = \frac{1}{2}\frac{\partial^2 G}{\partial T^2}\bigg|_{eq}$$

                        **From CPD factors**, using the separable structure:

                        $$A_\alpha = \frac{1}{2}\sigma \sum_r \lambda_r \cdot F_\alpha''(c_\alpha^{eq}) \cdot \prod_{\beta \neq \alpha} F_\beta(c_\beta^{eq}) \cdot D(T_m)$$

                        where σ is the denormalization standard deviation.

                        **Validity conditions:**
                        1. Equilibrium point is a true minimum (not saddle or boundary)
                        2. Perturbation range is small compared to curvature radius
                        3. No phase transitions occur within perturbation range
                        4. Higher-order terms (cubic, quartic) are negligible
                        """)


if __name__ == "__main__":
    main()
