#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Co-Cr-Fe-Ni Gibbs Free Energy Tensor Explorer
with Spherical Harmonics Visualization & Analysis

Features:
✓ Gibbs energy interpolation from CSV data
✓ Cartesian ↔ Spherical coordinate transforms
✓ Spherical harmonics (Yₗₘ) decomposition & reconstruction
✓ Interactive SH coefficient editing with live surface update
✓ Multi-temperature SH coefficient animation
✓ Phase boundary detection via SH gradient analysis
✓ Full tensor G[x_Co, x_Cr, x_Fe, T] calculations
✓ Phase matrix export (LIQUID/FCC stability map)
✓ e3nn-compatible coefficient export for equivariant ML

Dependencies: streamlit, numpy, pandas, plotly, scipy, torch (optional for e3nn)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
#from scipy.special import sph_harm
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_gradient
import warnings
warnings.filterwarnings('ignore')
# =============================================
# ROBUST SPHERICAL HARMONICS IMPORT
# =============================================
try:
    from scipy.special import sph_harm
    SCIPY_SH_AVAILABLE = True
except (ImportError, AttributeError):
    SCIPY_SH_AVAILABLE = False
    st.warning("⚠️ scipy.special.sph_harm not available. Using pure NumPy fallback.")
    
    # Pure NumPy implementation of associated Legendre polynomials & spherical harmonics
    def legendre_p(l, m, x):
        """Associated Legendre polynomial P_l^m(x) using recurrence."""
        x = np.clip(x, -1, 1)
        if m > l:
            return np.zeros_like(x)
        # Start with P_m^m
        pmm = np.ones_like(x)
        if m > 0:
            somx2 = np.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm *= -fact * somx2
                fact += 2
        if l == m:
            return pmm
        # Recurrence to get P_{m+1}^m
        pmmp1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            return pmmp1
        # General recurrence for l > m+1
        pll = np.zeros((l - m + 1, len(x)))
        pll[0] = pmm
        pll[1] = pmmp1
        for ll in range(m + 2, l + 1):
            pll[ll - m] = ((2 * ll - 1) * x * pll[ll - m - 1] - 
                          (ll + m - 1) * pll[ll - m - 2]) / (ll - m)
        return pll[l - m]
    
    def sph_harm(m, l, theta, phi):
        """
        Pure NumPy spherical harmonic Y_l^m(theta, phi).
        Convention: theta=azimuth (0→2π), phi=polar (0→π)
        Returns complex-valued array.
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        # Normalization factor
        norm = np.sqrt((2 * l + 1) / (4 * np.pi) * 
                      np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m)))
        # Associated Legendre
        plm = legendre_p(l, abs(m), np.cos(phi))
        # Azimuthal phase
        azimuth = np.exp(1j * m * theta)
        return norm * plm * azimuth * ((-1) ** m if m >= 0 else 1)

# =============================================
# PATH & PAGE CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(
    page_title="CoCrFeNi Gibbs Energy + SH Explorer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# COLORMAPS & SYMBOLS
# =============================================
COLORMAPS = sorted(list(set([
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu",
    "YlOrBr", "YlOrRd", "BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy",
    "RdYlBu", "RdYlGn", "Spectral", "Phase", "Twilight", "HSV", "Jet",
    "Rainbow", "Hot", "Cool", "Spring", "Summer", "Autumn", "Winter",
    "Bone", "Copper", "Cubehelix", "Terrain", "Ocean", "Sinebow", "Prism",
    "Flag", "Gnuplot", "Gnuplot2", "CMRmap", "Afmhot", "Gist_heat",
    "Gist_rainbow", "Gist_stern", "Gist_earth", "Gist_ncar", "Brg", "Bwr",
    "Seismic", "Coolwarm", "Blackbody", "Electric", "Algae", "Deep", "Dense",
    "Haline", "Ice", "Matter", "Speed", "Tempo", "Thermal", "Turbid",
    "Plotly3", "Portland", "Picnic", "Solar", "Balance", "Delta", "Curl",
    "IceFire", "Edge", "Fall", "Sunset", "Sunsetdark", "Teal", "Tealgrn",
    "Tropic", "Peach", "Oxy", "Mint", "Emrld", "Aggrnyl", "Agsunset",
    "Armyrose", "Bluered", "Blugrn", "Bluyl", "Brwnyl", "Burg", "Burgyl",
    "Darkmint", "Geysr", "Magenta", "Mrybm", "Mygbm", "Oryel", "Pinkyl",
    "Purp", "Purpor", "Redor", "Ylorrd", "Ylorbr", "Ylgnbu", "Ylgn"
])))

SYMBOLS = ["circle", "diamond", "cross", "x", "star", "square", "pentagon", "hexagon", 
           "hexagon2", "octagon", "star-diamond", "star-triangle-up", "star-square"]

# =============================================
# COORDINATE TRANSFORMATIONS
# =============================================
def cartesian_to_spherical(c1, c2, c3):
    """Convert composition (Co, Cr, Fe) to spherical (r, theta, phi)."""
    r = np.sqrt(c1**2 + c2**2 + c3**2)
    safe_r = np.where(r == 0, 1e-12, r)
    theta = np.arctan2(c2, c1)
    phi = np.arccos(np.clip(c3 / safe_r, -1.0, 1.0))
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

# =============================================
# SPHERICAL HARMONICS UTILITIES
# =============================================
def compute_real_spherical_harmonics(l_max, theta, phi):
    """
    Compute real spherical harmonics up to degree l_max.
    Returns array of shape (n_points, n_coeffs) where n_coeffs = (l_max+1)**2
    
    Real form convention:
    - m = 0: Y_l^0 (real)
    - m > 0: √2 * Re[Y_l^m]
    - m < 0: √2 * Im[Y_l^|m|]
    """
    n_pts = len(theta) if hasattr(theta, '__len__') else 1
    n_coeffs = (l_max + 1) ** 2
    Y = np.zeros((n_pts, n_coeffs))
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                Y[:, idx] = np.sqrt(2) * sph_harm(abs(m), l, theta, phi).imag
            elif m > 0:
                Y[:, idx] = np.sqrt(2) * sph_harm(m, l, theta, phi).real
            else:
                Y[:, idx] = sph_harm(0, l, theta, phi).real
            idx += 1
    return Y

def fit_spherical_harmonics(G_values, theta, phi, l_max=4, reg_lambda=1e-6):
    """
    Fit Gibbs energy values to spherical harmonics expansion with optional regularization.
    
    Returns:
        coeffs: SH coefficients array
        G_reconstructed: Reconstructed G values
        rmse: Root mean square error
        condition_number: Condition number of the design matrix
    """
    Y = compute_real_spherical_harmonics(l_max, theta, phi)
    
    # Regularized least squares: (YᵀY + λI)c = YᵀG
    YtY = Y.T @ Y
    YtG = Y.T @ G_values
    n_coeffs = YtY.shape[0]
    
    coeffs, residuals, rank, s = lstsq(YtY + reg_lambda * np.eye(n_coeffs), YtG)
    G_reconstructed = Y @ coeffs
    rmse = np.sqrt(np.mean((G_values - G_reconstructed)**2))
    condition_number = np.linalg.cond(YtY + reg_lambda * np.eye(n_coeffs))
    
    return coeffs, G_reconstructed, rmse, condition_number

def generate_sh_surface(coeffs, l_max, r_base=0.5, r_scale=0.3, resolution=60):
    """
    Generate mesh data for visualizing SH reconstruction on a sphere.
    
    Returns:
        X, Y, Z: Cartesian coordinates for surface plot
        G_values: Gibbs energy values on the surface
        theta_grid, phi_grid: Spherical coordinate grids
    """
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0.01, np.pi-0.01, resolution)  # Avoid poles for stability
    Theta, Phi = np.meshgrid(theta, phi)
    
    # Evaluate SH expansion on grid
    G_sh = np.zeros_like(Theta)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                basis = np.sqrt(2) * sph_harm(abs(m), l, Theta, Phi).imag
            elif m > 0:
                basis = np.sqrt(2) * sph_harm(m, l, Theta, Phi).real
            else:
                basis = sph_harm(0, l, Theta, Phi).real
            G_sh += coeffs[idx] * basis
            idx += 1
    
    # Radial deformation based on G values for visual effect
    G_norm = (G_sh - G_sh.min()) / (G_sh.max() - G_sh.min() + 1e-10)
    r = r_base + r_scale * G_norm
    
    # Convert to Cartesian
    X = r * np.sin(Phi) * np.cos(Theta)
    Y = r * np.sin(Phi) * np.sin(Theta)
    Z = r * np.cos(Phi)
    
    return X, Y, Z, G_sh, Theta, Phi

def compute_sh_gradient(coeffs, l_max, theta, phi, r=1.0):
    """
    Compute gradient of SH expansion on sphere (tangential components).
    
    Returns:
        dG_dtheta, dG_dphi: Gradient components in spherical coordinates
    """
    eps = 1e-6
    G_center = np.zeros_like(theta)
    
    # Evaluate at center point
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                basis = np.sqrt(2) * sph_harm(abs(m), l, theta, phi).imag
            elif m > 0:
                basis = np.sqrt(2) * sph_harm(m, l, theta, phi).real
            else:
                basis = sph_harm(0, l, theta, phi).real
            G_center += coeffs[idx] * basis
            idx += 1
    
    # Numerical gradient in theta
    G_theta_plus = np.zeros_like(theta)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                basis = np.sqrt(2) * sph_harm(abs(m), l, theta + eps, phi).imag
            elif m > 0:
                basis = np.sqrt(2) * sph_harm(m, l, theta + eps, phi).real
            else:
                basis = sph_harm(0, l, theta + eps, phi).real
            G_theta_plus += coeffs[idx] * basis
            idx += 1
    dG_dtheta = (G_theta_plus - G_center) / eps
    
    # Numerical gradient in phi
    G_phi_plus = np.zeros_like(theta)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                basis = np.sqrt(2) * sph_harm(abs(m), l, theta, phi + eps).imag
            elif m > 0:
                basis = np.sqrt(2) * sph_harm(m, l, theta, phi + eps).real
            else:
                basis = sph_harm(0, l, theta, phi + eps).real
            G_phi_plus += coeffs[idx] * basis
            idx += 1
    dG_dphi = (G_phi_plus - G_center) / eps
    
    return dG_dtheta, dG_dphi

# =============================================
# TENSOR & PHASE MATRIX CALCULATIONS
# =============================================
def compute_gibbs_tensor(df, T_values, comp_grid_res=20):
    """
    Compute full Gibbs energy tensor G[x_Co, x_Cr, x_Fe, T] for both phases.
    
    Returns:
        tensor_dict: Dictionary containing:
            - 'G_LIQ': 4D array [T_idx, Co_idx, Cr_idx, Fe_idx]
            - 'G_FCC': 4D array [T_idx, Co_idx, Cr_idx, Fe_idx]
            - 'G_stable': 4D array of min(G_LIQ, G_FCC)
            - 'phase_matrix': 4D array of phase labels (0=LIQUID, 1=FCC)
            - 'composition_grid': Meshgrid arrays for Co, Cr, Fe
            - 'T_array': Temperature values
    """
    T_array = np.array(sorted(df['T'].unique()))
    comp_vals = np.linspace(0, 1, comp_grid_res)
    Co_grid, Cr_grid, Fe_grid = np.meshgrid(comp_vals, comp_vals, comp_vals, indexing='ij')
    
    # Flatten composition grid for interpolation
    comp_pts = np.column_stack([
        Co_grid.ravel(), Cr_grid.ravel(), Fe_grid.ravel()
    ])
    valid_mask = np.sum(comp_pts, axis=1) <= 1.0
    
    n_T = len(T_array)
    n_valid = np.sum(valid_mask)
    
    # Initialize output arrays
    G_LIQ = np.full((n_T, comp_grid_res, comp_grid_res, comp_grid_res), np.nan)
    G_FCC = np.full((n_T, comp_grid_res, comp_grid_res, comp_grid_res), np.nan)
    phase_matrix = np.full((n_T, comp_grid_res, comp_grid_res, comp_grid_res), -1, dtype=int)
    
    for t_idx, T_val in enumerate(T_array):
        df_T = df[df['T'] == T_val].copy()
        if len(df_T) < 4:
            continue
            
        pts = df_T[['Co', 'Cr', 'Fe']].values
        interp_liq = LinearNDInterpolator(pts, df_T['G_LIQ'].values)
        interp_fcc = LinearNDInterpolator(pts, df_T['G_FCC'].values)
        
        # Evaluate on valid composition points
        g_liq_vals = interp_liq(comp_pts[valid_mask])
        g_fcc_vals = interp_fcc(comp_pts[valid_mask])
        
        # Reshape back to grid
        G_LIQ[t_idx][valid_mask.reshape(Co_grid.shape)] = g_liq_vals
        G_FCC[t_idx][valid_mask.reshape(Co_grid.shape)] = g_fcc_vals
        
        # Determine stable phase
        stable_mask = g_liq_vals <= g_fcc_vals
        phase_flat = np.where(stable_mask, 0, 1)  # 0=LIQUID, 1=FCC
        phase_matrix[t_idx][valid_mask.reshape(Co_grid.shape)] = phase_flat
    
    G_stable = np.minimum(G_LIQ, G_FCC)
    
    return {
        'G_LIQ': G_LIQ,
        'G_FCC': G_FCC,
        'G_stable': G_stable,
        'phase_matrix': phase_matrix,
        'composition_grid': (Co_grid, Cr_grid, Fe_grid),
        'T_array': T_array,
        'comp_vals': comp_vals
    }

def detect_phase_boundaries(phase_matrix, comp_grid, T_idx, smoothing_sigma=1.0):
    """
    Detect phase boundaries in composition space using gradient analysis.
    
    Returns:
        boundary_points: Array of [Co, Cr, Fe] coordinates on phase boundary
        boundary_confidence: Confidence score for each boundary point
    """
    Co_grid, Cr_grid, Fe_grid = comp_grid
    phase_slice = phase_matrix[T_idx]
    
    # Compute spatial gradients of phase indicator
    grad_co = gaussian_gradient(phase_slice.astype(float), axis=0, sigma=smoothing_sigma)
    grad_cr = gaussian_gradient(phase_slice.astype(float), axis=1, sigma=smoothing_sigma)
    grad_fe = gaussian_gradient(phase_slice.astype(float), axis=2, sigma=smoothing_sigma)
    
    # Boundary magnitude (high gradient = likely boundary)
    grad_mag = np.sqrt(grad_co**2 + grad_cr**2 + grad_fe**2)
    
    # Threshold for boundary detection
    threshold = np.percentile(grad_mag[~np.isnan(grad_mag)], 90)
    boundary_mask = (grad_mag > threshold) & (~np.isnan(phase_slice))
    
    # Extract boundary coordinates
    boundary_co = Co_grid[boundary_mask]
    boundary_cr = Cr_grid[boundary_mask]
    boundary_fe = Fe_grid[boundary_mask]
    boundary_conf = grad_mag[boundary_mask]
    
    boundary_points = np.column_stack([boundary_co, boundary_cr, boundary_fe])
    
    return boundary_points, boundary_conf

# =============================================
# DATA LOADING & INTERPOLATION
# =============================================
@st.cache_data(ttl=3600)
def load_all_data(csv_dir=CSV_FILES_DIR):
    """Load all Gibbs energy CSV files and concatenate."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        return None
        
    dfs = []
    for f in files:
        basename = os.path.basename(f)
        try:
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            df["T"] = T
            dfs.append(df)
        except Exception as e:
            st.warning(f"Skipping {f}: {e}")
    
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

@st.cache_data(ttl=1800)
def build_interpolators_for_T(df, T):
    """Build interpolation functions for LIQUID and FCC phases at temperature T."""
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
        
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values, fill_value=np.nan)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values, fill_value=np.nan)
    return interp_liq, interp_fcc

# =============================================
# MAIN APP
# =============================================
def main():
    # Load data
    df = load_all_data()
    if df is None:
        st.error("❌ No data files found. Please place Gibbs_*.csv files in the `csv_files` directory.")
        st.stop()
    
    T_list = sorted(df["T"].unique())
    if len(T_list) == 0:
        st.error("❌ No temperature data available.")
        st.stop()
    
    # ================= HEADER =================
    st.title("🔷 Co-Cr-Fe-Ni Gibbs Energy + SH Explorer")
    st.markdown(r"""
    **Tensor**: $G = G(x_{\text{Co}}, x_{\text{Cr}}, x_{\text{Fe}}, T)$  
    **Phases**: LIQUID, FCC | **Stable**: $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$  
    **Analysis**: Spherical harmonics decomposition, phase boundaries, tensor exports
    """)
    
    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # --- Query Point ---
        st.subheader("📍 Query Point")
        q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f", key="q_co")
        q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f", key="q_cr")
        q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f", key="q_fe")
        q_t = st.selectbox("Query T (K)", T_list, index=len(T_list)//2, key="q_t")
        
        comp_sum = q_co + q_cr + q_fe
        if comp_sum > 1.0:
            st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0")
        
        eval_query = st.button("🔍 Evaluate", use_container_width=True, key="eval_btn")
        st.divider()
        
        # --- Global Viz ---
        st.subheader("🌡️ Visualization")
        T_val = st.select_slider("Field T (K)", options=T_list, 
                                value=T_list[len(T_list)//2], key="T_viz")
        grid_res = st.slider("Grid Resolution", 15, 40, 25, step=5, key="grid_res")
        
        st.divider()
        
        # --- Coordinate System ---
        st.subheader("🌐 Coordinates")
        coord_sys = st.radio("System", 
                            ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                            index=0, key="coord_radio")
        st.divider()
        
        # --- Phase Display ---
        st.subheader("🎨 Phase Display")
        show_phase = st.radio("Mode", 
                             ["Stable Phase (Min G)", "LIQUID Only", "FCC Only", "Both Overlay"],
                             index=0, key="phase_mode")
        cmap = st.selectbox("Colormap", COLORMAPS, 
                           index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0,
                           key="cmap_select")
        
        col1, col2 = st.columns(2)
        marker_size = col1.slider("Marker Size", 1, 10, 3, key="mkr_size")
        opacity = col2.slider("Opacity", 0.1, 1.0, 0.75, 0.05, key="opacity")
        st.divider()
        
        # --- e3nn Symbols ---
        st.subheader("🔷 e3nn Symbols")
        symbol = st.selectbox("Marker Symbol", SYMBOLS, index=1, key="symbol_select")
        scale_by_g = st.toggle("Scale size by |G|", value=False, key="scale_g")
        show_ref_sphere = st.toggle("Show Reference Sphere", value=False, key="ref_sphere")
        show_axes = st.toggle("Show Coordinate Axes", value=False, key="show_axes")
        show_simplex = st.toggle("Show Composition Simplex", value=False, key="show_simplex")
        st.divider()
        
        # --- SPHERICAL HARMONICS ---
        st.subheader("🌀 Spherical Harmonics")
        sh_enabled = st.toggle("Enable SH Analysis", value=False, key="sh_toggle")
        
        if sh_enabled:
            l_max = st.slider("Max Degree l_max", 1, 8, 4, key="l_max_slider")
            sh_r_fixed = st.number_input("Analysis Radius r", 0.3, 1.0, 0.6, 0.05, key="sh_r")
            sh_viz_mode = st.radio("SH View", 
                                  ["Coefficients", "Reconstructed Surface", "Error Map", "Gradient Field"],
                                  index=0, key="sh_viz")
            
            # Interactive editing
            enable_edit = st.toggle("✏️ Edit Coefficients", value=False, key="sh_edit")
            if enable_edit:
                edit_factor = st.slider("Edit Multiplier", 0.5, 2.0, 1.0, 0.1, key="edit_mult")
                st.caption("Selected coefficients will be scaled by this factor")
            
            # Multi-T trends
            multi_t = st.toggle("📈 Multi-T Trends", value=False, key="multi_t_toggle")
            if multi_t:
                t_range = st.multiselect("Temperatures for Trends", T_list, 
                                        default=[T_list[0], T_list[-1]] if len(T_list) > 1 else T_list,
                                        key="t_trends")
            
            # Phase boundary
            detect_boundaries = st.toggle("🔍 Detect Phase Boundaries", value=False, key="detect_boundary")
            st.divider()
        
        # --- Tensor Export ---
        st.subheader("📦 Tensor Operations")
        export_tensor = st.button("📤 Compute Full Tensor G", key="compute_tensor")
        export_phase_matrix = st.button("📊 Export Phase Matrix", key="export_phase")
        st.caption("Tensor: G[x_Co, x_Cr, x_Fe, T] | Phase: 0=LIQUID, 1=FCC")
        st.divider()
        
        # --- Layout ---
        st.subheader("✏️ Layout")
        template = st.selectbox("Template", ["plotly_white", "plotly_dark", "seaborn", "none"], 
                               index=0, key="template")
        show_grid = st.toggle("Show Grid", value=True, key="show_grid_toggle")
        st.caption(f"📊 Data: {len(T_list)} temperatures | {len(df):,} points")
    
    # ================= QUERY EVALUATION =================
    query_result = None
    if eval_query:
        if comp_sum <= 1.0:
            interp_liq_q, interp_fcc_q = build_interpolators_for_T(df, q_t)
            if interp_liq_q is not None:
                pt = np.array([[q_co, q_cr, q_fe]])
                g_liq_q = float(interp_liq_q(pt)[0])
                g_fcc_q = float(interp_fcc_q(pt)[0])
                
                if not (np.isnan(g_liq_q) or np.isnan(g_fcc_q)):
                    g_stable_q = min(g_liq_q, g_fcc_q)
                    phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"
                    query_result = {
                        "T": q_t, "Co": q_co, "Cr": q_cr, "Fe": q_fe,
                        "Ni": round(1.0 - comp_sum, 4),
                        "G_LIQ": g_liq_q, "G_FCC": g_fcc_q,
                        "G_stable": g_stable_q, "Phase": phase_q
                    }
    
    if query_result:
        st.success(f"✅ Query @ T={query_result['T']} K")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
        c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
        c3.metric("G_stable", f"{query_result['G_stable']:,.0f}", "J/mol")
        c4.metric("Phase", query_result['Phase'])
        c5.metric("x_Ni", f"{query_result['Ni']:.3f}")
        st.divider()
    
    # ================= MAIN COMPUTATION =================
    interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
    if interp_liq is None:
        st.error(f"No data for T = {T_val} K")
        st.stop()
    
    # Generate grid
    x = np.linspace(0, 1, grid_res)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = np.sum(grid_pts, axis=1) <= 1.0
    pts_valid = grid_pts[valid_mask]
    
    # Evaluate Gibbs energies
    G_liq = interp_liq(pts_valid)
    G_fcc = interp_fcc(pts_valid)
    valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    
    pts = pts_valid[valid_eval]
    G_liq = G_liq[valid_eval]
    G_fcc = G_fcc[valid_eval]
    G_stable = np.minimum(G_liq, G_fcc)
    stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
    
    # Coordinate transform
    if coord_sys == "Spherical (r, θ, φ)":
        r_data, theta_data, phi_data = cartesian_to_spherical(pts[:, 0], pts[:, 1], pts[:, 2])
        x_data, y_data, z_data = r_data, theta_data, phi_data
        x_title, y_title, z_title = "r", "θ (rad)", "φ (rad)"
    else:
        x_data, y_data, z_data = pts[:, 0], pts[:, 1], pts[:, 2]
        x_title, y_title, z_title = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    # Marker sizing
    sizes = np.full(len(G_stable), marker_size)
    if scale_by_g:
        g_norm = np.abs(G_stable)
        g_min, g_max = g_norm.min(), g_max.max()
        if g_max > g_min:
            sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min)
    
    # ================= PLOTTING =================
    fig = go.Figure()
    
    def make_cbar(title_text):
        return dict(title=dict(text=title_text), thickness=20, len=0.7,
                   outlinecolor="black", outlinewidth=1)
    
    marker_cfg = dict(symbol=symbol, colorscale=cmap, opacity=opacity,
                     line=dict(width=1, color="#000000"))
    
    # Add phase traces
    if show_phase == "Stable Phase (Min G)":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode="markers",
            marker=dict(**marker_cfg, color=G_stable, size=sizes, colorbar=make_cbar("G (J/mol)")),
            name="Stable", hovertemplate=f"{x_title}=%{{x:.3f}}<br>{y_title}=%{{y:.3f}}<br>"
                                       f"{z_title}=%{{z:.3f}}<br>G=%{{marker.color:,.0f}}<br>"
                                       f"Phase=%{{text}}<extra></extra>",
            text=stable_label
        ))
    elif show_phase == "LIQUID Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode="markers",
            marker=dict(**marker_cfg, color=G_liq, size=sizes, colorbar=make_cbar("G_LIQ")),
            name="LIQUID"
        ))
    elif show_phase == "FCC Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode="markers",
            marker=dict(**marker_cfg, color=G_fcc, size=sizes, colorbar=make_cbar("G_FCC")),
            name="FCC"
        ))
    else:  # Both
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode="markers",
            marker=dict(**marker_cfg, color=G_liq, size=sizes, 
                       colorbar=make_cbar("G_LIQ"), opacity=opacity*0.7),
            name="LIQUID"
        ))
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode="markers",
            marker=dict(**marker_cfg, color=G_fcc, size=sizes,
                       colorbar=make_cbar("G_FCC"), opacity=opacity*0.7),
            name="FCC"
        ))
    
    # Query point overlay
    if query_result:
        if coord_sys == "Spherical (r, θ, φ)":
            x_q, y_q, z_q = cartesian_to_spherical(
                np.array([query_result["Co"]]),
                np.array([query_result["Cr"]]),
                np.array([query_result["Fe"]])
            )
        else:
            x_q, y_q, z_q = [query_result["Co"]], [query_result["Cr"]], [query_result["Fe"]]
        
        fig.add_trace(go.Scatter3d(
            x=x_q, y=y_q, z=z_q, mode="markers",
            marker=dict(size=14, color="red", symbol="diamond", line=dict(width=2, color="white")),
            name="Query",
            hovertemplate=f"<b>QUERY</b><br>T={query_result['T']} K<br>"
                         f"x_Co={query_result['Co']:.3f}<br>x_Cr={query_result['Cr']:.3f}<br>"
                         f"x_Fe={query_result['Fe']:.3f}<br>x_Ni={query_result['Ni']:.3f}<br>"
                         f"G={query_result['G_stable']:,.0f} J/mol<br>"
                         f"Phase={query_result['Phase']}<extra></extra>"
        ))
    
    # Reference shapes
    if show_ref_sphere:
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.05, colorscale=[[0,"gray"],[1,"gray"]],
                               showscale=False, hoverinfo="skip", name="Ref Sphere"))
    
    if show_axes:
        for ax, color, label in [([1.1,0,0], "red", "Co"), ([0,1.1,0], "green", "Cr"), ([0,0,1.1], "blue", "Fe")]:
            fig.add_trace(go.Scatter3d(x=[0,ax[0]], y=[0,ax[1]], z=[0,ax[2]],
                                      mode="lines+text", line=dict(color=color, width=3),
                                      text=["", label], textfont=dict(size=11, color=color),
                                      hoverinfo="skip", name=f"{label} axis"))
    
    if show_simplex:
        edges = [[(1,0,0),(0,1,0)], [(1,0,0),(0,0,1)], [(1,0,0),(0,0,0)],
                [(0,1,0),(0,0,1)], [(0,1,0),(0,0,0)], [(0,0,1),(0,0,0)]]
        ex, ey, ez = [], [], []
        for e in edges:
            ex += [e[0][0], e[1][0], None]
            ey += [e[0][1], e[1][1], None]
            ez += [e[0][2], e[1][2], None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                  line=dict(color="black", width=1.5, dash="dash"),
                                  name="Simplex", hoverinfo="skip"))
    
    # Layout
    axis_cfg = dict(showbackground=True, backgroundcolor="#ffffff" if template!="plotly_dark" else "#0e1117",
                   gridcolor="rgba(128,128,128,0.3)" if show_grid else "rgba(0,0,0,0)",
                   showgrid=show_grid, zerolinecolor="rgba(128,128,128,0.4)" if show_grid else "rgba(0,0,0,0)")
    
    fig.update_layout(
        template=template if template != "none" else None,
        scene=dict(xaxis=dict(title=x_title, **axis_cfg),
                  yaxis=dict(title=y_title, **axis_cfg),
                  zaxis=dict(title=z_title, **axis_cfg),
                  aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        title=dict(text=f"Gibbs Energy @ T={T_val} K | Points: {len(pts):,}", font=dict(size=16)),
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ================= SPHERICAL HARMONICS ANALYSIS =================
    if sh_enabled and interp_liq is not None:
        st.divider()
        st.subheader("🌀 Spherical Harmonics Analysis")
        
        # Prepare data for SH fit
        r_vals, theta_vals, phi_vals = cartesian_to_spherical(pts[:, 0], pts[:, 1], pts[:, 2])
        mask_r = np.abs(r_vals - sh_r_fixed) < 0.05
        pts_sh = pts[mask_r]
        G_sh_vals = G_stable[mask_r]
        theta_sh = theta_vals[mask_r]
        phi_sh = phi_vals[mask_r]
        
        if len(pts_sh) > (l_max + 1)**2:
            with st.spinner("Fitting spherical harmonics..."):
                coeffs, G_recon, rmse, cond_num = fit_spherical_harmonics(
                    G_sh_vals, theta_sh, phi_sh, l_max, reg_lambda=1e-5)
                
                # Interactive coefficient editing
                if enable_edit:
                    coeffs_display = coeffs.copy()
                    st.caption(f"✏️ Editing mode: coefficients scaled by {edit_factor}×")
                    
                    # Let user select which coefficients to edit
                    coeff_select = st.multiselect(
                        "Select coefficients to edit",
                        options=[f"Y_{l}^{m}" for l in range(l_max+1) for m in range(-l, l+1)],
                        default=[f"Y_{l}^{0}" for l in range(min(3, l_max+1))],
                        key="coeff_edit_select"
                    )
                    
                    # Apply edit factor to selected coefficients
                    idx = 0
                    for l in range(l_max + 1):
                        for m in range(-l, l + 1):
                            label = f"Y_{l}^{m}"
                            if label in coeff_select:
                                coeffs_display[idx] *= edit_factor
                            idx += 1
                    
                    # Regenerate surface with edited coefficients
                    X_sh, Y_sh, Z_sh, G_sh_edit, th_grid, ph_grid = generate_sh_surface(
                        coeffs_display, l_max, r_base=sh_r_fixed, r_scale=0.25)
                    
                    st.info(f"🔄 Surface updated with edited coefficients (RMSE: {rmse:.1f} J/mol)")
                else:
                    # Standard reconstruction
                    X_sh, Y_sh, Z_sh, G_sh_edit, th_grid, ph_grid = generate_sh_surface(
                        coeffs, l_max, r_base=sh_r_fixed, r_scale=0.25)
                
                # Display fit quality
                r2 = 1 - np.sum((G_sh_vals - G_recon)**2) / np.sum((G_sh_vals - np.mean(G_sh_vals))**2)
                st.success(f"✓ SH Fit: RMSE = {rmse:.1f} J/mol | R² = {r2:.4f} | Cond# = {cond_num:.1e}")
                
                # Visualization modes
                if sh_viz_mode == "Coefficients":
                    # Bar chart of coefficients
                    coeff_labels = [f"Y_{l}^{m}" for l in range(l_max+1) for m in range(-l, l+1)]
                    fig_coeffs = go.Figure(data=[go.Bar(
                        x=coeff_labels, y=coeffs_display if enable_edit else coeffs,
                        marker_color=coeffs_display if enable_edit else coeffs,
                        colorscale="RdBu", showscale=True
                    )])
                    fig_coeffs.update_layout(
                        title=f"SH Coefficients (l_max={l_max})" + (" ✏️ Edited" if enable_edit else ""),
                        xaxis_title="Spherical Harmonic Yₗₘ", yaxis_title="Coefficient (J/mol)",
                        xaxis_tickangle=-45, height=350, margin=dict(t=40, b=80)
                    )
                    st.plotly_chart(fig_coeffs, use_container_width=True)
                    
                    # Coefficient table
                    with st.expander("📋 Coefficient Table"):
                        coeff_df = pd.DataFrame({
                            'l': [l for l in range(l_max+1) for _ in range(2*l+1)],
                            'm': [m for l in range(l_max+1) for m in range(-l, l+1)],
                            'coefficient': coeffs_display if enable_edit else coeffs,
                            'label': coeff_labels
                        })
                        st.dataframe(coeff_df.style.format({'coefficient': '{:.3f}'}), height=300)
                
                elif sh_viz_mode == "Reconstructed Surface":
                    fig_sh = go.Figure(data=[go.Surface(
                        x=X_sh, y=Y_sh, z=Z_sh, surfacecolor=G_sh_edit,
                        colorscale=cmap, opacity=0.95, colorbar=dict(title="G (J/mol)")
                    )])
                    title_suffix = " ✏️ Edited" if enable_edit else ""
                    fig_sh.update_layout(
                        title=f"SH-Reconstructed G @ r={sh_r_fixed:.2f}{title_suffix}",
                        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig_sh, use_container_width=True)
                
                elif sh_viz_mode == "Error Map":
                    # Plot residuals on sphere
                    residual = G_sh_vals - G_recon
                    x_err, y_err, z_err = spherical_to_cartesian(
                        np.full_like(theta_sh, sh_r_fixed), theta_sh, phi_sh)
                    
                    fig_err = go.Figure(data=[go.Scatter3d(
                        x=x_err, y=y_err, z=z_err, mode='markers',
                        marker=dict(size=4, color=residual, colorscale="RdBu_r",
                                   colorbar=dict(title="Residual"), opacity=0.85)
                    )])
                    fig_err.update_layout(
                        title=f"Reconstruction Error @ r={sh_r_fixed:.2f}",
                        scene=dict(aspectmode="cube")
                    )
                    st.plotly_chart(fig_err, use_container_width=True)
                
                elif sh_viz_mode == "Gradient Field":
                    # Compute and visualize SH gradient (phase boundary indicator)
                    dG_dth, dG_dph = compute_sh_gradient(
                        coeffs_display if enable_edit else coeffs, l_max, theta_sh, phi_sh)
                    grad_mag = np.sqrt(dG_dth**2 + dG_dph**2)
                    
                    x_grad, y_grad, z_grad = spherical_to_cartesian(
                        np.full_like(theta_sh, sh_r_fixed), theta_sh, phi_sh)
                    
                    fig_grad = go.Figure(data=[go.Scatter3d(
                        x=x_grad, y=y_grad, z=z_grad, mode='markers',
                        marker=dict(size=5, color=grad_mag, colorscale="Viridis",
                                   colorbar=dict(title="|∇G|"), opacity=0.9)
                    )])
                    fig_grad.update_layout(
                        title=f"SH Gradient Magnitude |∇G| @ r={sh_r_fixed:.2f}",
                        scene=dict(aspectmode="cube")
                    )
                    st.plotly_chart(fig_grad, use_container_width=True)
                    st.caption("🔍 High gradient regions indicate potential phase boundaries")
                
                # Multi-temperature trends
                if multi_t and len(t_range) >= 2:
                    st.subheader("📈 Multi-Temperature SH Coefficient Trends")
                    
                    # Collect coefficients across temperatures
                    trend_data = []
                    for T_trend in t_range:
                        interp_liq_t, interp_fcc_t = build_interpolators_for_T(df, T_trend)
                        if interp_liq_t is None:
                            continue
                            
                        # Sample points at fixed radius
                        r_t, th_t, ph_t = cartesian_to_spherical(pts[:, 0], pts[:, 1], pts[:, 2])
                        mask_t = np.abs(r_t - sh_r_fixed) < 0.05
                        if np.sum(mask_t) < (l_max+1)**2:
                            continue
                            
                        G_t = G_stable[mask_t]
                        th_t = theta_vals[mask_t]
                        ph_t = phi_vals[mask_t]
                        
                        coeffs_t, _, _, _ = fit_spherical_harmonics(G_t, th_t, ph_t, l_max)
                        trend_data.append({'T': T_trend, 'coeffs': coeffs_t})
                    
                    if len(trend_data) >= 2:
                        # Plot coefficient evolution
                        for coeff_idx in range(min(6, len(coeffs))):  # Show first 6 coeffs
                            l = int(np.floor(np.sqrt(coeff_idx)))
                            m = coeff_idx - l*(l+1) - l
                            
                            fig_trend = go.Figure()
                            for td in trend_data:
                                fig_trend.add_trace(go.Scatter(
                                    x=[d['T'] for d in trend_data],
                                    y=[d['coeffs'][coeff_idx] for d in trend_data],
                                    mode='lines+markers', name=f"Y_{l}^{m}"
                                ))
                            fig_trend.update_layout(
                                title=f"Coefficient Y_{l}^{m} vs Temperature",
                                xaxis_title="Temperature (K)", yaxis_title="Coefficient (J/mol)",
                                height=300
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.warning("⚠️ Need ≥2 temperatures with sufficient data points for trends")
                
                # Export coefficients
                with st.expander("📥 Export SH Coefficients (e3nn-ready)"):
                    export_data = {
                        'l_max': l_max,
                        'r_reference': float(sh_r_fixed),
                        'temperature_K': int(T_val),
                        'coefficients': (coeffs_display if enable_edit else coeffs).tolist(),
                        'rmse_j_mol': float(rmse),
                        'r_squared': float(r2),
                        'n_data_points': int(len(pts_sh)),
                        'edited': enable_edit,
                        'edit_factor': edit_factor if enable_edit else None
                    }
                    st.json(export_data)
                    
                    coeff_df_export = pd.DataFrame({
                        'l': [l for l in range(l_max+1) for _ in range(2*l+1)],
                        'm': [m for l in range(l_max+1) for m in range(-l, l+1)],
                        'coefficient': coeffs_display if enable_edit else coeffs
                    })
                    st.download_button(
                        "📥 Download CSV",
                        data=coeff_df_export.to_csv(index=False),
                        file_name=f"SH_coeffs_T{T_val}K_r{sh_r_fixed:.2f}.csv",
                        mime="text/csv"
                    )
                    
                    st.download_button(
                        "📥 Download JSON (e3nn)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"SH_coeffs_T{T_val}K_r{sh_r_fixed:.2f}.json",
                        mime="application/json"
                    )
                
                # Phase boundary detection via SH gradient
                if detect_boundaries:
                    st.subheader("🔍 Phase Boundary Detection via SH Gradient")
                    
                    # Use gradient magnitude to identify boundaries
                    boundary_threshold = np.percentile(np.sqrt(dG_dth**2 + dG_dph**2), 85)
                    boundary_mask = np.sqrt(dG_dth**2 + dG_dph**2) > boundary_threshold
                    
                    if np.any(boundary_mask):
                        x_bnd, y_bnd, z_bnd = spherical_to_cartesian(
                            np.full_like(theta_sh[boundary_mask], sh_r_fixed),
                            theta_sh[boundary_mask], phi_sh[boundary_mask])
                        
                        fig_bnd = go.Figure()
                        # Original points
                        fig_bnd.add_trace(go.Scatter3d(
                            x=x_data, y=y_data, z=z_data, mode='markers',
                            marker=dict(size=2, color='lightgray', opacity=0.3),
                            name="All points", hoverinfo='skip'
                        ))
                        # Boundary points
                        fig_bnd.add_trace(go.Scatter3d(
                            x=x_bnd, y=y_bnd, z=z_bnd, mode='markers',
                            marker=dict(size=6, color='red', symbol='diamond'),
                            name="Phase Boundary",
                            hovertemplate="Boundary Point<br>θ=%{y:.3f}<br>φ=%{z:.3f}<extra></extra>"
                        ))
                        fig_bnd.update_layout(
                            title=f"Detected Phase Boundaries @ T={T_val} K, r={sh_r_fixed:.2f}",
                            scene=dict(aspectmode="cube")
                        )
                        st.plotly_chart(fig_bnd, use_container_width=True)
                        
                        st.caption(f"📍 {len(x_bnd)} boundary points detected (gradient threshold: {boundary_threshold:.2f})")
                        
                        # Export boundary coordinates
                        bnd_df = pd.DataFrame({
                            'x_Co': x_bnd, 'x_Cr': y_bnd, 'x_Fe': z_bnd,
                            'theta_rad': theta_sh[boundary_mask],
                            'phi_rad': phi_sh[boundary_mask]
                        })
                        st.download_button(
                            "📥 Download Boundary CSV",
                            data=bnd_df.to_csv(index=False),
                            file_name=f"phase_boundary_T{T_val}K_r{sh_r_fixed:.2f}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("ℹ️ No clear phase boundaries detected at this resolution. Try adjusting l_max or radius.")
        else:
            st.warning(f"⚠️ Need >{(l_max+1)**2} points at r≈{sh_r_fixed}; found {len(pts_sh)}. Try adjusting radius or grid resolution.")
    
    # ================= TENSOR & PHASE MATRIX EXPORT =================
    if export_tensor or export_phase_matrix:
        st.divider()
        st.subheader("📦 Tensor Calculations")
        
        with st.spinner("Computing full tensor G[x_Co, x_Cr, x_Fe, T]..."):
            tensor_result = compute_gibbs_tensor(df, T_list, comp_grid_res=15)
            
            if export_tensor:
                st.success("✅ Full Gibbs tensor computed")
                
                # Display tensor statistics
                G_stable_tensor = tensor_result['G_stable']
                valid_G = G_stable_tensor[~np.isnan(G_stable_tensor)]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min G", f"{np.nanmin(valid_G):,.0f}", "J/mol")
                c2.metric("Max G", f"{np.nanmax(valid_G):,.0f}", "J/mol")
                c3.metric("Mean |G|", f"{np.mean(np.abs(valid_G)):,.0f}", "J/mol")
                c4.metric("Valid Points", f"{len(valid_G):,}")
                
                # Download tensor
                tensor_dict = {
                    'G_LIQ': tensor_result['G_LIQ'].tolist(),
                    'G_FCC': tensor_result['G_FCC'].tolist(),
                    'G_stable': tensor_result['G_stable'].tolist(),
                    'composition_grid': [g.tolist() for g in tensor_result['composition_grid']],
                    'T_array': tensor_result['T_array'].tolist(),
                    'comp_vals': tensor_result['comp_vals'].tolist(),
                    'metadata': {
                        'shape': tensor_result['G_stable'].shape,
                        'units': 'J/mol',
                        'phases': ['LIQUID', 'FCC']
                    }
                }
                
                st.download_button(
                    "📥 Download Full Tensor (JSON)",
                    data=json.dumps(tensor_dict, indent=2),
                    file_name=f"Gibbs_tensor_CoCrFeNi.json",
                    mime="application/json"
                )
                st.caption("📦 Tensor shape: [n_T, n_Co, n_Cr, n_Fe] | Format: JSON with metadata")
            
            if export_phase_matrix:
                st.success("✅ Phase matrix computed")
                
                phase_mat = tensor_result['phase_matrix']
                Co_g, Cr_g, Fe_g = tensor_result['composition_grid']
                T_arr = tensor_result['T_array']
                
                # Display phase statistics for current T_val
                t_idx = np.argmin(np.abs(T_arr - T_val))
                phase_slice = phase_mat[t_idx]
                valid_phase = phase_slice[~np.isnan(phase_slice)]
                
                if len(valid_phase) > 0:
                    liq_pct = np.sum(valid_phase == 0) / len(valid_phase) * 100
                    fcc_pct = np.sum(valid_phase == 1) / len(valid_phase) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("LIQUID Region", f"{liq_pct:.1f}%")
                    c2.metric("FCC Region", f"{fcc_pct:.1f}%")
                    c3.metric("Boundary Points", f"{np.sum(np.abs(gaussian_gradient(phase_slice.astype(float))) > np.percentile(gaussian_gradient(phase_slice.astype(float)), 90)):,}")
                
                # 2D slice visualization
                fe_slice_idx = tensor_result['comp_vals'].size // 2
                phase_2d = phase_mat[t_idx, :, :, fe_slice_idx]
                co_vals = tensor_result['comp_vals']
                cr_vals = tensor_result['comp_vals']
                
                fig_phase = go.Figure(data=go.Heatmap(
                    z=phase_2d, x=co_vals, y=cr_vals,
                    colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
                    colorbar=dict(title="Phase<br>0=LIQ, 1=FCC", tickvals=[0, 1], ticktext=["LIQUID", "FCC"]),
                    showscale=True
                ))
                fig_phase.update_layout(
                    title=f"Phase Map @ T={T_arr[t_idx]} K, x_Fe≈{tensor_result['comp_vals'][fe_slice_idx]:.2f}",
                    xaxis_title="x_Co", yaxis_title="x_Cr",
                    width=500, height=450
                )
                st.plotly_chart(fig_phase)
                
                # Export phase matrix
                phase_export = {
                    'phase_matrix': phase_mat.tolist(),
                    'composition_grid': [g.tolist() for g in tensor_result['composition_grid']],
                    'T_array': T_arr.tolist(),
                    'comp_vals': tensor_result['comp_vals'].tolist(),
                    'phase_encoding': {0: 'LIQUID', 1: 'FCC'},
                    'metadata': {
                        'shape': phase_mat.shape,
                        'description': 'Phase stability: 0=LIQUID (G_LIQ <= G_FCC), 1=FCC'
                    }
                }
                
                st.download_button(
                    "📥 Download Phase Matrix (JSON)",
                    data=json.dumps(phase_export, indent=2),
                    file_name=f"phase_matrix_CoCrFeNi.json",
                    mime="application/json"
                )
                st.caption("📊 Phase encoding: 0 = LIQUID (stable), 1 = FCC (stable)")
    
    # ================= STATISTICS =================
    st.divider()
    st.subheader("📊 Current Grid Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min G (Stable)", f"{G_stable.min():,.0f} J/mol")
    c2.metric("Max G", f"{G_stable.max():,.0f} J/mol")
    c3.metric("Mean |G|", f"{np.mean(np.abs(G_stable)):,.0f} J/mol")
    
    if show_phase in ["Stable Phase (Min G)", "Both Overlay"]:
        liq_pct = np.sum(G_liq <= G_fcc) / len(G_liq) * 100
        c4.metric("LIQUID Fraction", f"{liq_pct:.1f}%")
    else:
        c4.metric("Points", f"{len(pts):,}")

if __name__ == "__main__":
    main()
