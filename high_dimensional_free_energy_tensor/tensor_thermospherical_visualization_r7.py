import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

# Try importing scipy.special, handle missing or deprecated
try:
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ `scipy` not available. Spherical Harmonic surface mode will be disabled. Install scipy with `pip install scipy` to enable it.")

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer - Physics Mode", layout="wide")

# =============================================
# VALIDATED COLORMAP LIBRARY (Plotly 6.x safe)
# =============================================
COLORMAPS = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu",
    "YlOrBr", "YlOrRd",
    "BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral",
    "Twilight", "HSV",
    "Jet", "Rainbow", "Hot", "Cool", "Blackbody", "Electric",
    "Plotly3", "Portland", "Picnic", "Solar", "Balance", "Delta", "Curl",
    "IceFire", "Edge", "Fall", "Sunset", "Sunsetdark", "Teal", "Tealgrn",
    "Tropic", "Peach", "Oxy", "Mint", "Emrld", "Aggrnyl", "Agsunset",
    "Armyrose", "Bluered", "Blugrn", "Bluyl", "Brwnyl", "Burg", "Burgyl",
    "Darkmint", "Geysr", "Magenta", "Mrybm", "Mygbm", "Oryel", "Pinkyl",
    "Purp", "Purpor", "Redor", "Ylorrd", "Ylorbr", "Ylgnbu", "Ylgn",
    "Haline", "Ice", "Matter", "Speed", "Tempo", "Thermal", "Turbid",
    "Algae", "Deep", "Dense", "Sinebow", "Phase"
]
COLORMAPS = sorted(list(set(COLORMAPS)))
SYMBOLS = ["circle", "diamond", "cross", "x", "star", "square", "pentagon", "hexagon"]

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
    """Convert spherical (r, theta, phi) back to Cartesian (Co, Cr, Fe)."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

# =============================================
# COLOR UTILITIES: HSV to RGB Conversion
# =============================================
def hsv_to_rgb(h, s, v):
    """Convert HSV values (arrays) to RGB tuples."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def hsv_to_rgb_vectorized(h, s, v):
    """Vectorized HSV to RGB conversion for arrays."""
    h = np.clip(h, 0, 1)
    s = np.clip(s, 0, 1)
    v = np.clip(v, 0, 1)
    
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    mask0 = (i == 0)
    mask1 = (i == 1)
    mask2 = (i == 2)
    mask3 = (i == 3)
    mask4 = (i == 4)
    mask5 = (i == 5)
    
    r[mask0] = v[mask0]; g[mask0] = t[mask0]; b[mask0] = p[mask0]
    r[mask1] = q[mask1]; g[mask1] = v[mask1]; b[mask1] = p[mask1]
    r[mask2] = p[mask2]; g[mask2] = v[mask2]; b[mask2] = t[mask2]
    r[mask3] = p[mask3]; g[mask3] = q[mask3]; b[mask3] = v[mask3]
    r[mask4] = t[mask4]; g[mask4] = p[mask4]; b[mask4] = v[mask4]
    r[mask5] = v[mask5]; g[mask5] = p[mask5]; b[mask5] = q[mask5]
    
    return np.stack([r, g, b], axis=-1)

# ================= DATA LOADING =================
@st.cache_data
def load_all_data(csv_dir=CSV_FILES_DIR):
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        st.error(f"No CSV files found in `{csv_dir}`.")
        st.stop()
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
    return pd.concat(dfs, ignore_index=True)

df = load_all_data()

# ================= GLOBAL TEMPERATURE STATISTICS =================
T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

# Global G ranges for consistent color scaling
G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

# Global ΔG statistics
df["dG"] = df["G_LIQ"] - df["G_FCC"]
dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))
dG_ref = 8.314 * T_max  # Reference: RT at highest T for normalization

# Temperature-dependent phase statistics
@st.cache_data
def compute_phase_statistics(df):
    stats = []
    for T in sorted(df["T"].unique()):
        df_T = df[df["T"] == T]
        liq_count = (df_T["dG"] <= 0).sum()
        total = len(df_T)
        liq_fraction = liq_count / total * 100 if total > 0 else 0
        avg_abs_dG = df_T["dG"].abs().mean()
        stats.append({
            "T": T,
            "LIQUID_fraction_%": liq_fraction,
            "avg_abs_dG_J_mol": avg_abs_dG,
            "total_points": total
        })
    return pd.DataFrame(stats)

phase_stats_df = compute_phase_statistics(df)

# ================= INTERPOLATION =================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None, None
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
    interp_stable = LinearNDInterpolator(pts, np.minimum(df_T["G_LIQ"], df_T["G_FCC"]).values)
    return interp_liq, interp_fcc, interp_stable

# Helper: interpolate G at arbitrary points from a DataFrame
def interp_G_at_pts(df_ref, pts, col="G_stable"):
    """Interpolate Gibbs energy at given composition points using LinearNDInterpolator."""
    if col not in df_ref.columns:
        if col == "G_stable":
            df_ref = df_ref.copy()
            df_ref["G_stable"] = np.minimum(df_ref["G_LIQ"], df_ref["G_FCC"])
        else:
            return np.full(len(pts), np.nan)
    
    pts_ref = df_ref[["Co", "Cr", "Fe"]].values
    vals_ref = df_ref[col].values
    interp = LinearNDInterpolator(pts_ref, vals_ref)
    return interp(pts)

# =============================================
# REAL SPHERICAL HARMONICS (compatible with old & new SciPy)
# =============================================
if SCIPY_AVAILABLE:
    def get_real_sph_harm(l, m, theta, phi):
        """
        Compute real spherical harmonic Y_lm (real) for given l,m and angles.
        theta: azimuthal angle (0 to 2π)
        phi:   polar angle (0 to π)
        """
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

    def sample_g_on_sphere(interp_liq, interp_fcc, R_fixed, n_theta=50, n_phi=50):
        """Evaluate stable G and dG on a sphere of fixed radius."""
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)
        x = R_fixed * np.sin(PH) * np.cos(TH)
        y = R_fixed * np.sin(PH) * np.sin(TH)
        z = R_fixed * np.cos(PH)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        valid = (pts[:,0] + pts[:,1] + pts[:,2]) <= 1.0

        G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
        G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)
        G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
        dG = G_liq - G_fcc

        valid = valid & ~np.isnan(G_stable)
        return TH, PH, G_stable.reshape(TH.shape), dG.reshape(TH.shape), valid.reshape(TH.shape)

    @st.cache_data(ttl=3600)
    def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=2):
        """Fit real spherical harmonic coefficients up to degree l_max."""
        theta_flat = theta_vals.ravel()
        phi_flat   = phi_vals.ravel()
        g_flat     = g_vals.ravel()
        valid = ~np.isnan(g_flat)
        theta_flat = theta_flat[valid]
        phi_flat   = phi_flat[valid]
        g_flat     = g_flat[valid]

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
        coeffs, _, _, _ = lstsq(A, g_flat)
        return coeffs, l_max

    def reconstruct_sh_surface(theta_grid, phi_grid, coeffs, l_max):
        """Reconstruct the function on a mesh using the fitted coefficients."""
        recon = np.zeros_like(theta_grid, dtype=float)
        idx = 0
        for l in range(l_max+1):
            for m in range(-l, l+1):
                Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
                recon += coeffs[idx] * Y
                idx += 1
        return recon

    # =============================================
    # PHYSICS-BASED: ENTROPY-DECOMPOSED SH MODEL
    # =============================================
    @st.cache_data(ttl=7200)
    def build_physics_sh_model(df, T_list, l_max_H=4, l_max_S=3, Tc=1200, delta_T=150):
        """
        Build entropy-decomposed spherical harmonic model.
        Fits H(x) and S(x) separately, then reconstructs G(x,T) = H(x) - T*S(x).
        """
        model = {}
        
        # Pre-sort temperatures for finite-difference derivatives
        T_sorted = sorted(T_list)
        
        for idx, T in enumerate(T_sorted):
            df_T = df[df["T"] == T].copy()
            if len(df_T) < 10:
                continue
                
            pts = df_T[["Co", "Cr", "Fe"]].values
            r, theta, phi = cartesian_to_spherical(pts[:,0], pts[:,1], pts[:,2])
            
            # Compute stable G at this T
            G_T = np.minimum(df_T["G_LIQ"].values, df_T["G_FCC"].values)
            
            # Estimate ∂G/∂T via central finite difference
            if idx == 0 or idx == len(T_sorted) - 1:
                # Forward/backward difference at endpoints
                if idx == 0 and len(T_sorted) > 1:
                    T_next = T_sorted[idx + 1]
                    df_next = df[df["T"] == T_next]
                    G_next = interp_G_at_pts(df_next, pts)
                    dG_dT = (G_next - G_T) / (T_next - T)
                elif idx == len(T_sorted) - 1 and len(T_sorted) > 1:
                    T_prev = T_sorted[idx - 1]
                    df_prev = df[df["T"] == T_prev]
                    G_prev = interp_G_at_pts(df_prev, pts)
                    dG_dT = (G_T - G_prev) / (T - T_prev)
                else:
                    continue
            else:
                # Central difference
                T_prev = T_sorted[idx - 1]
                T_next = T_sorted[idx + 1]
                df_prev = df[df["T"] == T_prev]
                df_next = df[df["T"] == T_next]
                
                G_prev = interp_G_at_pts(df_prev, pts)
                G_next = interp_G_at_pts(df_next, pts)
                
                dG_dT = (G_next - G_prev) / (T_next - T_prev)
            
            # Thermodynamic relations: S = -∂G/∂T, H = G + T*S
            S_vals = -dG_dT
            H_vals = G_T + T * S_vals
            
            # Clean NaN/inf values
            valid_mask = np.isfinite(H_vals) & np.isfinite(S_vals) & ~np.isnan(theta) & ~np.isnan(phi)
            if np.sum(valid_mask) < 20:
                continue
                
            theta_valid = theta[valid_mask]
            phi_valid = phi[valid_mask]
            H_valid = H_vals[valid_mask]
            S_valid = S_vals[valid_mask]
            
            # Fit spherical harmonics to H and S
            coeffs_H, l_max_H_used = fit_sh_coeffs(theta_valid, phi_valid, H_valid, l_max=l_max_H)
            coeffs_S, l_max_S_used = fit_sh_coeffs(theta_valid, phi_valid, S_valid, l_max=l_max_S)
            
            if coeffs_H is None or coeffs_S is None:
                continue
            
            model[T] = {
                "coeffs_H": coeffs_H,
                "coeffs_S": coeffs_S,
                "l_max_H": l_max_H_used,
                "l_max_S": l_max_S_used,
                "Tc": Tc,
                "delta_T": delta_T,
                "n_coeffs_H": len(coeffs_H),
                "n_coeffs_S": len(coeffs_S)
            }
        
        return model

    def interpolate_coeffs_across_T(model, T_target):
        """Linearly interpolate SH coefficients between neighboring temperatures."""
        T_keys = sorted(model.keys())
        
        if T_target <= T_keys[0]:
            return model[T_keys[0]]["coeffs_H"], model[T_keys[0]]["coeffs_S"]
        if T_target >= T_keys[-1]:
            return model[T_keys[-1]]["coeffs_H"], model[T_keys[-1]]["coeffs_S"]
        
        # Find bracketing temperatures
        for i in range(len(T_keys) - 1):
            if T_keys[i] <= T_target <= T_keys[i+1]:
                T_low, T_high = T_keys[i], T_keys[i+1]
                w = (T_target - T_low) / (T_high - T_low)
                
                coeffs_H_low = model[T_low]["coeffs_H"]
                coeffs_H_high = model[T_high]["coeffs_H"]
                coeffs_S_low = model[T_low]["coeffs_S"]
                coeffs_S_high = model[T_high]["coeffs_S"]
                
                # Pad to same length if needed
                max_len_H = max(len(coeffs_H_low), len(coeffs_H_high))
                max_len_S = max(len(coeffs_S_low), len(coeffs_S_high))
                
                coeffs_H_low = np.pad(coeffs_H_low, (0, max_len_H - len(coeffs_H_low)), 'constant')
                coeffs_H_high = np.pad(coeffs_H_high, (0, max_len_H - len(coeffs_H_high)), 'constant')
                coeffs_S_low = np.pad(coeffs_S_low, (0, max_len_S - len(coeffs_S_low)), 'constant')
                coeffs_S_high = np.pad(coeffs_S_high, (0, max_len_S - len(coeffs_S_high)), 'constant')
                
                coeffs_H_interp = (1-w) * coeffs_H_low + w * coeffs_H_high
                coeffs_S_interp = (1-w) * coeffs_S_low + w * coeffs_S_high
                
                return coeffs_H_interp, coeffs_S_interp
        
        return model[T_keys[0]]["coeffs_H"], model[T_keys[0]]["coeffs_S"]

    def temperature_weighted_coeffs(coeffs_raw, l_max, T, Tc, delta_T):
        """
        Apply Landau-type order parameter to suppress high-l terms at high T.
        η(T) = 1 / (1 + exp[(T - Tc)/ΔT])
        c_lm^eff = c_lm^0 * η^l
        """
        eta = 1.0 / (1.0 + np.exp((T - Tc) / delta_T))
        coeffs_weighted = np.zeros_like(coeffs_raw)
        idx = 0
        for l in range(l_max + 1):
            weight = eta ** l
            for m in range(-l, l + 1):
                if idx < len(coeffs_raw):
                    coeffs_weighted[idx] = coeffs_raw[idx] * weight
                idx += 1
        return coeffs_weighted, eta

    def reconstruct_G_physics(theta_grid, phi_grid, coeffs_H, coeffs_S, T, l_max):
        """
        Physics-based reconstruction: G(θ,φ,T) = H(θ,φ) - T·S(θ,φ)
        using spherical harmonic expansion.
        """
        G_recon = np.zeros_like(theta_grid)
        idx = 0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                if idx >= len(coeffs_H) or idx >= len(coeffs_S):
                    break
                Y_lm = get_real_sph_harm(l, m, theta_grid, phi_grid)
                # Thermodynamic superposition
                G_recon += (coeffs_H[idx] - T * coeffs_S[idx]) * Y_lm
                idx += 1
        return G_recon

    def compute_dG_surface_physics(model, T_render, theta_grid, phi_grid, l_max=4):
        """Compute ΔG = G_LIQ - G_FCC surface using physics-based reconstruction."""
        # For phase coloring, we need separate LIQ and FCC reconstructions
        # This is a simplified version: use stable G and sign from nearest data
        coeffs_H, coeffs_S = interpolate_coeffs_across_T(model, T_render)
        G_recon = reconstruct_G_physics(theta_grid, phi_grid, coeffs_H, coeffs_S, T_render, l_max)
        
        # Estimate dG sign from nearest temperature slice
        T_nearest = min(model.keys(), key=lambda t: abs(t - T_render))
        df_T = df[df["T"] == T_nearest]
        if len(df_T) == 0:
            return np.zeros_like(G_recon)
        
        # Sample dG at grid points for sign information
        pts_grid = np.column_stack([
            (np.sin(phi_grid) * np.cos(theta_grid)).ravel(),
            (np.sin(phi_grid) * np.sin(theta_grid)).ravel(),
            (np.cos(phi_grid)).ravel()
        ])
        valid_mask = np.sum(pts_grid**2, axis=1) > 1e-6
        pts_valid = pts_grid[valid_mask]
        
        # Interpolate dG from data
        dG_data = df_T["G_LIQ"].values - df_T["G_FCC"].values
        pts_data = df_T[["Co", "Cr", "Fe"]].values
        interp_dG = LinearNDInterpolator(pts_data, dG_data)
        dG_interp = interp_dG(pts_valid)
        
        # Create full dG field with sign from data, magnitude from reconstruction gradient
        dG_full = np.zeros(theta_grid.size)
        dG_full[valid_mask.ravel()] = dG_interp
        dG_full = dG_full.reshape(theta_grid.shape)
        
        return dG_full

    def compute_mean_curvature(G, theta, phi, R=1.0):
        """
        Approximate mean curvature of G(θ,φ) surface.
        Returns array of same shape as G.
        """
        # Finite-difference second derivatives
        dG_dtheta = np.gradient(G, axis=1)
        dG_dphi = np.gradient(G, axis=0)
        d2G_dtheta2 = np.gradient(dG_dtheta, axis=1)
        d2G_dphi2 = np.gradient(dG_dphi, axis=0)
        d2G_dthetadphi = np.gradient(dG_dtheta, axis=0)
        
        # Metric coefficients for spherical coordinates
        sin_phi = np.sin(phi)
        E = R**2 * sin_phi**2 + dG_dtheta**2
        F = dG_dtheta * dG_dphi
        G_met = R**2 + dG_dphi**2
        
        # Mean curvature formula (simplified for small gradients)
        denom = (1 + (dG_dtheta/(R*sin_phi))**2 + (dG_dphi/R)**2)**1.5
        H = 0.5 * (d2G_dtheta2/(R**2 * sin_phi**2) + d2G_dphi2/R**2) / denom
        
        return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

# ================= HEADER =================
st.title("🔷 Co-Cr-Fe-Ni Gibbs Energy Explorer: Physics-Based Mode")
st.markdown(r"""
This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface using **thermodynamically-consistent spherical harmonics**.

**Physics Model:** $G(\mathbf{x}, T) = H(\mathbf{x}) - T \cdot S(\mathbf{x})$  
**Order Parameter:** $\eta(T) = [1 + \exp((T-T_c)/\Delta T)]^{-1}$ modulates harmonic content  
**Visual Encoding:** Color saturation ∝ $|\Delta G|/RT$, surface roughness ∝ atomic mobility
""")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- Query Point ---
    st.subheader("📍 Query Point")
    q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")

    q_t = st.selectbox("Query T (K)", T_list, index=len(T_list)//2 if T_list else 0)

    comp_sum = q_co + q_cr + q_fe
    if comp_sum > 1.0:
        st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0 (x_Ni would be negative).")

    eval_query = st.button("🔍 Evaluate at Point", use_container_width=True)

    st.divider()

    # --- Spherical Harmonics Probe ---
    st.subheader("🔮 Spherical Harmonics Probe")
    show_sh_probe = st.toggle("Show SH Probe", value=True,
                              help="Display l=0 (scalar) and l=1 (vector) spherical harmonic glyphs at the query point")
    sh_probe_scale = st.slider("Probe Scale", 0.01, 0.5, 0.08, 0.01,
                               help="Radius of the spherical harmonic probe sphere")
    show_comp_vector = st.toggle("Show Composition Vector", value=True,
                                 help="l=1 arrow from origin to query point")

    st.divider()

    # --- Global Viz Params ---
    st.subheader("🌡️ Visualization Parameters")
    T_val = st.select_slider("Field T (K)", options=T_list,
                             value=T_list[len(T_list)//2] if T_list else 1000)
    grid_res = st.slider("Grid Resolution (for marker mode)", 15, 500, 25, step=5,
                         help="Higher = finer detail but slower rendering.")

    st.divider()

    # --- Coordinate System ---
    st.subheader("🌐 Coordinate System (marker mode)")
    coord_sys = st.radio("Select",
                         ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                         index=0,
                         help="Spherical: r=√(Co²+Cr²+Fe²), θ=atan2(Cr,Co), φ=acos(Fe/r)")

    st.divider()

    # --- Visualization Mode Selection ---
    st.subheader("🎨 Rendering Mode")
    if not SCIPY_AVAILABLE:
        st.error("⚠️ `scipy` not installed. Spherical Harmonic modes are disabled.")
        render_mode = "Markers (point cloud)"
        sh_disabled = True
        physics_disabled = True
    else:
        render_mode = st.radio("Visualization",
                               ["Markers (point cloud)", 
                                "Spherical Harmonic Surface (Empirical)", 
                                "Spherical Harmonic Surface (Physics-Based)"],
                               index=2,
                               help="Physics-Based: G=H-TS decomposition with Landau order parameter")
        sh_disabled = (render_mode == "Markers (point cloud)")
        physics_disabled = (render_mode != "Spherical Harmonic Surface (Physics-Based)")

    # --- Empirical SH Parameters (only if empirical mode selected) ---
    if render_mode == "Spherical Harmonic Surface (Empirical)" and not sh_disabled:
        st.subheader("🔧 Empirical SH Parameters")
        T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5
        sh_R_fixed = st.slider("Base sphere radius", 0.2, 0.9, 
                               float(0.35 + 0.25 * T_factor), 0.01)
        sh_alpha = st.slider("Radial distortion strength", 0.0, 0.8, 
                             float(0.1 + 0.4 * T_factor), 0.01)
        sh_l_max = st.slider("Max harmonic degree l", 0, 6, 
                             int(4 - 2 * T_factor))
        sh_n_theta = st.slider("Theta resolution", 20, 120, 60, step=10)
        sh_n_phi   = st.slider("Phi resolution", 20, 120, 60, step=10)
        surface_color_mode = st.radio("Color by",
                                      ["Stable G (global scale)", 
                                       "ΔG = G_LIQ - G_FCC (phase indicator)",
                                       "Temperature-encoded"],
                                      index=1)
        if surface_color_mode == "ΔG = G_LIQ - G_FCC (phase indicator)":
            cmap_sh = st.selectbox("Colormap", ["RdBu_r", "RdYlBu", "Spectral", "Balance", "Curl"], index=0)
        elif surface_color_mode == "Temperature-encoded":
            cmap_sh = st.selectbox("Colormap", ["Hot", "Thermal", "Inferno", "Magma"], index=0)
        else:
            cmap_sh = st.selectbox("Colormap", COLORMAPS, index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0)
        show_T_ghost = st.toggle("Show ghost surfaces at T±ΔT", value=False)
        if show_T_ghost:
            T_delta = st.slider("Temperature offset (K)", 50, 500, 200, 50)
            ghost_opacity = st.slider("Ghost opacity", 0.1, 0.5, 0.2, 0.05)

    # --- Physics-Based SH Parameters ---
    if render_mode == "Spherical Harmonic Surface (Physics-Based)" and not physics_disabled:
        st.subheader("⚛️ Physics Model Parameters")
        
        # Landau order parameter controls
        st.markdown("**Landau Order Parameter η(T)**")
        Tc_default = int(T_min + 0.6 * (T_max - T_min))  # Default: 60% through range
        Tc = st.slider("Critical temperature Tc (K)", T_min, T_max, Tc_default, 50,
                       help="Temperature where order parameter η = 0.5 (phase transition midpoint)")
        delta_T = st.slider("Transition width ΔT (K)", 50, 400, 150, 25,
                           help="Temperature range over which η transitions from 1→0")
        
        # Harmonic truncation limits
        st.markdown("**Harmonic Expansion Limits**")
        col1, col2 = st.columns(2)
        l_max_H = col1.slider("Max l for Enthalpy H", 0, 6, 4)
        l_max_S = col2.slider("Max l for Entropy S", 0, 6, 3)
        
        # Surface geometry controls
        st.markdown("**Surface Geometry**")
        base_radius = st.slider("Base radius (ordered state)", 0.2, 0.6, 0.35, 0.01,
                               help="Sphere radius when η→1 (low T, crystalline)")
        distortion_amp = st.slider("Distortion amplitude", 0.0, 0.5, 0.15, 0.01,
                                  help="How strongly G-variations deform the surface")
        
        # Visual encoding options
        st.markdown("**Visual Encoding**")
        phase_encoding = st.radio("Phase visualization",
                                  ["Color saturation by |ΔG|/RT", 
                                   "Curvature overlay for spinodal regions",
                                   "Combined: saturation + curvature"],
                                  index=0)
        show_mobility = st.toggle("Show atomic mobility (lighting)", value=True,
                                 help="Surface roughness ∝ exp(-Eₐ/RT)")
        
        # Resolution
        sh_n_theta = st.slider("Theta resolution", 30, 150, 80, step=10)
        sh_n_phi   = st.slider("Phi resolution", 30, 150, 80, step=10)
        
        # Build physics model (cached)
        with st.spinner("Building physics-based SH model..."):
            physics_model = build_physics_sh_model(df, T_list, 
                                                  l_max_H=l_max_H, 
                                                  l_max_S=l_max_S,
                                                  Tc=Tc, 
                                                  delta_T=delta_T)
        
        if len(physics_model) == 0:
            st.error("❌ Physics model construction failed. Try adjusting parameters or checking data.")
            physics_disabled = True
        else:
            st.success(f"✅ Physics model built: {len(physics_model)} temperature slices")
            with st.expander("📊 Model Diagnostics"):
                st.write(f"**Fitted temperatures:** {sorted(physics_model.keys())}")
                sample_T = sorted(physics_model.keys())[len(physics_model.keys())//2]
                st.write(f"**Sample coefficients at T={sample_T}K:**")
                st.write(f"- H coefficients: {physics_model[sample_T]['n_coeffs_H']}")
                st.write(f"- S coefficients: {physics_model[sample_T]['n_coeffs_S']}")

    # --- Marker-specific options ---
    if render_mode == "Markers (point cloud)":
        st.subheader("🎨 Marker Rendering")
        show_phase = st.radio("Phase Mode",
                              ["Stable Phase (Min G)", "LIQUID Only", "FCC Only", "Both Phases Overlay"],
                              index=0)
        cmap = st.selectbox("Colormap", COLORMAPS,
                            index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0)
        col1, col2 = st.columns(2)
        marker_size = col1.slider("Marker Size", 1, 10, 3)
        opacity = col2.slider("Opacity", 0.1, 1.0, 0.75, 0.05)
        col1, col2 = st.columns(2)
        marker_line_width = col1.slider("Outline Width", 0, 5, 0)
        marker_line_color = col2.color_picker("Outline Color", "#000000")
        symbol = st.selectbox("Marker Symbol", SYMBOLS, index=1)
        scale_size_by_g = st.toggle("Scale size by |G|", value=False)

    # --- Common visualization extras ---
    st.subheader("🔷 Geometric Aids")
    show_ref_sphere = st.toggle("Show Reference Sphere", value=False)
    ref_sphere_r = st.slider("Sphere Radius", 0.1, 1.5, 1.0, 0.05) if show_ref_sphere else 1.0
    show_axes_frame = st.toggle("Show Coordinate Axes", value=False)
    show_simplex = st.toggle("Show Composition Simplex", value=False)

    st.divider()

    # --- Layout & Typography ---
    st.subheader("✏️ Layout & Typography")
    template = st.selectbox("Template",
                            ["plotly_white", "plotly_dark", "plotly", "seaborn", "simple_white", "none"],
                            index=0)
    bg_color = st.color_picker("Background Color", "#ffffff")
    show_grid = st.toggle("Show Grid", value=True)
    col1, col2 = st.columns(2)
    tick_font = col1.slider("Tick Font", 8, 20, 12)
    axis_title_font = col2.slider("Axis Title Font", 10, 24, 14)
    title_font = st.slider("Chart Title Font", 12, 28, 16)
    st.markdown("**Colorbar**")
    cbar_title_txt = st.text_input("Title", "G (J/mol)")
    col1, col2 = st.columns(2)
    cbar_title_size = col1.slider("Title Font", 8, 20, 12)
    cbar_tick_size = col2.slider("Tick Font", 8, 20, 11)
    col1, col2 = st.columns(2)
    cbar_thickness = col1.slider("Thickness", 10, 50, 20)
    cbar_len = col2.slider("Length Fraction", 0.3, 1.0, 0.7, 0.05)
    col1, col2 = st.columns(2)
    cbar_xpad = col1.slider("X Pad (px)", 0, 50, 10)
    cbar_ypad = col2.slider("Y Pad (px)", 0, 50, 10)

    st.divider()
    st.caption(f"Loaded: {len(T_list)} temperatures ({T_min}-{T_max}K) | {len(df):,} total rows")

    with st.expander("📊 Phase Statistics by Temperature"):
        st.dataframe(phase_stats_df.style.format({
            "LIQUID_fraction_%": "{:.1f}",
            "avg_abs_dG_J_mol": "{:.0f}"
        }), use_container_width=True)

# ================= QUERY EVALUATION =================
query_result = None
if eval_query:
    if comp_sum > 1.0:
        st.error("Cannot evaluate: composition sum exceeds 1.0.")
    else:
        interp_liq_q, interp_fcc_q, interp_stable_q = build_interpolators_for_T(df, q_t)
        if interp_liq_q is not None:
            pt = np.array([[q_co, q_cr, q_fe]])
            g_liq_q = float(interp_liq_q(pt)[0])
            g_fcc_q = float(interp_fcc_q(pt)[0])

            if np.isnan(g_liq_q) or np.isnan(g_fcc_q):
                st.error("❌ Query point lies outside the convex hull of available data.")
            else:
                g_stable_q = min(g_liq_q, g_fcc_q)
                phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"
                dG_q = g_liq_q - g_fcc_q
                query_result = {
                    "T": q_t, "Co": q_co, "Cr": q_cr, "Fe": q_fe,
                    "Ni": 1.0 - q_co - q_cr - q_fe,
                    "G_LIQ": g_liq_q, "G_FCC": g_fcc_q,
                    "G_stable": g_stable_q, "Phase": phase_q,
                    "dG": dG_q
                }
        else:
            st.error(f"No interpolator available for T = {q_t} K.")

# ================= DISPLAY QUERY RESULTS =================
if query_result:
    st.success(f"✅ Query Result at T = {query_result['T']} K")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("G_stable", f"{query_result['G_stable']:,.0f}", "J/mol")
    c4.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol")
    c5.metric("Stable Phase", query_result['Phase'])
    c6.metric("x_Ni", f"{query_result['Ni']:.3f}")

    T_factor_q = (query_result['T'] - T_min) / T_range if T_range > 0 else 0.5
    if query_result['Phase'] == "LIQUID":
        st.info(f"🔥 High-temperature phase (LIQUID). T-factor: {T_factor_q:.2f}")
    else:
        st.info(f"❄️ Low-temperature phase (FCC). T-factor: {T_factor_q:.2f}")

    if T_val != query_result['T']:
        st.info(f"ℹ️ The 3D field is rendered at T = {T_val} K; query values are for T = {query_result['T']} K.")
    st.divider()

# ================= MAIN RENDERING =================
interp_liq, interp_fcc, interp_stable = build_interpolators_for_T(df, T_val)

if interp_liq is None:
    st.error(f"No data loaded for T = {T_val} K.")
    st.stop()

fig = go.Figure()

# Compute temperature factor
T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5

# ------------------------------------------------------------------
# MODE 3: PHYSICS-BASED SPHERICAL HARMONIC SURFACE
# ------------------------------------------------------------------
if render_mode == "Spherical Harmonic Surface (Physics-Based)" and SCIPY_AVAILABLE and not physics_disabled:

    def render_physics_sh_surface(T_render, opacity=0.9, is_ghost=False, ghost_label=""):
        """Render physics-based SH surface with thermodynamic coupling."""
        
        # Get interpolated coefficients
        coeffs_H, coeffs_S = interpolate_coeffs_across_T(physics_model, T_render)
        if coeffs_H is None or coeffs_S is None:
            return False
        
        # Get model parameters
        model_params = physics_model[list(physics_model.keys())[0]]
        l_max_H = model_params["l_max_H"]
        l_max_S = model_params["l_max_S"]
        l_max = max(l_max_H, l_max_S)
        Tc = model_params["Tc"]
        delta_T = model_params["delta_T"]
        
        # Apply Landau order parameter weighting
        coeffs_H_weighted, eta = temperature_weighted_coeffs(coeffs_H, l_max, T_render, Tc, delta_T)
        coeffs_S_weighted, _ = temperature_weighted_coeffs(coeffs_S, l_max, T_render, Tc, delta_T)
        
        # Create angular grid
        theta = np.linspace(0, 2*np.pi, sh_n_theta)
        phi = np.linspace(0, np.pi, sh_n_phi)
        TH, PH = np.meshgrid(theta, phi)
        
        # Reconstruct G via thermodynamic superposition: G = H - T*S
        G_recon = reconstruct_G_physics(TH, PH, coeffs_H_weighted, coeffs_S_weighted, T_render, l_max)
        
        # Compute dG for phase coloring
        dG_recon = compute_dG_surface_physics(physics_model, T_render, TH, PH, l_max)
        
        # Normalize G for radial deformation
        G_min = np.nanmin(G_recon)
        G_max = np.nanmax(G_recon)
        if G_max > G_min:
            G_norm = (G_recon - G_min) / (G_max - G_min)
        else:
            G_norm = np.zeros_like(G_recon)
        
        # Temperature-dependent geometry
        # Base radius contracts when ordered (low T), expands when disordered (high T)
        radius_base = base_radius + 0.2 * (1 - eta)  # Thermal expansion proxy
        radius = radius_base + distortion_amp * eta * G_norm
        
        # Convert to Cartesian coordinates
        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)
        
        # Visual encoding based on physics
        if phase_encoding == "Color saturation by |ΔG|/RT":
            # HSV: hue=phase, saturation=|ΔG|/RT, value=temperature brightness
            RT = 8.314 * T_render
            driving_force = np.clip(np.abs(dG_recon) / RT, 0, 2) / 2
            
            hue = np.where(dG_recon < 0, 0.0, 0.67)  # Red=LIQUID (dG<0), Blue=FCC (dG>0)
            saturation = driving_force
            value = 0.7 + 0.3 * (T_render - T_min) / (T_max - T_min)
            
            # Convert to RGB
            surfacecolor_rgb = hsv_to_rgb_vectorized(hue, saturation, value)
            color_mode = "rgb"
            cbar_title = "|ΔG|/RT (phase driving force)"
            cmin, cmax = 0, 1
            
        elif phase_encoding == "Curvature overlay for spinodal regions":
            # Compute mean curvature to highlight instability regions
            curvature = compute_mean_curvature(G_recon, TH, PH)
            curvature_abs = np.abs(curvature)
            
            # Normalize curvature for coloring
            curv_min = np.nanpercentile(curvature_abs, 5)
            curv_max = np.nanpercentile(curvature_abs, 95)
            curv_norm = np.clip((curvature_abs - curv_min) / (curv_max - curv_min + 1e-12), 0, 1)
            
            # Color by curvature magnitude (red = high curvature = near spinodal)
            surfacecolor_rgb = hsv_to_rgb_vectorized(
                h=np.zeros_like(curv_norm),  # Red hue
                s=curv_norm,                  # Saturation = curvature
                v=np.ones_like(curv_norm)     # Full brightness
            )
            color_mode = "rgb"
            cbar_title = "|Mean Curvature| (instability indicator)"
            cmin, cmax = 0, 1
            
        else:  # Combined: saturation + curvature
            RT = 8.314 * T_render
            driving_force = np.clip(np.abs(dG_recon) / RT, 0, 2) / 2
            curvature = compute_mean_curvature(G_recon, TH, PH)
            curvature_abs = np.abs(curvature)
            curv_min = np.nanpercentile(curvature_abs, 5)
            curv_max = np.nanpercentile(curvature_abs, 95)
            curv_norm = np.clip((curvature_abs - curv_min) / (curv_max - curv_min + 1e-12), 0, 1)
            
            # Combined encoding: hue=phase, saturation=driving_force * (1 + curvature)
            hue = np.where(dG_recon < 0, 0.0, 0.67)
            saturation = np.clip(driving_force * (1 + 0.5 * curv_norm), 0, 1)
            value = 0.7 + 0.3 * (T_render - T_min) / (T_max - T_min)
            
            surfacecolor_rgb = hsv_to_rgb_vectorized(hue, saturation, value)
            color_mode = "rgb"
            cbar_title = "Combined: |ΔG|/RT × (1+|κ|)"
            cmin, cmax = 0, 1
        
        # Temperature-dependent rendering properties
        if show_mobility:
            # Atomic mobility proxy: Arrhenius behavior
            E_a = 150e3  # Activation energy ~150 kJ/mol (typical for metallic diffusion)
            R_gas = 8.314
            mobility = np.exp(-E_a / (R_gas * T_render))
            
            # Map mobility to surface properties
            surface_roughness = 0.2 + 0.7 * (1 - mobility)  # Low T: smooth/specular
            surface_diffuse = 0.3 if mobility < 0.1 else 0.6  # High T: more diffuse
            surface_opacity = opacity * (0.7 + 0.3 * mobility)  # More transparent when mobile
        else:
            surface_roughness = 0.4
            surface_diffuse = 0.5
            surface_opacity = opacity
        
        # Surface name and colorbar settings
        if is_ghost:
            surface_name = f"T={T_render}K {ghost_label}"
            show_scale = False
            surface_opacity *= 0.5
        else:
            surface_name = f"Physics-SH: T={T_render}K, η={eta:.2f}"
            show_scale = True
        
        # Add surface trace
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surfacecolor_rgb if color_mode=="rgb" else G_recon,
            colorscale=cmap_sh if color_mode=="scale" else None,
            cmin=cmin if color_mode=="rgb" else G_global_min,
            cmax=cmax if color_mode=="rgb" else G_global_max,
            opacity=surface_opacity,
            name=surface_name,
            showscale=show_scale,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(size=cbar_title_size)),
                thickness=cbar_thickness,
                len=cbar_len,
                tickfont=dict(size=cbar_tick_size),
                xpad=cbar_xpad,
                ypad=cbar_ypad
            ) if show_scale else None,
            hovertemplate=(
                f"<b>{surface_name}</b><br>"
                f"G = %{{z:.1f}} (radius)<br>"
                f"η = {eta:.2f}<br>"
                f"<extra></extra>"
            ) if not is_ghost else None,
            lighting=dict(
                ambient=0.5,
                diffuse=surface_diffuse,
                roughness=surface_roughness,
                specular=0.3 if eta > 0.5 else 0.1,
                fresnel=1.0
            ),
            lightposition=dict(x=100, y=100, z=50)
        ))
        return True

    # Render main surface
    success = render_physics_sh_surface(T_val, opacity=0.92)

    # Render ghost surfaces if enabled (for empirical mode compatibility)
    # In physics mode, we could show T±ΔT for comparison
    if 'show_T_ghost' in locals() and show_T_ghost and success:
        T_ghost_list = []
        if T_val - T_delta >= T_min:
            T_ghost_list.append((T_val - T_delta, "(cold)"))
        if T_val + T_delta <= T_max:
            T_ghost_list.append((T_val + T_delta, "(hot)"))

        for T_ghost, label in T_ghost_list:
            render_physics_sh_surface(T_ghost, opacity=0.3, is_ghost=True, ghost_label=label)

    if not success:
        st.warning("Physics-based SH rendering failed. Try adjusting harmonic limits or temperature parameters.")
        
        # Fallback: show empirical mode parameters as hint
        st.info("💡 Tip: Ensure your CSV data spans sufficient temperature range for derivative estimation.")

# ------------------------------------------------------------------
# MODE 2: EMPIRICAL SPHERICAL HARMONIC SURFACE
# ------------------------------------------------------------------
elif render_mode == "Spherical Harmonic Surface (Empirical)" and SCIPY_AVAILABLE and not sh_disabled:

    def render_sh_surface(T_render, opacity=0.9, is_ghost=False, ghost_label=""):
        """Render empirical SH surface (original implementation)."""
        interp_liq_T, interp_fcc_T, _ = build_interpolators_for_T(df, T_render)
        if interp_liq_T is None:
            return False

        TH, PH, G_vals, dG_vals, valid_mask = sample_g_on_sphere(
            interp_liq_T, interp_fcc_T, sh_R_fixed,
            n_theta=sh_n_theta, n_phi=sh_n_phi
        )

        if not np.any(valid_mask):
            return False

        coeffs, l_max_used = fit_sh_coeffs(TH, PH, G_vals, l_max=sh_l_max)
        if coeffs is None:
            return False

        G_smooth = reconstruct_sh_surface(TH, PH, coeffs, l_max_used)

        T_local_factor = (T_render - T_min) / T_range if T_range > 0 else 0.5
        alpha_effective = sh_alpha * (1.0 + 0.5 * np.sin(np.pi * T_local_factor))

        G_min = G_smooth.min()
        G_max = G_smooth.max()
        if G_max > G_min:
            G_norm = (G_smooth - G_min) / (G_max - G_min)
        else:
            G_norm = np.zeros_like(G_smooth)

        radius = sh_R_fixed + alpha_effective * G_norm
        thermal_expansion = 1.0 + 0.1 * T_local_factor
        radius *= thermal_expansion

        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)

        if surface_color_mode == "ΔG = G_LIQ - G_FCC (phase indicator)":
            coeffs_dG, _ = fit_sh_coeffs(TH, PH, dG_vals, l_max=sh_l_max)
            if coeffs_dG is not None:
                surfacecolor = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max_used)
                surfacecolor = np.clip(surfacecolor, -dG_global_abs_max, dG_global_abs_max)
                cbar_title = "ΔG = G_LIQ - G_FCC (J/mol)"
                cmin = -dG_global_abs_max
                cmax = dG_global_abs_max
            else:
                surfacecolor = G_smooth
                cbar_title = cbar_title_txt
                cmin = G_global_min
                cmax = G_global_max
        elif surface_color_mode == "Temperature-encoded":
            surfacecolor = np.full_like(G_smooth, T_render)
            cbar_title = "Temperature (K)"
            cmin = T_min
            cmax = T_max
        else:
            surfacecolor = G_smooth
            cbar_title = cbar_title_txt
            cmin = G_global_min
            cmax = G_global_max

        if is_ghost:
            surface_name = f"T={T_render}K {ghost_label}"
            surface_opacity = ghost_opacity
            show_scale = False
        else:
            surface_name = f"SH surface (T={T_render}K)"
            surface_opacity = opacity
            show_scale = True

        if T_local_factor < 0.3:
            surface_roughness = 0.3
            lighting_effect = "specular"
        elif T_local_factor > 0.7:
            surface_roughness = 0.8
            lighting_effect = "diffuse"
        else:
            surface_roughness = 0.5
            lighting_effect = "flat"

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surfacecolor,
            colorscale=cmap_sh,
            cmin=cmin,
            cmax=cmax,
            opacity=surface_opacity,
            name=surface_name,
            showscale=show_scale,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(size=cbar_title_size)),
                thickness=cbar_thickness,
                len=cbar_len,
                tickfont=dict(size=cbar_tick_size),
                xpad=cbar_xpad,
                ypad=cbar_ypad
            ) if show_scale else None,
            hovertemplate=(
                f"<b>{surface_name}</b><br>"
                f"G_stable = %{{surfacecolor:,.0f}} J/mol<br>"
                f"T = {T_render} K<br>"
                f"<extra></extra>"
            ) if not is_ghost else None,
            lighting=dict(ambient=0.6, diffuse=0.4, roughness=surface_roughness),
            lightposition=dict(x=100, y=100, z=50)
        ))
        return True

    success = render_sh_surface(T_val, opacity=0.9)

    if 'show_T_ghost' in locals() and show_T_ghost and success:
        T_ghost_list = []
        if T_val - T_delta >= T_min:
            T_ghost_list.append((T_val - T_delta, "(cold)"))
        if T_val + T_delta <= T_max:
            T_ghost_list.append((T_val + T_delta, "(hot)"))

        for T_ghost, label in T_ghost_list:
            render_sh_surface(T_ghost, opacity=ghost_opacity, is_ghost=True, ghost_label=label)

    if not success:
        st.warning("Spherical harmonic fitting failed. Try adjusting parameters.")

# ------------------------------------------------------------------
# MODE 1: MARKER POINT CLOUD
# ------------------------------------------------------------------
else:
    # Generate tetrahedral grid
    x = np.linspace(0, 1, grid_res)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    pts_valid = grid_pts[valid_mask]

    G_liq = interp_liq(pts_valid)
    G_fcc = interp_fcc(pts_valid)
    valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    pts = pts_valid[valid_eval]
    G_liq = G_liq[valid_eval]
    G_fcc = G_fcc[valid_eval]
    G_stable = np.minimum(G_liq, G_fcc)
    stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
    dG = G_liq - G_fcc

    if coord_sys == "Spherical (r, θ, φ)":
        r, theta, phi = cartesian_to_spherical(pts[:, 0], pts[:, 1], pts[:, 2])
        x_data, y_data, z_data = r, theta, phi
        x_title, y_title, z_title = "r", "θ (rad)", "φ (rad)"
        if query_result:
            x_q, y_q, z_q = cartesian_to_spherical(
                np.array([query_result["Co"]]),
                np.array([query_result["Cr"]]),
                np.array([query_result["Fe"]])
            )
        else:
            x_q = y_q = z_q = np.array([np.nan])
    else:
        x_data, y_data, z_data = pts[:, 0], pts[:, 1], pts[:, 2]
        x_title, y_title, z_title = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
        if query_result:
            x_q = np.array([query_result["Co"]])
            y_q = np.array([query_result["Cr"]])
            z_q = np.array([query_result["Fe"]])
        else:
            x_q = y_q = z_q = np.array([np.nan])

    if scale_size_by_g:
        g_norm = np.abs(G_stable)
        g_min, g_max = g_norm.min(), g_norm.max()
        if g_max > g_min:
            sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min)
        else:
            sizes = np.full_like(g_norm, marker_size)
    else:
        sizes = np.full(len(G_stable), marker_size)

    def make_marker_config(color_data, size_data, cbar_title):
        return dict(
            size=size_data,
            color=color_data,
            colorscale=cmap,
            opacity=opacity,
            symbol=symbol,
            line=dict(width=marker_line_width, color=marker_line_color),
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(size=cbar_title_size)),
                thickness=cbar_thickness,
                len=cbar_len,
                tickfont=dict(size=cbar_tick_size),
                xpad=cbar_xpad,
                ypad=cbar_ypad
            )
        )

    if show_phase == "Stable Phase (Min G)":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_stable, sizes, cbar_title_txt),
            name="Stable Phase",
            hovertemplate=(f"<b>Stable</b><br>{x_title}=%{{x:.3f}}<br>{y_title}=%{{y:.3f}}<br>"
                           f"{z_title}=%{{z:.3f}}<br>G=%{{marker.color:,.0f}} J/mol<br>"
                           f"Phase=%{{text}}<extra></extra>"),
            text=stable_label
        ))
    elif show_phase == "LIQUID Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_liq, sizes, cbar_title_txt),
            name="LIQUID",
            hovertemplate=(f"<b>LIQUID</b><br>{x_title}=%{{x:.3f}}<br>{y_title}=%{{y:.3f}}<br>"
                           f"{z_title}=%{{z:.3f}}<br>G_LIQ=%{{marker.color:,.0f}} J/mol<extra></extra>")
        ))
    elif show_phase == "FCC Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_fcc, sizes, cbar_title_txt),
            name="FCC",
            hovertemplate=(f"<b>FCC</b><br>{x_title}=%{{x:.3f}}<br>{y_title}=%{{y:.3f}}<br>"
                           f"{z_title}=%{{z:.3f}}<br>G_FCC=%{{marker.color:,.0f}} J/mol<extra></extra>")
        ))
    else:  # Both Phases Overlay
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_liq, sizes, cbar_title_txt.replace("G", "G_LIQ")),
            name="LIQUID"
        ))
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_fcc, sizes, cbar_title_txt.replace("G", "G_FCC")),
            name="FCC"
        ))

# ------------------------------------------------------------------
# COMMON ELEMENTS (Query point, composition vector, SH probe, geometric aids)
# ------------------------------------------------------------------

# Query point overlay
if query_result is not None and not np.isnan(x_q[0]):
    fig.add_trace(go.Scatter3d(
        x=x_q, y=y_q, z=z_q,
        mode="markers",
        marker=dict(size=14, color="red", symbol="diamond",
                    line=dict(width=2, color="white")),
        name="Query Point",
        hovertemplate=(f"<b>QUERY POINT</b><br>T={query_result['T']} K<br>"
                       f"x_Co={query_result['Co']:.3f}<br>x_Cr={query_result['Cr']:.3f}<br>"
                       f"x_Fe={query_result['Fe']:.3f}<br>x_Ni={query_result['Ni']:.3f}<br>"
                       f"G_stable={query_result['G_stable']:,.0f} J/mol<br>"
                       f"ΔG={query_result['dG']:,.0f} J/mol<br>"
                       f"Phase={query_result['Phase']}<extra></extra>")
    ))

# Spherical harmonics probe with temperature-dependent styling
if query_result is not None and show_sh_probe:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    g_val = abs(query_result["G_stable"])
    g_max_all = max(abs(df["G_LIQ"]).max(), abs(df["G_FCC"]).max())

    T_query_factor = (query_result['T'] - T_min) / T_range if T_range > 0 else 0.5
    radius = sh_probe_scale * (0.5 + 0.5 * g_val / g_max_all) * (1.0 + 0.3 * T_query_factor) if g_max_all > 0 else sh_probe_scale

    if query_result["Phase"] == "LIQUID":
        base_color = "#e74c3c"
        phase_color = f"rgb(255, {int(100 * (1 - T_query_factor))}, {int(100 * (1 - T_query_factor))})"
    else:
        base_color = "#3498db"
        phase_color = f"rgb({int(100 * T_query_factor)}, {int(150 * T_query_factor)}, 255)"

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sh = qx + radius * np.outer(np.cos(u), np.sin(v))
    y_sh = qy + radius * np.outer(np.sin(u), np.sin(v))
    z_sh = qz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sh, y=y_sh, z=z_sh,
        opacity=0.25 + 0.15 * T_query_factor,
        colorscale=[[0, phase_color], [1, phase_color]],
        showscale=False,
        name=f"SH l=0 ({query_result['Phase']}, T={query_result['T']}K)",
        hovertemplate=(f"<b>Spherical Harmonic l=0</b><br>"
                       f"G_stable={query_result['G_stable']:,.0f} J/mol<br>"
                       f"Phase={query_result['Phase']}<br>"
                       f"T={query_result['T']} K<br>"
                       f"Radius={radius:.4f}<extra></extra>")
    ))

    n_probe = 12
    theta_p = np.linspace(0, 2*np.pi, n_probe, endpoint=False)
    phi_p = np.linspace(0, np.pi, n_probe)
    theta_p, phi_p = np.meshgrid(theta_p, phi_p)
    theta_p, phi_p = theta_p.ravel(), phi_p.ravel()
    x_p = qx + radius * np.sin(phi_p) * np.cos(theta_p)
    y_p = qy + radius * np.sin(phi_p) * np.sin(theta_p)
    z_p = qz + radius * np.cos(phi_p)
    fig.add_trace(go.Scatter3d(
        x=x_p, y=y_p, z=z_p,
        mode="markers",
        marker=dict(size=5, color=phase_color, symbol="circle",
                    line=dict(width=1, color="white"), opacity=0.8),
        name="SH Probe Grid",
        hoverinfo="skip"
    ))

# Composition vector (l=1 arrow)
if query_result is not None and show_comp_vector:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    fig.add_trace(go.Scatter3d(
        x=[0, qx], y=[0, qy], z=[0, qz],
        mode="lines+text",
        line=dict(color="gold", width=5),
        text=["", "l=1"],
        textposition="top center",
        textfont=dict(size=14, color="gold"),
        name="Composition Vector (l=1)",
        hovertemplate=(f"<b>l=1 Composition Vector</b><br>"
                       f"Co={qx:.3f}<br>Cr={qy:.3f}<br>Fe={qz:.3f}<extra></extra>")
    ))
    fig.add_trace(go.Scatter3d(
        x=[qx], y=[qy], z=[qz],
        mode="markers",
        marker=dict(size=8, color="gold", symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Vector Head",
        hoverinfo="skip"
    ))

# Reference sphere
if show_ref_sphere:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sph = ref_sphere_r * np.outer(np.cos(u), np.sin(v))
    y_sph = ref_sphere_r * np.outer(np.sin(u), np.sin(v))
    z_sph = ref_sphere_r * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=x_sph, y=y_sph, z=z_sph,
        opacity=0.08,
        colorscale=[[0, "gray"], [1, "gray"]],
        showscale=False,
        name="Ref Sphere",
        hoverinfo="skip"
    ))

# Coordinate axes
if show_axes_frame:
    axis_len = 1.1
    fig.add_trace(go.Scatter3d(
        x=[0, axis_len], y=[0, 0], z=[0, 0],
        mode="lines+text",
        line=dict(color="red", width=4),
        text=["", "Co"],
        textposition="top center",
        textfont=dict(size=12, color="red"),
        name="Co axis",
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_len], z=[0, 0],
        mode="lines+text",
        line=dict(color="green", width=4),
        text=["", "Cr"],
        textposition="top center",
        textfont=dict(size=12, color="green"),
        name="Cr axis",
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_len],
        mode="lines+text",
        line=dict(color="blue", width=4),
        text=["", "Fe"],
        textposition="top center",
        textfont=dict(size=12, color="blue"),
        name="Fe axis",
        hoverinfo="skip"
    ))

# Composition simplex wireframe
if show_simplex:
    simplex_edges = [
        [(1,0,0), (0,1,0)], [(1,0,0), (0,0,1)], [(1,0,0), (0,0,0)],
        [(0,1,0), (0,0,1)], [(0,1,0), (0,0,0)], [(0,0,1), (0,0,0)]
    ]
    edge_x, edge_y, edge_z = [], [], []
    for e in simplex_edges:
        edge_x += [e[0][0], e[1][0], None]
        edge_y += [e[0][1], e[1][1], None]
        edge_z += [e[0][2], e[1][2], None]
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Simplex Boundary",
        hoverinfo="skip"
    ))

# ------------------------------------------------------------------
# FIGURE LAYOUT
# ------------------------------------------------------------------
def make_axis(title_text):
    return dict(
        title=dict(text=title_text, font=dict(size=axis_title_font)),
        tickfont=dict(size=tick_font),
        showbackground=True,
        backgroundcolor=bg_color,
        gridcolor="rgba(128,128,128,0.3)" if show_grid else "rgba(0,0,0,0)",
        showgrid=show_grid,
        zerolinecolor="rgba(128,128,128,0.5)" if show_grid else "rgba(0,0,0,0)",
        zerolinewidth=1 if show_grid else 0
    )

if render_mode == "Markers (point cloud)":
    scene_x_title = x_title
    scene_y_title = y_title
    scene_z_title = z_title
else:
    scene_x_title = "x<sub>Co</sub>"
    scene_y_title = "x<sub>Cr</sub>"
    scene_z_title = "x<sub>Fe</sub>"

# Temperature-dependent title with physics context
T_factor_display = (T_val - T_min) / T_range if T_range > 0 else 0.5

if render_mode == "Spherical Harmonic Surface (Physics-Based)" and not physics_disabled:
    # Compute order parameter for title
    model_params = physics_model[list(physics_model.keys())[0]]
    Tc = model_params["Tc"]
    delta_T = model_params["delta_T"]
    eta_display = 1.0 / (1.0 + np.exp((T_val - Tc) / delta_T))
    
    if eta_display > 0.7:
        phase_char = "Crystalline FCC (ordered)"
    elif eta_display < 0.3:
        phase_char = "Isotropic LIQUID (disordered)"
    else:
        phase_char = "Phase transition region"
    
    title_text = f"Physics-SH: G=H-TS at T={T_val}K | η={eta_display:.2f} | {phase_char}"
else:
    phase_dominant = "LIQUID-dominated" if T_factor_display > 0.6 else "FCC-dominated" if T_factor_display < 0.4 else "Transition"
    title_text = f"Gibbs Energy at T = {T_val} K ({phase_dominant}) | Mode: {render_mode}"

fig.update_layout(
    template=template if template != "none" else None,
    scene=dict(
        xaxis=make_axis(scene_x_title),
        yaxis=make_axis(scene_y_title),
        zaxis=make_axis(scene_z_title),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    title=dict(text=title_text, font=dict(size=title_font)),
    margin=dict(l=0, r=0, b=0, t=50),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)")
)

try:
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Plot rendering error: {e}")
    st.info("Try selecting a different colormap or reducing grid resolution.")

# ================= STATISTICS =================
if render_mode == "Markers (point cloud)":
    st.subheader("📊 Phase Statistics at Current Grid")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min G (Stable)", f"{G_stable.min():,.0f} J/mol")
    col2.metric("Max G", f"{G_stable.max():,.0f} J/mol")
    col3.metric("Mean |G|", f"{np.mean(np.abs(G_stable)):,.0f} J/mol")
    if show_phase in ["Stable Phase (Min G)", "Both Phases Overlay"]:
        liq_pct = np.sum(G_liq <= G_fcc) / len(G_liq) * 100
        col4.metric("LIQUID Region", f"{liq_pct:.1f}%")
    else:
        col4.metric("Points Rendered", f"{len(pts):,}")

elif render_mode == "Spherical Harmonic Surface (Empirical)":
    st.subheader("📊 Empirical SH Surface Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{T_val} K")
    col2.metric("T-factor", f"{T_factor:.2f}")
    col3.metric("Distortion α", f"{sh_alpha:.2f}")
    col4.metric("Harmonic l_max", f"{sh_l_max}")

else:  # Physics-based mode
    st.subheader("⚛️ Physics-Based SH Model Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Compute order parameter
    model_params = physics_model[list(physics_model.keys())[0]]
    Tc = model_params["Tc"]
    delta_T = model_params["delta_T"]
    eta = 1.0 / (1.0 + np.exp((T_val - Tc) / delta_T))
    
    col1.metric("Temperature", f"{T_val} K")
    col2.metric("Order parameter η", f"{eta:.3f}")
    col3.metric("Effective l_max", f"{int(eta * max(l_max_H, l_max_S))}")
    col4.metric("Model slices", f"{len(physics_model)}")
    
    # Physics interpretation panel
    with st.expander("📖 Physics Interpretation"):
        st.markdown(f"""
        **Thermodynamic Decomposition at T = {T_val} K:**
        
        - **Order parameter η = {eta:.3f}**: 
          - η → 1: Crystalline order (FCC), sharp angular features
          - η → 0: Liquid disorder, isotropic smooth surface
          
        - **Harmonic truncation**: High-l terms suppressed by factor η^l
          - Enthalpy H: retains features up to l ≈ {int(eta * l_max_H)}
          - Entropy S: retains features up to l ≈ {int(eta * l_max_S)}
          
        - **Visual encoding**:
          - Color saturation ∝ |ΔG|/RT: stronger driving force = more saturated
          - Surface roughness ∝ atomic mobility: exp(-Eₐ/RT)
          - Curvature highlights: regions near spinodal decomposition
        
        **Landau parameters**: Tc = {Tc} K, ΔT = {delta_T} K
        """)
    
    # Temperature evolution hint
    st.info(f"""
    **Temperature Evolution Preview:**
    - **Low T ({T_min}K)**: η ≈ 1.0 → FCC-dominated, crystalline facets, high-l harmonics active
    - **Current T ({T_val}K)**: η = {eta:.2f} → {phase_char}
    - **High T ({T_max}K)**: η ≈ 0.0 → LIQUID-dominated, smooth sphere, only l=0,1 survive
    """)

# ================= FOOTER =================
st.divider()
st.caption("""
**Physics-Based Mode**: Implements G(x,T) = H(x) - T·S(x) decomposition with Landau order parameter η(T) 
for thermodynamically-consistent spherical harmonics visualization. 
Code by Dr. Sasikumar Subramanian | Functional Materials Research
""")
