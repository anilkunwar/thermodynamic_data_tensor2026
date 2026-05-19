import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.linalg import lstsq

# Try importing scipy.special, handle missing or deprecated
try:
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =============================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
# =============================================
st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer", layout="wide")

# =============================================
# PHYSICAL CONSTANTS
# =============================================
R_GAS = 8.314  # J/(mol·K)

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

if not SCIPY_AVAILABLE:
    st.warning("⚠️ `scipy` not available. Spherical Harmonic surface mode will be disabled. "
               "Install scipy with `pip install scipy` to enable it.")

# =============================================
# COLORMAP LIBRARY (Plotly 6.x safe)
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
    r = np.sqrt(c1**2 + c2**2 + c3**2)
    safe_r = np.where(r == 0, 1e-12, r)
    theta = np.arctan2(c2, c1)
    phi = np.arccos(np.clip(c3 / safe_r, -1.0, 1.0))
    return r, theta, phi

# =============================================
# SPHERICAL HARMONICS (old & new SciPy compatible)
# =============================================
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

def sample_g_on_sphere_full(interp_liq, interp_fcc, R_fixed, n_theta=50, n_phi=50):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    x = R_fixed * np.sin(PH) * np.cos(TH)
    y = R_fixed * np.sin(PH) * np.sin(TH)
    z = R_fixed * np.cos(PH)
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    valid = (pts[:, 0] + pts[:, 1] + pts[:, 2]) <= 1.0

    G_liq = interp_liq(pts)
    G_fcc = interp_fcc(pts)
    # Mask outside simplex / convex hull
    G_liq = np.where(valid, G_liq, np.nan)
    G_fcc = np.where(valid, G_fcc, np.nan)

    G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
    dG = G_liq - G_fcc
    return TH, PH, G_liq, G_fcc, G_stable, dG, valid

def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=2):
    theta_flat = theta_vals.ravel()
    phi_flat = phi_vals.ravel()
    g_flat = g_vals.ravel()
    valid = ~np.isnan(g_flat)
    if np.sum(valid) == 0:
        return None, l_max
    theta_flat = theta_flat[valid]
    phi_flat = phi_flat[valid]
    g_flat = g_flat[valid]

    A = []
    for t, p in zip(theta_flat, phi_flat):
        row = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                row.append(get_real_sph_harm(l, m, t, p))
        A.append(row)
    A = np.array(A)
    coeffs, _, _, _ = lstsq(A, g_flat)
    return coeffs, l_max

def reconstruct_sh_surface(theta_grid, phi_grid, coeffs, l_max):
    recon = np.zeros_like(theta_grid, dtype=float)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if idx >= len(coeffs):
                break
            Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
            recon += coeffs[idx] * Y
            idx += 1
    return recon

# =============================================
# DATA LOADING & GLOBAL STATS
# =============================================
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

T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

# Global ranges for consistent colorbars
G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

df["dG"] = df["G_LIQ"] - df["G_FCC"]
dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))

@st.cache_data
def compute_phase_statistics(df):
    stats = []
    for T in sorted(df["T"].unique()):
        df_T = df[df["T"] == T]
        liq_count = (df_T["dG"] <= 0).sum()
        total = len(df_T)
        stats.append({
            "T": T,
            "LIQUID_fraction_%": liq_count / total * 100 if total > 0 else 0,
            "avg_abs_dG_J_mol": df_T["dG"].abs().mean(),
            "total_points": total
        })
    return pd.DataFrame(stats)

phase_stats_df = compute_phase_statistics(df)

# =============================================
# INTERPOLATION
# =============================================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
    return interp_liq, interp_fcc

# =============================================
# FAST CONTINUOUS T-FIELD VIA SH COEFFICIENT SPLINES
# =============================================
@st.cache_data(ttl=3600)
def build_sh_splines(df, T_list, l_max, R_fixed, n_theta, n_phi):
    """
    For every discrete T, fit SH coefficients to G_liq, G_fcc, G_stable, dG.
    Build a 1D linear interpolator (axis=0) for each coefficient across T.
    """
    n_coeffs = (l_max + 1) ** 2
    fields = {"G_liq": [], "G_fcc": [], "G_stable": [], "dG": []}
    for T in T_list:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T)
        if interp_liq is None:
            for k in fields:
                fields[k].append(np.zeros(n_coeffs))
            continue
        TH, PH, G_liq, G_fcc, G_stable, dG, _ = sample_g_on_sphere_full(
            interp_liq, interp_fcc, R_fixed, n_theta, n_phi
        )
        for key, vals in [("G_liq", G_liq), ("G_fcc", G_fcc),
                          ("G_stable", G_stable), ("dG", dG)]:
            coeffs, _ = fit_sh_coeffs(TH, PH, vals, l_max)
            if coeffs is None:
                coeffs = np.zeros(n_coeffs)
            fields[key].append(coeffs)

    T_arr = np.array(T_list, dtype=float)
    splines = {}
    for key, coeff_list in fields.items():
        arr = np.array(coeff_list)  # shape (n_T, n_coeffs)
        splines[key] = interp1d(T_arr, arr, kind='linear', axis=0,
                                fill_value='extrapolate')
    return splines, l_max

def reconstruct_fields_at_t(T_query, splines, l_max, TH, PH):
    """Reconstruct all Gibbs fields at arbitrary float T from coefficient splines."""
    out = {}
    for key, spline in splines.items():
        coeffs = spline(T_query)
        out[key] = reconstruct_sh_surface(TH, PH, coeffs, l_max)
    return out

# =============================================
# PHYSICS-BASED RADIUS & QUERY EVALUATION
# =============================================
def compute_physics_radius(dG, T, R_base, alpha, w):
    """
    Gibbs–Helmholtz spherical deformation:
        ρ = R_base * [ 1 + α * tanh( η / w ) ]
    where η = ΔG / (RT).
    Returns: (radius_array, eta_array, R_base)
    """
    RT = R_GAS * T
    eta = np.where(RT > 1e-6, dG / RT, 0.0)
    deformation = np.tanh(eta / w)
    radius = R_base * (1.0 + alpha * deformation)
    return radius, eta, R_base

def evaluate_query_point(df, q_co, q_cr, q_fe, T_query):
    """Evaluate thermodynamic state at query composition and arbitrary T."""
    T_list = sorted(df["T"].unique())
    pt = np.array([[q_co, q_cr, q_fe]])

    if T_query in T_list:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_query)
        if interp_liq is None:
            return None
        g_liq = float(interp_liq(pt)[0])
        g_fcc = float(interp_fcc(pt)[0])
    else:
        lower = [t for t in T_list if t <= T_query]
        upper = [t for t in T_list if t >= T_query]
        T_lo = max(lower) if lower else T_list[0]
        T_hi = min(upper) if upper else T_list[-1]

        if T_lo == T_hi:
            interp_liq, interp_fcc = build_interpolators_for_T(df, T_lo)
            g_liq = float(interp_liq(pt)[0])
            g_fcc = float(interp_fcc(pt)[0])
        else:
            w = (T_query - T_lo) / (T_hi - T_lo)
            il_lo, if_lo = build_interpolators_for_T(df, T_lo)
            il_hi, if_hi = build_interpolators_for_T(df, T_hi)
            gl_lo, gf_lo = float(il_lo(pt)[0]), float(if_lo(pt)[0])
            gl_hi, gf_hi = float(il_hi(pt)[0]), float(if_hi(pt)[0])

            g_liq = gl_lo if np.isnan(gl_hi) else (gl_hi if np.isnan(gl_lo) else gl_lo + w * (gl_hi - gl_lo))
            g_fcc = gf_lo if np.isnan(gf_hi) else (gf_hi if np.isnan(gf_lo) else gf_lo + w * (gf_hi - gf_lo))

    if np.isnan(g_liq) or np.isnan(g_fcc):
        return None

    g_stable = min(g_liq, g_fcc)
    phase = "LIQUID" if g_liq <= g_fcc else "FCC"
    dG = g_liq - g_fcc
    eta = dG / (R_GAS * T_query) if T_query > 0 else 0.0

    return {
        "T": T_query, "Co": q_co, "Cr": q_cr, "Fe": q_fe,
        "Ni": 1.0 - q_co - q_cr - q_fe,
        "G_LIQ": g_liq, "G_FCC": g_fcc,
        "G_stable": g_stable, "Phase": phase,
        "dG": dG, "eta": eta
    }

def interpolate_grid_to_t(grid_pts, df, T_query):
    """Interpolate G_LIQ and G_FCC onto a fixed composition grid at arbitrary T."""
    T_list = sorted(df["T"].unique())
    if T_query in T_list:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_query)
        return interp_liq(grid_pts), interp_fcc(grid_pts)

    lower = [t for t in T_list if t <= T_query]
    upper = [t for t in T_list if t >= T_query]
    T_lo = max(lower) if lower else T_list[0]
    T_hi = min(upper) if upper else T_list[-1]

    if T_lo == T_hi:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_lo)
        return interp_liq(grid_pts), interp_fcc(grid_pts)

    w = (T_query - T_lo) / (T_hi - T_lo)
    il_lo, if_lo = build_interpolators_for_T(df, T_lo)
    il_hi, if_hi = build_interpolators_for_T(df, T_hi)

    G_liq_lo, G_fcc_lo = il_lo(grid_pts), if_lo(grid_pts)
    G_liq_hi, G_fcc_hi = il_hi(grid_pts), if_hi(grid_pts)

    G_liq = np.where(np.isnan(G_liq_lo), G_liq_hi,
            np.where(np.isnan(G_liq_hi), G_liq_lo, G_liq_lo + w * (G_liq_hi - G_liq_lo)))
    G_fcc = np.where(np.isnan(G_fcc_lo), G_fcc_hi,
            np.where(np.isnan(G_fcc_hi), G_fcc_lo, G_fcc_lo + w * (G_fcc_hi - G_fcc_lo)))
    return G_liq, G_fcc

# =============================================
# HEADER
# =============================================
st.title("🔷 Co-Cr-Fe-Ni Gibbs–Helmholtz Tensor Visualization")
st.markdown(r"""
Continuous $G(\mathbf{x}, T)$ hypersurface reconstructed via **spherical-harmonic coefficient splines**.  
Phase stability driven by the dimensionless coordinate $\eta = \Delta G / RT$.  
Surface deforms according to $\rho = R_{\text{base}}\bigl[1 + \alpha \tanh(\eta/w)\bigr]$.
""")

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- Unified Temperature ---
    st.subheader("🌡️ Thermodynamic Temperature")
    T_query = st.slider("Temperature (K)", int(T_min), int(T_max),
                        int((T_min + T_max) // 2), step=10,
                        help="Single T for both query evaluation and field rendering")

    st.divider()

    # --- Query Point ---
    st.subheader("📍 Query Point")
    q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")

    comp_sum = q_co + q_cr + q_fe
    if comp_sum > 1.0:
        st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0 (x_Ni would be negative).")

    eval_query = st.button("🔍 Evaluate at T", use_container_width=True)

    st.divider()

    # --- SH Probe ---
    st.subheader("🔮 Local Probe")
    show_sh_probe = st.toggle("Show SH Probe", value=True,
                              help="Sphere at query point scaled by |η|")
    sh_probe_scale = st.slider("Probe Scale", 0.01, 0.5, 0.08, 0.01)
    show_comp_vector = st.toggle("Show Composition Vector", value=True)

    st.divider()

    # --- Visualization Mode ---
    st.subheader("🎨 Rendering Mode")
    if not SCIPY_AVAILABLE:
        st.error("⚠️ `scipy` not installed. SH Surface mode is disabled.")
        render_mode = "Markers (point cloud)"
        sh_disabled = True
    else:
        render_mode = st.radio("Visualization",
                               ["Markers (point cloud)", "SH Surface (Physics)"],
                               index=1)
        sh_disabled = (render_mode == "Markers (point cloud)")

    # --- SH Controls ---
    if render_mode == "SH Surface (Physics)" and not sh_disabled:
        st.subheader("🔧 Physics Parameters")
        st.markdown(r"$\rho = R_{\text{base}}\bigl[1 + \alpha \tanh(\eta/w)\bigr]$")

        sh_R_base = st.slider("Base radius $R_{base}$", 0.2, 0.9, 0.50, 0.05)
        sh_alpha = st.slider("Distortion amplitude α", 0.0, 0.50, 0.15, 0.01,
                             help="Max radial expansion/contraction from η")
        sh_w = st.slider("Transition width w (η-space)", 0.1, 3.0, 1.0, 0.1,
                         help="Smaller w → sharper phase front")
        sh_l_max = st.slider("Max harmonic degree l", 0, 6, 4,
                             help="Higher l = more crystalline facets")
        sh_n_theta = st.slider("Theta resolution", 30, 120, 50, step=10,
                               help="Lower = faster render")
        sh_n_phi = st.slider("Phi resolution", 30, 120, 50, step=10)

        st.markdown("**🎨 Surface Coloring**")
        color_mode = st.selectbox("Color by",
                                   ["G_stable (J/mol)",
                                    "ΔG (J/mol)",
                                    "η = ΔG/RT (dimensionless)",
                                    "Phase (LIQUID/FCC)"],
                                   index=2)

        st.markdown("**📊 Overlays**")
        show_phase_boundary = st.toggle("Show η≈0 phase boundary", value=True,
                                        help="Gold markers where ΔG ≈ 0")
        show_T_ghost = st.toggle("Show ghost surfaces at T±ΔT", value=False)
        if show_T_ghost:
            T_delta = st.slider("ΔT offset (K)", 50, 500, 200, 50)
            ghost_opacity = st.slider("Ghost opacity", 0.1, 0.5, 0.2, 0.05)

    # --- Marker Controls ---
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

        st.subheader("🌐 Coordinate System")
        coord_sys = st.radio("Select",
                             ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                             index=0)

    # --- Geometric Aids ---
    st.subheader("🔷 References")
    show_ref_sphere = st.toggle("Show Reference Sphere", value=False)
    ref_sphere_r = st.slider("Sphere Radius", 0.1, 1.5, 1.0, 0.05) if show_ref_sphere else 1.0
    show_axes_frame = st.toggle("Show Coordinate Axes", value=False)
    show_simplex = st.toggle("Show Composition Simplex", value=False)

    st.divider()

    # --- Layout ---
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

# =============================================
# QUERY EVALUATION
# =============================================
query_result = None
if eval_query:
    if comp_sum > 1.0:
        st.error("Cannot evaluate: composition sum exceeds 1.0.")
    else:
        query_result = evaluate_query_point(df, q_co, q_cr, q_fe, T_query)
        if query_result is None:
            st.error("❌ Query point lies outside the convex hull at this temperature.")

# =============================================
# DISPLAY QUERY RESULTS
# =============================================
if query_result:
    st.success(f"✅ Query Result at T = {query_result['T']} K")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("G_stable", f"{query_result['G_stable']:,.0f}", "J/mol")
    c4.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol")
    c5.metric("Stable Phase", query_result['Phase'])
    c6.metric("η", f"{query_result['eta']:.3f}")

    eta_q = query_result['eta']
    if abs(eta_q) < 0.1:
        st.info(f"⚖️ Near phase boundary (η = {eta_q:.3f}) — coexistence / melting front")
    elif eta_q < -1.0:
        st.info(f"🔥 Deep liquid (η = {eta_q:.3f}) — stable melt")
    elif eta_q > 1.0:
        st.info(f"❄️ Deep FCC solid (η = {eta_q:.3f}) — stable crystal")
    else:
        st.info(f"🔶 Transition region (η = {eta_q:.3f})")
    st.divider()

# =============================================
# MAIN RENDERING
# =============================================
fig = go.Figure()
T_factor_display = (T_query - T_min) / T_range if T_range > 0 else 0.5

# ------------------------------------------------------------------
# SH SURFACE MODE (fast physics)
# ------------------------------------------------------------------
if render_mode == "SH Surface (Physics)" and SCIPY_AVAILABLE and not sh_disabled:

    # Build coefficient splines once (cached)
    splines, l_max = build_sh_splines(df, T_list, sh_l_max, sh_R_base, sh_n_theta, sh_n_phi)

    theta = np.linspace(0, 2 * np.pi, sh_n_theta)
    phi = np.linspace(0, np.pi, sh_n_phi)
    TH, PH = np.meshgrid(theta, phi)

    def add_sh_surface(T_render, opacity=0.9, is_ghost=False, ghost_label=""):
        fields = reconstruct_fields_at_t(T_render, splines, l_max, TH, PH)
        G_stable = fields["G_stable"]
        dG = fields["dG"]

        # Physics-based deformation
        radius, eta, R_base = compute_physics_radius(dG, T_render, sh_R_base, sh_alpha, sh_w)
        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)

        # Color mode selection
        if color_mode == "G_stable (J/mol)":
            surfacecolor = G_stable
            colorscale = "Viridis"
            cmin, cmax = G_global_min, G_global_max
            cbar_title = "G_stable (J/mol)"
        elif color_mode == "ΔG (J/mol)":
            surfacecolor = dG
            colorscale = "RdBu_r"
            cmin, cmax = -dG_global_abs_max, dG_global_abs_max
            cbar_title = "ΔG = G_LIQ - G_FCC (J/mol)"
        elif color_mode == "η = ΔG/RT (dimensionless)":
            surfacecolor = eta
            colorscale = "RdBu_r"
            eta_abs_max = max(abs(eta.min()), abs(eta.max()), 0.1)
            cmin, cmax = -eta_abs_max, eta_abs_max
            cbar_title = "η = ΔG / RT"
        else:  # Phase
            surfacecolor = np.where(dG < 0, -1.0, 1.0)
            colorscale = [[0, "rgb(231,76,60)"], [0.5, "white"], [1, "rgb(52,152,219)"]]
            cmin, cmax = -1, 1
            cbar_title = "Phase (-1=LIQUID, +1=FCC)"

        surface_name = f"SH T={T_render:.0f}K {ghost_label}".strip()

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            name=surface_name,
            showscale=not is_ghost,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(size=cbar_title_size)),
                thickness=cbar_thickness,
                len=cbar_len,
                tickfont=dict(size=cbar_tick_size),
                xpad=cbar_xpad,
                ypad=cbar_ypad
            ) if not is_ghost else None,
            hovertemplate=(
                f"<b>{surface_name}</b><br>"
                f"G_stable = %{{z:.0f}} (radius proxy)<br>"
                f"<extra></extra>"
            ) if not is_ghost else None,
            lighting=dict(ambient=0.6, diffuse=0.4, roughness=0.5),
            lightposition=dict(x=100, y=100, z=50)
        ))

        # Phase boundary: η ≈ 0 markers on undeformed sphere
        if show_phase_boundary and not is_ghost:
            eta_threshold = 0.05
            mask_b = np.abs(eta) < eta_threshold
            if np.any(mask_b):
                # Subsample boundary to avoid overcrowding
                step = max(1, sh_n_theta // 25)
                mask_b[::step, ::step] = mask_b[::step, ::step]
                mask_b_decimated = mask_b & (np.random.rand(*mask_b.shape) < 0.3)  # stochastic decimation for very dense grids
                if np.any(mask_b_decimated):
                    X_b = R_base * np.sin(PH)[mask_b_decimated] * np.cos(TH)[mask_b_decimated]
                    Y_b = R_base * np.sin(PH)[mask_b_decimated] * np.sin(TH)[mask_b_decimated]
                    Z_b = R_base * np.cos(PH)[mask_b_decimated]
                    fig.add_trace(go.Scatter3d(
                        x=X_b, y=Y_b, z=Z_b,
                        mode="markers",
                        marker=dict(
                            size=6, color="gold", symbol="diamond",
                            line=dict(width=1, color="black"), opacity=0.9
                        ),
                        name=f"Phase Boundary @ {T_render:.0f}K",
                        hovertemplate=(
                            f"<b>Phase Boundary</b><br>"
                            f"T = {T_render:.0f} K<br>"
                            f"η ≈ 0 (ΔG ≈ 0)<br>"
                            f"<extra></extra>"
                        )
                    ))
        return True

    # Main surface
    add_sh_surface(T_query, opacity=0.92)

    # Ghost surfaces
    if show_T_ghost:
        if T_query - T_delta >= T_min:
            add_sh_surface(T_query - T_delta, opacity=ghost_opacity,
                           is_ghost=True, ghost_label="(cold)")
        if T_query + T_delta <= T_max:
            add_sh_surface(T_query + T_delta, opacity=ghost_opacity,
                           is_ghost=True, ghost_label="(hot)")

# ------------------------------------------------------------------
# MARKER POINT CLOUD MODE
# ------------------------------------------------------------------
else:
    grid_res = st.sidebar.slider("Grid Resolution", 15, 100, 25, step=5) \
        if 'grid_res' not in locals() else 25

    x = np.linspace(0, 1, grid_res)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    pts_valid = grid_pts[valid_mask]

    G_liq, G_fcc = interpolate_grid_to_t(pts_valid, df, T_query)
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
        sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min) if g_max > g_min else np.full_like(g_norm, marker_size)
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
            hovertemplate=(f"<b>Stable</b><br>{x_title}=%{{x:.3f}}<<br>{y_title}=%{{y:.3f}}<<br>"
                           f"{z_title}=%{{z:.3f}}<<br>G=%{{marker.color:,.0f}} J/mol<br>"
                           f"Phase=%{{text}}<<extra></extra>"),
            text=stable_label
        ))
    elif show_phase == "LIQUID Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_liq, sizes, cbar_title_txt),
            name="LIQUID",
            hovertemplate=(f"<b>LIQUID</b><br>{x_title}=%{{x:.3f}}<<br>{y_title}=%{{y:.3f}}<<br>"
                           f"{z_title}=%{{z:.3f}}<<br>G_LIQ=%{{marker.color:,.0f}} J/mol<<extra></extra>")
        ))
    elif show_phase == "FCC Only":
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers",
            marker=make_marker_config(G_fcc, sizes, cbar_title_txt),
            name="FCC",
            hovertemplate=(f"<b>FCC</b><br>{x_title}=%{{x:.3f}}<<br>{y_title}=%{{y:.3f}}<<br>"
                           f"{z_title}=%{{z:.3f}}<<br>G_FCC=%{{marker.color:,.0f}} J/mol<<extra></extra>")
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
# COMMON OVERLAYS (Query point, probe, vector, references)
# ------------------------------------------------------------------
if query_result is not None:
    if render_mode == "Markers (point cloud)" and coord_sys == "Spherical (r, θ, φ)":
        x_q, y_q, z_q = cartesian_to_spherical(
            np.array([query_result["Co"]]),
            np.array([query_result["Cr"]]),
            np.array([query_result["Fe"]])
        )
    else:
        x_q = np.array([query_result["Co"]])
        y_q = np.array([query_result["Cr"]])
        z_q = np.array([query_result["Fe"]])

    if not np.isnan(x_q[0]):
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
                           f"η={query_result['eta']:.3f}<br>"
                           f"Phase={query_result['Phase']}<<extra></extra>")
        ))

    if show_sh_probe:
        qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
        eta_q = query_result.get("eta", 0.0)
        probe_r = sh_probe_scale * (1.0 + 0.5 * min(abs(eta_q), 5.0))

        if eta_q < -0.5:
            probe_color = "#e74c3c"
        elif eta_q > 0.5:
            probe_color = "#3498db"
        else:
            probe_color = "#f39c12"

        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 24)
        x_pr = qx + probe_r * np.outer(np.cos(u), np.sin(v))
        y_pr = qy + probe_r * np.outer(np.sin(u), np.sin(v))
        z_pr = qz + probe_r * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_pr, y=y_pr, z=z_pr,
            opacity=0.25,
            colorscale=[[0, probe_color], [1, probe_color]],
            showscale=False,
            name=f"SH Probe (η={eta_q:.2f})",
            hovertemplate=(f"<b>SH Probe</b><br>η={eta_q:.3f}<br>Phase={query_result['Phase']}<<extra></extra>")
        ))

    if show_comp_vector:
        qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
        fig.add_trace(go.Scatter3d(
            x=[0, qx], y=[0, qy], z=[0, qz],
            mode="lines+text",
            line=dict(color="gold", width=5),
            text=["", "l=1"],
            textposition="top center",
            textfont=dict(size=14, color="gold"),
            name="Composition Vector (l=1)",
            hovertemplate=f"<b>l=1 Vector</b><br>Co={qx:.3f}<br>Cr={qy:.3f}<br>Fe={qz:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter3d(
            x=[qx], y=[qy], z=[qz],
            mode="markers",
            marker=dict(size=8, color="gold", symbol="diamond",
                        line=dict(width=1, color="white")),
            name="Vector Head",
            hoverinfo="skip"
        ))

if show_ref_sphere:
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
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

if show_axes_frame:
    axis_len = 1.1
    for ax, col, lab in [([axis_len,0,0], "red", "Co"),
                         ([0,axis_len,0], "green", "Cr"),
                         ([0,0,axis_len], "blue", "Fe")]:
        fig.add_trace(go.Scatter3d(
            x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]],
            mode="lines+text",
            line=dict(color=col, width=4),
            text=["", lab],
            textposition="top center",
            textfont=dict(size=12, color=col),
            name=f"{lab} axis",
            hoverinfo="skip"
        ))

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
# LAYOUT
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
    scene_x_title, scene_y_title, scene_z_title = x_title, y_title, z_title
else:
    scene_x_title, scene_y_title, scene_z_title = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

phase_dominant = "LIQUID-dominated" if T_factor_display > 0.6 else "FCC-dominated" if T_factor_display < 0.4 else "Transition"
title_text = f"Gibbs Energy at T = {T_query} K ({phase_dominant}) | Mode: {render_mode}"

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
    st.info("Try lowering resolution or switching color mode.")

# =============================================
# STATISTICS
# =============================================
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

else:
    st.subheader("📊 Spherical Harmonic Surface Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{T_query} K")
    col2.metric("T-factor", f"{T_factor_display:.2f}")
    col3.metric("Distortion α", f"{sh_alpha:.2f}")
    col4.metric("Harmonic l_max", f"{sh_l_max}")

    if SCIPY_AVAILABLE and 'splines' in dir():
        fields = reconstruct_fields_at_t(T_query, splines, sh_l_max, TH, PH)
        dG_surf = fields["dG"]
        eta_surf = dG_surf / (R_GAS * T_query)
        col1.metric("Mean |η|", f"{np.mean(np.abs(eta_surf)):.2f}")
        col2.metric("Phase boundary pts", f"{np.sum(np.abs(eta_surf) < 0.05):,}")
        col3.metric("Max |η|", f"{np.max(np.abs(eta_surf)):.2f}")
        col4.metric("Min η", f"{np.min(eta_surf):.2f}")

    st.info(f"""
    **Physics Summary:**
    - **Low T ({T_min}K)**: FCC-dominated, crystalline, expanded sphere (η > 0)
    - **High T ({T_max}K)**: LIQUID-dominated, fluid, contracted sphere (η < 0)
    - **Current**: {phase_dominant} with T-factor = {T_factor_display:.2f}
    - **Gold diamonds**: Phase boundary where η ≈ 0 (ΔG ≈ 0)
    """)
