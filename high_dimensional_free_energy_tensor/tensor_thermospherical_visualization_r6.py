
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator

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

st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer", layout="wide")

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
# Compute global statistics across all temperatures for consistent normalization
T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

# Global G ranges for consistent color scaling across temperatures
G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

# Global ΔG = G_LIQ - G_FCC statistics
df["dG"] = df["G_LIQ"] - df["G_FCC"]
dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))

# Temperature-dependent phase statistics (for verification)
@st.cache_data
def compute_phase_statistics(df):
    """Compute LIQUID fraction and average |dG| per temperature."""
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
        return None, None
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
    return interp_liq, interp_fcc

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
        # Try modern sph_harm_y first (SciPy >= 1.17)
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

        # Valid only if sum <= 1 (Ni non‑negative)
        valid = (pts[:,0] + pts[:,1] + pts[:,2]) <= 1.0

        G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
        G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)
        G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
        dG = G_liq - G_fcc  # Negative = LIQUID stable, Positive = FCC stable

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
        from scipy.linalg import lstsq
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

# ================= HEADER =================
st.title("🔷 Co-Cr-Fe-Ni Gibbs Free Energy Tensor Visualization")
st.markdown(r"""
This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface from discrete CSV data.  
The stable phase is determined by $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$.
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

    # --- Coordinate System (only affects marker mode) ---
    st.subheader("🌐 Coordinate System (marker mode)")
    coord_sys = st.radio("Select",
                         ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                         index=0,
                         help="Spherical: r=√(Co²+Cr²+Fe²), θ=atan2(Cr,Co), φ=acos(Fe/r)")

    st.divider()

    # --- Visualization Mode: Markers vs Spherical Harmonic Surface ---
    st.subheader("🎨 Rendering Mode")
    if not SCIPY_AVAILABLE:
        st.error("⚠️ `scipy` not installed. Spherical Harmonic Surface mode is disabled.")
        render_mode = "Markers (point cloud)"
        sh_disabled = True
    else:
        render_mode = st.radio("Visualization",
                               ["Markers (point cloud)", "Spherical Harmonic Surface (SH)"],
                               index=1,  # Default to SH mode now that it's improved
                               help="SH surface shows continuous angular variation of G with temperature‑dependent deformation")
        sh_disabled = False

    if render_mode == "Spherical Harmonic Surface (SH)" and not sh_disabled:
        st.subheader("🔧 Spherical Harmonic Parameters")

        # Temperature-dependent distortion controls
        st.markdown("**🌡️ Temperature Effects**")

        # Auto-compute temperature factor (0 = coldest, 1 = hottest)
        T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5

        # Temperature-dependent base radius (thermal expansion)
        # Low T: smaller radius (contracted, solid-like)
        # High T: larger radius (expanded, liquid-like)
        sh_R_fixed = st.slider("Base sphere radius", 0.2, 0.9, 
                               float(0.35 + 0.25 * T_factor),  # Default based on T
                               help="Nominal radius before distortion. Auto-adjusts with temperature (thermal expansion effect)")

        # Temperature-dependent distortion strength
        # Low T: crystalline, sharp features (lower alpha)
        # High T: fluid, smooth features (higher alpha)
        sh_alpha = st.slider("Radial distortion strength", 0.0, 0.8, 
                             float(0.1 + 0.4 * T_factor),  # Default based on T
                             help="How much G variation expands/contracts the sphere. Higher at high T (fluid-like)")

        # Temperature-dependent harmonic degree
        # Low T: need higher l to capture crystalline anisotropy
        # High T: lower l is sufficient for smooth liquid surface
        sh_l_max = st.slider("Max harmonic degree l", 0, 6, 
                             int(4 - 2 * T_factor),  # Default: high l at low T, low l at high T
                             help="Higher l captures more anisotropy (crystalline). Lower l for smooth (liquid-like)")

        sh_n_theta = st.slider("Theta resolution", 20, 120, 60, step=10)
        sh_n_phi   = st.slider("Phi resolution", 20, 120, 60, step=10)

        # Surface coloring mode
        st.markdown("**🎨 Surface Coloring**")
        surface_color_mode = st.radio("Color by",
                                      ["Stable G (global scale)", 
                                       "ΔG = G_LIQ - G_FCC (phase indicator)",
                                       "Temperature-encoded"],
                                      index=1,
                                      help="ΔG mode: blue=FCC stable, red=LIQUID stable")

        # Colormap selection based on mode
        if surface_color_mode == "ΔG = G_LIQ - G_FCC (phase indicator)":
            cmap_sh = st.selectbox("Colormap", ["RdBu_r", "RdYlBu", "Spectral", "Balance", "Curl"], index=0,
                                   help="Diverging colormap: red=LIQUID, blue=FCC")
        elif surface_color_mode == "Temperature-encoded":
            cmap_sh = st.selectbox("Colormap", ["Hot", "Thermal", "Inferno", "Magma"], index=0,
                                   help="Sequential colormap showing temperature intensity")
        else:
            cmap_sh = st.selectbox("Colormap", COLORMAPS, index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0)

        # Temperature comparison mode
        st.markdown("**📊 Multi-Temperature Comparison**")
        show_T_ghost = st.toggle("Show ghost surfaces at T±ΔT", value=False,
                                 help="Display semi-transparent surfaces at neighboring temperatures")
        if show_T_ghost:
            T_delta = st.slider("Temperature offset (K)", 50, 500, 200, 50)
            ghost_opacity = st.slider("Ghost opacity", 0.1, 0.5, 0.2, 0.05)

    # --- Marker-specific options (shown only if marker mode selected) ---
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
        scale_size_by_g = st.toggle("Scale size by |G|", value=False,
                                    help="Marker size ∝ |G| (tensor‑glyph style)")

    # --- Common visualization extras (e3nn style) ---
    st.subheader("🔷 e3nn / Geometric Shapes")
    show_ref_sphere = st.toggle("Show Reference Sphere", value=False,
                                help="Wireframe sphere at max composition radius")
    ref_sphere_r = st.slider("Sphere Radius", 0.1, 1.5, 1.0, 0.05) if show_ref_sphere else 1.0
    show_axes_frame = st.toggle("Show Coordinate Axes", value=False,
                                help="E(3) coordinate frame arrows at origin")
    show_simplex = st.toggle("Show Composition Simplex", value=False,
                             help="Wireframe of the Co-Cr-Fe-Ni tetrahedron boundary")

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

    # Show phase statistics
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
        interp_liq_q, interp_fcc_q = build_interpolators_for_T(df, q_t)
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

    # Temperature context
    T_factor_q = (query_result['T'] - T_min) / T_range if T_range > 0 else 0.5
    if query_result['Phase'] == "LIQUID":
        st.info(f"🔥 High-temperature phase (LIQUID). T-factor: {T_factor_q:.2f}")
    else:
        st.info(f"❄️ Low-temperature phase (FCC). T-factor: {T_factor_q:.2f}")

    if T_val != query_result['T']:
        st.info(f"ℹ️ The 3D field is rendered at T = {T_val} K; query values are for T = {query_result['T']} K.")
    st.divider()

# ================= MAIN RENDERING =================
interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)

if interp_liq is None:
    st.error(f"No data loaded for T = {T_val} K.")
    st.stop()

fig = go.Figure()

# Compute temperature factor for this rendering
T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5

# ------------------------------------------------------------------
# MODE 1: SPHERICAL HARMONIC SURFACE (only if scipy available)
# ------------------------------------------------------------------
if render_mode == "Spherical Harmonic Surface (SH)" and SCIPY_AVAILABLE:

    def render_sh_surface(T_render, opacity=0.9, is_ghost=False, ghost_label=""):
        """Render SH surface for a given temperature with temperature-dependent styling."""
        interp_liq_T, interp_fcc_T = build_interpolators_for_T(df, T_render)
        if interp_liq_T is None:
            return False

        TH, PH, G_vals, dG_vals, valid_mask = sample_g_on_sphere(
            interp_liq_T, interp_fcc_T, sh_R_fixed,
            n_theta=sh_n_theta, n_phi=sh_n_phi
        )

        if not np.any(valid_mask):
            return False

        # Fit spherical harmonic coefficients
        coeffs, l_max_used = fit_sh_coeffs(TH, PH, G_vals, l_max=sh_l_max)
        if coeffs is None:
            return False

        # Reconstruct smooth G field
        G_smooth = reconstruct_sh_surface(TH, PH, coeffs, l_max_used)

        # Temperature-dependent deformation:
        # Low T: crystalline, sharp features → use higher frequency components
        # High T: fluid, smooth → stronger radial distortion
        T_local_factor = (T_render - T_min) / T_range if T_range > 0 else 0.5

        # Base distortion from user slider (already T-dependent default)
        # Add temperature-modulated component
        alpha_effective = sh_alpha * (1.0 + 0.5 * np.sin(np.pi * T_local_factor))

        # Normalize G for deformation
        G_min = G_smooth.min()
        G_max = G_smooth.max()
        if G_max > G_min:
            G_norm = (G_smooth - G_min) / (G_max - G_min)
        else:
            G_norm = np.zeros_like(G_smooth)

        # Apply temperature-dependent deformation:
        # Low T: smaller distortion, more "rigid"
        # High T: larger distortion, more "fluid"
        radius = sh_R_fixed + alpha_effective * G_norm

        # Thermal expansion: higher T = larger base radius
        thermal_expansion = 1.0 + 0.1 * T_local_factor
        radius *= thermal_expansion

        # Convert to Cartesian
        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)

        # Determine surface color based on mode
        if surface_color_mode == "ΔG = G_LIQ - G_FCC (phase indicator)":
            # Fit dG surface for phase coloring
            coeffs_dG, _ = fit_sh_coeffs(TH, PH, dG_vals, l_max=sh_l_max)
            if coeffs_dG is not None:
                surfacecolor = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max_used)
                # Clamp to global range for consistent colors across T
                surfacecolor = np.clip(surfacecolor, -dG_global_abs_max, dG_global_abs_max)
                cbar_title = "ΔG = G_LIQ - G_FCC (J/mol)"
                # Center color scale at 0
                cmin = -dG_global_abs_max
                cmax = dG_global_abs_max
            else:
                surfacecolor = G_smooth
                cbar_title = cbar_title_txt
                cmin = G_global_min
                cmax = G_global_max
        elif surface_color_mode == "Temperature-encoded":
            # Encode temperature directly on surface
            surfacecolor = np.full_like(G_smooth, T_render)
            cbar_title = "Temperature (K)"
            cmin = T_min
            cmax = T_max
        else:  # Stable G (global scale)
            surfacecolor = G_smooth
            cbar_title = cbar_title_txt
            cmin = G_global_min
            cmax = G_global_max

        # Surface styling based on temperature
        if is_ghost:
            surface_name = f"T={T_render}K {ghost_label}"
            surface_opacity = ghost_opacity
            show_scale = False
        else:
            surface_name = f"SH surface (T={T_render}K)"
            surface_opacity = opacity
            show_scale = True

        # Temperature-dependent surface properties
        # Low T: more metallic/reflective (solid)
        # High T: more diffuse/transparent (liquid)
        if T_local_factor < 0.3:
            surface_roughness = 0.3  # Smooth, crystalline
            lighting_effect = "specular"
        elif T_local_factor > 0.7:
            surface_roughness = 0.8  # Rough, liquid
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

    # Render main surface
    success = render_sh_surface(T_val, opacity=0.9)

    # Render ghost surfaces if enabled
    if show_T_ghost and success:
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
# MODE 2: MARKER POINT CLOUD (original functionality)
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
# COMMON ELEMENTS (Query point, composition vector, SH probe, e3nn shapes)
# ------------------------------------------------------------------

# Query point overlay (always diamond)
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

    # Temperature-dependent probe radius
    T_query_factor = (query_result['T'] - T_min) / T_range if T_range > 0 else 0.5
    radius = sh_probe_scale * (0.5 + 0.5 * g_val / g_max_all) * (1.0 + 0.3 * T_query_factor) if g_max_all > 0 else sh_probe_scale

    # Phase color with temperature modulation
    if query_result["Phase"] == "LIQUID":
        base_color = "#e74c3c"  # Red for liquid
        # Hotter = more intense
        phase_color = f"rgb(255, {int(100 * (1 - T_query_factor))}, {int(100 * (1 - T_query_factor))})"
    else:
        base_color = "#3498db"  # Blue for FCC
        # Colder = more intense blue
        phase_color = f"rgb({int(100 * T_query_factor)}, {int(150 * T_query_factor)}, 255)"

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sh = qx + radius * np.outer(np.cos(u), np.sin(v))
    y_sh = qy + radius * np.outer(np.sin(u), np.sin(v))
    z_sh = qz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sh, y=y_sh, z=z_sh,
        opacity=0.25 + 0.15 * T_query_factor,  # More visible at high T
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

# e3nn Reference Sphere
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

# Temperature-dependent title
T_factor_display = (T_val - T_min) / T_range if T_range > 0 else 0.5
phase_dominant = "LIQUID-dominated" if T_factor_display > 0.6 else "FCC-dominated" if T_factor_display < 0.4 else "Transition"

fig.update_layout(
    template=template if template != "none" else None,
    scene=dict(
        xaxis=make_axis(scene_x_title),
        yaxis=make_axis(scene_y_title),
        zaxis=make_axis(scene_z_title),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    title=dict(
        text=f"Gibbs Energy Tensor at T = {T_val} K ({phase_dominant}) | Mode: {render_mode}",
        font=dict(size=title_font)
    ),
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
else:
    # SH mode statistics
    st.subheader("📊 Spherical Harmonic Surface Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{T_val} K")
    col2.metric("T-factor", f"{T_factor:.2f}")
    col3.metric("Distortion α", f"{sh_alpha:.2f}")
    col4.metric("Harmonic l_max", f"{sh_l_max}")

    # Show temperature evolution hint
    st.info(f"""
    **Temperature Effects:**
    - **Low T ({T_min}K)**: FCC-dominated (solid), crystalline structure, smaller radius, sharp features
    - **High T ({T_max}K)**: LIQUID-dominated (melted), fluid structure, larger radius, smooth features
    - **Current**: {phase_dominant} region with T-factor = {T_factor:.2f}
    """)
