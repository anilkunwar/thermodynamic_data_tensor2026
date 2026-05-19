import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator, CubicSpline

# Try importing scipy.special, handle missing or deprecated
try:
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ `scipy` not available. Spherical Harmonic surface mode will be disabled. "
               "Install scipy with `pip install scipy` to enable it.")

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

    def sample_g_on_sphere_full(interp_liq, interp_fcc, R_fixed, n_theta=50, n_phi=50):
        """Evaluate G_LIQ, G_FCC, stable G and dG on a sphere of fixed radius."""
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
        dG = G_liq - G_fcc

        valid = valid & ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        return (TH, PH,
                G_liq.reshape(TH.shape), G_fcc.reshape(TH.shape),
                G_stable.reshape(TH.shape), dG.reshape(TH.shape),
                valid.reshape(TH.shape))

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

    # =============================================
    # CONTINUOUS T-FIELD VIA SPLINE-INTERPOLATED SH COEFFICIENTS
    # =============================================
    @st.cache_data(ttl=3600)
    def build_continuous_sh_field(df, l_max=4, R_fixed=0.5, n_theta=60, n_phi=60):
        """
        Pre-fit SH coefficients at every discrete T, then build cubic splines
        so that T becomes a continuous parameter for the Gibbs energy field.
        """
        T_list = sorted(df["T"].unique())
        T_array = np.array(T_list, dtype=float)
        n_coeffs = (l_max + 1) ** 2

        coeff_data = {"G_liq": [], "G_fcc": [], "dG": []}

        for T in T_list:
            interp_liq, interp_fcc = build_interpolators_for_T(df, T)
            if interp_liq is None:
                for key in coeff_data:
                    coeff_data[key].append(np.zeros(n_coeffs))
                continue

            TH, PH, G_liq, G_fcc, G_stable, dG, valid = sample_g_on_sphere_full(
                interp_liq, interp_fcc, R_fixed, n_theta, n_phi
            )

            for key, vals in [("G_liq", G_liq), ("G_fcc", G_fcc), ("dG", dG)]:
                coeffs, _ = fit_sh_coeffs(TH, PH, vals, l_max=l_max)
                if coeffs is None:
                    coeffs = np.zeros(n_coeffs)
                coeff_data[key].append(coeffs)

        splines = {}
        for key in coeff_data:
            arr = np.array(coeff_data[key])  # shape: (n_T, n_coeffs)
            splines[key] = [CubicSpline(T_array, arr[:, i]) for i in range(n_coeffs)]

        return splines, float(T_array[0]), float(T_array[-1]), l_max

    def reconstruct_fields_at_t(TH, PH, T_query, splines, l_max):
        """Reconstruct G_LIQ, G_FCC and dG at an arbitrary float temperature."""
        fields = {}
        for key, spline_list in splines.items():
            coeffs = np.array([sp(T_query) for sp in spline_list])
            fields[key] = reconstruct_sh_surface(TH, PH, coeffs, l_max)
        return fields

# =============================================
# PHYSICS-BASED RADIUS & QUERY EVALUATION
# =============================================
def compute_physics_radius(dG, T, R_base, alpha, w, beta_thermal=0.15, T_ref=1000.0):
    """
    Gibbs–Helmholtz spherical deformation:
        ρ = R₀(T) · [ 1 + α · tanh( η / w ) ]
    where η = ΔG / (RT) is the dimensionless phase driving force.
    """
    RT = R_GAS * T
    eta = np.where(RT > 1e-6, dG / RT, 0.0)

    # Thermal expansion of reference radius
    R_0 = R_base * (1.0 + beta_thermal * (T - T_ref) / T_ref)

    # Sigmoidal deformation: expanded for FCC (η > 0), contracted for LIQUID (η < 0)
    deformation = np.tanh(eta / w)
    radius = R_0 * (1.0 + alpha * deformation)
    return radius, eta, R_0


def evaluate_query_point(df, q_co, q_cr, q_fe, T_query):
    """
    Evaluate thermodynamic state at query composition and temperature.
    If T_query is not in the discrete dataset, linearly interpolate between
    the two bracketing temperatures.
    """
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
        T_lower = max(lower) if lower else T_list[0]
        T_upper = min(upper) if upper else T_list[-1]

        if T_lower == T_upper:
            interp_liq, interp_fcc = build_interpolators_for_T(df, T_lower)
            g_liq = float(interp_liq(pt)[0])
            g_fcc = float(interp_fcc(pt)[0])
        else:
            w = (T_query - T_lower) / (T_upper - T_lower)
            il_l, if_l = build_interpolators_for_T(df, T_lower)
            il_u, if_u = build_interpolators_for_T(df, T_upper)
            gl_l, gf_l = float(il_l(pt)[0]), float(if_l(pt)[0])
            gl_u, gf_u = float(il_u(pt)[0]), float(if_u(pt)[0])

            # Handle possible NaN from out-of-hull at one T but not the other
            g_liq = gl_l if np.isnan(gl_u) else (gl_u if np.isnan(gl_l) else gl_l + w * (gl_u - gl_l))
            g_fcc = gf_l if np.isnan(gf_u) else (gf_u if np.isnan(gf_l) else gf_l + w * (gf_u - gf_l))

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
    """
    Interpolate G_LIQ and G_FCC onto a fixed composition grid at arbitrary T_query
    by linearly weighting the two nearest discrete temperature slices.
    """
    T_list = sorted(df["T"].unique())
    if T_query in T_list:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_query)
        return interp_liq(grid_pts), interp_fcc(grid_pts)

    lower = [t for t in T_list if t <= T_query]
    upper = [t for t in T_list if t >= T_query]
    T_lower = max(lower) if lower else T_list[0]
    T_upper = min(upper) if upper else T_list[-1]

    if T_lower == T_upper:
        interp_liq, interp_fcc = build_interpolators_for_T(df, T_lower)
        return interp_liq(grid_pts), interp_fcc(grid_pts)

    w = (T_query - T_lower) / (T_upper - T_lower)
    il_l, if_l = build_interpolators_for_T(df, T_lower)
    il_u, if_u = build_interpolators_for_T(df, T_upper)

    G_liq_l, G_fcc_l = il_l(grid_pts), if_l(grid_pts)
    G_liq_u, G_fcc_u = il_u(grid_pts), if_u(grid_pts)

    # NaN-safe blending
    G_liq = np.where(np.isnan(G_liq_l), G_liq_u,
             np.where(np.isnan(G_liq_u), G_liq_l, G_liq_l + w * (G_liq_u - G_liq_l)))
    G_fcc = np.where(np.isnan(G_fcc_l), G_fcc_u,
             np.where(np.isnan(G_fcc_u), G_fcc_l, G_fcc_l + w * (G_fcc_u - G_fcc_l)))
    return G_liq, G_fcc


# ================= HEADER =================
st.title("🔷 Co-Cr-Fe-Ni Gibbs–Helmholtz Tensor Visualization")
st.markdown(r"""
This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface from discrete CSV data.  
The phase landscape is driven by the dimensionless coordinate $\eta = \Delta G / RT$.  
The stable phase is determined by $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$.
""")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- Unified Temperature ---
    st.subheader("🌡️ Thermodynamic Temperature")
    T_query = st.slider("Temperature (K)", int(T_min), int(T_max),
                        int((T_min + T_max) / 2), step=10,
                        help="Single unified T for both query evaluation and field rendering")

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

    # --- Spherical Harmonics Probe ---
    st.subheader("🔮 Local SH Probe")
    show_sh_probe = st.toggle("Show SH Probe", value=True,
                              help="Display spherical harmonic glyph at the query point scaled by |η|")
    sh_probe_scale = st.slider("Probe Scale", 0.01, 0.5, 0.08, 0.01,
                               help="Base radius of the probe sphere")
    show_comp_vector = st.toggle("Show Composition Vector", value=True,
                                 help="l=1 arrow from origin to query point")

    st.divider()

    # --- Visualization Mode ---
    st.subheader("🎨 Rendering Mode")
    if not SCIPY_AVAILABLE:
        st.error("⚠️ `scipy` not installed. Spherical Harmonic Surface mode is disabled.")
        render_mode = "Markers (point cloud)"
        sh_disabled = True
    else:
        render_mode = st.radio("Visualization",
                               ["Markers (point cloud)", "Spherical Harmonic Surface (SH)"],
                               index=1,
                               help="SH surface shows continuous angular variation with physics-based deformation")
        sh_disabled = False

    # --- SH-specific controls ---
    if render_mode == "Spherical Harmonic Surface (SH)" and not sh_disabled:
        st.subheader("🔧 Physics-Based SH Parameters")

        st.markdown(r"""
        **Theory:** $\rho = R_0(T)\bigl[1 + \alpha \tanh(\eta / w)\bigr]$  
        where $\eta = \Delta G / RT$. At $\eta = 0$ the radius sits exactly at $R_0$ — the phase boundary.
        """)

        sh_R_base = st.slider("Base radius $R_0$", 0.2, 0.9, 0.50, 0.05,
                              help="Reference radius; thermal expansion scales this")
        sh_alpha = st.slider("Distortion amplitude α", 0.0, 0.50, 0.15, 0.01,
                             help="Maximum radial expansion/contraction from η")
        sh_w = st.slider("Transition width w (η-space)", 0.1, 3.0, 1.0, 0.1,
                         help="Smaller w → sharper phase front")
        sh_beta_thermal = st.slider("Thermal expansion β", 0.0, 0.30, 0.15, 0.01,
                                    help="Fractional radius change per ΔT/T_ref")
        sh_T_ref = st.number_input("Reference T_ref (K)", 500.0, 3000.0, 1000.0, 50.0)

        sh_l_max = st.slider("Max harmonic degree l", 0, 6, 4,
                             help="Higher l captures crystalline anisotropy")
        sh_n_theta = st.slider("Theta resolution", 20, 120, 60, step=10)
        sh_n_phi   = st.slider("Phi resolution", 20, 120, 60, step=10)

        st.markdown("**🎨 Surface Coloring**")
        surface_color_mode = st.radio("Color by",
                                      ["η = ΔG/RT (dimensionless)",
                                       "ΔG = G_LIQ - G_FCC (J/mol)",
                                       "Stable G (global scale)",
                                       "Temperature-encoded"],
                                      index=0,
                                      help="η mode: diverging colormap centered at 0 (phase boundary)")

        if surface_color_mode == "η = ΔG/RT (dimensionless)":
            cmap_sh = st.selectbox("Colormap", ["RdBu_r", "Balance", "Curl", "Spectral"], index=0,
                                   help="Diverging: blue=FCC (η>0), red=LIQUID (η<<0)")
        elif surface_color_mode in ["ΔG = G_LIQ - G_FCC (J/mol)", "Stable G (global scale)"]:
            cmap_sh = st.selectbox("Colormap", COLORMAPS,
                                   index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0)
        else:
            cmap_sh = st.selectbox("Colormap", ["Hot", "Thermal", "Inferno", "Magma"], index=0,
                                   help="Sequential colormap showing temperature intensity")

        st.markdown("**📊 Multi-Temperature Comparison**")
        show_T_ghost = st.toggle("Show ghost surfaces at T±ΔT", value=False,
                                 help="Display semi-transparent surfaces at neighboring temperatures")
        if show_T_ghost:
            T_delta = st.slider("Temperature offset (K)", 50, 500, 200, 50)
            ghost_opacity = st.slider("Ghost opacity", 0.1, 0.5, 0.2, 0.05)

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
        scale_size_by_g = st.toggle("Scale size by |G|", value=False,
                                    help="Marker size ∝ |G| (tensor‑glyph style)")

        st.subheader("🌐 Coordinate System")
        coord_sys = st.radio("Select",
                             ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                             index=0)

    # --- Common visualization extras ---
    st.subheader("🔷 Geometric References")
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
        query_result = evaluate_query_point(df, q_co, q_cr, q_fe, T_query)
        if query_result is None:
            st.error("❌ Query point lies outside the convex hull of available data at this temperature.")

# ================= DISPLAY QUERY RESULTS =================
if query_result:
    st.success(f"✅ Query Result at T = {query_result['T']} K")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("G_stable", f"{query_result['G_stable']:,.0f}", "J/mol")
    c4.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol")
    c5.metric("Stable Phase", query_result['Phase'])
    c6.metric("η", f"{query_result['eta']:.3f}")

    # Physical interpretation
    eta_q = query_result['eta']
    if abs(eta_q) < 0.1:
        st.info(f"⚖️ Near phase boundary (η = {eta_q:.3f}) — coexistence / melting front")
    elif eta_q < -1.0:
        st.info(f"🔥 Deep liquid (η = {eta_q:.3f}) — thermodynamically stable melt")
    elif eta_q > 1.0:
        st.info(f"❄️ Deep FCC solid (η = {eta_q:.3f}) — thermodynamically stable crystal")
    else:
        st.info(f"🔶 Transition region (η = {eta_q:.3f}) — moderate driving force")

    st.divider()

# ================= MAIN RENDERING =================
fig = go.Figure()

# Compute temperature factor for layout annotations
T_factor_display = (T_query - T_min) / T_range if T_range > 0 else 0.5
phase_dominant = "LIQUID-dominated" if T_factor_display > 0.6 else "FCC-dominated" if T_factor_display < 0.4 else "Transition"

# ------------------------------------------------------------------
# MODE 1: SPHERICAL HARMONIC SURFACE (continuous T-field)
# ------------------------------------------------------------------
if render_mode == "Spherical Harmonic Surface (SH)" and SCIPY_AVAILABLE:

    # Build continuous field (cached)
    sh_field_data = build_continuous_sh_field(
        df, l_max=sh_l_max, R_fixed=sh_R_base,
        n_theta=sh_n_theta, n_phi=sh_n_phi
    )
    splines, T_min_f, T_max_f, l_max_f = sh_field_data

    # Base theta-phi grid
    theta = np.linspace(0, 2*np.pi, sh_n_theta)
    phi = np.linspace(0, np.pi, sh_n_phi)
    TH, PH = np.meshgrid(theta, phi)

    def render_sh_surface_continuous(T_render, opacity=0.9, is_ghost=False, ghost_label=""):
        """Render SH surface at arbitrary float temperature."""
        fields = reconstruct_fields_at_t(TH, PH, T_render, splines, l_max_f)
        G_liq = fields["G_liq"]
        G_fcc = fields["G_fcc"]
        dG = fields["dG"]
        G_stable = np.minimum(G_liq, G_fcc)

        # Physics-based radius with Gibbs–Helmholtz deformation
        radius, eta, R_0 = compute_physics_radius(
            dG, T_render, sh_R_base, sh_alpha, sh_w,
            beta_thermal=sh_beta_thermal, T_ref=sh_T_ref
        )

        # Cartesian coordinates on the deformed manifold
        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)

        # Determine surface coloring
        if surface_color_mode == "η = ΔG/RT (dimensionless)":
            surfacecolor = eta
            cbar_title = "η = ΔG / RT"
            eta_abs_max = max(abs(eta.min()), abs(eta.max()))
            if eta_abs_max < 0.01:
                eta_abs_max = 1.0
            cmin = -eta_abs_max
            cmax = eta_abs_max
            cmap_use = cmap_sh
        elif surface_color_mode == "ΔG = G_LIQ - G_FCC (J/mol)":
            surfacecolor = dG
            cbar_title = "ΔG (J/mol)"
            cmin = -dG_global_abs_max
            cmax = dG_global_abs_max
            cmap_use = cmap_sh
        elif surface_color_mode == "Temperature-encoded":
            surfacecolor = np.full_like(eta, T_render)
            cbar_title = "Temperature (K)"
            cmin = T_min
            cmax = T_max
            cmap_use = cmap_sh
        else:  # Stable G
            surfacecolor = G_stable
            cbar_title = cbar_title_txt
            cmin = G_global_min
            cmax = G_global_max
            cmap_use = cmap_sh

        surface_name = f"SH surface T={T_render:.0f}K {ghost_label}".strip()

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surfacecolor,
            colorscale=cmap_use,
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
                f"η = %{{surfacecolor:.3f}}<<br>"
                f"G_stable = {G_stable.mean():,.0f} J/mol (avg)<br>"
                f"T = {T_render:.0f} K<br>"
                f"<extra></extra>"
            ) if not is_ghost else None,
            lighting=dict(ambient=0.6, diffuse=0.4, roughness=0.5),
            lightposition=dict(x=100, y=100, z=50)
        ))

        # Phase boundary: η ≈ 0 naturally sits at radius R_0 (undeformed sphere)
        if not is_ghost:
            eta_threshold = 0.05
            mask_b = np.abs(eta) < eta_threshold
            if np.any(mask_b):
                X_b = R_0 * np.sin(PH[mask_b]) * np.cos(TH[mask_b])
                Y_b = R_0 * np.sin(PH[mask_b]) * np.sin(TH[mask_b])
                Z_b = R_0 * np.cos(PH[mask_b])
                fig.add_trace(go.Scatter3d(
                    x=X_b, y=Y_b, z=Z_b,
                    mode="markers",
                    marker=dict(
                        size=7, color="gold", symbol="diamond",
                        line=dict(width=2, color="black"), opacity=0.95
                    ),
                    name=f"Phase Boundary (η≈0) @ {T_render:.0f}K",
                    hovertemplate=(
                        f"<b>Phase Boundary</b><br>"
                        f"T = {T_render:.0f} K<br>"
                        f"η ≈ 0 (ΔG ≈ 0)<br>"
                        f"<extra></extra>"
                    )
                ))
        return True

    # Main surface at unified query T
    success = render_sh_surface_continuous(T_query, opacity=0.9)

    # Ghost surfaces at neighboring temperatures
    if show_T_ghost and success:
        if T_query - T_delta >= T_min:
            render_sh_surface_continuous(T_query - T_delta, opacity=ghost_opacity,
                                         is_ghost=True, ghost_label="(cold)")
        if T_query + T_delta <= T_max:
            render_sh_surface_continuous(T_query + T_delta, opacity=ghost_opacity,
                                         is_ghost=True, ghost_label="(hot)")

    if not success:
        st.warning("Spherical harmonic fitting failed. Try adjusting parameters.")

# ------------------------------------------------------------------
# MODE 2: MARKER POINT CLOUD
# ------------------------------------------------------------------
else:
    # Generate tetrahedral grid
    x = np.linspace(0, 1, grid_res)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    pts_valid = grid_pts[valid_mask]

    # Interpolate to the unified T_query
    G_liq, G_fcc = interpolate_grid_to_t(pts_valid, df, T_query)

    valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    pts = pts_valid[valid_eval]
    G_liq = G_liq[valid_eval]
    G_fcc = G_fcc[valid_eval]
    G_stable = np.minimum(G_liq, G_fcc)
    stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
    dG = G_liq - G_fcc
    eta = dG / (R_GAS * T_query) if T_query > 0 else np.zeros_like(dG)

    if render_mode == "Markers (point cloud)":
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
# COMMON ELEMENTS (Query point, composition vector, SH probe, e3nn shapes)
# ------------------------------------------------------------------

# Query point overlay (always diamond)
if query_result is not None:
    # Determine coordinates for query point depending on current mode
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

# Spherical harmonics probe with η-dependent styling
if query_result is not None and show_sh_probe:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    eta_q = query_result.get("eta", 0.0)

    # Radius scales with thermodynamic depth |η|
    eta_depth = min(abs(eta_q), 5.0) / 5.0  # cap at 5
    radius = sh_probe_scale * (0.5 + 0.5 * eta_depth)

    # Color: red liquid / blue FCC / gold transition
    if eta_q < -0.5:
        phase_color = "#e74c3c"
    elif eta_q > 0.5:
        phase_color = "#3498db"
    else:
        phase_color = "#f39c12"

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sh = qx + radius * np.outer(np.cos(u), np.sin(v))
    y_sh = qy + radius * np.outer(np.sin(u), np.sin(v))
    z_sh = qz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sh, y=y_sh, z=z_sh,
        opacity=0.3,
        colorscale=[[0, phase_color], [1, phase_color]],
        showscale=False,
        name=f"SH Probe (η={eta_q:.2f})",
        hovertemplate=(f"<b>SH Probe</b><br>"
                       f"η = {eta_q:.3f}<br>"
                       f"Phase = {query_result['Phase']}<<br>"
                       f"Radius = {radius:.4f}<extra></extra>")
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
        text=f"Gibbs–Helmholtz Tensor at T = {T_query} K ({phase_dominant}) | Mode: {render_mode}",
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
    col3.metric("Mean |η|", f"{np.mean(np.abs(eta)):.2f}")
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

    # Reconstruct surface fields quickly for stats
    if SCIPY_AVAILABLE and 'splines' in locals():
        fields = reconstruct_fields_at_t(TH, PH, T_query, splines, l_max_f)
        dG_surf = fields["dG"]
        eta_surf = dG_surf / (R_GAS * T_query)
        col1.metric("Mean |η|", f"{np.mean(np.abs(eta_surf)):.2f}")
        col2.metric("Phase boundary pts", f"{np.sum(np.abs(eta_surf) < 0.05):,}")
        col3.metric("Max |η|", f"{np.max(np.abs(eta_surf)):.2f}")
        col4.metric("Min η", f"{np.min(eta_surf):.2f}")

    st.info(f"""
    **Physics Summary:**
    - **η = ΔG / RT**: dimensionless phase driving force
    - **η ≈ 0** (gold markers): Phase boundary where LIQUID and FCC coexist
    - **η < 0** (red surface): LIQUID stable — sphere contracts (fluid-like)
    - **η > 0** (blue surface): FCC stable — sphere expands (crystalline)
    - **Thermal expansion**: Base radius $R_0$ scales with $(1 + \\beta \\Delta T/T_{{ref}})$
    """)
