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
    st.warning("⚠️ `scipy` not available. Spherical Harmonic surface mode will be disabled.")

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(page_title="CoCrFeNi Phase Stability Explorer", layout="wide")

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

# Phase-specific symbols and colors for DISTINCT visual identity
PHASE_SYMBOLS = {"LIQUID": "circle", "FCC": "diamond"}
PHASE_COLORS = {"LIQUID": "#e74c3c", "FCC": "#2980b9"}  # Red vs Blue
PHASE_COLORS_LIGHT = {"LIQUID": "rgba(231, 76, 60, 0.3)", "FCC": "rgba(41, 128, 185, 0.3)"}

# =============================================
# DATA LOADING
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

# Global G ranges for consistent color scaling
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
# TETRAHEDRAL GRID UTILITIES (Scientific Mode)
# =============================================
def generate_tetrahedral_grid(resolution=25):
    """Generate valid composition points inside the Co-Cr-Fe-Ni simplex."""
    x = np.linspace(0, 1, resolution)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    return grid_pts[valid_mask]

def find_phase_boundary_points(pts, dG_values, threshold=50.0):
    """Find points near ΔG = 0 (the phase boundary)."""
    boundary_mask = np.abs(dG_values) < threshold
    return pts[boundary_mask], dG_values[boundary_mask]

# =============================================
# SPHERICAL HARMONICS (kept for aesthetic modes)
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
        return TH, PH, G_stable.reshape(TH.shape), dG.reshape(TH.shape), valid.reshape(TH.shape), pts

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
        from scipy.linalg import lstsq
        coeffs, _, _, _ = lstsq(A, g_flat)
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

    def extract_dg_zero_contour(TH, PH, dG_grid, R_fixed, threshold_factor=0.3):
        """Extract approximate ΔG=0 contour lines from spherical grid."""
        # Find edges where sign changes
        dG_norm = dG_grid / (np.nanmax(np.abs(dG_grid)) + 1e-12)
        contours_x, contours_y, contours_z = [], [], []

        # Horizontal edges (theta direction)
        for i in range(dG_grid.shape[0]):
            for j in range(dG_grid.shape[1]-1):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i,j+1])):
                    continue
                if dG_grid[i,j] * dG_grid[i,j+1] < 0:  # Sign change
                    # Interpolate
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i,j+1]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i,j+1] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i,j+1] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))

        # Vertical edges (phi direction)
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
# HEADER
# =============================================
st.title("🔷 Co-Cr-Fe-Ni Phase Stability Explorer")
st.markdown(r"""
**Single-Temperature Phase Comparison Tool.** At a chosen temperature, clearly distinguish 
**LIQUID** (🔴 red, circles, smooth) vs **FCC** (🔵 blue, diamonds, wireframe) stability regions.  
The **ΔG = 0 boundary** (🟡 gold line/surface) marks the exact phase transition frontier.
""")

# =============================================
# SIDEBAR - SINGLE T FOCUS
# =============================================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- SINGLE TEMPERATURE SELECTOR ---
    st.subheader("🌡️ Temperature")
    T_val = st.select_slider("Select T (K)", options=T_list,
                             value=T_list[len(T_list)//2] if T_list else 1000)
    T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5

    phase_dominant = "LIQUID" if T_factor > 0.6 else "FCC" if T_factor < 0.4 else "Transition"
    st.info(f"T = {T_val}K | Expected: {phase_dominant}")

    st.divider()

    # --- QUERY POINT ---
    st.subheader("📍 Query Composition")
    q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    comp_sum = q_co + q_cr + q_fe
    if comp_sum > 1.0:
        st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0")
    eval_query = st.button("🔍 Evaluate", use_container_width=True)

    st.divider()

    # --- RENDERING MODE (Phase Comparison Focus) ---
    st.subheader("🎨 Visualization Mode")

    mode_options = [
        "Phase Boundary (Scientific)",
        "Dual SH Surfaces (Aesthetic)",
        "ΔG Difference Surface",
        "Ternary Flat Projection",
        "Markers (Point Cloud)"
    ]
    if not SCIPY_AVAILABLE:
        mode_options = [m for m in mode_options if "SH" not in m and "Difference" not in m]
        st.error("SciPy missing: SH modes disabled")

    render_mode = st.radio("Mode", mode_options, index=0,
                           help="""
                           **Phase Boundary**: True tetrahedral ΔG=0 surface + phase-colored volume. Most scientifically accurate.  
                           **Dual SH**: Two spherical harmonic surfaces with distinct styles.  
                           **ΔG Surface**: Single surface where bumps = driving force.  
                           **Ternary Flat**: Standard 2D ternary diagram.  
                           **Markers**: Classic 3D scatter.
                           """)

    st.divider()

    # --- MODE-SPECIFIC CONTROLS ---
    if render_mode == "Phase Boundary (Scientific)":
        st.subheader("🔧 Scientific Mode Settings")
        grid_res = st.slider("Grid Resolution", 15, 60, 30, step=5,
                             help="Higher = finer boundary but slower")
        boundary_threshold = st.slider("Boundary Width (J/mol)", 10, 200, 50, 10,
                                       help="Tolerance for ΔG ≈ 0")
        show_phase_volume = st.toggle("Show Phase Volume", value=True,
                                      help="Color interior points by stable phase")
        volume_opacity = st.slider("Volume Opacity", 0.05, 0.5, 0.15, 0.05)
        volume_size = st.slider("Volume Point Size", 1, 6, 2)
        show_simplex = st.toggle("Show Simplex Frame", value=True)

    elif render_mode in ["Dual SH Surfaces (Aesthetic)", "ΔG Difference Surface"]:
        st.subheader("🔧 Spherical Harmonic Settings")
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.5, 0.05)
        sh_l_max = st.slider("Max Harmonic Degree", 1, 6, 3)
        sh_n_theta = st.slider("Theta Resolution", 30, 120, 60, step=10)
        sh_n_phi = st.slider("Phi Resolution", 30, 120, 60, step=10)

        if render_mode == "Dual SH Surfaces (Aesthetic)":
            st.markdown("**Phase Surface Styles**")
            liq_opacity = st.slider("LIQUID Surface Opacity", 0.1, 1.0, 0.6, 0.05)
            fcc_opacity = st.slider("FCC Surface Opacity", 0.1, 1.0, 0.4, 0.05)
            show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
        else:  # ΔG Difference Surface
            dg_scale = st.slider("ΔG Deformation Scale", 0.001, 0.1, 0.02, 0.001,
                                 help="How much ΔG magnitude distorts radius")

    elif render_mode == "Ternary Flat Projection":
        st.subheader("🔧 Ternary Settings")
        flat_color_by = st.radio("Color By", ["Stable Phase", "ΔG (diverging)", "G_magnitude"], index=1)
        flat_marker_size = st.slider("Marker Size", 2, 15, 6)
        flat_opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05)
        show_ternary_grid = st.toggle("Show Grid Lines", value=True)

    else:  # Markers
        st.subheader("🔧 Marker Settings")
        grid_res = st.slider("Grid Resolution", 15, 100, 30, step=5)
        marker_size = st.slider("Marker Size", 1, 10, 4)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05)
        show_phase = st.radio("Display", ["Stable Phase Only", "Both Phases (Distinct Shapes)"], index=1)
        cmap = st.selectbox("Colormap", COLORMAPS, index=COLORMAPS.index("RdBu_r") if "RdBu_r" in COLORMAPS else 0)

    st.divider()

    # --- GLOBAL EXTRAS ---
    st.subheader("🔷 Reference Geometry")
    show_axes_frame = st.toggle("Coordinate Axes", value=True)
    show_query_probe = st.toggle("Query Probe Sphere", value=True)

    st.divider()
    st.subheader("✏️ Layout")
    template = st.selectbox("Template", ["plotly_white", "plotly_dark", "seaborn", "simple_white"], index=0)
    bg_color = st.color_picker("Background", "#ffffff")
    title_font = st.slider("Title Font", 12, 24, 16)

    st.divider()
    st.caption(f"Data: {len(T_list)} temperatures | {len(df):,} rows")

# =============================================
# QUERY EVALUATION
# =============================================
query_result = None
if eval_query:
    if comp_sum > 1.0:
        st.error("Composition sum exceeds 1.0")
    else:
        interp_liq_q, interp_fcc_q = build_interpolators_for_T(df, T_val)
        if interp_liq_q is not None:
            pt = np.array([[q_co, q_cr, q_fe]])
            g_liq_q = float(interp_liq_q(pt)[0])
            g_fcc_q = float(interp_fcc_q(pt)[0])
            if np.isnan(g_liq_q) or np.isnan(g_fcc_q):
                st.error("Query point outside data convex hull")
            else:
                g_stable_q = min(g_liq_q, g_fcc_q)
                phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"
                dG_q = g_liq_q - g_fcc_q
                query_result = {
                    "T": T_val, "Co": q_co, "Cr": q_cr, "Fe": q_fe,
                    "Ni": 1.0 - comp_sum,
                    "G_LIQ": g_liq_q, "G_FCC": g_fcc_q,
                    "G_stable": g_stable_q, "Phase": phase_q, "dG": dG_q
                }
        else:
            st.error(f"No data for T={T_val}K")

# Display query results
if query_result:
    st.success(f"Query at T={query_result['T']}K, x_Ni={query_result['Ni']:.3f}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol",
              delta_color="inverse" if query_result['dG'] < 0 else "normal")
    c4.metric("Stable", query_result['Phase'],
              delta="LIQUID" if query_result['Phase']=="LIQUID" else "FCC")
    c5.metric("Driving Force", f"{abs(query_result['dG']):,.0f}", "J/mol")
    st.divider()

# =============================================
# MAIN FIGURE
# =============================================
interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
if interp_liq is None:
    st.error(f"No interpolator for T={T_val}K")
    st.stop()

fig = go.Figure()

# ------------------------------------------------------------------
# MODE 1: PHASE BOUNDARY (SCIENTIFIC - TETRAHEDRAL)
# ------------------------------------------------------------------
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

    # Phase volume - distinct shapes per phase
    for phase in ["LIQUID", "FCC"]:
        mask = stable == phase
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
            mode="markers",
            marker=dict(
                size=volume_size,
                color=PHASE_COLORS[phase],
                symbol=PHASE_SYMBOLS[phase],
                opacity=volume_opacity,
                line=dict(width=0.5, color="white")
            ),
            name=f"{phase} Region",
            hovertemplate=(f"<b>{phase}</b><br>" +
                           "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                           f"ΔG={{'+' if phase=='FCC' else ''}}%{{customdata:.0f}} J/mol<extra></extra>"),
            customdata=dG[mask]
        ))

    # Phase boundary (ΔG ≈ 0) - GOLD line for clarity
    boundary_pts, boundary_dG = find_phase_boundary_points(pts, dG, boundary_threshold)
    if len(boundary_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
            mode="markers",
            marker=dict(
                size=4, color="#f1c40f", symbol="x",
                line=dict(width=1, color="#b7950b")
            ),
            name="ΔG = 0 Boundary",
            hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG ≈ 0<br>x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<extra></extra>"
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
        # Vertex labels
        vertices = [(1,0,0,"Co"), (0,1,0,"Cr"), (0,0,1,"Fe"), (0,0,0,"Ni")]
        for vx, vy, vz, vl in vertices:
            fig.add_trace(go.Scatter3d(
                x=[vx], y=[vy], z=[vz], mode="text", text=[vl],
                textposition="top center", textfont=dict(size=14, color="black"),
                hoverinfo="skip", showlegend=False
            ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 2: DUAL SH SURFACES (AESTHETIC)
# ------------------------------------------------------------------
elif render_mode == "Dual SH Surfaces (Aesthetic)" and SCIPY_AVAILABLE:
    TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
        interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
    )

    # Fit SH for each phase separately
    coeffs_liq, l_max = fit_sh_coeffs(TH, PH, interp_liq(sphere_pts).reshape(TH.shape), l_max=sh_l_max)
    coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc(sphere_pts).reshape(TH.shape), l_max=sh_l_max)

    if coeffs_liq is not None and coeffs_fcc is not None:
        G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max)
        G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max)

        # Normalize for deformation
        def normalize_g(g):
            g_min, g_max = np.nanmin(g), np.nanmax(g)
            return (g - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(g)

        # LIQUID surface: warm, smooth, solid
        R_liq = sh_R_fixed * (0.9 + 0.2 * normalize_g(G_liq_sh))
        X_liq = R_liq * np.sin(PH) * np.cos(TH)
        Y_liq = R_liq * np.sin(PH) * np.sin(TH)
        Z_liq = R_liq * np.cos(PH)

        fig.add_trace(go.Surface(
            x=X_liq, y=Y_liq, z=Z_liq,
            surfacecolor=G_liq_sh,
            colorscale=[[0, "#ffcccc"], [1, "#cc0000"]],  # Warm red scale
            cmin=G_global_min, cmax=G_global_max,
            opacity=liq_opacity,
            name="G_LIQ Surface",
            showscale=False,
            hovertemplate="<b>LIQUID</b><br>G_LIQ=%{surfacecolor:,.0f} J/mol<extra></extra>",
            lighting=dict(ambient=0.5, diffuse=0.5, roughness=0.2, specular=0.8),
            lightposition=dict(x=100, y=100, z=50)
        ))

        # FCC surface: cool, wireframe-style (using lower opacity + grid)
        R_fcc = sh_R_fixed * (0.85 + 0.2 * normalize_g(G_fcc_sh))
        X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
        Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
        Z_fcc = R_fcc * np.cos(PH)

        fig.add_trace(go.Surface(
            x=X_fcc, y=Y_fcc, z=Z_fcc,
            surfacecolor=G_fcc_sh,
            colorscale=[[0, "#ccccff"], [1, "#0000cc"]],  # Cool blue scale
            cmin=G_global_min, cmax=G_global_max,
            opacity=fcc_opacity,
            name="G_FCC Surface",
            showscale=False,
            hovertemplate="<b>FCC</b><br>G_FCC=%{surfacecolor:,.0f} J/mol<extra></extra>",
            contours=dict(
                x=dict(show=True, color="blue", width=2, highlight=False),
                y=dict(show=True, color="blue", width=2, highlight=False),
                z=dict(show=True, color="blue", width=2, highlight=False)
            ),
            lighting=dict(ambient=0.6, diffuse=0.4, roughness=0.6, specular=0.2)
        ))

        # ΔG = 0 contour (gold boundary line)
        if show_dg_contour:
            cx, cy, cz = extract_dg_zero_contour(TH, PH, dG_grid, sh_R_fixed)
            if len(cx) > 0:
                fig.add_trace(go.Scatter3d(
                    x=cx, y=cy, z=cz,
                    mode="markers",
                    marker=dict(size=5, color="#f1c40f", symbol="diamond",
                                line=dict(width=2, color="#b7950b")),
                    name="ΔG = 0 Boundary",
                    hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG = 0<extra></extra>"
                ))
    else:
        st.warning("SH fitting failed. Try lower l_max or check data coverage.")

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 3: ΔG DIFFERENCE SURFACE
# ------------------------------------------------------------------
elif render_mode == "ΔG Difference Surface" and SCIPY_AVAILABLE:
    TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
        interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
    )

    coeffs_dG, l_max = fit_sh_coeffs(TH, PH, dG_grid, l_max=sh_l_max)
    if coeffs_dG is not None:
        dG_smooth = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max)

        # Radius = base + scale * ΔG (positive = FCC bulges out, negative = LIQUID dents in)
        dG_norm = dG_smooth / (dG_global_abs_max + 1e-12)
        radius = sh_R_fixed + dg_scale * dG_smooth
        radius = np.clip(radius, 0.1, 2.0)  # Prevent negative/too large

        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)

        # Color by stable phase: red for LIQUID (dG<0), blue for FCC (dG>0)
        surfacecolor = dG_smooth

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surfacecolor,
            colorscale="RdBu_r",  # Red=LIQUID, Blue=FCC
            cmin=-dG_global_abs_max, cmax=dG_global_abs_max,
            opacity=0.9,
            name="ΔG Surface",
            colorbar=dict(
                title=dict(text="ΔG = G_LIQ - G_FCC (J/mol)", font=dict(size=12)),
                thickness=20, len=0.7
            ),
            hovertemplate="<b>ΔG Surface</b><br>ΔG=%{surfacecolor:,.0f} J/mol<br>" +
                          ("LIQUID stable" if "%{surfacecolor}" < "0" else "FCC stable") + "<extra></extra>",
            contours=dict(
                z=dict(show=True, highlightcolor="#f1c40f", highlightwidth=3,
                       project=dict(z=True), usecolormap=False, color="#f1c40f")
            )
        ))

        st.info("""
        **How to read this:** The surface is a sphere deformed by ΔG magnitude.  
        🔴 **Red regions** (dented inward) = LIQUID stable (negative ΔG).  
        🔵 **Blue regions** (bulged outward) = FCC stable (positive ΔG).  
        The **gold contour** marks ΔG = 0.
        """)
    else:
        st.warning("SH fitting failed for ΔG")

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 4: TERNARY FLAT PROJECTION
# ------------------------------------------------------------------
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

    # Project: x=Co, y=Cr, z=Fe but we use z for Ni or ΔG or flat
    if flat_color_by == "Stable Phase":
        colors = [PHASE_COLORS[p] for p in stable]
        colorbar_title = "Phase"
    elif flat_color_by == "ΔG (diverging)":
        colors = dG
        colorbar_title = "ΔG (J/mol)"
    else:
        colors = np.minimum(G_liq, G_fcc)
        colorbar_title = "G_stable (J/mol)"

    # Use z-axis for Ni content to give slight 3D feel, or flat
    z_data = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]  # Ni

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=z_data,
        mode="markers",
        marker=dict(
            size=flat_marker_size,
            color=colors,
            colorscale="RdBu_r" if flat_color_by == "ΔG (diverging)" else None,
            cmin=-dG_global_abs_max if flat_color_by == "ΔG (diverging)" else None,
            cmax=dG_global_abs_max if flat_color_by == "ΔG (diverging)" else None,
            opacity=flat_opacity,
            symbol=[PHASE_SYMBOLS[p] for p in stable],
            line=dict(width=0.5, color="white")
        ),
        name="Ternary View",
        hovertemplate=("x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Ni=%{z:.3f}<br>" +
                       "Phase=%{text}<extra></extra>"),
        text=stable
    ))

    if show_ternary_grid:
        # Add grid lines at constant Ni
        for ni in [0.0, 0.25, 0.5, 0.75]:
            mask = np.abs(z_data - ni) < 0.02
            if mask.sum() > 10:
                fig.add_trace(go.Scatter3d(
                    x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                    mode="markers", marker=dict(size=1, color="gray", opacity=0.3),
                    hoverinfo="skip", showlegend=False
                ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Ni</sub>"

# ------------------------------------------------------------------
# MODE 5: MARKERS (POINT CLOUD - IMPROVED)
# ------------------------------------------------------------------
else:
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

    if show_phase == "Stable Phase Only":
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=G_stable,
                colorscale=cmap,
                opacity=opacity,
                line=dict(width=1, color="white")
            ),
            name="Stable Phase",
            hovertemplate="<b>%{text}</b><br>G=%{marker.color:,.0f} J/mol<extra></extra>",
            text=stable
        ))
    else:
        # DISTINCT SHAPES per phase - this is the key improvement
        for phase in ["LIQUID", "FCC"]:
            if phase == "LIQUID":
                mask = G_liq <= G_fcc
                g_vals = G_liq[mask]
            else:
                mask = G_fcc < G_liq
                g_vals = G_fcc[mask]

            if mask.sum() == 0:
                continue

            fig.add_trace(go.Scatter3d(
                x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=PHASE_COLORS[phase],
                    symbol=PHASE_SYMBOLS[phase],
                    opacity=opacity,
                    line=dict(width=1, color="white")
                ),
                name=f"{phase} Phase",
                hovertemplate=(f"<b>{phase}</b><br>" +
                               "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                               f"G_{phase}=%{{marker.color:,.0f}} J/mol<extra></extra>"),
                customdata=g_vals
            ))

        # Add ΔG = 0 boundary points in gold
        boundary_mask = np.abs(dG) < 100
        if boundary_mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=pts[boundary_mask, 0], y=pts[boundary_mask, 1], z=pts[boundary_mask, 2],
                mode="markers",
                marker=dict(size=6, color="#f1c40f", symbol="x",
                            line=dict(width=2, color="#b7950b")),
                name="ΔG ≈ 0 Boundary",
                hovertemplate="<b>BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
            ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# COMMON OVERLAYS (Query point, axes, etc.)
# ------------------------------------------------------------------

# Query point with phase-appropriate styling
if query_result is not None:
    q_color = PHASE_COLORS[query_result["Phase"]]
    q_symbol = PHASE_SYMBOLS[query_result["Phase"]]

    fig.add_trace(go.Scatter3d(
        x=[query_result["Co"]], y=[query_result["Cr"]], z=[query_result["Fe"]],
        mode="markers+text",
        marker=dict(size=16, color=q_color, symbol=q_symbol,
                    line=dict(width=3, color="white")),
        text=["QUERY"],
        textposition="top center",
        textfont=dict(size=12, color=q_color),
        name=f"Query ({query_result['Phase']})",
        hovertemplate=(f"<b>QUERY</b><br>T={query_result['T']}K<br>" +
                       f"x_Co={query_result['Co']:.3f}<br>x_Cr={query_result['Cr']:.3f}<br>" +
                       f"x_Fe={query_result['Fe']:.3f}<br>x_Ni={query_result['Ni']:.3f}<br>" +
                       f"G_stable={query_result['G_stable']:,.0f}<br>ΔG={query_result['dG']:,.0f}<br>" +
                       f"Phase={query_result['Phase']}<extra></extra>")
    ))

    # Query probe sphere
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
        x_line = [0, 1.0 if coord==0 else 0]
        y_line = [0, 1.0 if coord==1 else 0]
        z_line = [0, 1.0 if coord==2 else 0]
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode="lines+text",
            line=dict(color=color, width=5),
            text=["", label],
            textposition="top center",
            textfont=dict(size=14, color=color, family="Arial Black"),
            hoverinfo="skip", showlegend=False
        ))

# ------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------
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
        text=f"Co-Cr-Fe-Ni at T = {T_val} K | {render_mode} | {phase_dominant}",
        font=dict(size=title_font)
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
    )
)

try:
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Render error: {e}")

# =============================================
# EXPLANATION FOOTER
# =============================================
with st.expander("📖 How to Interpret Each Mode", expanded=True):
    st.markdown("""
    ### Phase Boundary (Scientific) — **Recommended**
    Uses the actual **Co-Cr-Fe-Ni tetrahedral composition space**.  
    - 🔴 **Red circles** = LIQUID-stable compositions  
    - 🔵 **Blue diamonds** = FCC-stable compositions  
    - 🟡 **Gold X's** = ΔG ≈ 0 phase boundary (the transition frontier)  
    This is the most thermodynamically faithful representation.

    ### Dual SH Surfaces (Aesthetic)
    Two spherical harmonic surfaces with **distinct visual styles**:  
    - 🔴 **LIQUID**: Warm red, smooth, specular (liquid-like shine)  
    - 🔵 **FCC**: Cool blue, wireframe contours, diffuse (crystalline feel)  
    - 🟡 **Gold diamonds** = ΔG = 0 contour line where they intersect  
    *Note: Spherical projection conflates composition and energy; use for qualitative intuition only.*

    ### ΔG Difference Surface
    A single sphere **deformed by ΔG magnitude**:  
    - 🔴 **Red / dented inward** = LIQUID favored (negative ΔG)  
    - 🔵 **Blue / bulged outward** = FCC favored (positive ΔG)  
    The deformation amplitude shows **driving force strength**.

    ### Ternary Flat Projection
    Standard materials-science view: x=Co, y=Cr, z=Ni.  
    Marker **shape** encodes phase (circle vs diamond), **color** encodes ΔG or stable phase.

    ### Markers (Point Cloud)
    Classic 3D scatter with **distinct shapes per phase** — circles for LIQUID, diamonds for FCC.
    """)
