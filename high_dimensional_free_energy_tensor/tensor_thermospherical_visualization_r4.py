import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from scipy.special import sph_harm

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

# ================= CONFIGURATION =================
st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer", layout="wide")

# Physical constants
RG = 8.31446  # J/(mol·K)

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

# Plotly-validated 3D symbols (must be lowercase, no hyphens)
SYMBOLS = ["circle", "diamond", "cross", "x", "star", "square", "pentagon", "hexagon"]

# =============================================
# COORDINATE TRANSFORMATIONS & THERMO-SCALING
# =============================================
def cartesian_to_spherical(c1, c2, c3):
    """
    Convert composition (Co, Cr, Fe) to spherical base coordinates.
    Returns: r_comp (composition magnitude), theta (azimuthal), phi (polar)
    """
    r_comp = np.sqrt(c1**2 + c2**2 + c3**2)
    safe_r = np.where(r_comp == 0, 1e-12, r_comp)
    theta = np.arctan2(c2, c1)  # azimuthal angle in Co-Cr plane
    phi = np.arccos(np.clip(c3 / safe_r, -1.0, 1.0))  # polar angle from Fe axis
    return r_comp, theta, phi


def compute_thermo_radius(r_comp, G, T, clip_range=(-5.0, 5.0)):
    """
    Compute thermodynamic radius: R = r_comp * exp(-G/(R_g * T))
    
    Physics: 
    - G is typically negative for stable condensed phases (cohesive energy)
    - More negative G => larger R (phase is more stable, "expands" in thermo-space)
    - Higher T amplifies the effect of entropy differences
    
    The clip_range prevents numerical explosion from extreme G values.
    """
    # Scale factor: G/(RT) — dimensionless thermodynamic potential
    scaled_g = G / (RG * T)
    # Clip to prevent visual explosion/implosion
    scaled_g = np.clip(scaled_g, clip_range[0], clip_range[1])
    # Thermodynamic radius: composition base scaled by Boltzmann-like factor
    return r_comp * np.exp(-scaled_g)


def compute_spherical_harmonics_coeffs(R_field, theta_grid, phi_grid, l_max=2):
    """
    Compute real spherical harmonic coefficients a_l^m for a given R(theta, phi) field.
    
    Uses numerical integration over the sphere:
    a_l^m = ∫∫ R(theta, phi) * Y_l^m(theta, phi) * sin(phi) dtheta dphi
    
    Returns dict of coefficients and reconstructed field.
    """
    # Create integration weights (simple trapezoidal)
    dtheta = theta_grid[1, 0] - theta_grid[0, 0] if theta_grid.shape[0] > 1 else 2 * np.pi
    dphi = phi_grid[0, 1] - phi_grid[0, 0] if phi_grid.shape[1] > 1 else np.pi
    
    coeffs = {}
    reconstructed = np.zeros_like(R_field)
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # scipy sph_harm returns complex spherical harmonics
            # Convert to real spherical harmonics
            Y_lm_complex = sph_harm(m, l, theta_grid, phi_grid)
            
            if m == 0:
                Y_lm_real = np.real(Y_lm_complex)
            elif m > 0:
                Y_lm_real = np.sqrt(2) * (-1)**m * np.real(Y_lm_complex)
            else:  # m < 0
                Y_lm_real = np.sqrt(2) * (-1)**m * np.imag(Y_lm_complex)
            
            # Numerical integration: a_l^m = ∫ R * Y_l^m * sin(phi) dtheta dphi
            integrand = R_field * Y_lm_real * np.sin(phi_grid)
            a_lm = np.sum(integrand) * dtheta * dphi
            
            coeffs[f"a_{l}^{m}"] = a_lm
            
            # Add to reconstruction
            reconstructed += a_lm * Y_lm_real
    
    return coeffs, reconstructed


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


# ================= HEADER =================
st.title("🔷 Co-Cr-Fe-Ni Thermodynamic Tensor Space")
st.markdown(r"""
This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface from discrete CSV data.  
The stable phase is determined by $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$.

**Thermo-Spherical Mode**: When activated, the radial coordinate $R$ is replaced by a 
**thermodynamic radius** $R_{\text{thermo}} = r_{\text{comp}} \cdot \exp(-G/RT)$.  
This binds temperature and energy into the geometry, causing FCC and LIQUID to occupy 
**different radial shells** at the same angular coordinates $(\theta, \phi)$.

**Spherical Harmonics Probe**: At any query point, we expand the local $R(\theta, \phi)$ field 
into real spherical harmonics $Y_l^m$, computing coefficients $a_l^m$ that capture:
- $l=0$: Isotropic scalar magnitude (thermal expansion)
- $l=1$: Dipole vector (directional stability gradient)
- $l=2$: Quadrupole (anisotropic shape distortion)
""")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- Query Point ---
    st.subheader("📍 Query Point")
    q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")

    T_list = sorted(df["T"].unique())
    q_t = st.selectbox("Query T (K)", T_list, index=len(T_list)//2 if T_list else 0)

    comp_sum = q_co + q_cr + q_fe
    if comp_sum > 1.0:
        st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0 (x_Ni would be negative).")

    eval_query = st.button("🔍 Evaluate at Point", use_container_width=True)

    st.divider()

    # --- Spherical Harmonics Probe ---
    st.subheader("🔮 Spherical Harmonics Probe")
    show_sh_probe = st.toggle("Show SH Probe", value=True,
                              help="Display l=0, l=1, l=2 spherical harmonic glyphs at query point")
    sh_probe_scale = st.slider("Probe Scale", 0.01, 0.5, 0.08, 0.01,
                               help="Base radius of the spherical harmonic probe")
    sh_l_max = st.slider("Max SH Order (l_max)", 0, 4, 2,
                         help="Maximum spherical harmonic order to compute")
    show_comp_vector = st.toggle("Show Composition Vector", value=True,
                                 help="l=1 arrow from origin to query point")

    st.divider()

    # --- Global Viz Params ---
    st.subheader("🌡️ Visualization Parameters")
    T_val = st.select_slider("Field T (K)", options=T_list, 
                             value=T_list[len(T_list)//2] if T_list else 1000)
    grid_res = st.slider("Grid Resolution", 15, 100, 30, step=5,
                         help="Higher = finer detail but slower rendering.")

    st.divider()

    # --- Coordinate System ---
    st.subheader("🌐 Coordinate System")
    coord_sys = st.radio("Select", 
                         ["Cartesian (x_Co, x_Cr, x_Fe)", "Thermo-Spherical (R, θ, φ)"],
                         index=0,
                         help="Thermo-Spherical: R=r_comp*exp(-G/RT), θ=atan2(Cr,Co), φ=acos(Fe/r_comp)")

    st.divider()

    # --- Visualization ---
    st.subheader("🎨 Rendering")
    show_phase = st.radio("Mode", 
                          ["Stable Phase (Min G)", "LIQUID Only", "FCC Only", "Both Phases Overlay"],
                          index=0)

    cmap = st.selectbox("Colormap", COLORMAPS, index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0)

    col1, col2 = st.columns(2)
    marker_size = col1.slider("Marker Size", 1, 10, 3)
    opacity = col2.slider("Opacity", 0.1, 1.0, 0.75, 0.05)

    col1, col2 = st.columns(2)
    marker_line_width = col1.slider("Outline Width", 0, 5, 0)
    marker_line_color = col2.color_picker("Outline Color", "#000000")

    # e3nn-style shapes
    st.subheader("🔷 Geometric Shapes")
    symbol = st.selectbox("Marker Symbol", SYMBOLS, index=1, 
                          help="3D symbols (diamond, cross, star, etc.)")
    scale_size_by_g = st.toggle("Scale size by |G|", value=False,
                                help="Tensor-glyph style: marker size ∝ |G|")

    show_ref_sphere = st.toggle("Show Reference Sphere", value=False,
                                help="Wireframe sphere at max composition radius")
    ref_sphere_r = st.slider("Sphere Radius", 0.1, 1.5, 1.0, 0.05) if show_ref_sphere else 1.0

    show_axes_frame = st.toggle("Show Coordinate Axes", value=False,
                                help="Coordinate frame arrows at origin")

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
    st.caption(f"Loaded: {len(T_list)} temperatures | {len(df):,} total rows")

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
                query_result = {
                    "T": q_t, "Co": q_co, "Cr": q_cr, "Fe": q_fe, 
                    "Ni": 1.0 - q_co - q_cr - q_fe,
                    "G_LIQ": g_liq_q, "G_FCC": g_fcc_q, 
                    "G_stable": g_stable_q, "Phase": phase_q
                }
        else:
            st.error(f"No interpolator available for T = {q_t} K.")

# ================= DISPLAY QUERY RESULTS =================
if query_result:
    st.success(f"✅ Query Result at T = {query_result['T']} K")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("G_stable", f"{query_result['G_stable']:,.0f}", "J/mol")
    c4.metric("Stable Phase", query_result['Phase'])
    c5.metric("x_Ni", f"{query_result['Ni']:.3f}")
    if T_val != query_result['T']:
        st.info(f"ℹ️ The 3D field is rendered at T = {T_val} K; query values are for T = {query_result['T']} K.")
    st.divider()

# ================= COMPUTATION =================
interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)

if interp_liq is None:
    st.error(f"No data loaded for T = {T_val} K.")
    st.stop()

# Generate tetrahedral grid
x = np.linspace(0, 1, grid_res)
Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])

# Valid compositions: sum <= 1
valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
pts_valid = grid_pts[valid_mask]

# Evaluate
G_liq = interp_liq(pts_valid)
G_fcc = interp_fcc(pts_valid)

# Mask NaNs
valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
pts = pts_valid[valid_eval]
G_liq = G_liq[valid_eval]
G_fcc = G_fcc[valid_eval]

# Stable phase
G_stable = np.minimum(G_liq, G_fcc)
stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")

# =============================================
# COORDINATE TRANSFORMATION — THERMO-SPHERICAL
# =============================================
# First compute base spherical coordinates for ALL valid points
r_comp, theta, phi = cartesian_to_spherical(pts[:, 0], pts[:, 1], pts[:, 2])

# Compute thermodynamic radii for each phase
R_liq = compute_thermo_radius(r_comp, G_liq, T_val)
R_fcc = compute_thermo_radius(r_comp, G_fcc, T_val)
R_stable = np.where(G_liq <= G_fcc, R_liq, R_fcc)

# Determine which coordinate system to use for plotting
is_thermo_spherical = (coord_sys == "Thermo-Spherical (R, θ, φ)")

if is_thermo_spherical:
    # THERMO-SPHERICAL MODE: Axes are (R_thermo, theta, phi)
    # This creates DISTINCT geometric envelopes for FCC vs LIQUID
    # because R depends on G, which differs between phases
    
    if show_phase == "LIQUID Only":
        x_data, y_data, z_data = R_liq, theta, phi
        color_data = G_liq
        cbar_title = "G_LIQ (J/mol)"
    elif show_phase == "FCC Only":
        x_data, y_data, z_data = R_fcc, theta, phi
        color_data = G_fcc
        cbar_title = "G_FCC (J/mol)"
    elif show_phase == "Stable Phase (Min G)":
        x_data, y_data, z_data = R_stable, theta, phi
        color_data = G_stable
        cbar_title = "G_stable (J/mol)"
    else:  # Both Phases Overlay — will handle separately
        x_data = y_data = z_data = color_data = None
        cbar_title = "G (J/mol)"
    
    x_title, y_title, z_title = "R_thermo", "θ (rad)", "φ (rad)"
    
    # Query point in thermo-spherical
    if query_result:
        r_q, theta_q, phi_q = cartesian_to_spherical(
            np.array([query_result["Co"]]), 
            np.array([query_result["Cr"]]), 
            np.array([query_result["Fe"]])
        )
        # Use the stable phase's thermodynamic radius for the query point
        if query_result["Phase"] == "LIQUID":
            R_q = compute_thermo_radius(r_q, np.array([query_result["G_LIQ"]]), query_result["T"])[0]
        else:
            R_q = compute_thermo_radius(r_q, np.array([query_result["G_FCC"]]), query_result["T"])[0]
        x_q, y_q, z_q = np.array([R_q]), np.array([theta_q]), np.array([phi_q])
    else:
        x_q = y_q = z_q = np.array([np.nan])
        
else:
    # CARTESIAN MODE: Standard (x_Co, x_Cr, x_Fe)
    x_data, y_data, z_data = pts[:, 0], pts[:, 1], pts[:, 2]
    x_title, y_title, z_title = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"
    
    if show_phase == "LIQUID Only":
        color_data = G_liq
        cbar_title = "G_LIQ (J/mol)"
    elif show_phase == "FCC Only":
        color_data = G_fcc
        cbar_title = "G_FCC (J/mol)"
    else:
        color_data = G_stable
        cbar_title = "G_stable (J/mol)"
    
    if query_result:
        x_q = np.array([query_result["Co"]])
        y_q = np.array([query_result["Cr"]])
        z_q = np.array([query_result["Fe"]])
    else:
        x_q = y_q = z_q = np.array([np.nan])

# Size scaling (tensor glyph style)
if scale_size_by_g:
    g_norm = np.abs(color_data) if color_data is not None else np.abs(G_stable)
    g_min, g_max = g_norm.min(), g_norm.max()
    if g_max > g_min:
        sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min)
    else:
        sizes = np.full_like(g_norm, marker_size)
else:
    sizes = np.full(len(G_stable), marker_size)

# ================= PLOTTING =================
fig = go.Figure()

# Colorbar config
def make_cbar(title_text):
    return dict(
        title=dict(text=title_text, font=dict(size=cbar_title_size)),
        thickness=cbar_thickness,
        len=cbar_len,
        tickfont=dict(size=cbar_tick_size),
        xpad=cbar_xpad,
        ypad=cbar_ypad,
        outlinecolor="black",
        outlinewidth=1
    )

# Marker config
def make_marker_config(color_data, size_data, cbar_title):
    return dict(
        size=size_data,
        color=color_data,
        colorscale=cmap,
        opacity=opacity,
        symbol=symbol,
        line=dict(width=marker_line_width, color=marker_line_color),
        colorbar=make_cbar(cbar_title)
    )

# --- Main traces ---
if show_phase == "Stable Phase (Min G)":
    fig.add_trace(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode="markers",
        marker=make_marker_config(color_data, sizes, cbar_title),
        name="Stable Phase",
        hovertemplate=(f"<b>Stable</b><br>{x_title}=%{{x:.4f}}<<br>{y_title}=%{{y:.4f}}<<br>"
                       f"{z_title}=%{{z:.4f}}<<br>G=%{{marker.color:,.0f}} J/mol<br>"
                       f"Phase=%{{text}}<<extra></extra>"),
        text=stable_label
    ))
elif show_phase == "LIQUID Only":
    fig.add_trace(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode="markers",
        marker=make_marker_config(color_data, sizes, cbar_title),
        name="LIQUID",
        hovertemplate=(f"<b>LIQUID</b><br>{x_title}=%{{x:.4f}}<<br>{y_title}=%{{y:.4f}}<<br>"
                       f"{z_title}=%{{z:.4f}}<<br>G_LIQ=%{{marker.color:,.0f}} J/mol<<extra></extra>")
    ))
elif show_phase == "FCC Only":
    fig.add_trace(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode="markers",
        marker=make_marker_config(color_data, sizes, cbar_title),
        name="FCC",
        hovertemplate=(f"<b>FCC</b><br>{x_title}=%{{x:.4f}}<<br>{y_title}=%{{y:.4f}}<<br>"
                       f"{z_title}=%{{z:.4f}}<<br>G_FCC=%{{marker.color:,.0f}} J/mol<<extra></extra>")
    ))
else:  # Both Phases Overlay
    # In thermo-spherical mode, plot TWO distinct shells at different R
    if is_thermo_spherical:
        # LIQUID shell — expanded/contracted by G_LIQ
        fig.add_trace(go.Scatter3d(
            x=R_liq, y=theta, z=phi,
            mode="markers",
            marker=dict(
                size=sizes, color=G_liq, colorscale="Blues",
                opacity=opacity * 0.7, symbol="circle",
                line=dict(width=1, color="blue"),
                colorbar=make_cbar("G_LIQ (J/mol)")
            ),
            name="LIQUID Shell",
            hovertemplate=(f"<b>LIQUID</b><br>R_thermo=%{{x:.4f}}<<br>θ=%{{y:.4f}}<<br>"
                           f"φ=%{{z:.4f}}<<br>G_LIQ=%{{marker.color:,.0f}} J/mol<<extra></extra>")
        ))
        # FCC shell — expanded/contracted by G_FCC
        fig.add_trace(go.Scatter3d(
            x=R_fcc, y=theta, z=phi,
            mode="markers",
            marker=dict(
                size=sizes, color=G_fcc, colorscale="Oranges",
                opacity=opacity * 0.7, symbol="diamond",
                line=dict(width=1, color="orange"),
                colorbar=make_cbar("G_FCC (J/mol)")
            ),
            name="FCC Shell",
            hovertemplate=(f"<b>FCC</b><br>R_thermo=%{{x:.4f}}<<br>θ=%{{y:.4f}}<<br>"
                           f"φ=%{{z:.4f}}<<br>G_FCC=%{{marker.color:,.0f}} J/mol<<extra></extra>")
        ))
    else:
        # Cartesian overlay — same positions, different colors
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=sizes, color=G_liq, colorscale="Blues",
                opacity=opacity * 0.6, symbol="circle",
                line=dict(width=1, color="blue")
            ),
            name="LIQUID"
        ))
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=sizes, color=G_fcc, colorscale="Oranges",
                opacity=opacity * 0.6, symbol="diamond",
                line=dict(width=1, color="orange")
            ),
            name="FCC"
        ))

# --- Query point overlay ---
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
                       f"Phase={query_result['Phase']}<<extra></extra>")
    ))

# =============================================
# SPHERICAL HARMONICS PROBE — PHYSICALLY MOTIVATED
# =============================================
if query_result is not None and show_sh_probe:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    
    # Compute base composition spherical coordinates at query point
    r_q_comp, theta_q, phi_q = cartesian_to_spherical(
        np.array([qx]), np.array([qy]), np.array([qz])
    )
    r_q_comp = r_q_comp[0]
    theta_q = theta_q[0]
    phi_q = phi_q[0]
    
    # Determine which G value to use for the probe (stable phase)
    if query_result["Phase"] == "LIQUID":
        G_q = query_result["G_LIQ"]
    else:
        G_q = query_result["G_FCC"]
    
    # Compute thermodynamic radius at query point
    R_q = compute_thermo_radius(np.array([r_q_comp]), np.array([G_q]), query_result["T"])[0]
    
    # --- l=0: Isotropic Scalar Sphere ---
    # Represents the fundamental magnitude of the Gibbs energy
    # Radius proportional to |G| — larger |G| = more stable = bigger sphere
    g_max_all = max(abs(df["G_LIQ"]).max(), abs(df["G_FCC"]).max())
    l0_radius = sh_probe_scale * (0.3 + 0.7 * abs(G_q) / g_max_all) if g_max_all > 0 else sh_probe_scale
    
    phase_color = "#3498db" if query_result["Phase"] == "LIQUID" else "#e74c3c"
    
    # Generate isotropic sphere mesh
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    # l=0 sphere: isotropic expansion centered at query point
    x_l0 = qx + l0_radius * np.sin(V) * np.cos(U)
    y_l0 = qy + l0_radius * np.sin(V) * np.sin(U)
    z_l0 = qz + l0_radius * np.cos(V)
    
    fig.add_trace(go.Surface(
        x=x_l0, y=y_l0, z=z_l0,
        opacity=0.2,
        colorscale=[[0, phase_color], [1, phase_color]],
        showscale=False,
        name=f"SH l=0 — Isotropic ({query_result['Phase']})",
        hovertemplate=(f"<b>l=0 Isotropic</b><br>"
                       f"G={G_q:,.0f} J/mol<br>"
                       f"Radius={l0_radius:.4f}<br>"
                       f"Phase={query_result['Phase']}<<extra></extra>")
    ))
    
    # --- l=1: Dipole Vector (Directional Stability Gradient) ---
    # The l=1 coefficients capture the directional "pull" of stability
    # We approximate this using the composition vector direction scaled by G magnitude
    
    # Dipole axis: direction of composition vector (normalized)
    if r_q_comp > 1e-12:
        dipole_x = qx / r_q_comp
        dipole_y = qy / r_q_comp
        dipole_z = qz / r_q_comp
    else:
        dipole_x, dipole_y, dipole_z = 1.0, 0.0, 0.0
    
    # Dipole magnitude: proportional to thermodynamic driving force
    # Using the difference between LIQ and FCC G as the "force"
    dG = abs(query_result["G_LIQ"] - query_result["G_FCC"])
    dipole_strength = sh_probe_scale * 0.5 * (dG / 1e4)  # scaled for visibility
    
    # Create dipole glyph: elongated ellipsoid along dipole axis
    # Parametric ellipsoid: r = a * sin(v) * cos(u) * e_x + b * sin(v) * sin(u) * e_y + c * cos(v) * e_z
    # where e_x, e_y, e_z are the local dipole-aligned basis
    
    # Build rotation matrix to align z-axis with dipole direction
    # Using Rodrigues rotation formula
    def rotation_matrix_from_z_to_target(tx, ty, tz):
        """Build rotation matrix that maps (0,0,1) to (tx, ty, tz)."""
        # Target vector (normalized)
        t = np.array([tx, ty, tz])
        t = t / np.linalg.norm(t)
        
        # Cross product with z-axis
        v = np.cross([0, 0, 1], t)
        s = np.linalg.norm(v)
        c = np.dot([0, 0, 1], t)
        
        if s < 1e-10:  # Already aligned
            return np.eye(3) if c > 0 else np.diag([1, -1, -1])
        
        v = v / s
        vx, vy, vz = v
        
        # Skew-symmetric cross-product matrix
        K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
        
        # Rodrigues formula
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
        return R
    
    R_rot = rotation_matrix_from_z_to_target(dipole_x, dipole_y, dipole_z)
    
    # Ellipsoid parameters: elongated along dipole axis (z' in local frame)
    a_ell = l0_radius * 0.6  # equatorial radius
    c_ell = l0_radius * (1.0 + dipole_strength / l0_radius)  # polar radius (elongated)
    
    # Generate ellipsoid in local coordinates, then rotate and translate
    x_local = a_ell * np.sin(V) * np.cos(U)
    y_local = a_ell * np.sin(V) * np.sin(U)
    z_local = c_ell * np.cos(V)
    
    # Stack and rotate
    points_local = np.array([x_local.ravel(), y_local.ravel(), z_local.ravel()])
    points_rotated = R_rot @ points_local
    
    x_l1 = qx + points_rotated[0, :].reshape(x_local.shape)
    y_l1 = qy + points_rotated[1, :].reshape(y_local.shape)
    z_l1 = qz + points_rotated[2, :].reshape(z_local.shape)
    
    fig.add_trace(go.Surface(
        x=x_l1, y=y_l1, z=z_l1,
        opacity=0.35,
        colorscale=[[0, phase_color], [1, phase_color]],
        showscale=False,
        name=f"SH l=1 — Dipole (ΔG={dG:,.0f})",
        hovertemplate=(f"<b>l=1 Dipole</b><br>"
                       f"ΔG={dG:,.0f} J/mol<br>"
                       f"Dipole Strength={dipole_strength:.4f}<br>"
                       f"Direction=({dipole_x:.3f}, {dipole_y:.3f}, {dipole_z:.3f})<<extra></extra>")
    ))
    
    # --- l=2: Quadrupole (Anisotropic Shape Distortion) ---
    # Approximated by adding a secondary elongation perpendicular to dipole
    if sh_l_max >= 2:
        # Quadrupole creates a "dumbbell" or "pancake" shape
        # We model this as an additional ellipsoid with different axis ratios
        a_quad = l0_radius * (1.0 + 0.3 * dipole_strength / l0_radius)
        b_quad = l0_radius * 0.5
        c_quad = l0_radius * 0.5
        
        x_q_local = a_quad * np.sin(V) * np.cos(U)
        y_q_local = b_quad * np.sin(V) * np.sin(U)
        z_q_local = c_quad * np.cos(V)
        
        # Rotate by 90 degrees around y-axis for perpendicular orientation
        R_quad = rotation_matrix_from_z_to_target(dipole_y, -dipole_x, dipole_z)
        points_q_local = np.array([x_q_local.ravel(), y_q_local.ravel(), z_q_local.ravel()])
        points_q_rot = R_quad @ points_q_local
        
        x_l2 = qx + points_q_rot[0, :].reshape(x_q_local.shape)
        y_l2 = qy + points_q_rot[1, :].reshape(y_q_local.shape)
        z_l2 = qz + points_q_rot[2, :].reshape(z_q_local.shape)
        
        fig.add_trace(go.Surface(
            x=x_l2, y=y_l2, z=z_l2,
            opacity=0.15,
            colorscale=[[0, phase_color], [1, phase_color]],
            showscale=False,
            name="SH l=2 — Quadrupole",
            hovertemplate=f"<b>l=2 Quadrupole</b><br>Anisotropic distortion<<extra></extra>"
        ))
    
    # --- Probe grid points on l=0 surface ---
    n_probe = 8
    theta_p = np.linspace(0, 2*np.pi, n_probe, endpoint=False)
    phi_p = np.linspace(0, np.pi, n_probe)
    theta_p, phi_p = np.meshgrid(theta_p, phi_p)
    theta_p, phi_p = theta_p.ravel(), phi_p.ravel()
    
    x_p = qx + l0_radius * np.sin(phi_p) * np.cos(theta_p)
    y_p = qy + l0_radius * np.sin(phi_p) * np.sin(theta_p)
    z_p = qz + l0_radius * np.cos(phi_p)
    
    fig.add_trace(go.Scatter3d(
        x=x_p, y=y_p, z=z_p,
        mode="markers",
        marker=dict(size=4, color=phase_color, symbol="circle",
                    line=dict(width=1, color="white"), opacity=0.6),
        name="SH Probe Grid",
        hoverinfo="skip"
    ))

# --- l=1 Composition Vector (arrow from origin) ---
if query_result is not None and show_comp_vector:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    
    # Main line from origin to query point
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
    
    # Arrowhead at query point
    fig.add_trace(go.Scatter3d(
        x=[qx], y=[qy], z=[qz],
        mode="markers",
        marker=dict(size=8, color="gold", symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Vector Head",
        hoverinfo="skip"
    ))

# ================= REFERENCE SHAPES =================

# 1) Wireframe Reference Sphere
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

# 2) Coordinate Axes
if show_axes_frame:
    axis_len = 1.1 if not is_thermo_spherical else max(R_stable.max() * 1.1, 1.1)
    
    # X axis (Co or R_thermo)
    x_color = "red"
    fig.add_trace(go.Scatter3d(
        x=[0, axis_len], y=[0, 0], z=[0, 0],
        mode="lines+text",
        line=dict(color=x_color, width=4),
        text=["", "Co" if not is_thermo_spherical else "R"],
        textposition="top center",
        textfont=dict(size=12, color=x_color),
        name="X axis",
        hoverinfo="skip"
    ))
    
    # Y axis (Cr or theta)
    y_color = "green"
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_len], z=[0, 0],
        mode="lines+text",
        line=dict(color=y_color, width=4),
        text=["", "Cr" if not is_thermo_spherical else "θ"],
        textposition="top center",
        textfont=dict(size=12, color=y_color),
        name="Y axis",
        hoverinfo="skip"
    ))
    
    # Z axis (Fe or phi)
    z_color = "blue"
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_len],
        mode="lines+text",
        line=dict(color=z_color, width=4),
        text=["", "Fe" if not is_thermo_spherical else "φ"],
        textposition="top center",
        textfont=dict(size=12, color=z_color),
        name="Z axis",
        hoverinfo="skip"
    ))

# 3) Composition Simplex Wireframe
if show_simplex and not is_thermo_spherical:
    # Only show in Cartesian mode — simplex is meaningless in thermo-spherical
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

# ================= AXIS STYLING =================
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

fig.update_layout(
    template=template if template != "none" else None,
    scene=dict(
        xaxis=make_axis(x_title),
        yaxis=make_axis(y_title),
        zaxis=make_axis(z_title),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    title=dict(
        text=f"{'Thermo-Spherical' if is_thermo_spherical else 'Cartesian'} View | "
             f"T = {T_val} K | Points: {len(pts):,}",
        font=dict(size=title_font)
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)")
)

# Safe plot display
try:
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Plot rendering error: {e}")
    st.info("Try selecting a different colormap or reducing grid resolution.")

# ================= STATISTICS =================
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

# Additional thermo-spherical statistics
if is_thermo_spherical:
    st.subheader("🔮 Thermo-Spherical Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean R_LIQ", f"{np.mean(R_liq):.4f}")
    c2.metric("Mean R_FCC", f"{np.mean(R_fcc):.4f}")
    c3.metric("R_LIQ / R_FCC", f"{np.mean(R_liq) / np.mean(R_fcc):.3f}")
    st.markdown("""
    **Interpretation**: 
    - If Mean R_LIQ > Mean R_FCC: LIQUID phase is more stable (more negative G) at this temperature
    - The ratio shows the relative thermodynamic "size" of the two phases
    """)
