import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

# ================= CONFIGURATION =================
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

# Plotly-validated 3D symbols (must be lowercase, no hyphens)
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
    grid_res = st.slider("Grid Resolution", 15, 500, 25, step=5,
                         help="Higher = finer detail but slower rendering.")

    st.divider()

    # --- Coordinate System ---
    st.subheader("🌐 Coordinate System")
    coord_sys = st.radio("Select", 
                         ["Cartesian (x_Co, x_Cr, x_Fe)", "Spherical (r, θ, φ)"],
                         index=0,
                         help="Spherical: r=√(Co²+Cr²+Fe²), θ=atan2(Cr,Co), φ=acos(Fe/r)")

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
    st.subheader("🔷 e3nn / Geometric Shapes")
    symbol = st.selectbox("Marker Symbol", SYMBOLS, index=1, 
                          help="e3nn-inspired 3D symbols (diamond, cross, star, etc.)")
    scale_size_by_g = st.toggle("Scale size by |G|", value=False,
                                help="Tensor-glyph style: marker size ∝ |G|")

    show_ref_sphere = st.toggle("Show Reference Sphere", value=False,
                                help="Wireframe sphere at max composition radius (e3nn-style)")
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

# Coordinate transformation
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

# Size scaling (tensor glyph style)
if scale_size_by_g:
    g_norm = np.abs(G_stable)
    g_min, g_max = g_norm.min(), g_norm.max()
    if g_max > g_min:
        sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min)
    else:
        sizes = np.full_like(g_norm, marker_size)
else:
    sizes = np.full(len(G_stable), marker_size)

# ================= PLOTTING =================
fig = go.Figure()

# Colorbar config (Plotly 5.x / 6.x compatible)
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

# Marker config - safe dict construction
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

# Query point overlay (always diamond for query)
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
                       f"Phase={query_result['Phase']}<extra></extra>")
    ))

# ================= SPHERICAL HARMONICS PROBE (e3nn-style) =================
if query_result is not None and show_sh_probe:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    g_val = abs(query_result["G_stable"])
    g_max_all = max(abs(df["G_LIQ"]).max(), abs(df["G_FCC"]).max())
    radius = sh_probe_scale * (0.5 + 0.5 * g_val / g_max_all) if g_max_all > 0 else sh_probe_scale

    phase_color = "#3498db" if query_result["Phase"] == "LIQUID" else "#e74c3c"

    # l=0 spherical harmonic: scalar sphere (constant on surface)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sh = qx + radius * np.outer(np.cos(u), np.sin(v))
    y_sh = qy + radius * np.outer(np.sin(u), np.sin(v))
    z_sh = qz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sh, y=y_sh, z=z_sh,
        opacity=0.25,
        colorscale=[[0, phase_color], [1, phase_color]],
        showscale=False,
        name=f"SH l=0 ({query_result['Phase']})",
        hovertemplate=(f"<b>Spherical Harmonic l=0</b><br>"
                       f"G_stable={query_result['G_stable']:,.0f} J/mol<br>"
                       f"Phase={query_result['Phase']}<br>"
                       f"Radius={radius:.4f}<extra></extra>")
    ))

    # Add SH probe points (geometric markers on the sphere surface)
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

# l=1 Composition Vector (arrow from origin)
if query_result is not None and show_comp_vector:
    qx, qy, qz = query_result["Co"], query_result["Cr"], query_result["Fe"]
    # Main line
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
    # Arrowhead (cone approximation using a small sphere)
    fig.add_trace(go.Scatter3d(
        x=[qx], y=[qy], z=[qz],
        mode="markers",
        marker=dict(size=8, color="gold", symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Vector Head",
        hoverinfo="skip"
    ))

# ================= e3nn REFERENCE SHAPES =================

# 1) Wireframe Reference Sphere (e3nn-style)
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

# 2) E(3) Coordinate Axes (arrows/lines from origin)
if show_axes_frame:
    axis_len = 1.1
    # X axis (Co)
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
    # Y axis (Cr)
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
    # Z axis (Fe)
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

# 3) Composition Simplex Wireframe (tetrahedron edges)
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

# Axis styling (Plotly 6.x compatible)
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
        text=f"Gibbs Energy Tensor at T = {T_val} K | Points: {len(pts):,}",
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
