import streamlit as st
import plotly.graph_objects as go
import numpy as np
from itertools import permutations

# --- PAGE SETUP ---
st.set_page_config(page_title="3D Grain Geometry Visualizer", layout="wide")
st.title("3D Tetrakaidecahedron Grain Visualizer")import streamlit as st
import plotly.graph_objects as go
import numpy as np
from itertools import permutations
import math

# --- PAGE SETUP ---
st.set_page_config(page_title="3D Grain Geometry Visualizer", layout="wide")
st.title("3D Tetrakaidecahedron Grain Visualizer")
st.markdown("""
This interactive tool generates a **tetrakaidecahedron** (truncated octahedron), 
the ideal space-filling geometry used to model single-phase FCC alloy grains like **CoCrFeNi**.
""")

# --- GEOMETRY GENERATION ---
# Vertices of a truncated octahedron are all permutations of (0, ±1, ±2)
base_coords = []
for p in set(permutations([0, 1, 2])):
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                x = p[0] * s1 if p[0] != 0 else 0
                y = p[1] * s2 if p[1] != 0 else 0
                z = p[2] * s3 if p[2] != 0 else 0
                base_coords.append((x, y, z))

# Remove duplicates to get exactly 24 unique vertices
vertices = np.array(list(set(base_coords)), dtype=float)

# Define the 14 faces by indexing the vertices array
square_faces = [
    [u for u in range(24) if vertices[u, 0] == 2],
    [u for u in range(24) if vertices[u, 0] == -2],
    [u for u in range(24) if vertices[u, 1] == 2],
    [u for u in range(24) if vertices[u, 1] == -2],
    [u for u in range(24) if vertices[u, 2] == 2],
    [u for u in range(24) if vertices[u, 2] == -2],
]

# Sort planar vertices in circular order so lines/meshes render cleanly
def sort_planar_vertices(face_indices, verts):
    pts = verts[face_indices]
    center = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - center)
    coords_2d = U[:, :2]
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    return [face_indices[i] for i in np.argsort(angles)]

sorted_square_faces = [sort_planar_vertices(f, vertices) for f in square_faces]

# Identify the 8 Hexagonal faces
hex_faces = []
signs = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
for s in signs:
    hex_indices = []
    for u in range(24):
        if np.isclose(s[0]*vertices[u,0] + s[1]*vertices[u,1] + s[2]*vertices[u,2], 3):
            hex_indices.append(u)
    if len(hex_indices) == 6:
        hex_faces.append(sort_planar_vertices(hex_indices, vertices))

all_faces = sorted_square_faces + hex_faces

# ============================================================
# EXPANDED COLORMAP OPTIONS (50+ colormaps)
# ============================================================
COLORMAPS = {
    # Perceptually Uniform Sequential
    "Viridis": "Viridis",
    "Plasma": "Plasma", 
    "Inferno": "Inferno",
    "Magma": "Magma",
    "Cividis": "Cividis",
    "Turbo": "Turbo",
    # Sequential
    "Blues": "Blues",
    "BuGn": "BuGn",
    "BuPu": "BuPu",
    "GnBu": "GnBu",
    "Greens": "Greens",
    "Greys": "Greys",
    "Oranges": "Oranges",
    "OrRd": "OrRd",
    "PuBu": "PuBu",
    "PuBuGn": "PuBuGn",
    "PuRd": "PuRd",
    "Purples": "Purples",
    "RdPu": "RdPu",
    "Reds": "Reds",
    "YlGn": "YlGn",
    "YlGnBu": "YlGnBu",
    "YlOrBr": "YlOrBr",
    "YlOrRd": "YlOrRd",
    # Diverging
    "BrBG": "BrBG",
    "PRGn": "PRGn",
    "PiYG": "PiYG",
    "PuOr": "PuOr",
    "RdBu": "RdBu",
    "RdGy": "RdGy",
    "RdYlBu": "RdYlBu",
    "RdYlGn": "RdYlGn",
    "Spectral": "Spectral",
    # Qualitative
    "Accent": "Accent",
    "Dark2": "Dark2",
    "Paired": "Paired",
    "Pastel1": "Pastel1",
    "Pastel2": "Pastel2",
    "Set1": "Set1",
    "Set2": "Set2",
    "Set3": "Set3",
    # Sequential (2)
    "Autumn": "autumn",
    "Bone": "bone",
    "Cool": "cool",
    "Copper": "copper",
    "Hot": "hot",
    "HSV": "hsv",
    "Jet": "jet",
    "Pink": "pink",
    "Spring": "spring",
    "Summer": "summer",
    "Winter": "winter",
    # Custom named
    "Rainbow": "Rainbow",
    "Electric": "Electric",
    "Blackbody": "Blackbody",
    "Earth": "Earth",
}

# ============================================================
# MICROSTRUCTURAL PARAMETER CALCULATIONS
# ============================================================

def calculate_tetrakaidecahedron_properties(a):
    """
    Calculate geometric properties of a tetrakaidecahedron (truncated octahedron)
    with edge length 'a'.

    For our canonical form with vertices at permutations of (0, ±1, ±2):
    The edge length a = sqrt(2)
    """
    # Edge length for canonical form
    a_canonical = np.sqrt(2)

    # Scaling factor if user provides a different edge length
    scale = a / a_canonical

    # Volume of tetrakaidecahedron: V = 8 * sqrt(2) * a^3
    volume = 8 * np.sqrt(2) * (a ** 3)

    # Surface area: A = 6*(1 + 2*sqrt(3)) * a^2
    # 6 square faces (area a^2 each) + 8 hexagonal faces (area 3*sqrt(3)/2 * a^2 each)
    surface_area = 6 * (a ** 2) + 8 * (3 * np.sqrt(3) / 2) * (a ** 2)
    surface_area = 6 * (1 + 2 * np.sqrt(3)) * (a ** 2)

    # Number of faces, edges, vertices
    n_faces = 14  # 6 squares + 8 hexagons
    n_edges = 36
    n_vertices = 24

    # Euler characteristic
    euler = n_vertices - n_edges + n_faces

    # Equivalent sphere diameter (d)
    # d = (6V/π)^(1/3)
    d_eq = ((6 * volume) / np.pi) ** (1/3)

    # Mean linear intercept (MLI) - stereological parameter
    # For space-filling polyhedra: MLI = 4V / A
    mli = 4 * volume / surface_area

    # Surface area to volume ratio
    sv_ratio = surface_area / volume

    # Mean caliper diameter
    mean_caliper = (3 * volume) / (2 * surface_area) * 4  # Approximation

    # Dihedral angles
    # Square-hexagon dihedral angle
    dihedral_sq_hex = np.arccos(-1 / np.sqrt(3))  # ≈ 125.26°
    # Hexagon-hexagon dihedral angle  
    dihedral_hex_hex = np.arccos(-1 / 3)  # ≈ 109.47°

    return {
        "edge_length": a,
        "volume": volume,
        "surface_area": surface_area,
        "n_faces": n_faces,
        "n_edges": n_edges,
        "n_vertices": n_vertices,
        "euler_characteristic": euler,
        "equivalent_diameter": d_eq,
        "mean_linear_intercept": mli,
        "surface_volume_ratio": sv_ratio,
        "mean_caliper_diameter": mean_caliper,
        "dihedral_sq_hex_deg": np.degrees(dihedral_sq_hex),
        "dihedral_hex_hex_deg": np.degrees(dihedral_hex_hex),
    }

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Visualization Options")

# Font and label controls
st.sidebar.subheader("Font & Label Settings")
font_family = st.sidebar.selectbox(
    "Font Family",
    ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Helvetica", "Palatino", "Garamond"],
    index=0
)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
font_color = st.sidebar.color_picker("Font Color", "#000000")

show_vertices = st.sidebar.checkbox("Show Vertex Labels", value=True)
show_faces = st.sidebar.checkbox("Fill Faces", value=True)
show_wireframe = st.sidebar.checkbox("Show Wireframe", value=True)

# Colormap selection
colormap_name = st.sidebar.selectbox(
    "Face Colormap",
    list(COLORMAPS.keys()),
    index=list(COLORMAPS.keys()).index("Viridis")
)
selected_colormap = COLORMAPS[colormap_name]

opacity = st.sidebar.slider("Face Opacity", 0.1, 1.0, 0.6)

# Edge styling
edge_color = st.sidebar.color_picker("Edge Color", "#000000")
edge_width = st.sidebar.slider("Edge Width", 1, 10, 4)

# Marker styling
marker_size = st.sidebar.slider("Marker Size", 2, 15, 5)
marker_color = st.sidebar.color_picker("Marker Color", "#DC143C")

# ============================================================
# MICROSTRUCTURAL PARAMETERS SIDEBAR
# ============================================================
st.sidebar.header("Microstructural Parameters")

# User input for edge length in micrometers
edge_length_um = st.sidebar.number_input(
    "Edge Length (μm)",
    min_value=0.001,
    max_value=1000.0,
    value=1.0,
    step=0.1,
    format="%.3f"
)

# Calculate properties
props = calculate_tetrakaidecahedron_properties(edge_length_um)

st.sidebar.markdown("---")
st.sidebar.markdown("**Calculated Properties:**")
st.sidebar.markdown(f"- **Volume:** `{props['volume']:.4f}` μm³")
st.sidebar.markdown(f"- **Surface Area:** `{props['surface_area']:.4f}` μm²")
st.sidebar.markdown(f"- **Equivalent Diameter (d):** `{props['equivalent_diameter']:.4f}` μm")
st.sidebar.markdown(f"- **Mean Linear Intercept:** `{props['mean_linear_intercept']:.4f}` μm")
st.sidebar.markdown(f"- **S/V Ratio:** `{props['surface_volume_ratio']:.4f}` μm⁻¹")
st.sidebar.markdown(f"- **Faces:** {props['n_faces']} (6 squares + 8 hexagons)")
st.sidebar.markdown(f"- **Edges:** {props['n_edges']}")
st.sidebar.markdown(f"- **Vertices:** {props['n_vertices']}")
st.sidebar.markdown(f"- **Euler Characteristic:** {props['euler_characteristic']}")
st.sidebar.markdown(f"- **Square-Hex Dihedral:** `{props['dihedral_sq_hex_deg']:.2f}°`")
st.sidebar.markdown(f"- **Hex-Hex Dihedral:** `{props['dihedral_hex_hex_deg']:.2f}°`")

# ============================================================
# PLOTLY 3D OBJECT CONSTRUCTION
# ============================================================
fig = go.Figure()

# 1. Draw Mesh Faces with colormap
if show_faces:
    I, J, K = [], [], []
    face_colors = []

    for idx, face in enumerate(all_faces):
        for t in range(1, len(face) - 1):
            I.append(face[0])
            J.append(face[t])
            K.append(face[t+1])
            # Assign color based on face index for colormap
            face_colors.append(idx)

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=I, j=J, k=K,
        opacity=opacity,
        colorscale=selected_colormap,
        intensity=face_colors,
        colorbar=dict(title="Face Index", titleside="right"),
        name="Grain Volume",
        showlegend=True,
        hovertemplate="Face Index: %{intensity}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
    ))

# 2. Draw Edges (Wireframe)
if show_wireframe:
    for face in all_faces:
        loop = face + [face[0]]
        edge_coords = vertices[loop]
        fig.add_trace(go.Scatter3d(
            x=edge_coords[:, 0], y=edge_coords[:, 1], z=edge_coords[:, 2],
            mode='lines',
            line=dict(color=edge_color, width=edge_width),
            showlegend=False,
            hoverinfo='skip'
        ))

# 3. Draw Nodes / Vertex Labels
if show_vertices:
    labels = [f"V{i}: ({vertices[i,0]:.0f},{vertices[i,1]:.0f},{vertices[i,2]:.0f})" for i in range(24)]
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers+text',
        marker=dict(size=marker_size, color=marker_color),
        text=[f"V{i}" for i in range(24)],
        textposition="top center",
        textfont=dict(family=font_family, size=font_size, color=font_color),
        hovertext=labels,
        name="Vertices / Triple Junctions"
    ))

# --- LAYOUT CONFIGURATION ---
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)",
                   titlefont=dict(family=font_family, size=font_size+2, color=font_color)),
        yaxis=dict(title='Y (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)",
                   titlefont=dict(family=font_family, size=font_size+2, color=font_color)),
        zaxis=dict(title='Z (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)",
                   titlefont=dict(family=font_family, size=font_size+2, color=font_color)),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
    font=dict(family=font_family, size=font_size, color=font_color),
    legend=dict(
        font=dict(family=font_family, size=font_size, color=font_color)
    )
)

# --- DISPLAY IN STREAMLIT ---
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# THEORETICAL SECTION WITH LATEX RENDERING
# ============================================================
st.header("📐 Microstructural Theory & Geometric Properties")

with st.expander("🔬 Tetrakaidecahedron (Truncated Octahedron) - Mathematical Foundation", expanded=True):
    st.markdown("""
    The **tetrakaidecahedron** (Greek: *tetra* = four, *kai* = and, *deca* = ten, *hedron* = face) 
    is a space-filling polyhedron with exactly **14 faces**: 6 squares and 8 regular hexagons. 
    It is also known as the **truncated octahedron**, an Archimedean solid.
    """)

    st.subheader("Vertex Generation")
    st.markdown("The 24 vertices are all permutations of $(0, \pm 1, \pm 2)$:")
    st.latex(r"""
    V = \left\{ (\pm 2, \pm 1, 0), (\pm 2, 0, \pm 1), (\pm 1, \pm 2, 0), 
    (\pm 1, 0, \pm 2), (0, \pm 2, \pm 1), (0, \pm 1, \pm 2) \right\}
    """)

    st.subheader("Geometric Properties")
    st.markdown("For a tetrakaidecahedron with edge length $a$:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Volume:**")
        st.latex(r"V = 8\sqrt{2} \, a^3")

        st.markdown("**Surface Area:**")
        st.latex(r"A = 6(1 + 2\sqrt{3}) \, a^2")

        st.markdown("**Equivalent Sphere Diameter:**")
        st.latex(r"d_{eq} = \left(\frac{6V}{\pi}\right)^{1/3} = \left(\frac{48\sqrt{2}}{\pi}\right)^{1/3} a")

    with col2:
        st.markdown("**Mean Linear Intercept:**")
        st.latex(r"\bar{L} = \frac{4V}{A} = \frac{16\sqrt{2}}{6(1+2\sqrt{3})} \, a")

        st.markdown("**Surface-to-Volume Ratio:**")
        st.latex(r"\frac{S}{V} = \frac{6(1+2\sqrt{3})}{8\sqrt{2} \, a}")

        st.markdown("**Number of Elements:**")
        st.latex(r"F = 14, \quad E = 36, \quad V = 24")

    st.markdown("**Euler Characteristic:**")
    st.latex(r"\chi = V - E + F = 24 - 36 + 14 = 2")

with st.expander("🔩 Application to CoCrFeNi High-Entropy Alloy (HEA)", expanded=True):
    st.markdown("""
    ### CoCrFeNi Context

    **CoCrFeNi** is a single-phase **Face-Centered Cubic (FCC)** high-entropy alloy (HEA) 
    that exhibits exceptional mechanical properties, corrosion resistance, and thermal stability. 
    The tetrakaidecahedron model is particularly relevant for this alloy system because:

    1. **Space-Filling Efficiency:** The truncated octahedron is the only Archimedean solid 
       that can fill 3D space without gaps, making it the idealized grain shape for 
       equiaxed microstructures.

    2. **Surface Energy Minimization:** In single-phase alloys like CoCrFeNi, grain boundaries 
       tend to minimize their total interfacial energy. The tetrakaidecahedron approximates 
       the equilibrium shape predicted by the **Wulff construction** for isotropic surface energy.

    3. **Triple Junction Geometry:** The 24 vertices represent **quadruple points** where 
       4 grains meet in 3D, while the 36 edges represent **triple lines** where 3 grains intersect.
    """)

    st.subheader("Dihedral Angles")
    st.markdown("""
    The equilibrium dihedral angles at grain boundary intersections are determined by 
    the balance of surface tensions (Young's equation):
    """)
    st.latex(r"""
    \cos\left(\frac{\phi_{sq-hx}}{2}\right) = \frac{1}{\sqrt{3}} 
    \quad \Rightarrow \quad \phi_{sq-hx} = \arccos\left(-\frac{1}{\sqrt{3}}\right) \approx 125.26°
    """)
    st.latex(r"""
    \cos\left(\frac{\phi_{hx-hx}}{2}\right) = \frac{1}{3} 
    \quad \Rightarrow \quad \phi_{hx-hx} = \arccos\left(-\frac{1}{3}\right) \approx 109.47°
    """)

    st.subheader("Grain Size Relationships")
    st.markdown("""
    For a polycrystalline CoCrFeNi microstructure with average grain size $d_{eq}$:
    """)
    st.latex(r"""
    \text{Hall-Petch Relation:} \quad \sigma_y = \sigma_0 + \frac{k_y}{\sqrt{d_{eq}}}
    """)
    st.markdown("""
    where:
    - $\sigma_y$ = yield strength
    - $\sigma_0$ = lattice friction stress  
    - $k_y$ = Hall-Petch coefficient (material constant)
    - $d_{eq}$ = equivalent grain diameter (μm)
    """)

with st.expander("📊 Stereological Relationships", expanded=False):
    st.markdown("""
    ### Quantitative Microstructure Analysis

    Stereology provides the mathematical framework to relate 3D microstructural features 
    to 2D measurements from metallographic cross-sections.
    """)

    st.subheader("Fundamental Stereological Equations")
    st.latex(r"""
    \text{Volume Fraction:} \quad V_V = A_A = L_L = P_P
    """)
    st.latex(r"""
    \text{Surface Area per Volume:} \quad S_V = \frac{4}{\pi} \cdot \bar{L}_A = 2P_L
    """)
    st.latex(r"""
    \text{Mean Linear Intercept:} \quad \bar{L} = \frac{4V_V}{S_V} = \frac{2}{P_L}
    """)

    st.subheader("Tetrakaidecahedron-Specific Relations")
    st.latex(r"""
    \text{For space-filling tetrakaidecahedra:} \quad \bar{L} = \frac{4V}{A} = \frac{16\sqrt{2}}{6(1+2\sqrt{3})} a \approx 0.779 \cdot a
    """)
    st.latex(r"""
    \text{Grain boundary area per unit volume:} \quad S_V = \frac{A}{V} = \frac{6(1+2\sqrt{3})}{8\sqrt{2} \, a} \approx 1.283 \cdot \frac{1}{a}
    """)

with st.expander("🧮 Current Calculation Results", expanded=True):
    st.subheader(f"Results for Edge Length $a = {edge_length_um:.3f}$ μm")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Volume", f"{props['volume']:.4f} μm³")
        st.metric("Surface Area", f"{props['surface_area']:.4f} μm²")
        st.metric("Equivalent Diameter (d)", f"{props['equivalent_diameter']:.4f} μm")

    with res_col2:
        st.metric("Mean Linear Intercept", f"{props['mean_linear_intercept']:.4f} μm")
        st.metric("S/V Ratio", f"{props['surface_volume_ratio']:.4f} μm⁻¹")
        st.metric("Mean Caliper Diameter", f"{props['mean_caliper_diameter']:.4f} μm")

    with res_col3:
        st.metric("Square-Hex Dihedral", f"{props['dihedral_sq_hex_deg']:.2f}°")
        st.metric("Hex-Hex Dihedral", f"{props['dihedral_hex_hex_deg']:.2f}°")
        st.metric("Euler Characteristic", f"{props['euler_characteristic']}")

st.info("""
💡 **Microstructure Context:** 
- The **Crimson Markers (Vertices)** represent structural triple junctions where 4 separate grains meet at a point in 3D space.
- The **Black Outlines (Edges)** represent the intersecting boundaries where 3 individual grains share an edge interface. 
- You can left-click and drag to rotate the cell, scroll to zoom, or use the sidebar controls to customize the display parameters.
""")
st.markdown("""
This interactive tool generates a **tetrakaidecahedron** (truncated octahedron), 
the ideal space-filling geometry used to model single-phase FCC alloy grains like **CoCrFeNi**.
""")

# --- GEOMETRY GENERATION ---
# Vertices of a truncated octahedron are all permutations of (0, ±1, ±2)
base_coords = []
for p in set(permutations([0, 1, 2])):
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                x = p[0] * s1 if p[0] != 0 else 0
                y = p[1] * s2 if p[1] != 0 else 0
                z = p[2] * s3 if p[2] != 0 else 0
                base_coords.append((x, y, z))

# Remove duplicates to get exactly 24 unique vertices
vertices = np.array(list(set(base_coords)), dtype=float)

# Define the 14 faces by indexing the vertices array
square_faces = [
    [u for u in range(24) if vertices[u, 0] == 2],
    [u for u in range(24) if vertices[u, 0] == -2],
    [u for u in range(24) if vertices[u, 1] == 2],
    [u for u in range(24) if vertices[u, 1] == -2],
    [u for u in range(24) if vertices[u, 2] == 2],
    [u for u in range(24) if vertices[u, 2] == -2],
]

# Sort planar vertices in circular order so lines/meshes render cleanly
def sort_planar_vertices(face_indices, verts):
    pts = verts[face_indices]
    center = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - center)
    coords_2d = U[:, :2]
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    return [face_indices[i] for i in np.argsort(angles)]

sorted_square_faces = [sort_planar_vertices(f, vertices) for f in square_faces]

# Identify the 8 Hexagonal faces
hex_faces = []
signs = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
for s in signs:
    hex_indices = []
    for u in range(24):
        if np.isclose(s[0]*vertices[u,0] + s[1]*vertices[u,1] + s[2]*vertices[u,2], 3):
            hex_indices.append(u)
    if len(hex_indices) == 6:
        hex_faces.append(sort_planar_vertices(hex_indices, vertices))

all_faces = sorted_square_faces + hex_faces

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Visualization Options")
show_vertices = st.sidebar.checkbox("Show Vertex Labels", value=True)
show_faces = st.sidebar.checkbox("Fill Faces", value=True)
opacity = st.sidebar.slider("Face Opacity", 0.1, 1.0, 0.6)

# --- PLOTLY 3D OBJECT CONSTRUCTION ---
# Initialize a proper Figure container object
fig = go.Figure()

# 1. Draw Mesh Faces
if show_faces:
    I, J, K = [], [], []
    # Triangulate our polygons for the Mesh3d engine
    for face in all_faces:
        for t in range(1, len(face) - 1):
            I.append(face[0])
            J.append(face[t])
            K.append(face[t+1])

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=I, j=J, k=K,
        opacity=opacity,
        color='rgb(100, 180, 255)',
        name="Grain Volume",
        showlegend=True
    ))

# 2. Draw Edges (Wireframe)
for face in all_faces:
    loop = face + [face[0]]
    edge_coords = vertices[loop]
    fig.add_trace(go.Scatter3d(
        x=edge_coords[:, 0], y=edge_coords[:, 1], z=edge_coords[:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        showlegend=False,
        hoverinfo='skip'
    ))

# 3. Draw Nodes / Vertex Labels
if show_vertices:
    labels = [f"V{i}: ({vertices[i,0]:.0f},{vertices[i,1]:.0f},{vertices[i,2]:.0f})" for i in range(24)]
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='crimson'),
        text=[f"V{i}" for i in range(24)],
        textposition="top center",
        hovertext=labels,
        name="Vertices / Triple Junctions"
    ))

# --- LAYOUT CONFIGURATION ---
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(title='Y (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(title='Z (Crystal Axis)', backgroundcolor="rgba(0,0,0,0)"),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
)

# --- DISPLAY IN STREAMLIT ---
st.plotly_chart(fig, use_container_width=True)

st.info("""
💡 **Microstructure Context:** 
- The **Crimson Markers (Vertices)** represent structural triple junctions where 4 separate grains meet at a point in 3D space.
- The **Black Outlines (Edges)** represent the intersecting boundaries where 3 individual grains share an edge interface. 
- You can left-click and drag to rotate the cell, scroll to zoom, or use the sidebar controls to customize the display parameters.
""")
