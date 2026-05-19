import streamlit as st
import plotly.graph_objects as px
import numpy as np
from itertools import permutations

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
                # Skip multiplying by zero sign changes to prevent duplicates
                x = p[0] * s1 if p[0] != 0 else 0
                y = p[1] * s2 if p[1] != 0 else 0
                z = p[2] * s3 if p[2] != 0 else 0
                base_coords.append((x, y, z))

# Remove duplicates to get exactly 24 unique vertices
vertices = np.array(list(set(base_coords)), dtype=float)

# Define the 14 faces by indexing the vertices array
# 6 Square faces
squares = [
    [x for x in range(24) if np.isclose(vertices[x, 0], 2) or np.isclose(vertices[x, 0], -2)],
    [x for x in range(24) if np.isclose(vertices[x, 1], 2) or np.isclose(vertices[x, 1], -2)],
    [x for x in range(24) if np.isclose(vertices[x, 2], 2) or np.isclose(vertices[x, 2], -2)]
]
# Split grouped coordinate extremes into individual 4-vertex square loops
square_faces = [
    [u for u in range(24) if vertices[u, 0] == 2],
    [u for u in range(24) if vertices[u, 0] == -2],
    [u for u in range(24) if vertices[u, 1] == 2],
    [u for u in range(24) if vertices[u, 1] == -2],
    [u for u in range(24) if vertices[u, 2] == 2],
    [u for u in range(24) if vertices[u, 2] == -2],
]

# Sort square face vertices in circular order so they render correctly
def sort_planar_vertices(face_indices, verts):
    pts = verts[face_indices]
    center = pts.mean(axis=0)
    # Find the two principal axes of variation for the plane
    U, S, Vt = np.linalg.svd(pts - center)
    coords_2d = U[:, :2]
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    return [face_indices[i] for i in np.argsort(angles)]

sorted_square_faces = [sort_planar_vertices(f, vertices) for f in square_faces]

# 8 Hexagonal faces are defined by cutting the corners of an octahedron (x + y + z = constant)
hex_faces = []
signs = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
for s in signs:
    face = [u for u in range(24) if np.isclose(abs(vertices[u,0]) + abs(vertices[u,1]) + abs(vertices[u,2]), 3) 
            and np.sign(vertices[u,0])*s[0] >= 0 
            and np.sign(vertices[u,1])*s[s[1]>=0] >= 0 # general sign check helper
            and np.sign(vertices[u,2])*s[2] >= 0]
    
    # Strictly isolate the 8 clean hexagonal planes by normal vector equations
    # Plane equations: +/-x +/-y +/-z = 3
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
fig = px.Mesh3d() # Base empty object to build layers onto

# 1. Draw Mesh Faces
if show_faces:
    # Plotly Mesh3d requires triangles (I, J, K indices). We must triangulate our polygons.
    I, J, K = [], [], []
    face_type_color = []
    
    # Triangulate squares (2 triangles each) and hexagons (4 triangles each)
    for f_idx, face in enumerate(all_faces):
        for t in range(1, len(face) - 1):
            I.append(face[0])
            J.append(face[t])
            K.append(face[t+1])

    fig = px.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=I, j=J, k=K,
        opacity=opacity,
        color='rgb(100, 180, 255)',
        name="Grain Volume",
        showscale=False
    )

# 2. Draw Edges (Wireframe)
for face in all_faces:
    # Close the loop by appending the first vertex index to the end
    loop = face + [face[0]]
    edge_coords = vertices[loop]
    fig.add_trace(px.Scatter3d(
        x=edge_coords[:, 0], y=edge_coords[:, 1], z=edge_coords[:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        showlegend=False,
        hoverinfo='skip'
    ).data[0])

# 3. Draw Nodes / Vertex Labels
if show_vertices:
    labels = [f"V{i}: ({vertices[i,0]:.0f},{vertices[i,1]:.0f},{vertices[i,2]:.0f})" for i in range(24)]
    fig.add_trace(px.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='crimson'),
        text=[f"V{i}" for i in range(24)],
        textposition="top center",
        hovertext=labels,
        name="Vertices / Triple Junctions"
    ).data[0])

# --- LAYOUT CONFIGURATION ---
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (Crystal Axis)', range=[-3, 3], backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(title='Y (Crystal Axis)', range=[-3, 3], backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(title='Z (Crystal Axis)', range=[-3, 3], backgroundcolor="rgba(0,0,0,0)"),
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
