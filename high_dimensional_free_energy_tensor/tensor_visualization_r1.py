import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator

# ================= CONFIGURATION =================
st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer", layout="wide")
st.title("🔷 Co-Cr-Fe-Ni Gibbs Free Energy Tensor Visualization")
st.markdown("""
This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface from discrete CSV data.  
The stable phase is determined by $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$.
""")

# ================= DATA LOADING (UPGRADED) =================
@st.cache_data
def load_data_from_files(files=None, use_example=False):
    """
    Load Gibbs data either from uploaded files, local csv_files/, or generate example.
    Returns a pandas DataFrame with columns Co, Cr, Fe, Ni, G_LIQ, G_FCC, T.
    """
    dfs = []
    
    # Case 1: Use uploaded files (Streamlit UploadedFile objects)
    if files:
        for uploaded_file in files:
            if uploaded_file.name.endswith(".csv"):
                try:
                    # Extract temperature from filename "Gibbs_XXXXK.csv"
                    basename = uploaded_file.name
                    T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
                    df = pd.read_csv(uploaded_file, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
                    df["T"] = T
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {uploaded_file.name}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    
    # Case 2: Read from local "csv_files/" directory
    csv_dir = "csv_files"
    if os.path.exists(csv_dir):
        files = glob.glob(os.path.join(csv_dir, "Gibbs_*.csv"))
        if files:
            for f in files:
                basename = os.path.basename(f)
                try:
                    T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
                    df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
                    df["T"] = T
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                return pd.concat(dfs, ignore_index=True)
    
    # Case 3: Generate example data (synthetic)
    if use_example:
        st.info("Generating synthetic Gibbs energy data for demonstration...")
        return generate_example_data()
    
    return pd.DataFrame()  # empty

def generate_example_data():
    """Create a plausible synthetic dataset for Co-Cr-Fe-Ni at multiple temperatures."""
    np.random.seed(42)
    n_compositions = 200
    temps = [800, 1000, 1200, 1400]
    rows = []
    
    for T in temps:
        # Random compositions (Co, Cr, Fe, Ni) that sum to 1
        comps = np.random.dirichlet(np.ones(4), size=n_compositions)
        for comp in comps:
            Co, Cr, Fe, Ni = comp
            # Simple quadratic model for G: ideal mixing + regular solution + temperature term
            # LIQ has lower G at high T, FCC lower at low T
            G_liq = -50 * T + 10000 * (Co**2 + Cr**2 + Fe**2 + Ni**2) + 5000 * (Co*Cr + Co*Fe + Cr*Fe)
            G_fcc = -30 * T + 8000 * (Co**2 + Cr**2 + Fe**2 + Ni**2) + 4000 * (Co*Cr + Co*Fe + Cr*Fe)
            rows.append([Co, Cr, Fe, Ni, G_liq, G_fcc, T])
    
    df = pd.DataFrame(rows, columns=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC", "T"])
    return df

# ================= INTERPOLATION =================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    """Build 3D interpolators for a specific temperature slice."""
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
    return interp_liq, interp_fcc

# ================= UI CONTROLS =================
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_files = st.file_uploader("Upload Gibbs_*K.csv files", type="csv", accept_multiple_files=True)
    generate_example = st.button("🔧 Generate Example Data (Synthetic)")
    
    st.header("🎛️ Visualization Parameters")
    
    # Placeholder for T slider – will be updated after data loads
    T_val = None
    grid_res = st.slider("Composition Grid Resolution", 15, 40, 25, step=5)
    show_phase = st.radio("Visualization Mode", 
                          ["Stable Phase (Min G)", "LIQUID Only", "FCC Only", "Both Phases Overlay"])
    marker_size = st.slider("Marker Size", 1, 6, 3)
    opacity = st.slider("Opacity", 0.3, 1.0, 0.75, 0.05)

# ================= MAIN APP LOGIC =================
# Load data (cached)
df = load_data_from_files(files=uploaded_files if uploaded_files else None,
                          use_example=generate_example)

if df.empty:
    st.warning("⚠️ No data loaded. Please upload CSV files or click 'Generate Example Data'.")
    st.stop()

# Temperature selection
T_list = sorted(df["T"].unique())
if len(T_list) == 0:
    st.error("Data contains no valid temperature entries.")
    st.stop()

T_val = st.sidebar.select_slider("Temperature (K)", options=T_list, value=T_list[len(T_list)//2])

# ================= COMPUTATION & PLOTTING =================
if T_val is not None:
    interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
    
    if interp_liq is None:
        st.error(f"No data points for T = {T_val} K.")
    else:
        # Generate tetrahedral grid
        x = np.linspace(0, 1, grid_res)
        Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
        grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
        
        # Mask: valid quaternary compositions must satisfy sum <= 1
        valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
        pts_valid = grid_pts[valid_mask]
        
        # Evaluate interpolators
        G_liq = interp_liq(pts_valid)
        G_fcc = interp_fcc(pts_valid)
        
        # Handle extrapolation/NaNs (outside convex hull)
        valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        pts = pts_valid[valid_eval]
        G_liq = G_liq[valid_eval]
        G_fcc = G_fcc[valid_eval]
        
        if len(pts) == 0:
            st.error("No valid interpolated points – try increasing grid resolution or check data coverage.")
            st.stop()
        
        # Determine stable phase & minimum G
        G_stable = np.minimum(G_liq, G_fcc)
        stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
        
        # Build Plotly Figure
        fig = go.Figure()
        
        if show_phase == "Stable Phase (Min G)":
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=G_stable, colorscale="Viridis", 
                           showscale=True, colorbar=dict(title="G (J/mol)", thickness=20)),
                name="Stable Phase",
                opacity=opacity
            ))
        elif show_phase == "LIQUID Only":
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=G_liq, colorscale="Blues", 
                           showscale=True, colorbar=dict(title="G_LIQ (J/mol)", thickness=20)),
                name="LIQUID", opacity=opacity
            ))
        elif show_phase == "FCC Only":
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=G_fcc, colorscale="Reds", 
                           showscale=True, colorbar=dict(title="G_FCC (J/mol)", thickness=20)),
                name="FCC", opacity=opacity
            ))
        else:  # Both Phases Overlay
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=marker_size, color=G_liq, colorscale="Blues", opacity=opacity),
                name="LIQUID"
            ))
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=marker_size, color=G_fcc, colorscale="Reds", opacity=opacity),
                name="FCC"
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis_title="x<sub>Co</sub>", yaxis_title="x<sub>Cr</sub>", zaxis_title="x<sub>Fe</sub>",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            title=f"Gibbs Energy Tensor at T = {T_val} K | Points: {len(pts):,}",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        st.subheader("📊 Phase Statistics at Current Grid")
        col1, col2, col3 = st.columns(3)
        col1.metric("Min G (Stable)", f"{G_stable.min():,.0f} J/mol")
        col2.metric("Max G", f"{G_stable.max():,.0f} J/mol")
        if show_phase in ["Stable Phase (Min G)", "Both Phases Overlay"]:
            liq_pct = np.sum(G_liq <= G_fcc) / len(G_liq) * 100
            col3.metric("LIQUID Region", f"{liq_pct:.1f}%")
