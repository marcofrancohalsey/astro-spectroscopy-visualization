import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from bpt_plot import compute_bpt_df, make_bpt_density_fig
from hr_diagram import compute_hr_df, make_hr_density_fig, make_spectral_histogram, compute_physical_hr_df, make_physical_hr_bubble_fig, make_gaia_histogram_fig


# --- Page config
st.set_page_config(
    page_title="Astro Spectroscopy Visualization",
    layout="wide"
)

st.title("Astrophysical Diagram Visualization")


# ============================================================
# SDSS — BPT DIAGRAM
# ============================================================

st.header("BPT Diagram — SDSS Galaxies")

# --- Load SDSS dataset
df_sdss = pd.read_csv("sdss_galaxies.csv")

st.write("Dataset preview:")
st.dataframe(df_sdss, use_container_width=True)

st.write("Rows:", df_sdss.shape[0])
st.write("Columns:", df_sdss.shape[1])

# --- Column explorer
col_sdss = st.selectbox(
    "Inspect SDSS column statistics",
    df_sdss.columns,
    key="sdss_column"
)
st.write(df_sdss[col_sdss].describe())


st.divider()
st.subheader("BPT Density Map")

# --- Compute BPT dataframe
bpt_df = compute_bpt_df(df_sdss)

st.write(
    f"Valid points after filtering: "
    f"**{len(bpt_df):,}** / {len(df_sdss):,}"
)

# --- Build density figure
fig_bpt = make_bpt_density_fig(
    bpt_df,
    nbinsx=700,
    nbinsy=700
)

# --- Center plot
c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    st.plotly_chart(fig_bpt)


# ============================================================
# GAIA — HR DIAGRAM
# ============================================================

st.header("Hertzsprung–Russell Diagram — Gaia DR3")

# --- Load Gaia stellar dataset
df_stars = pd.read_csv("gaia_stars.csv")

st.write("Dataset preview:")
st.dataframe(df_stars, use_container_width=True)

st.write("Rows:", df_stars.shape[0])
st.write("Columns:", df_stars.shape[1])

# --- Column explorer
col_gaia = st.selectbox(
    "Inspect Gaia column statistics",
    df_stars.columns,
    key="gaia_column"
)
st.write(df_stars[col_gaia].describe())


st.divider()
st.subheader("HR Density Map")

# --- Compute HR dataframe
hr_df = compute_hr_df(df_stars)

st.write(
    f"Valid points after filtering: "
    f"**{len(hr_df):,}** / {len(df_stars):,}"
)

# --- Build density figure
fig_hr = make_hr_density_fig(
    hr_df,
    nbinsx=700,
    nbinsy=700
)

# --- Center plot
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    st.plotly_chart(fig_hr)


st.subheader("Spectral type distribution")

n_stars = st.slider(
    "number of stars to sample for histogram",
    1000, len(df_stars), 25000, 1000
)

df_subset = df_stars.sample(n=n_stars, random_state=164684)

valid = df_subset["teff_gspphot"].notna().sum()

st.write(f"Stars with valid Teff: **{valid:,}** / {n_stars:,}")
fig_spec = make_spectral_histogram(df_subset)

fig_spec.update_traces(texttemplate="%{y:,}", textposition="outside", cliponaxis=False)
fig_spec.update_yaxes(rangemode="tozero")

c1, c2, c3 = st.columns([1, 1.5, 1])
with c2:
    st.plotly_chart(fig_spec)

st.subheader("Physical HR Diagram")

# --- Sample size slider

n_stars_phys = st.slider(
    "Number of stars to include",
    min_value=5000,
    max_value=len(df_stars),
    value=min(50000, len(df_stars)),
    step=5000,
    key="phys_hr_n"
)

df_subset = df_stars.sample(n=n_stars_phys, random_state=1997)

# --- Color mode selector
color_mode = st.radio(
    "Color by:",
    options=["mass", "spectral"],
    format_func= lambda x: "Mass (continuous)" if x == "mass" else "Spectral type (OBAFGKM)",
    horizontal=True,
    key="phys_hr_color"
)
df_phys = compute_physical_hr_df(df_subset)

fig_phys_hr = make_physical_hr_bubble_fig(
    df_phys,
    color_mode=color_mode,
    title="Physical HR Diagram (Gaia DR3, d<= 500 pc) - Bubble"
)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.plotly_chart(fig_phys_hr, use_container_width=True)


st.subheader("Gaia distributions (histograms)")

#Sample size slider
n_stars_hist = st.slider(
    "Number of stars for histograms",
    min_value=5000,
    max_value=len(df_stars),
    value=min(100000, len(df_stars)),
    step=5000,
    key="hist_n"
)

df_hist = df_stars.sample(n=n_stars_hist, random_state=8041)

bins = st.slider("Bins", 20, 200, 80, 10, key="hist_bins")

h1 = make_gaia_histogram_fig(df_hist, "mass_flame", nbins=bins, title="Mass (Solar masses)")
h2 = make_gaia_histogram_fig(df_hist, "teff_gspphot", nbins=bins, title="Effective Temperature (K)")
h3 = make_gaia_histogram_fig(df_hist, "radius_gspphot", nbins=bins, title="Radius (Solar radii)")
h4 = make_gaia_histogram_fig(df_hist, "lum_flame", nbins = bins, title="Luminosity (Solar luminosities) - log counts", log_y=True)
h5 = make_gaia_histogram_fig(df_hist, "distance_pc", nbins=bins, title="Distance (pc)")

# layout grid
r1c1, r1c2 = st.columns(2)
with r1c1: st.plotly_chart(h1, use_container_width=True)
with r1c2: st.plotly_chart(h2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1: st.plotly_chart(h3, use_container_width=True)
with r2c2: st.plotly_chart(h4, use_container_width=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2: st.plotly_chart(h5, use_container_width=True)