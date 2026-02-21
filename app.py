import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from bpt_plot import compute_bpt_df, make_bpt_density_fig
from hr_diagram import (
    compute_hr_df,
    make_hr_density_fig,
    make_spectral_histogram,
    compute_physical_hr_df,
    make_physical_hr_bubble_fig,
    make_gaia_histogram_fig,
    add_gaia_categories,
    make_gaia_treemap_fig,
    make_topn_bar_fig
)

# Must execute before any other Streamlit call
st.set_page_config(
    page_title="Astro Spectroscopy Visualization",
    page_icon="🔭",
    layout="wide"
)

# Global padding for wide-layout readability
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 10rem;
        padding-right: 10rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom hero header (HTML for centering + typography control)
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1 style="color:black; margin-bottom: 6px;">Astro Spectroscopy Visualization</h1>
        <p style="color:#444; font-size: 18px; margin-top: 0;">
            Gaia DR3 HR diagrams & SDSS BPT diagnostics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Load datasets once; reused across tabs
df_sdss = pd.read_csv("sdss_galaxies.csv")
df_stars = pd.read_csv("gaia_stars.csv")

# Main navigation
tab_gaia, tab_sdss, tab_rankings = st.tabs(
    ["🌌 Gaia (HR)", "🛰️ SDSS (BPT)", "🏆 Rankings & Summary"]
)

with tab_gaia:
    st.header("Hertzsprung–Russell Diagrams (Gaia DR3)")

    # Dataset snapshot for quick sanity checks
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{len(df_stars):,}")
    k2.metric("Columns", f"{df_stars.shape[1]}")
    k3.metric("Missing Teff", f"{df_stars['teff_gspphot'].isna().sum():,}")

    # Optional full table view
    show_gaia_table = st.checkbox("Show Gaia dataset", value=True)
    if show_gaia_table:
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.dataframe(df_stars, use_container_width=True)

    st.divider()
    st.subheader("HR Density Map")

    # Filter to the subset used by the HR density plot (finite, plot-ready rows)
    hr_df = compute_hr_df(df_stars)
    st.write(f"Valid stars after filtering: **{len(hr_df):,}** / {len(df_stars):,}")

    fig_hr = make_hr_density_fig(hr_df, nbinsx=700, nbinsy=700)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.plotly_chart(fig_hr, use_container_width=True)

    st.info(
        """Density map of 100,000 Gaia DR3 stars within a spherical volume of radius 500 pc centered on the Solar System, 
        displayed in the color–absolute magnitude diagram. The color scale indicates stellar number density, 
        revealing evolutionary sequences such as the main sequence and the giant branch."""
    )

    st.divider()
    st.subheader("Physical HR Diagram and Spectral Type Distribution — Gaia DR3")

    # Controls define a reproducible sample shared by both panels
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 2])

    with ctrl1:
        n_stars = st.slider(
            "Stars to sample",
            1000,
            len(df_stars),
            min(25000, len(df_stars)),
            1000,
            key="linked_sample_n"
        )

    with ctrl2:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=164684,
            step=1,
            key="linked_seed"
        )

    with ctrl3:
        color_mode = st.radio(
            "Color physical HR by",
            options=["spectral", "mass"],
            format_func=lambda x:
                "Spectral type (OBAFGKM)" if x == "spectral" else "Mass (continuous)",
            horizontal=True,
            key="linked_phys_color"
        )

    df_subset = df_stars.sample(n=n_stars, random_state=int(seed)).copy()

    # Side-by-side panels with matched height for direct comparison
    left, spacer, right = st.columns([1, 0.08, 1])
    PANEL_HEIGHT = 650

    with left:
        st.markdown("Physical HR diagram")

        # Compute physical quantities for the bubble plot version
        df_phys = compute_physical_hr_df(df_subset)

        fig_phys_hr = make_physical_hr_bubble_fig(
            df_phys,
            color_mode=color_mode,
            title="Physical HR Diagram — Gaia DR3 (d ≤ 500 pc)"
        )
        fig_phys_hr.update_layout(
            height=PANEL_HEIGHT,
            margin=dict(l=40, r=25, t=60, b=45)
        )
        st.plotly_chart(fig_phys_hr, use_container_width=True)

    with right:
        st.markdown("Spectral type histogram")

        # Teff availability limits which stars can be assigned a spectral type
        valid = df_subset["teff_gspphot"].notna().sum()
        st.markdown(f"Stars with valid Teff: **{valid:,}** / {n_stars:,}")

        fig_spec = make_spectral_histogram(df_subset)
        fig_spec.update_layout(
            height=PANEL_HEIGHT,
            margin=dict(l=40, r=25, t=60, b=45)
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown(
        "Both panels display the same sampled stars. "
        "The HR diagram shows physical properties, and the histogram summarizes spectral types."
    )

    st.divider()
    st.subheader("Stellar Parameter Distributions")

    # Sampling keeps the UI responsive for large Gaia tables
    n_stars_hist = st.slider(
        "Stars for histograms",
        min_value=5000,
        max_value=len(df_stars),
        value=min(5000, len(df_stars)),
        step=5000,
        key="hist_n"
    )
    bins = st.slider("Bins", 20, 200, 140, 10, key="hist_bins")

    df_hist = df_stars.sample(n=n_stars_hist, random_state=8041)

    # Precompute figures once; the multiselect controls what gets rendered
    figs = {
        "Mass": make_gaia_histogram_fig(df_hist, "mass_flame", nbins=bins),
        "Effective Temperature": make_gaia_histogram_fig(df_hist, "teff_gspphot", nbins=bins),
        "Radius": make_gaia_histogram_fig(df_hist, "radius_gspphot", nbins=bins),
        "Luminosity": make_gaia_histogram_fig(df_hist, "lum_flame", nbins=bins, log_y=True),
        "Distance": make_gaia_histogram_fig(df_hist, "distance_pc", nbins=bins),
    }

    hist_options = st.multiselect(
        "Select histograms",
        options=list(figs.keys()),
        default=list(figs.keys()),
        key="hist_select"
    )

    cols = st.columns(2)
    for i, name in enumerate(hist_options):
        with cols[i % 2]:
            st.plotly_chart(figs[name], use_container_width=True)

    st.divider()
    st.subheader("Stellar Demographics in the Solar Neighborhood")

    # Category assignment requires a sample; N trades off detail vs speed
    n_tree = st.slider(
        "Stars to include",
        min_value=1000,
        max_value=len(df_stars),
        value=min(1000, len(df_stars)),
        step=1000,
        key="tree_n"
    )

    df_cat = add_gaia_categories(df_stars.sample(n=n_tree, random_state=42))

    fig_tree = make_gaia_treemap_fig(
        df_cat,
        title="Stellar population hierarchy"
    )

    c1, c2, c3 = st.columns([1, 10, 1])
    with c2:
        st.plotly_chart(fig_tree, use_container_width=True)

with tab_sdss:
    st.header("BPT Diagnostic Diagram (Sloan Digital Sky Survey)")

    # Dataset snapshot focused on key emission-line availability
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{len(df_sdss):,}")
    k2.metric("Columns", f"{df_sdss.shape[1]}")
    k3.metric(
        "Missing Hα",
        f"{df_sdss['h_alpha_flux'].isna().sum():,}"
        if "h_alpha_flux" in df_sdss.columns else "—"
    )

    # Table preview controls
    cA, cB = st.columns([1, 1])
    with cA:
        show_sdss_table = st.checkbox("Show SDSS dataset", value=True)
    with cB:
        sdss_rows = st.slider("Rows to display", 5, 200, 25, 5, key="sdss_rows")

    if show_sdss_table:
        c1, c2, c3 = st.columns([1, 11, 1])
        with c2:
            st.dataframe(df_sdss.head(sdss_rows), use_container_width=True)

    st.divider()
    st.subheader("BPT Density Map")

    # Compute log-ratio space with positivity/finite checks for safe logs
    bpt_df = compute_bpt_df(df_sdss)
    st.write(f"Valid points after filtering: **{len(bpt_df):,}** / {len(df_sdss):,}")

    fig_bpt = make_bpt_density_fig(bpt_df, nbinsx=700, nbinsy=700)
    c1, c2, c3 = st.columns([1, 5, 1])
    with c2:
        st.plotly_chart(fig_bpt, use_container_width=True)

    st.info(
        """Density map of SDSS emission-line galaxies in the BPT diagnostic plane (log([O III]/Hβ) vs log([N II]/Hα)). 
        The density scale highlights the star-forming sequence and the AGN/LINER region, enabling classification of 
        ionization mechanisms in nearby galaxies."""
    )

    st.divider()

    # Quick per-column diagnostics without leaving the app
    show_sdss_stats = st.checkbox("Show SDSS column statistics", value=True)
    if show_sdss_stats:
        col_sdss = st.selectbox("Inspect SDSS column", df_sdss.columns, key="sdss_column")
        c1, c2, c3 = st.columns([1, 0.8, 1])
        with c2:
            st.write(df_sdss[col_sdss].describe())

with tab_rankings:
    st.header("Stellar Property Rankings (Gaia DR3)")

    # Shared ranking depth across metrics
    top_n = st.slider("Select ranking depth", 5, 50, 10, 1, key="topn")

    tab1, tab2, tab3 = st.tabs(["Most luminous", "Most massive", "Largest"])

    with tab1:
        fig = make_topn_bar_fig(
            df_stars,
            metric="lum_flame",
            top_n=top_n,
            title=f"Top {top_n} Most luminous stars (L☉)"
        )
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_topn_bar_fig(
            df_stars,
            metric="mass_flame",
            top_n=top_n,
            title=f"Top {top_n} Most massive stars (M☉)"
        )
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_topn_bar_fig(
            df_stars,
            metric="radius_gspphot",
            top_n=top_n,
            title=f"Top {top_n} largest stars (R☉)"
        )
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.plotly_chart(fig, use_container_width=True)