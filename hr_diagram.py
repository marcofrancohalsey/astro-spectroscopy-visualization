import pandas as pd
import numpy as np
import plotly.express as px

# Custom colorscale used across density plots
INFERNO_PINK_FADE = [
    [0.00, "white"],
    [0.02, "#ff007f"], [0.06, "#ff1a8c"], [0.10, "#ff3399"], [0.14, "#ff4da6"],
    [0.18, "#ff66b3"], [0.22, "#ff80bf"], [0.26, "#ff99cc"], [0.30, "#ffb3d9"],
    [0.36, "#ffc2d1"], [0.42, "#ffd1c9"], [0.48, "#ffdfbf"], [0.54, "#ffeab3"],
    [0.60, "#fff2a6"], [0.66, "#fff599"], [0.72, "#fff88c"], [0.78, "#fffbb3"],
    [0.84, "#fffdd0"], [0.88, "#fffde0"], [0.92, "#fffef0"], [0.95, "#fffef7"],
    [0.97, "#fffefc"], [0.99, "#ffffff"], [1.00, "#fffffb"]
]


def classify_spectral_type(teff: float) -> str:
    """
    Map effective temperature (K) to a coarse OBAFGKM spectral class.
    """
    if teff > 30000:
        return "O"
    elif teff > 10000:
        return "B"
    elif teff > 7500:
        return "A"
    elif teff > 6000:
        return "F"
    elif teff > 5200:
        return "G"
    elif teff > 3700:
        return "K"
    else:
        return "M"


def compute_hr_df(
    df: pd.DataFrame,
    col_parallax: str = "parallax",
    col_gmag: str = "phot_g_mean_mag",
    col_bp_rp: str = "bp_rp",
) -> pd.DataFrame:
    """
    Compute HR-diagram coordinates from Gaia columns.

    Returns bp_rp and absolute magnitude M_G. Parallax is assumed to be in mas:
        M_G = G + 5 log10(parallax_mas) - 10

    Rows are filtered to finite values and positive parallax to keep the transform valid.
    """
    parallax = pd.to_numeric(df[col_parallax], errors="coerce")
    gmag = pd.to_numeric(df[col_gmag], errors="coerce")
    bp_rp = pd.to_numeric(df[col_bp_rp], errors="coerce")

    # Positive parallax is required for the distance-modulus form used here
    mask = (
        np.isfinite(parallax) & (parallax > 0) &
        np.isfinite(gmag) &
        np.isfinite(bp_rp)
    )

    parallax = parallax[mask]
    gmag = gmag[mask]
    bp_rp = bp_rp[mask]

    M_G = gmag + 5*np.log10(parallax) - 10

    return pd.DataFrame({"bp_rp": bp_rp, "M_G": M_G})


def make_hr_density_fig(
    hr_df: pd.DataFrame,
    nbinsx: int = 700,
    nbinsy: int = 700,
    x_range=(-0.5, 4.5),
    y_range=(15, -5),  # inverted (bright at top)
    title: str = "Gaia DR3 Hertzsprung–Russell Diagram (Density, d ≤ 500 pc)",
):
    """Return a Plotly density heatmap figure for an HR diagram."""
    fig = px.density_heatmap(
        hr_df,
        x="bp_rp",
        y="M_G",
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        title=title,
    )

    # Layout tuned for readability; Streamlit controls width via use_container_width
    fig.update_layout(
        template="none",
        height=650,
        margin=dict(l=90, r=60, t=90, b=80),
        font=dict(size=14, color="black"),
        title=dict(
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            pad=dict(t=10),
            font=dict(size=18, color="black")
        ),
        coloraxis=dict(
            colorscale=INFERNO_PINK_FADE,
            cmin=1,
            colorbar=dict(
                title="Star count",
                title_font=dict(size=14, color="black"),
                tickfont=dict(size=12, color="black"),
                tickformat=",",
                len=0.85,
                outlinecolor="black",
                outlinewidth=1,
            ),
        ),
    )

    fig.update_xaxes(
        range=list(x_range),
        title="BP − RP Color index",
        ticks="outside",
        ticklen=14,
        tickwidth=2,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=False,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )

    fig.update_yaxes(
        range=list(y_range),
        title="Absolute Magnitude (M_G)",
        ticks="outside",
        ticklen=14,
        tickwidth=2,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=False,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )

    # Hover reports HR coordinates + bin count for density heatmap
    fig.update_traces(
        hovertemplate=(
            "BP − RP: %{x:.3f}<br>"
            "M_G: %{y:.3f}<br>"
            "Star Count: %{z:,}<extra></extra>"
        )
    )

    return fig


def make_spectral_histogram(df: pd.DataFrame):
    """Return bar chart of star counts by spectral type (white theme)."""
    df_spec = df.copy()

    # Spectral assignment relies on Teff; restrict to finite entries
    mask = np.isfinite(df_spec["teff_gspphot"])
    df_spec = df_spec[mask]

    df_spec["spectral_type"] = df_spec["teff_gspphot"].apply(classify_spectral_type)

    order = ["O", "B", "A", "F", "G", "K", "M"]

    counts = (
        df_spec["spectral_type"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .reset_index()
    )
    counts.columns = ["spectral_type", "count"]

    fig = px.bar(
        counts,
        x="spectral_type",
        y="count",
        title="<b>Number of Stars by Spectral Type</b>",
        text="count"
    )

    # Bar labels are placed outside; cliponaxis=False prevents truncation
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker_color="#1f77b4",
        marker_line_color="black",
        marker_line_width=0.5
    )

    fig.update_layout(
        template="none",
        height=650,
        margin=dict(l=95, r=40, t=70, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(color="black", size=16),
        ),
        xaxis_title="<b>Spectral Type</b>",
        yaxis_title="<b>Star Count</b>",
        showlegend=False
    )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        tickfont=dict(color="black", size=12),
        title_font=dict(color="black", size=13),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        tickfont=dict(color="black", size=12),
        title_font=dict(color="black", size=13),
        showgrid=False,
        zeroline=False,
        tickformat=",",
        automargin=True,
    )

    return fig


def compute_physical_hr_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a clean dataframe for the physical HR bubble chart.

    Required inputs are Teff, luminosity, and radius (positive/finite). Mass and distance
    are preserved for hover/context if present.
    """
    out = df.copy()

    # Coerce expected numeric columns; invalid entries become NaN
    num_cols = ["teff_gspphot", "lum_flame", "radius_gspphot", "mass_flame", "distance_pc"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Log plots and marker sizing require positive values
    mask = (
        np.isfinite(out["teff_gspphot"]) & (out["teff_gspphot"] > 0) &
        np.isfinite(out["lum_flame"]) & (out["lum_flame"] > 0) &
        np.isfinite(out["radius_gspphot"]) & (out["radius_gspphot"] > 0)
    )

    out = out.loc[mask, [
        "source_id",
        "distance_pc",
        "teff_gspphot",
        "lum_flame",
        "radius_gspphot",
        "mass_flame"
    ]].copy()

    # Coarse spectral labels used when color_mode="spectral"
    out["spectral_type"] = out["teff_gspphot"].apply(classify_spectral_type)

    return out


def make_physical_hr_bubble_fig(
    df_phys: pd.DataFrame,
    color_mode: str = "mass",  # "mass" or "spectral"
    title: str = "Physical HR Diagram — Gaia DR3 (d ≤ 500 pc)",
):
    """Return a Plotly physical HR bubble figure (Teff vs Luminosity) on a dark background."""
    # color_mode chooses between categorical spectral classes and continuous mass
    if color_mode == "spectral":
        color_col = "spectral_type"
        color_args = dict(category_orders={"spectral_type": ["O", "B", "A", "F", "G", "K", "M"]})
    else:
        color_col = "mass_flame"
        color_args = {}

    fig = px.scatter(
        df_phys,
        x="teff_gspphot",
        y="lum_flame",
        size="radius_gspphot",
        color=color_col,
        hover_data={
            "source_id": True,
            "distance_pc": ":.1f",
            "teff_gspphot": ":.0f",
            "lum_flame": ":.3g",
            "radius_gspphot": ":.3g",
            "mass_flame": ":.3g",
        },
        title=title,
        **color_args,
    )

    # HR convention: temperature decreases to the right; luminosity is shown in log space
    fig.update_yaxes(type="log", title="Luminosity (L\u2609)")
    fig.update_xaxes(autorange="reversed", title="Effective temperature (K)")

    fig.update_layout(
        template="none",
        height=650,
        margin=dict(l=80, r=40, t=80, b=70),

        plot_bgcolor="black",
        paper_bgcolor="black",

        font=dict(color="white"),
        title=dict(x=0.5, xanchor="center", font=dict(color="white")),

        legend_title_text="",
        legend=dict(
            font=dict(color="white"),
            title_font=dict(color="white"),
        ),
    )

    fig.update_xaxes(
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        tickcolor="white",
        linecolor="white",
        showline=True,
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        tickcolor="white",
        linecolor="white",
        showline=True,
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showgrid=False,
        zeroline=False,
    )

    return fig


def make_gaia_histogram_fig(
        df: pd.DataFrame,
        column: str,
        nbins: int = 80,
        title: str | None = None,
        log_y: bool = False,
):
    """
    Return a Plotly histogram (dark theme) for a numeric Gaia parameter.

    The input series is coerced to numeric and filtered to finite values
    to avoid binning artifacts.
    """
    s = pd.to_numeric(df[column], errors="coerce")
    s = s[np.isfinite(s)]
    plot_df = pd.DataFrame({column: s})

    label_map = {
        "mass_flame": "Stellar Mass (M☉)",
        "radius_gspphot": "Stellar Radius (R☉)",
        "lum_flame": "Stellar Luminosity (L☉)",
        "teff_gspphot": "Effective Temperature (K)",
        "distance_pc": "Distance (pc)",
    }
    x_label = label_map.get(column, column.replace("_", " ").title())

    # Default titles keep the UI consistent across parameters
    title_map = {
        "mass_flame": "Distribution of Stellar Mass",
        "radius_gspphot": "Distribution of Stellar Radius",
        "lum_flame": "Distribution of Stellar Luminosity",
        "teff_gspphot": "Distribution of Effective Temperature",
        "distance_pc": "Distribution of Stellar Distance",
    }
    auto_title = title or title_map.get(column, f"Distribution of {x_label}")

    fig = px.histogram(
        plot_df,
        x=column,
        nbins=nbins,
        title=f"<b>{auto_title}</b>",
    )

    fig.update_traces(
        marker_color="#FF8C00",
        marker_line_color="white",
        marker_line_width=0.4,
    )

    fig.update_layout(
        template="none",
        height=450,
        margin=dict(l=90, r=40, t=70, b=70),

        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),

        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=16),
        ),

        xaxis_title=f"<b>{x_label}</b>",
        yaxis_title="<b>Star Count</b>",
    )

    fig.update_xaxes(
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="white",
        tickfont=dict(color="white", size=12),
        title_font=dict(color="white", size=13),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="white",
        tickfont=dict(color="white", size=12),
        title_font=dict(color="white", size=13),
        showgrid=False,
        zeroline=False,
        automargin=True,
        tickformat=",",
    )

    if log_y:
        fig.update_yaxes(type="log")

    return fig


def add_gaia_categories(
    df: pd.DataFrame,
    mass_bins=(0.0, 0.5, 1.0, 1.5, 2.0, 5.0, np.inf),
    dist_bins=(0, 50, 100, 200, 300, 400, 500, np.inf),
) -> pd.DataFrame:
    """Return df with spectral_type, mass_bin, distance_bin columns."""
    out = df.copy()

    out["teff_gspphot"] = pd.to_numeric(out["teff_gspphot"], errors="coerce")
    out["mass_flame"] = pd.to_numeric(out["mass_flame"], errors="coerce")
    out["distance_pc"] = pd.to_numeric(out["distance_pc"], errors="coerce")

    # Categories require valid temperature, mass, and distance
    mask = (
        np.isfinite(out["teff_gspphot"]) & (out["teff_gspphot"] > 0) &
        np.isfinite(out["mass_flame"]) & (out["mass_flame"] > 0) &
        np.isfinite(out["distance_pc"]) & (out["distance_pc"] > 0)
    )
    out = out.loc[mask].copy()

    out["spectral_type"] = out["teff_gspphot"].apply(classify_spectral_type)

    mass_labels = [f"{mass_bins[i]}–{mass_bins[i+1]} M☉" if np.isfinite(mass_bins[i+1]) else f">{mass_bins[i]} M☉"
                   for i in range(len(mass_bins) - 1)]
    dist_labels = [f"{dist_bins[i]}–{dist_bins[i+1]} pc" if np.isfinite(dist_bins[i+1]) else f">{dist_bins[i]} pc"
                   for i in range(len(dist_bins) - 1)]

    out["mass_bin"] = pd.cut(out["mass_flame"], bins=list(mass_bins), labels=mass_labels, right=False, include_lowest=True)
    out["distance_bin"] = pd.cut(out["distance_pc"], bins=list(dist_bins), labels=dist_labels, right=False, include_lowest=True)

    # Force deterministic ordering in Plotly
    out["spectral_type"] = pd.Categorical(out["spectral_type"], categories=["O","B","A","F","G","K","M"], ordered=True)
    out["mass_bin"] = pd.Categorical(out["mass_bin"], categories=mass_labels, ordered=True)
    out["distance_bin"] = pd.Categorical(out["distance_bin"], categories=dist_labels, ordered=True)

    return out


def make_gaia_treemap_fig(
    df_cat: pd.DataFrame,
    title: str = "Stellar Population Hierarchy"
):
    fig = px.treemap(
        df_cat,
        path=["spectral_type", "mass_bin", "distance_bin"],
        color="spectral_type",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textinfo="label+percent entry",
        marker=dict(
            line=dict(width=1, color="black")
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Share: %{percentEntry:.2%}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=24,
                color="#8EC7F0"
            )
        ),
        template="simple_white",
        width=850,
        height=650,
        plot_bgcolor="black",
        paper_bgcolor="black",
        margin=dict(t=90, l=20, r=20, b=20),
        font=dict(color="white")
    )

    return fig


def make_topn_bar_fig(
    df: pd.DataFrame,
    metric: str,
    top_n: int = 20,
    title: str | None = None,
):
    """Return a styled horizontal bar chart for top-N objects by a metric."""
    cols = ["source_id", "distance_pc", "bp_rp", "teff_gspphot", "mass_flame", "radius_gspphot", "lum_flame"]

    data = df.copy()
    for c in cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="ignore")

    s = pd.to_numeric(data[metric], errors="coerce")
    data = data[np.isfinite(s)].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")

    top = (
        data.sort_values(metric, ascending=False)
            .head(top_n)
            .copy()
    )

    # Prefix prevents Plotly from treating numeric-looking IDs as a continuous axis
    top["source_id_label"] = "ID " + top["source_id"].astype(str)

    fig = px.bar(
        top.sort_values(metric, ascending=True),
        x=metric,
        y="source_id_label",
        orientation="h",
        title=title or f"Top {top_n} by {metric}",
        hover_data={
            "source_id_label": False,
            "distance_pc": ":.1f",
            "bp_rp": ":.3g",
            "teff_gspphot": ":.0f",
            "mass_flame": ":.3g",
            "radius_gspphot": ":.3g",
            "lum_flame": ":.3g",
        },
    )

    fig.update_traces(
        marker_color="#9BE90A",
        marker_line=dict(color="white", width=0.6),
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )

    fig.update_layout(
        width=900,
        height=650,
        plot_bgcolor="black",
        paper_bgcolor="black",
        title=dict(x=0.5, xanchor="center", font=dict(size=24, color="white")),
        font=dict(color="white"),
        xaxis_title=metric,
        yaxis_title="Source ID",
        margin=dict(t=80, l=70, r=40, b=60),
        bargap=0.25,
    )

    fig.update_xaxes(
        showline=True,
        linecolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white"),
        ticks="outside",
        tickcolor="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)",
        zeroline=False,
        mirror=True,
    )

    fig.update_yaxes(
        type="category",
        showline=True,
        linecolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white"),
        ticks="outside",
        tickcolor="white",
        showgrid=False,
        mirror=True,
    )

    return fig