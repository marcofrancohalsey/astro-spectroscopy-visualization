import pandas as pd
import numpy as np
import plotly.express as px

INFERNO_PINK_FADE = [
    [0.00, "white"],
    [0.02, "#ff007f"], [0.06, "#ff1a8c"], [0.10, "#ff3399"], [0.14, "#ff4da6"],
    [0.18, "#ff66b3"], [0.22, "#ff80bf"], [0.26, "#ff99cc"], [0.30, "#ffb3d9"],
    [0.36, "#ffc2d1"], [0.42, "#ffd1c9"], [0.48, "#ffdfbf"], [0.54, "#ffeab3"],
    [0.60, "#fff2a6"], [0.66, "#fff599"], [0.72, "#fff88c"], [0.78, "#fffbb3"],
    [0.84, "#fffdd0"], [0.88, "#fffde0"], [0.92, "#fffef0"], [0.95, "#fffef7"],
    [0.97, "#fffefc"], [0.99, "#ffffff"], [1.00, "#fffffb"]
]


# ============================================================
# SPECTRAL CLASSIFICATION
# ============================================================

def classify_spectral_type(teff: float) -> str:
    """Return approximate spectral type from effective temperature."""
    
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
    Return a dataframe with BP-RP color and absolute G magnitude (M_G).
    Assumes parallax is in mas.
    """
    parallax = pd.to_numeric(df[col_parallax], errors="coerce")
    gmag = pd.to_numeric(df[col_gmag], errors="coerce")
    bp_rp = pd.to_numeric(df[col_bp_rp], errors="coerce")

    mask = (
        np.isfinite(parallax) & (parallax > 0) &
        np.isfinite(gmag) &
        np.isfinite(bp_rp)
    )

    parallax = parallax[mask]
    gmag = gmag[mask]
    bp_rp = bp_rp[mask]

    # M_G = G + 5 log10(parallax_mas) - 10
    M_G = gmag + 5*np.log10(parallax) - 10

    return pd.DataFrame({"bp_rp": bp_rp, "M_G": M_G})

def make_hr_density_fig(
    hr_df: pd.DataFrame,
    nbinsx: int = 700,
    nbinsy: int = 700,
    x_range = (-0.5, 4.5),
    y_range = (15, -5),  # inverted (bright at top)
    title: str = "HR Diagram (Gaia DR3, d ≤ 500 pc) — Density",
):
    """Return a Plotly density heatmap figure for an HR diagram."""
    fig = px.density_heatmap(
        hr_df,
        x="bp_rp",
        y="M_G",
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        title=title
    )

    fig.update_layout(
        template="simple_white",
        width=300,
        height=600,
        plot_bgcolor="black",
        paper_bgcolor="black",
        coloraxis=dict(
            colorscale=INFERNO_PINK_FADE,
            cmin=1,
            colorbar=dict(title="counts")
        )
    )

    fig.update_xaxes(
        range=list(x_range),
        title="BP − RP (Color)",
        showline=True,
        linecolor="black",
        ticks="outside",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)"
    )

    # NOTE: give an inverted range instead of invert_yaxis (matplotlib style)
    fig.update_yaxes(
        range=list(y_range),
        title="Absolute Magnitude M_G",
        showline=True,
        linecolor="black",
        ticks="outside",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)"
    )

    return fig


# ============================================================
# SPECTRAL HISTOGRAM
# ============================================================

def make_spectral_histogram(df: pd.DataFrame):
    """Return bar chart of star counts by spectral type."""

    df_spec = df.copy()

    mask = np.isfinite(df_spec["teff_gspphot"])
    df_spec = df_spec[mask]

    df_spec["spectral_type"] = df_spec["teff_gspphot"].apply(
        classify_spectral_type
    )

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
        title="Number of Stars by Spectral Type"
    )

    return fig
# ============================================================
# PHYSICAL HR DIAGRAM
# ============================================================

def compute_physical_hr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return clean dataframe for physical HR bubble chart."""
    out = df.copy()

    for c in ["teff_gspphot", "lum_flame", "radius_gspphot", "mass_flame", "distance_pc"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

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

    out["spectral_type"] = out["teff_gspphot"].apply(classify_spectral_type)

    return out


def make_physical_hr_bubble_fig(
    df_phys: pd.DataFrame,
    color_mode: str = "mass",  # "mass" or "spectral"
    title: str = "Physical HR Diagram — Bubble Chart",
):
    """Return Plotly bubble HR figure (Teff vs L)."""

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
        **color_args
    )

    fig.update_yaxes(type="log", title="Luminosity (L☉)")
    fig.update_xaxes(autorange="reversed", title="Effective Temperature (K)")

    fig.update_layout(
        template="simple_white",
        width=850,
        height=650,
        plot_bgcolor="black",
        paper_bgcolor="black",
        legend_title_text="",
    )

    return fig

def make_gaia_histogram_fig(
        df: pd.DataFrame,
        column: str,
        nbins: int = 80,
        title: str | None = None,
        log_y: bool = False,
):
    """Return a Plotly histogram for gaia numeric column. """
    s = pd.to_numeric(df[column], errors="coerce")
    s = s[np.isfinite(s)]

    plot_df = pd.DataFrame({column: s})

    fig = px.histogram(
        plot_df,
        x=column,
        nbins=nbins,
        title=title or f"Histogram of {column}",
    )

    fig.update_layout(
        template = "simple_white",
        width=700,
        height=450,
        plot_bgcolor="black",
        paper_bgcolor="black",
        xaxis_title=column,
        yaxis_title="counts",
    )

    if log_y:
        fig.update_yaxes(type="log")

    return fig
