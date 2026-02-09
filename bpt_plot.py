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

def compute_bpt_df(
    df: pd.DataFrame,
    col_nii: str = "nii_6584_flux",
    col_ha: str = "h_alpha_flux",
    col_oiii: str = "oiii_5007_flux",
    col_hb: str = "h_beta_flux",
) -> pd.DataFrame:
    """ Return a dataframe with log ratios for BPT diagram plotting. """
    nii = pd.to_numeric(df[col_nii], errors="coerce")
    ha = pd.to_numeric(df[col_ha], errors="coerce")
    oiii = pd.to_numeric(df[col_oiii], errors="coerce")
    hb = pd.to_numeric(df[col_hb], errors="coerce")

    ratio_x = nii / ha
    ratio_y = oiii / hb

    mask = (
        np.isfinite(ratio_x) & (ratio_x > 0) &
        np.isfinite(ratio_y) & (ratio_y > 0)
    )

    x = np.log10(ratio_x[mask])
    y = np.log10(ratio_y[mask])

    return pd.DataFrame({"log_NII_Ha": x, "log_OIII_Hb": y})

def make_bpt_density_fig(
    plot_df: pd.DataFrame,
    nbinsx: int = 700,
    nbinsy: int = 700,
    x_range=(-2.0, 0.5),
    y_range=(-1, 1.5),
    title: str = "BPT diagnostic diagram — Sloan Digital Sky Survey density map",
):
    """Return a styled Plotly density heatmap figure for the BPT diagram."""

    fig = px.density_heatmap(
        plot_df,
        x="log_NII_Ha",
        y="log_OIII_Hb",
        nbinsx=nbinsx,
        nbinsy=nbinsy,
    )

    # --- Layout general (dark astro style)
    fig.update_layout(
        width=720,
        height=720,
        plot_bgcolor="black",
        paper_bgcolor="black",

        # --- Título centrado y visible
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=24,
                color="white"
            )
        ),

        # --- Colorbar styling
        coloraxis=dict(
            colorscale=INFERNO_PINK_FADE,
            cmin=1,
            colorbar=dict(
                title="counts",
                title_font=dict(color="white"),
                tickfont=dict(color="white")
            )
        ),

        margin=dict(t=80, l=60, r=40, b=60),
        font=dict(color="white")
    )

    # --- Eje X
    fig.update_xaxes(
        range=list(x_range),
        title="log([N II] λ6584 / Hα)",
        title_font=dict(size=14, color="white"),
        tickfont=dict(color="white"),
        showline=True,
        linecolor="white",
        ticks="outside",
        tickcolor="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)"
    )

    # --- Eje Y
    fig.update_yaxes(
        range=list(y_range),
        title="log([O III] λ5007 / Hβ)",
        title_font=dict(size=14, color="white"),
        tickfont=dict(color="white"),
        showline=True,
        linecolor="white",
        ticks="outside",
        tickcolor="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)"
    )

    return fig