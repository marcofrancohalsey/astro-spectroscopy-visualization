# Astrophysics Data Dashboard 

Streamlit app for exploratory astrophysics visualizations using Gaia DR3 and the Sloan Digital Sky Survey (SDSS).

## What’s inside
- Gaia DR3 HR density map (color–absolute magnitude)
- Physical HR diagram (Teff vs luminosity, radius as size; optional coloring by mass or spectral type)
- Spectral type distribution (OBAFGKM)
- Stellar parameter histograms (mass, Teff, radius, luminosity, distance)
- SDSS BPT diagnostic density map (log([O III]/Hβ) vs log([N II]/Hα))
- Top-N rankings for Gaia stellar properties

## Run locally
pip install -r requirements.txt

streamlit run app.py

## Run on Render

Access the deployed app here:
https://astro-spectroscopy-visualization.onrender.com
