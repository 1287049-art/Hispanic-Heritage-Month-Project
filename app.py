# === FILE: app.py ===
"""
Streamlit app: Historical data + polynomial regression (degree >=3) for Latin American countries.
- Fetches real historical series from OurWorldInData (and World Bank where noted).
- Allows selecting up to 3 countries (defaults: Uruguay, Chile, Argentina).
- Category dropdown covers the requested categories; the app will attempt to fetch the best available dataset.
- Editable raw-data table, selectable regression degree (>=3), selectable plotting increment (1-10 years), extrapolation years.
- Shows regression equation, plots fit (extrapolated portion in different color), allows interpolation/extrapolation for a chosen year, computes average rate of change between years, and performs simple function analysis (local extrema, increasing/decreasing intervals, where increasing fastest).
- Printer-friendly HTML report downloadable.

Notes:
- This app fetches public CSVs from OurWorldInData. For some indicators there may be limited historical coverage; the app reports available years.
- Designed to be self-contained in a single file for Streamlit deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from datetime import datetime
import base64

st.set_page_config(page_title="Latin-country historical regression explorer", layout="wide")

# ------------------ Helper: mapping categories to OWID grapher IDs (best-effort) ------------------
CATEGORY_MAP = {
    "Population": "population-unwpp",
    "Unemployment rate": "unemployment-total",
    "Education levels from 0-25 where 25 is the highest level of education": "mean-years-of-schooling-long-run",
    "Life expectancy": "life-expectancy",
    "Average wealth": "wealth-per-adult",
    "Average income": "gdp-per-capita",
    "Birth rate": "crude-birth-rate",
    "Immigration out of the country": "net-migration",
    "Murder Rate": "homicide-rate-unodc",
}

# Default countries: choose three high-income Latin American countries with long data coverage
DEFAULT_COUNTRIES = ["Uruguay", "Chile", "Argentina"]

# Utility: fetch OWID grapher CSV for a chart id
def fetch_owid_chart_csv(chart_id):
    url = f"https://ourworldindata.org/grapher/{chart_id}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    return df

# Extract country series from OWID grapher table (wide format where columns are countries)
def extract_series_from_owid(df, country, value_column=None):
    # OWID chart CSV often is in long format: 'Entity','Year','Value'.
    # Handle common formats.
    if {"country","Entity","Country","entity","Entity Name"} & set(df.columns):
        # find likely entity column
        entity_col = None
        for c in ["Entity","entity","Country","country","Entity Name"]:
            if c in df.columns:
                entity_col = c
                break
        year_col = None
        for c in ["Year","year","Year AD"]:
            if c in df.columns:
                year_col = c
                break
        # value column heuristics
        val_col = None
        for c in ["Value","value","Numeric","Total"]:
            if c in df.columns:
                val_col = c
                break
        if val_col is None:
            # maybe wide format where country names are columns
            if country in df.columns:
                s = df[[year_col, country]].dropna()
                s.columns = ["year","value"]
                return s
            raise ValueError("Could not find value column in OWID CSV")
        s = df[df[entity_col] == country][[year_col, val_col]].copy()
        s.columns = ["year","value"]
        s = s.dropna()
        return s
    else:
        # wide format: columns Year, <country1>, <country2>...
        if "Year" in df.columns:
            if country in df.columns:
                s = df[["Year", country]].dropna()
                s.columns = ["year","value"]
                return s
        # fallback: try first two columns
        s = df.iloc[:, :2].dropna()
        s.columns = ["year","value"]
        return s

# Fit polynomial regression (degree >=3) returning Polynomial object (coef in increasing order)
def fit_polynomial(years, values, degree=3):
    # scale years to improve conditioning: subtract a reference year
    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    ref = x.mean()
    xp = x - ref
    coeffs = np.polyfit(xp, y, deg=degree)
    # np.polyfit returns highest-first, but Polynomial wants lowest-first
    p = Polynomial(coeffs[::-1])
    # store reference in polynomial object as attribute
    p.ref = ref
    return p

# Evaluate polynomial at years (absolute years)
def eval_poly(p, years):
    x = np.array(years, dtype=float) - p.ref
    return p(x)

# Find critical points (local extrema) on given continuous domain [xmin, xmax]
def analyze_polynomial(p, xmin, xmax):
    # p is Polynomial (lowest-first), p.deriv() etc available
    dp = p.deriv()
    ddp = dp.deriv()
    # find real roots of dp (critical points)
    crit_roots = np.roots(dp.coef[::-1]) if len(dp.coef)>0 else np.array([])
    # careful: dp.coef is lowest-first; np.roots expects highest-first
    crit_real = []
    for r in np.roots(dp.coef[::-1]):
        if np.isreal(r):
            x = float(np.real(r)) + p.ref
            if xmin - 1e-8 <= x <= xmax + 1e-8:
                crit_real.append(x)
    crit_real = sorted(set(np.round(crit_real,6)))
    # local maxima/minima by second derivative sign
    extrema = []
    for x in crit_real:
        val_dd = ddp(x - p.ref)
        kind = 'minimum' if val_dd > 0 else ('maximum' if val_dd < 0 else 'inflection')
        extrema.append((x, eval_poly(p, [x])[0], kind))
    # find where derivative attains maximum on domain (fastest increase)
    # derivative is a polynomial; find its critical points by ddp roots
    ddp_roots = []
    if len(ddp.coef) >= 2:
        for r in np.roots(ddp.coef[::-1]):
            if np.isreal(r):
                x = float(np.real(r)) + p.ref
                if xmin <= x <= xmax:
                    ddp_roots.append(x)
    # candidate points for extrema of derivative: domain ends + ddp_roots
    candidates = [xmin, xmax] + ddp_roots
    deriv_vals = [(x, dp(x - p.ref)) for x in candidates]
    # find max derivative
    max_deriv = max(deriv_vals, key=lambda t: t[1])
    min_deriv = min(deriv_vals, key=lambda t: t[1])
    return {
        "critical_points": extrema,
        "max_derivative_point": (max_deriv[0], float(max_deriv[1])), # (year, rate)
        "min_derivative_point": (min_deriv[0], float(min_deriv[1])),
    }

# Generate printer-friendly HTML report
def generate_report_html(title, country, category, years, values, poly, analysis_text, equation_text, extrap_years=None):
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    html = f"""
    <html><head><meta charset='utf-8'><title>{title}</title></head><body>
    <h1>{title}</h1>
    <p>Generated: {now}</p>
    <h2>{country} — {category}</h2>
    <h3>Raw data (year,value)</h3>
    <pre>{pd.DataFrame({'year':years,'value':values}).to_csv(index=False)}</pre>
    <h3>Regression equation</h3>
    <pre>{equation_text}</pre>
    <h3>Analysis</h3>
    <pre>{analysis_text}</pre>
    </body></html>
    """
    return html

# ------------------ Streamlit UI ------------------
st.title("Latin-country historical regression explorer")
st.markdown("Use real historical data (OurWorldInData / World Bank) to build a polynomial regression (degree ≥ 3), analyze it, and extrapolate/interpolate.")

col1, col2 = st.columns([1,2])
with col1:
    country_sel = st.multiselect("Select up to 3 countries (start typing)", options=None, default=DEFAULT_COUNTRIES, max_selections=3)
    # set options to country list from OWID dynamically: fetch list once from population chart
    try:
        owid_pop = fetch_owid_chart_csv('population-unwpp')
        countries_available = sorted(owid_pop['Entity'].unique()) if 'Entity' in owid_pop.columns else sorted([c for c in owid_pop.columns if c!='Year'])
        # replace multiselect with updated options if needed
        if country_sel is None or len(country_sel)==0:
            country_sel = DEFAULT_COUNTRIES
        # enforce intersection with available
        country_sel = [c for c in country_sel if c in countries_available]
    except Exception as e:
        # fallback
        countries_available = DEFAULT_COUNTRIES

with col2:
    category = st.selectbox("Select data category", list(CATEGORY_MAP.keys()))
    degree = st.slider("Polynomial degree (min 3)", min_value=3, max_value=8, value=3)
    increment = st.slider("Plot increments (years)", min_value=1, max_value=10, value=1)
    extrapolate_years = st.number_input("Extrapolate forward (years)", min_value=0, max_value=100, value=10)
    show_multiple = st.checkbox("Plot multiple countries on same graph", value=True)

st.markdown("---")

# Fetch data for selected category and countries
chart_id = CATEGORY_MAP.get(category)
if chart_id is None:
    st.error("No data mapping for selected category.")
    st.stop()

# Attempt to fetch OWID chart; if fails, show error
try:
    df_chart = fetch_owid_chart_csv(chart_id)
except Exception as e:
    st.error(f"Could not fetch dataset for {category} (chart id {chart_id}). Error: {e}")
    st.stop()

# For each selected country, extract series and show editable table
country_series = {}
min_year = 9999
max_year = -9999
for c in country_sel:
    try:
        s = extract_series_from_owid(df_chart, c)
        if s.empty:
            st.warning(f"No data for {c} in category {category}.")
            continue
        s = s.sort_values('year')
        s['year'] = s['year'].astype(int)
        country_series[c] = s
        min_year = min(min_year, s['year'].min())
        max_year = max(max_year, s['year'].max())
    except Exception as e:
        st.warning(f"Failed to extract series for {c}: {e}")

if len(country_series)==0:
    st.error("No series available for the chosen countries/category.")
    st.stop()

st.write(f"Data coverage across selected countries: {min_year} — {max_year}")

# Show editable raw data table for the first country (user can switch)
first_country = list(country_series.keys())[0]
st.subheader(f"Raw data table (editable) — {first_country} — {category}")
editable_df = st.data_editor(country_series[first_country].rename(columns={'year':'Year','value':'Value'}), num_rows="dynamic")

# User may have edited values: use editable_df as base for first country
country_series[first_country] = editable_df.rename(columns={'Year':'year','Value':'value'})

# Fit and plot for one or multiple countries
fig, ax = plt.subplots(figsize=(10,5))
colors = ['tab:blue','tab:orange','tab:green']
legend_entries = []
analysis_texts = {}
report_blobs = {}
for i,(c, s) in enumerate(country_series.items()):
    years = s['year'].values
    vals = s['value'].values.astype(float)
    # drop NaNs
    mask = ~np.isnan(vals)
    years = years[mask]
    vals = vals[mask]
    if len(years) < degree+1:
        st.warning(f"Not enough data points for {c} to fit degree {degree} (need at least degree+1). Skipping.")
        continue
    p = fit_polynomial(years, vals, degree=degree)
    # prepare plotting range
    x_plot = np.arange(years.min(), years.max()+extrapolate_years+1, 0.5)
    y_plot = eval_poly(p, x_plot)
    # split into observed and extrapolated
    cutoff = years.max()
    obs_mask = x_plot <= cutoff
    ext_mask = x_plot > cutoff
    ax.scatter(years, vals, label=f"{c} data", color=colors[i%len(colors)], s=20)
    ax.plot(x_plot[obs_mask], y_plot[obs_mask], color=colors[i%len(colors)], linestyle='-')
    if extrapolate_years>0:
        ax.plot(x_plot[ext_mask], y_plot[ext_mask], color=colors[i%len(colors)], linestyle='--', alpha=0.7)
    legend_entries.append(c)
    # Equation text
    # get polynomial coefficients highest-first for human-readable
    coeffs = p.coef[::-1]  # highest-first
    terms = []
    deg = len(coeffs)-1
    for j,coef in enumerate(coeffs):
        power = deg - j
        terms.append(f"{coef:.6g}*x^{power}")
    equation = " + ".join(terms)
    # perform analysis
    analysis = analyze_polynomial(p, years.min(), years.max())
    # human readable analysis
    text = []
    for (xv, yv, kind) in analysis['critical_points']:
        d = datetime(int(round(xv)),1,1).date() if (xmin:=xv) and False else f"{xv:.1f}"
        text.append(f"The {category.lower()} of {c} reached a local {kind} at year {xv:.1f}. The modeled value was {yv:.3f}.")
    mdpt = analysis['max_derivative_point']
    text.append(f"The model indicates the {category.lower()} was increasing the fastest (highest derivative) at year {mdpt[0]:.1f}, at a rate of {mdpt[1]:.3g} units/year.")
    # domain and range
    sample_x = np.linspace(years.min(), years.max(), 500)
    sample_y = eval_poly(p, sample_x)
    dom = f"years {int(years.min())} to {int(years.max())}"
    ran = f"approx {sample_y.min():.3g} to {sample_y.max():.3g}"
    text.append(f"Model domain: {dom}. Estimated model range over domain: {ran}.")
    # extrapolation example
    extrap_year = cutoff + extrapolate_years
    extrap_val = float(eval_poly(p, [extrap_year])[0])
    text.append(f"According to the regression model, the {category.lower()} for {c} is predicted to be {extrap_val:.3g} in year {int(extrap_year)} (extrapolation).")
    analysis_texts[c] = "\n".join(text)
    # report blob
    report_blobs[c] = {
        'years': years.tolist(),
        'values': vals.tolist(),
        'equation': equation,
        'analysis': "\n".join(text)
    }

ax.set_xlabel('Year')
ax.set_ylabel(category)
ax.set_title(f"{category} — polynomial degree {degree}")
ax.legend()
st.pyplot(fig)

# Display equations and analyses
st.subheader('Model equations & analyses')
for c, blob in report_blobs.items():
    st.markdown(f"**{c}**")
    st.code(blob['equation'])
    st.text(blob['analysis'])

# Tools: interpolate/extrapolate a value given an input year
st.subheader('Interpolation / Extrapolation / Average rate of change')
sel_country = st.selectbox('Choose country for numeric queries', options=list(country_series.keys()))
model_degree = degree
years = country_series[sel_country]['year'].values
vals = country_series[sel_country]['value'].values.astype(float)
mask = ~np.isnan(vals)
years = years[mask]; vals = vals[mask]
if len(years) >= model_degree+1:
    p_sel = fit_polynomial(years, vals, degree=model_degree)
    query_year = st.number_input('Year to estimate value for (can be outside observed range)', value=int(years.max())+5)
    est = float(eval_poly(p_sel, [query_year])[0])
    st.write(f"Estimated {category} for {sel_country} in year {int(query_year)}: {est:.3f}")
    # average rate of change
    st.write('Average rate of change between two years:')
    y1 = st.number_input('Year 1', value=int(years.max())-10)
    y2 = st.number_input('Year 2', value=int(years.max()))
    if y2!=y1:
        v1 = float(eval_poly(p_sel, [y1])[0]); v2 = float(eval_poly(p_sel, [y2])[0])
        avg_rate = (v2 - v1) / (y2 - y1)
        st.write(f"Average rate of change from {y1} to {y2}: {avg_rate:.6g} units per year")
else:
    st.warning('Not enough points to build model for numeric queries.')

# Printer-friendly report
st.subheader('Printer-friendly report')
report_country = st.selectbox('Choose country to make report (per-country)', options=list(report_blobs.keys()))
if st.button('Generate & download HTML report'):
    blob = report_blobs[report_country]
    html = generate_report_html('Regression report', report_country, category, blob['years'], blob['values'], None, blob['analysis'], blob['equation'])
    b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    href = f"data:text/html;base64,{b64}"
    st.markdown(f"[Download report]({href})")

st.markdown('---')
st.caption('Data sources: Our World in Data (https://ourworldindata.org). This app fetches CSVs directly from OWID grapher endpoints.')

# === END app.py ===


# === FILE: requirements ===
# Minimal unpinned requirements for Streamlit deployment
streamlit
pandas
numpy
matplotlib
requests

# === END requirements ===
