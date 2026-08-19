import streamlit as st

from who_model import (
    predict_life_expectancy,
    predict_life_expectancy_min,
    VALID_REGIONS,
    FEATURE_RANGES,
    model_comparison,
)

st.set_page_config(page_title='Life Expectancy Predictor', layout='centered')

st.markdown("""<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
      --who-blue: #007eb4;
      --who-rule: rgba(128, 150, 162, 0.32);
  }
  @media (prefers-color-scheme: dark) {
      :root { --who-blue: #4fb9e3; }
  }
  html, body, [class*="css"], .stMarkdown, label, input, select, button {
      font-family: 'IBM Plex Sans', -apple-system, sans-serif;
  }
  .block-container { padding-top: 3rem; max-width: 780px; }
  .mast { border-bottom: 3px solid var(--who-blue); padding-bottom: 0.9rem; margin-bottom: 0.4rem; }
  .mast .eyebrow {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase;
      color: var(--who-blue); margin-bottom: 0.5rem;
  }
  .mast h1 {
      font-size: 2.1rem; font-weight: 600; margin: 0; letter-spacing: -0.02em;
      color: inherit;
  }
  .mast .sub { font-size: 0.9rem; opacity: 0.62; margin-top: 0.35rem; }
  .sect {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
      opacity: 0.62; margin: 2rem 0 0.2rem 0;
      border-top: 1px solid var(--who-rule); padding-top: 0.9rem;
  }
  .sect-note { font-size: 0.82rem; opacity: 0.62; margin-bottom: 0.6rem; }
  .consent-q {
      font-size: 1.02rem; font-weight: 500; line-height: 1.5;
      margin: 1.4rem 0 0.7rem 0;
  }
  div[role="radiogroup"] { gap: 0.4rem; }
  .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
      font-family: 'IBM Plex Mono', monospace !important;
      border-radius: 0 !important;
  }
  .stNumberInput input { font-size: 0.95rem !important; }
  .stNumberInput input:focus { border-color: var(--who-blue) !important; }
  .stNumberInput label, .stSelectbox label {
      font-size: 0.78rem !important; opacity: 0.72; font-weight: 400 !important;
  }
  .readout { margin-top: 2.2rem; border-top: 1px solid var(--who-rule); padding-top: 1.6rem; }
  .readout .cap {
      font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
      letter-spacing: 0.15em; text-transform: uppercase; opacity: 0.62;
  }
  .readout .figure { display: flex; align-items: baseline; gap: 0.6rem; margin: 0.5rem 0 0.1rem 0; }
  .readout .value {
      font-family: 'IBM Plex Mono', monospace; font-size: 3.6rem; font-weight: 500;
      line-height: 1; letter-spacing: -0.03em;
  }
  .readout .unit { font-size: 1rem; opacity: 0.62; }
  .readout .pm {
      font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem;
      color: var(--who-blue); margin-left: 0.4rem;
  }
  .readout .basis { font-size: 0.84rem; opacity: 0.62; margin-bottom: 1rem; }
  .readout .foot { font-size: 0.8rem; opacity: 0.62; margin-top: 0.5rem; line-height: 1.55; }
  .disclaimer { font-size: 0.72rem; opacity: 0.45; margin-top: 2.4rem; line-height: 1.6; }
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="mast">
<div class="eyebrow">Global Health Observatory &middot; Estimator</div>
<h1>Life expectancy at birth</h1>
<div class="sub">Modelled from WHO country statistics, 179 countries, 2000&ndash;2015</div>
</div>""", unsafe_allow_html=True)

st.markdown(
    '<div class="consent-q">Do you consent to using advanced population data, '
    'which may include protected information, for better accuracy?</div>',
    unsafe_allow_html=True,
)

consent = st.radio(
    'Consent',
    ['Yes — include health statistics', 'No — economic and demographic data only'],
    label_visibility='collapsed',
)
use_advanced = consent.startswith('Yes')

rmse = model_comparison.set_index('model')['rmse']
adv_rmse, min_rmse = float(rmse.iloc[0]), float(rmse.iloc[1])
error = adv_rmse if use_advanced else min_rmse

st.markdown('<div class="sect">Population statistics</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    region = st.selectbox('Region', VALID_REGIONS)
    gdp_per_capita = st.number_input('GDP per capita, USD', min_value=1.0,
                                     value=9000.0, step=100.0, format='%.0f')
with c2:
    year = st.number_input('Year', min_value=2000, max_value=2015, value=2015, step=1)
    population_mln = st.number_input('Population, millions', min_value=0.01,
                                     value=45.0, step=0.5, format='%.2f')
with c3:
    economy_status = st.selectbox('Economy status', ['Developing', 'Developed'])
    schooling = st.number_input('Schooling, years', min_value=0.0, max_value=25.0,
                                value=11.0, step=0.1, format='%.1f')

economy_status_developed = 1 if economy_status == 'Developed' else 0

if use_advanced:
    st.markdown('<div class="sect">Health statistics</div>'
                '<div class="sect-note">The protected fields covered by your consent.</div>',
                unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        adult_mortality = st.number_input('Adult mortality, per 1,000', min_value=0.0,
                                          value=160.0, step=1.0, format='%.1f')
        hepatitis_b = st.number_input('Hepatitis B coverage, %', min_value=0.0,
                                      max_value=100.0, value=90.0, step=1.0, format='%.0f')
        thinness_10_19 = st.number_input('Thinness, ages 10–19, %', min_value=0.0,
                                         value=8.0, step=0.1, format='%.1f')
    with h2:
        under_five_deaths = st.number_input('Under-five deaths, per 1,000', min_value=0.0,
                                            value=25.0, step=1.0, format='%.1f')
        polio = st.number_input('Polio coverage, %', min_value=0.0, max_value=100.0,
                                value=92.0, step=1.0, format='%.0f')
        thinness_5_9 = st.number_input('Thinness, ages 5–9, %', min_value=0.0,
                                       value=8.5, step=0.1, format='%.1f')
    with h3:
        incidents_hiv = st.number_input('HIV incidence, per 1,000', min_value=0.0,
                                        value=0.15, step=0.01, format='%.2f')
        diphtheria = st.number_input('Diphtheria coverage, %', min_value=0.0,
                                     max_value=100.0, value=91.0, step=1.0, format='%.0f')
        measles = st.number_input('Measles coverage, %', min_value=0.0, max_value=100.0,
                                  value=88.0, step=1.0, format='%.0f')

    b1, b2, _ = st.columns(3)
    with b1:
        bmi = st.number_input('Average BMI', min_value=10.0, max_value=50.0,
                              value=25.5, step=0.1, format='%.1f')
    with b2:
        alcohol_consumption = st.number_input('Alcohol, litres per capita', min_value=0.0,
                                              value=4.0, step=0.1, format='%.1f')

if use_advanced:
    prediction = predict_life_expectancy(
        region=region, year=year,
        adult_mortality=adult_mortality, under_five_deaths=under_five_deaths,
        hepatitis_b=hepatitis_b, polio=polio, diphtheria=diphtheria, measles=measles,
        bmi=bmi, incidents_hiv=incidents_hiv, alcohol_consumption=alcohol_consumption,
        thinness_10_19=thinness_10_19, thinness_5_9=thinness_5_9,
        gdp_per_capita=gdp_per_capita, population_mln=population_mln,
        schooling=schooling, economy_status_developed=economy_status_developed,
    )
    basis = 'Estimated from 17 inputs including health statistics'
else:
    prediction = predict_life_expectancy_min(
        region=region, year=year, gdp_per_capita=gdp_per_capita,
        population_mln=population_mln, schooling=schooling,
        economy_status_developed=economy_status_developed,
    )
    basis = 'Estimated from 6 non-medical inputs'

LO, HI = 36.0, 86.0
W, PAD = 700.0, 34.0


def x_of(v):
    return PAD + (v - LO) / (HI - LO) * (W - 2 * PAD)


band_lo, band_hi = prediction - error, prediction + error
ticks = [40, 50, 60, 70, 80]

tick_svg = ''.join(
    f'<line x1="{x_of(t):.1f}" y1="46" x2="{x_of(t):.1f}" y2="52" '
    f'stroke="currentColor" stroke-opacity="0.32"/>'
    f'<text x="{x_of(t):.1f}" y="66" text-anchor="middle" '
    f'font-family="IBM Plex Mono, monospace" font-size="11" '
    f'fill="currentColor" fill-opacity="0.55">{t}</text>'
    for t in ticks
)

st.markdown(f"""<div class="readout">
<div class="cap">Estimate</div>
<div class="figure">
<span class="value">{prediction:.1f}</span>
<span class="unit">years</span>
<span class="pm">&plusmn;{error:.2f}</span>
</div>
<div class="basis">{basis}</div>
<svg viewBox="0 0 {W:.0f} 74" width="100%" role="img"
aria-label="Estimate {prediction:.1f} years, range {band_lo:.1f} to {band_hi:.1f}">
<rect x="{x_of(band_lo):.1f}" y="18" width="{x_of(band_hi) - x_of(band_lo):.1f}"
height="22" fill="var(--who-blue)" fill-opacity="0.28"/>
<line x1="{PAD}" y1="46" x2="{W - PAD:.0f}" y2="46" stroke="currentColor" stroke-opacity="0.32"/>
{tick_svg}
<line x1="{x_of(prediction):.1f}" y1="10" x2="{x_of(prediction):.1f}" y2="46"
stroke="var(--who-blue)" stroke-width="3"/>
<text x="{x_of(band_lo):.1f}" y="13" text-anchor="middle"
font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor"
fill-opacity="0.7">{band_lo:.1f}</text>
<text x="{x_of(band_hi):.1f}" y="13" text-anchor="middle"
font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor"
fill-opacity="0.7">{band_hi:.1f}</text>
</svg>
<div class="foot">
Shaded band shows typical model error. Withholding health statistics widens it
from &plusmn;{adv_rmse:.2f} to &plusmn;{min_rmse:.2f} years.
</div>
</div>
<p class="disclaimer">
Student project using WHO life expectancy data. Not affiliated with, endorsed by,
or an official product of the World Health Organization.
</p>""", unsafe_allow_html=True)