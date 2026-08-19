import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor

import style
from who_model import df, X_cols, X_train_s, MINIMAL, model_comparison

st.set_page_config(page_title='Feature Choices', layout='centered')

style.apply()

AXIS = '#8096a2'
TREND = '#e4572e'
REGION_COLOURS = {
    'Africa': '#d9a441',
    'Asia': '#3f8ea7',
    'Central America and Caribbean': '#8d6cab',
    'European Union': '#2f6f9f',
    'Middle East': '#c96a4f',
    'North America': '#4c9f70',
    'Oceania': '#b5606f',
    'Rest of Europe': '#6fa8c7',
    'South America': '#9c8b3d',
}

st.markdown("""<div class="mast">
<div class="eyebrow">Global Health Observatory &middot; Method</div>
<h1>Feature Choices</h1>
<div class="sub">What we kept, combined and dropped</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="lede">
The raw file has 21 columns for 179 countries, 2000 to 2015. We tested each decision
below against the data rather than assuming it.
</div>""", unsafe_allow_html=True)


@st.cache_data
def correlation_pairs():
    c = df.corr(numeric_only=True)
    mask = np.triu(np.ones(c.shape), k=1).astype(bool)
    pairs = c.where(mask).stack()
    pairs = pairs[pairs.abs() > 0.75].sort_values(key=abs, ascending=False)
    out = pd.DataFrame({
        'Feature A': [a for a, b in pairs.index],
        'Feature B': [b for a, b in pairs.index],
        'r': pairs.values.round(3),
    })
    return out[~out[['Feature A', 'Feature B']].isin(['Life_expectancy']).any(axis=1)]


@st.cache_data
def vif_table():
    Xv = sm.add_constant(X_train_s[X_cols].astype(float), has_constant='add')
    v = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])],
                  index=Xv.columns).drop('const')
    return v.sort_values(ascending=False).round(2).rename('VIF').to_frame()


def transform_plot(col, label):
    panels = [(df[col], label, False), (np.log1p(df[col]), f'log({label})', True)]
    titles = [f'{name} &nbsp;&nbsp; r = {vals.corr(df.Life_expectancy):.2f}'
              for vals, name, _ in panels]

    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.09)

    for i, (vals, name, logged) in enumerate(panels, start=1):
        for region, colour in REGION_COLOURS.items():
            sub = df['Region'] == region
            fig.add_trace(go.Scattergl(
                x=vals[sub], y=df.loc[sub, 'Life_expectancy'],
                mode='markers', name=region, legendgroup=region,
                showlegend=(i == 1),
                marker=dict(color=colour, size=5, opacity=0.62,
                            line=dict(width=0)),
                customdata=np.stack([df.loc[sub, 'Country'],
                                     df.loc[sub, 'Year']], axis=-1),
                hovertemplate='<b>%{customdata[0]}</b> %{customdata[1]}'
                              '<br>' + name + ': %{x:.2f}'
                              '<br>Life expectancy: %{y:.1f}<extra></extra>',
            ), row=1, col=i)

        slope, intercept = np.polyfit(vals, df['Life_expectancy'], 1)
        xs = np.linspace(vals.min(), vals.max(), 60)
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode='lines',
            line=dict(color=TREND, width=2.5), showlegend=False,
            hoverinfo='skip',
        ), row=1, col=i)

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=64, b=90),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=AXIS),
        legend=dict(orientation='h', yanchor='top', y=-0.16, x=0,
                    font=dict(size=10.5), itemsizing='constant',
                    title_text=''),
        hoverlabel=dict(font_family='IBM Plex Sans, sans-serif'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,150,162,0.18)',
                     zeroline=False, linecolor='rgba(128,150,162,0.4)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,150,162,0.18)',
                     zeroline=False, linecolor='rgba(128,150,162,0.4)')
    fig.update_yaxes(title_text='Life expectancy', title_font_size=11, row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig


st.markdown('<div class="sect">1 &middot; Data quality</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Nothing needed cleaning. No missing values, no duplicate rows, and every country has
all sixteen years. Range checks found no impossible values either.
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="decision">
<div class="what">We kept the outliers</div>
<div class="why">An IQR rule would delete 47% of the data, mostly from the poorest
regions. Those extreme values are real events, not recording errors. Dropping them
lowers the error score but gives you a model that only works on average countries.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">2 &middot; Overlapping columns</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Some columns measure almost the same thing. Left alone they make the coefficients
unstable, and in one case the model will not solve at all.
</div>""", unsafe_allow_html=True)

st.dataframe(correlation_pairs(), hide_index=True, width='stretch')

st.markdown("""<div class="decision">
<div class="what">Dropped Economy_status_Developing</div>
<div class="why">It is Economy_status_Developed inverted, r = &minus;1.000. The two
always sum to one, which breaks the model. This is a duplicate column, not a
judgement call.
</div></div>
<div class="decision">
<div class="what">Dropped Infant_deaths, kept Under_five_deaths</div>
<div class="why">Correlated at 0.986. Under-five is the wider measure and overlaps
slightly less with adult mortality.
</div></div>
<div class="decision">
<div class="what">Averaged the four vaccines into Vaccination_coverage</div>
<div class="why">Hepatitis B, polio, diphtheria and measles are all coverage
percentages for one-year-olds, so they rise and fall together.
</div></div>
<div class="decision">
<div class="what">Averaged the two thinness columns into Thinness_avg</div>
<div class="why">Overlapping age bands measuring the same thing, correlated at 0.939.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">3 &middot; Log transforms</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Linear regression fits one straight line per feature. If the real relationship curves,
the line misses at both ends. Logging straightens it.
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="decision">
<div class="what">GDP per capita</div>
<div class="why">Going from $600 to $3,000 adds years of life. Going from $60,000 to
$63,000 adds almost nothing. The left plot bends; the right one does not.
</div></div>""", unsafe_allow_html=True)

st.plotly_chart(transform_plot('GDP_per_capita', 'GDP per capita'), width='stretch')

st.markdown("""<div class="decision">
<div class="what">HIV incidence</div>
<div class="why">Skewed at 4.98, so most countries sit squashed against the left axis.
After logging, its coefficient is no longer significant, because adult mortality
already accounts for HIV deaths.
</div></div>""", unsafe_allow_html=True)

st.plotly_chart(transform_plot('Incidents_HIV', 'HIV incidence'), width='stretch')

st.markdown("""<div class="decision">
<div class="what">Population, left alone</div>
<div class="why">Logging fixes the skew but changes nothing useful. Correlation with
life expectancy is 0.026 before and &minus;0.012 after. Population size does not
predict life expectancy.
</div></div>""", unsafe_allow_html=True)

st.plotly_chart(transform_plot('Population_mln', 'Population'), width='stretch')

st.markdown("""<div class="decision">
<div class="what">BMI, left alone</div>
<div class="why">Skew of &minus;0.12 is already symmetric and the values only span
19.8 to 32.1. Logging it here would be changing a number to get a result we liked.
</div></div>
<div class="decision">
<div class="what">Region, one-hot encoded</div>
<div class="why">Regions have no order, so numbering them 0 to 8 would invent one. We
drop one dummy, making Africa the baseline every other region is compared against.
Country is left out entirely: 179 dummies would just memorise each country's average,
and the calculator has to work for countries the model has never seen.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">4 &middot; Checking it worked</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Anything above 10 here would mean a feature is still redundant. Nothing is, so we
stopped dropping columns. A separate test removing each feature one at a time agreed:
none of them improved the model by leaving.
</div>""", unsafe_allow_html=True)

st.dataframe(vif_table(), width='stretch', height=300)

st.markdown("""<div class="body-text">
The unscaled model reported a condition number near 936,000, which usually signals
collinearity. Here it was just the mismatched units. Scaling dropped it below 10 and
left every error metric identical.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">5 &middot; The two feature sets</div>', unsafe_allow_html=True)

adv = list(X_cols)
health_only = [c for c in adv if c not in MINIMAL]
rmse = model_comparison.set_index('model')['rmse']

st.markdown(f"""<div class="body-text">
<strong>Advanced model, {len(adv)} columns.</strong> Everything available.
</div>
<div class="featlist">{', '.join(adv)}</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="body-text" style="margin-top:1.2rem;">
<strong>Minimal model, {len(MINIMAL)} columns.</strong> These
{len(health_only)} health features are withheld without consent.
</div>
<div class="featlist"><span class="drop">{', '.join(health_only)}</span></div>""",
            unsafe_allow_html=True)

st.markdown(f"""<div class="body-text" style="margin-top:1.2rem;">
Losing them takes typical error from &plusmn;{rmse.iloc[0]:.2f} to
&plusmn;{rmse.iloc[1]:.2f} years. The gap is also uneven. The minimal model is
about three times worse for countries with low life expectancy than for those with
high, so it fails hardest where accuracy matters most.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">6 &middot; Reading the coefficients</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Life expectancy is calculated from mortality rates in the first place. Adult mortality
and under-five deaths are not really predictors then, and the model is partly redoing
that arithmetic. It explains why they dominate and why the advanced model scores so
well.
</div>
<div class="body-text">
It also means some coefficients point the opposite way to the plain correlation. BMI
tracks longer life on its own, because well-fed countries are wealthy ones. Once the
model knows about wealth and mortality, what is left in BMI is obesity, and the sign
flips. These are effects with everything else held still, not causes, and the
calculator's inputs should not be read as advice.
</div>""", unsafe_allow_html=True)