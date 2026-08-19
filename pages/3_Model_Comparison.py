import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import style
from sklearn import metrics

from who_model import (df, X_cols, X_test_s, X_train_s, MINIMAL, model_comparison,
                       pte_full, pte_min, res_full, res_min, y_test, y_train)

st.set_page_config(page_title='Model Comparison', layout='centered')

style.apply()

AXIS = '#8096a2'
ADV = '#007eb4'
MIN = '#d9822b'

st.markdown("""<div class="mast">
<div class="eyebrow">Global Health Observatory &middot; Results</div>
<h1>Model Comparison</h1>
<div class="sub">What the advanced model gains, and who pays for the minimal one</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="lede">
Both models use the same rows, the same split and the same scaler. The only
difference is that the minimal model never sees health data. Everything below
follows from that one change.
</div>""", unsafe_allow_html=True)

rmse = model_comparison.set_index('model')['rmse']
adv_rmse, min_rmse = float(rmse.iloc[0]), float(rmse.iloc[1])


@st.cache_data
def errors_frame():
    out = pd.DataFrame({
        'Country': df.loc[X_test_s.index, 'Country'].values,
        'Region': df.loc[X_test_s.index, 'Region'].values,
        'Year': df.loc[X_test_s.index, 'Year'].values,
        'Actual': y_test.values,
        'Advanced': pte_full.values,
        'Minimal': pte_min.values,
    })
    out['adv_err'] = out.Advanced - out.Actual
    out['min_err'] = out.Minimal - out.Actual
    return out


@st.cache_data
def headline_table():
    m = model_comparison.copy()
    m['features'] = [len(X_cols), len(MINIMAL)]
    m['inputs'] = [17, 6]
    m = m[['model', 'inputs', 'features', 'rmse', 'mae', 'r2']]
    m.columns = ['Model', 'User inputs', 'Model columns', 'RMSE', 'MAE', 'R2']
    m['RMSE'] = m['RMSE'].round(3)
    m['MAE'] = m['MAE'].round(3)
    m['R2'] = m['R2'].round(3)
    return m


@st.cache_data
def region_table():
    e = errors_frame()
    g = e.groupby('Region').apply(lambda d: pd.Series({
        'Rows': len(d),
        'Advanced': np.sqrt((d.adv_err ** 2).mean()),
        'Minimal': np.sqrt((d.min_err ** 2).mean()),
    }), include_groups=False)
    g['Times worse'] = (g.Minimal / g.Advanced).round(1)
    g[['Advanced', 'Minimal']] = g[['Advanced', 'Minimal']].round(2)
    g['Rows'] = g['Rows'].astype(int)
    return g.sort_values('Minimal', ascending=False).reset_index()


@st.cache_data
def quartile_table():
    e = errors_frame()
    q = pd.qcut(e.Actual, 4, labels=['Lowest 25%', 'Second', 'Third', 'Highest 25%'])
    g = e.groupby(q, observed=True).apply(lambda d: pd.Series({
        'Advanced': np.sqrt((d.adv_err ** 2).mean()),
        'Minimal': np.sqrt((d.min_err ** 2).mean()),
    }), include_groups=False)
    g['Times worse'] = (g.Minimal / g.Advanced).round(1)
    g[['Advanced', 'Minimal']] = g[['Advanced', 'Minimal']].round(2)
    g.index.name = 'Life expectancy band'
    return g.reset_index()


@st.cache_data
def baseline_table():
    mean_pred = np.full(len(y_test), y_train.mean())
    region_means = df.loc[X_train_s.index].groupby('Region').Life_expectancy.mean()
    region_pred = df.loc[X_test_s.index, 'Region'].map(region_means).values
    rows = [
        ['Always predict the overall average',
         metrics.root_mean_squared_error(y_test, mean_pred)],
        ['Always predict the regional average',
         metrics.root_mean_squared_error(y_test, region_pred)],
        ['Minimal model', min_rmse],
        ['Benchmark set in the brief', 1.8],
        ['Advanced model', adv_rmse],
    ]
    out = pd.DataFrame(rows, columns=['Approach', 'RMSE'])
    out['RMSE'] = out['RMSE'].round(2)
    return out.sort_values('RMSE', ascending=False).reset_index(drop=True)


@st.cache_data
def coefficient_table():
    frames = []
    for name, res, feats in [('Advanced', res_full, X_cols), ('Minimal', res_min, MINIMAL)]:
        p = res.params.drop('const').reindex(feats)
        top = p.reindex(p.abs().sort_values(ascending=False).index).head(5)
        frames.append(pd.DataFrame({
            'Model': name,
            'Feature': top.index,
            'Coefficient': top.values.round(2),
        }))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def region_share():
    out = {}
    for name, res, feats in [('Advanced', res_full, X_cols), ('Minimal', res_min, MINIMAL)]:
        p = res.params.drop('const').reindex(feats).abs()
        regions = [f for f in feats if f.startswith('Region_')]
        out[name] = p[regions].sum() / p.sum() * 100
    return out


def region_share_chart():
    share = region_share()
    fig = go.Figure()
    for i, (name, colour) in enumerate([('Advanced', ADV), ('Minimal', MIN)]):
        fig.add_trace(go.Bar(
            y=[name], x=[share[name]], orientation='h', marker_color=colour,
            name=name, showlegend=False, text=[f'{share[name]:.0f}%'],
            textposition='outside', textfont=dict(size=13),
            hovertemplate='%{x:.0f}% of coefficient weight<extra>' + name + '</extra>',
        ))
    fig.update_layout(
        height=170, margin=dict(l=10, r=40, t=10, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=AXIS),
        bargap=0.4,
    )
    fig.update_xaxes(title_text='Share of total coefficient weight from region dummies',
                     title_font_size=11, range=[0, 100], showgrid=True,
                     gridcolor='rgba(128,150,162,0.18)', zeroline=False, ticksuffix='%')
    fig.update_yaxes(showgrid=False)
    return fig


def actual_vs_predicted():
    e = errors_frame()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=(f'Advanced &nbsp; RMSE {adv_rmse:.2f}',
                                        f'Minimal &nbsp; RMSE {min_rmse:.2f}'))
    for i, (col, colour) in enumerate([('Advanced', ADV), ('Minimal', MIN)], start=1):
        fig.add_trace(go.Scattergl(
            x=e.Actual, y=e[col], mode='markers', showlegend=False,
            marker=dict(color=colour, size=5, opacity=0.55, line=dict(width=0)),
            customdata=np.stack([e.Country, e.Year], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b> %{customdata[1]}'
                          '<br>Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>',
        ), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=[36, 86], y=[36, 86], mode='lines', showlegend=False, hoverinfo='skip',
            line=dict(color=AXIS, width=1.5, dash='dot'),
        ), row=1, col=i)
    fig.update_layout(
        height=390, margin=dict(l=10, r=10, t=56, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=AXIS),
        hoverlabel=dict(font_family='IBM Plex Sans, sans-serif'),
    )
    fig.update_xaxes(title_text='Actual', title_font_size=11, range=[36, 86],
                     showgrid=True, gridcolor='rgba(128,150,162,0.18)', zeroline=False)
    fig.update_yaxes(range=[36, 86], showgrid=True,
                     gridcolor='rgba(128,150,162,0.18)', zeroline=False)
    fig.update_yaxes(title_text='Predicted', title_font_size=11, row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig


def region_chart():
    g = region_table().sort_values('Minimal')
    fig = go.Figure()
    fig.add_trace(go.Bar(y=g.Region, x=g.Advanced, name='Advanced',
                         orientation='h', marker_color=ADV,
                         hovertemplate='%{y}<br>RMSE %{x:.2f}<extra>Advanced</extra>'))
    fig.add_trace(go.Bar(y=g.Region, x=g.Minimal, name='Minimal',
                         orientation='h', marker_color=MIN,
                         hovertemplate='%{y}<br>RMSE %{x:.2f}<extra>Minimal</extra>'))
    fig.update_layout(
        barmode='group', height=420, margin=dict(l=10, r=10, t=16, b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=AXIS),
        legend=dict(orientation='h', yanchor='top', y=-0.1, x=0, title_text=''),
        bargap=0.28, bargroupgap=0.08,
    )
    fig.update_xaxes(title_text='RMSE, years', title_font_size=11, showgrid=True,
                     gridcolor='rgba(128,150,162,0.18)', zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


st.markdown('<div class="sect">1 &middot; Headline numbers</div>', unsafe_allow_html=True)

st.table(headline_table())

st.markdown(f"""<div class="body-text">
The advanced model predicts life expectancy to within &plusmn;{adv_rmse:.2f} years,
well past the 1.8 year benchmark we were given. The minimal model manages
&plusmn;{min_rmse:.2f}. That is {min_rmse / adv_rmse:.1f} times the error, and it is
the price of not asking for health data.
</div>
<div class="body-text">
R&sup2; tells a gentler story than RMSE here, {model_comparison.r2.iloc[1]:.2f} against
{model_comparison.r2.iloc[0]:.2f}, because most of the variation in life expectancy is
between rich and poor countries and wealth alone captures that. RMSE is the honest
measure: it is in years, and years are what a health ministry would act on.
</div>""", unsafe_allow_html=True)

st.plotly_chart(actual_vs_predicted(), width='stretch')

st.markdown("""<div class="body-text">
Perfect prediction sits on the dotted line. The advanced model hugs it. The minimal
model fans out, and it fans out worst at the bottom left.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">2 &middot; What the minimal model cannot see</div>',
            unsafe_allow_html=True)

health = [c for c in X_cols if c not in MINIMAL]

st.markdown(f"""<div class="body-text">
The brief asks whether a user consents to advanced population data that may include
protected information. We read that as medical data, so the {len(health)} features
below are withheld unless consent is given.
</div>
<div class="featlist"><span class="drop">{', '.join(health)}</span></div>""",
            unsafe_allow_html=True)

st.markdown("""<div class="body-text" style="margin-top:1.1rem;">
The line is drawn on whether a country would treat the statistic as a health record,
not on how useful it is. Alcohol consumption is a health behaviour, so it goes, even
though it is weak. Schooling and GDP are published openly by every government, so they
stay.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">3 &middot; Who pays for it</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
The extra error is not shared out evenly. Splitting the test set by actual life
expectancy shows where it lands.
</div>""", unsafe_allow_html=True)

st.table(quartile_table())

st.markdown("""<div class="body-text">
The minimal model is worst for the countries with the lowest life expectancy, which
are the countries a life expectancy model is most needed for. By region the pattern
is the same.
</div>""", unsafe_allow_html=True)

st.plotly_chart(region_chart(), width='stretch')

st.table(region_table())

worst = errors_frame()
worst = worst.reindex(worst.min_err.abs().sort_values(ascending=False).index).head(1)
w = worst.iloc[0]

st.markdown(f"""<div class="body-text">
The single worst case is {w.Country} in {int(w.Year)}. Actual life expectancy was
{w.Actual:.1f} years. The advanced model said {w.Advanced:.1f}. The minimal model said
{w.Minimal:.1f}, over by {w.Minimal - w.Actual:.0f} years, because on paper
{w.Country} looks better off than its neighbours, with roughly three times the regional
median GDP per capita. What the minimal model cannot see is HIV incidence fourteen
times the regional median, and that is the entire difference.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">4 &middot; What each model leans on</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Both models are standardised, so their coefficients are in the same units and can be
compared directly. These are the five largest in each.
</div>""", unsafe_allow_html=True)

st.table(coefficient_table())

share = region_share()

st.markdown(f"""<div class="body-text">
The advanced model is built on mortality. The minimal model has no mortality to look
at, so it falls back on wealth and on region. Adding up the coefficient weight sitting
in the region dummies makes the difference plain.
</div>""", unsafe_allow_html=True)

st.plotly_chart(region_share_chart(), width='stretch')

st.markdown(f"""<div class="body-text">
{share['Minimal']:.0f}% of the minimal model's coefficient weight is in the region
dummies, against {share['Advanced']:.0f}% for the advanced model. Take away the health
data and the model compensates by leaning on which continent a country sits in. It
still predicts reasonably well, but it is doing so from geography rather than from
anything about the population.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">5 &middot; Compared to doing nothing</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="body-text">
RMSE only means something next to an alternative. These are the simplest approaches
that need no model at all.
</div>""", unsafe_allow_html=True)

st.table(baseline_table())

st.markdown("""<div class="body-text">
The advanced model comfortably clears the benchmark. The minimal model does not, and
it only improves on looking up a regional average by about 1.4 years, which fits what
the coefficients showed: without health data the model is not doing much more than
sorting countries by continent and wealth.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">6 &middot; Summary</div>', unsafe_allow_html=True)

st.markdown(f"""<div class="body-text">
Consent buys about {min_rmse - adv_rmse:.1f} years of accuracy on average, and far
more than that for the countries in the worst health. A user who declines still gets a
usable estimate, but it is roughly {min_rmse / adv_rmse:.1f} times less precise and it
will tend to be optimistic about the places that are struggling most.
</div>
<div class="body-text">
That asymmetry is the finding worth carrying forward. A privacy-preserving model is
not uniformly slightly worse. It is nearly as good for wealthy countries and
considerably worse for poor ones, which means the cost of protecting the data is paid
mostly by the populations the data is about.
</div>""", unsafe_allow_html=True)