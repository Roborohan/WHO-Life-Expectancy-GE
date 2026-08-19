import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

import style
from who_model import df, X_cols, X_train_s, MINIMAL, res_full, model_comparison

st.set_page_config(page_title='Feature Choices', layout='centered')

style.apply()

st.markdown("""<div class="mast">
<div class="eyebrow">Global Health Observatory &middot; Method</div>
<h1>Feature choices</h1>
<div class="sub">What was kept, what was combined, what was dropped &mdash; and why</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="lede">
The raw dataset holds 21 columns for 179 countries across 2000&ndash;2015. Two models
are built from it: an advanced model using every available statistic, and a minimal
model with all health data withheld. Each transformation below was tested against
the data rather than assumed.
</div>""", unsafe_allow_html=True)


@st.cache_data
def top_correlations(threshold=0.75):
    c = df.corr(numeric_only=True)
    mask = np.triu(np.ones(c.shape), k=1).astype(bool)
    pairs = c.where(mask).stack()
    pairs = pairs[pairs.abs() > threshold].sort_values(key=abs, ascending=False)
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


@st.cache_data
def skew_table():
    cols = ['GDP_per_capita', 'Incidents_HIV', 'Population_mln', 'BMI', 'Schooling']
    rows = []
    for c in cols:
        rows.append({
            'Feature': c,
            'Skew (raw)': round(df[c].skew(), 2),
            'Skew (logged)': round(np.log1p(df[c]).skew(), 2),
            'r with target (raw)': round(df[c].corr(df.Life_expectancy), 3),
            'r with target (logged)': round(np.log1p(df[c]).corr(df.Life_expectancy), 3),
        })
    return pd.DataFrame(rows)


st.markdown('<div class="sect">1 &middot; Data quality</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
No cleaning was required. The file contains no missing values, no duplicate rows and
no duplicate country&ndash;year pairs. Every country has all sixteen years, so the
panel is balanced. Range checks confirmed no impossible values: immunisation and
thinness percentages stay within bounds, mortality and GDP are never negative, and
infant deaths never exceed under-five deaths.
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="decision">
<div class="what">Outliers were retained, not removed</div>
<div class="why">An interquartile-range rule would have deleted 47% of the data,
and the rows it flags are concentrated in the poorest regions. The extreme values
trace to real events rather than recording errors. Removing them would improve the
error metric while producing a model that only works on unremarkable countries.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">2 &middot; Redundant columns</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Several columns measure nearly the same thing. Left in place they inflate standard
errors and, in one case, make the design matrix singular.
</div>""", unsafe_allow_html=True)

st.dataframe(top_correlations(), hide_index=True, width='stretch')

st.markdown("""<div class="decision">
<div class="what">Economy_status_Developing dropped</div>
<div class="why">A perfect mirror of Economy_status_Developed at r = &minus;1.000.
The two always sum to one, which is collinear with the intercept and makes the model
unsolvable. This is a duplicate column rather than a modelling choice.
</div></div>
<div class="decision">
<div class="what">Infant_deaths dropped, Under_five_deaths kept</div>
<div class="why">Correlated at 0.986. Under-five deaths is the broader measure and
overlaps slightly less with adult mortality.
</div></div>
<div class="decision">
<div class="what">Four immunisation columns averaged into Vaccination_coverage</div>
<div class="why">Hepatitis B, polio, diphtheria and measles are all percentage
coverage among one-year-olds, delivered through the same programmes, and move
together. Averaging keeps the signal without four near-identical columns.
</div></div>
<div class="decision">
<div class="what">Two thinness columns averaged into Thinness_avg</div>
<div class="why">Overlapping age bands measuring the same prevalence, correlated
at 0.939.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">3 &middot; Transformations</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Linear regression fits one slope per feature, so a curved relationship is fitted
badly. Two columns were log-transformed; the others were tested and left alone.
</div>""", unsafe_allow_html=True)

st.dataframe(skew_table(), hide_index=True, width='stretch')

st.markdown("""<div class="decision">
<div class="what">GDP_per_capita logged</div>
<div class="why">The clearest case. Moving from $600 to $3,000 per capita buys large
gains in life expectancy; moving from $60,000 to $63,000 buys almost none. Logging
straightens that curve and lifts the correlation with the target from 0.58 to 0.80.
</div></div>
<div class="decision">
<div class="what">Incidents_HIV logged</div>
<div class="why">Skew of 4.98 on the same criterion as GDP. After the transform its
coefficient is not significantly different from zero, because adult mortality already
captures HIV's effect on survival &mdash; the column has little left to explain.
</div></div>
<div class="decision">
<div class="what">Population_mln left raw</div>
<div class="why">Logging fixes the skew but changes nothing predictively: the
correlation with life expectancy is 0.026 raw and &minus;0.012 logged. Population
size simply does not predict life expectancy once wealth and region are known.
</div></div>
<div class="decision">
<div class="what">BMI left raw</div>
<div class="why">Skew of &minus;0.12 is already symmetric, and the range spans a
factor of 1.6 with no long tail. A log transform here would change the coefficient
without any justification from the data.
</div></div>
<div class="decision">
<div class="what">Region one-hot encoded</div>
<div class="why">A nominal category with no ordering, so label encoding would imply
a false ranking. One dummy is dropped to avoid collinearity with the intercept,
making Africa the reference category that all other coefficients are measured
against. Country itself is excluded: 179 dummies over 2,864 rows would memorise
each country's mean rather than generalise, and the prediction function must work
for countries outside the training data.
</div></div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">4 &middot; Verification</div>', unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Variance inflation factors on the final feature set. Values above 10 indicate a
feature largely predictable from the others; nothing reaches that threshold, so no
further columns were removed. A leave-one-out test confirmed this independently:
no feature's removal improved the model.
</div>""", unsafe_allow_html=True)

st.dataframe(vif_table(), width='stretch', height=300)

st.markdown("""<div class="body-text">
The high condition number reported by the unscaled model was traced to differing
feature scales rather than collinearity. Standardising reduced it from roughly
936,000 to under 10 while leaving every error metric identical to five decimal
places, confirming the diagnosis.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">5 &middot; The two feature sets</div>', unsafe_allow_html=True)

adv = [c for c in X_cols]
minimal = list(MINIMAL)
health_only = [c for c in adv if c not in minimal]

st.markdown(f"""<div class="body-text">
<strong>Advanced model &mdash; {len(adv)} columns.</strong> Every available statistic.
</div>
<div class="featlist">{', '.join(adv)}</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="body-text" style="margin-top:1.2rem;">
<strong>Minimal model &mdash; {len(minimal)} columns.</strong> The
{len(health_only)} health features below are withheld unless the user consents.
</div>
<div class="featlist"><span class="drop">{', '.join(health_only)}</span></div>""",
            unsafe_allow_html=True)

rmse = model_comparison.set_index('model')['rmse']
st.markdown(f"""<div class="body-text" style="margin-top:1.2rem;">
Withholding those {len(health_only)} features raises typical error from
&plusmn;{rmse.iloc[0]:.2f} to &plusmn;{rmse.iloc[1]:.2f} years. That gap is the
measurable cost of consent, and it is not evenly distributed: the minimal model's
errors are roughly three times larger for countries with low life expectancy than
for those with high, so the privacy-preserving model is least reliable for the
populations most at risk.
</div>""", unsafe_allow_html=True)


st.markdown('<div class="sect">6 &middot; A caution on reading coefficients</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="body-text">
Life expectancy at birth is calculated from age-specific mortality rates. Adult
mortality and under-five deaths are therefore not ordinary predictors: the model is
partly recovering an arithmetic identity, which is why they dominate every measure
of importance and why the advanced model is so accurate.
</div>
<div class="body-text">
A consequence is that several health features show a coefficient whose sign is
opposite to their simple correlation with life expectancy. BMI correlates positively
with life expectancy on its own, because a well-fed population is a wealthy one, but
once wealth and mortality are accounted for the remaining signal is obesity, and the
coefficient turns negative. These are conditional effects holding everything else
fixed, not causal claims, and the estimator's inputs should not be read as such.
</div>""", unsafe_allow_html=True)
