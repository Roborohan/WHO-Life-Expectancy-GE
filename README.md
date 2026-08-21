# WHO Life Expectancy

**[Try the live app →](https://who-life-expectancy-app.streamlit.app)**

Two linear regression models estimating national life expectancy from WHO country
statistics, built around a data consent problem.

## The brief

WHO member countries will share economic and demographic statistics, but are reluctant
to share anything derived from medical records. The task was to build two models — one
using every available statistic, one with all health data withheld — and a prediction
function that asks the user for consent and selects between them. The client set an
accuracy benchmark of 1.8 years RMSE.

## The models

| | Features | RMSE |
|---|---|---|
| Advanced | 20 | ±1.23 years |
| Minimal | 13 | ±4.55 years |

Both are ordinary least squares, trained on 179 countries across 2000–2015. The minimal
model excludes mortality, immunisation coverage, HIV incidence, BMI, thinness and
alcohol consumption.

## Key finding

The cost of withholding health data is not shared evenly. For the quarter of countries
with the lowest life expectancy the minimal model is over four times worse than the
advanced one, against under two and a half times for the highest quarter. Around three
quarters of its predictive weight falls on regional dummies, so a privacy-preserving
model ends up substituting geography for evidence — and fails hardest where accuracy
matters most.

## Running it locally

```bash
git clone https://github.com/Roborohan/WHO-Life-Expectancy.git
cd WHO-Life-Expectancy
pip install -r requirements.txt
streamlit run HOME.py
```

## Structure

```
HOME.py                  landing page
who_model.py             data pipeline, both fitted models, prediction functions
style.py                 shared CSS
pages/
  1_Life_Expectancy_Calculator.py    the estimator, with the consent prompt
  2_Feature_Choices.py               cleaning and feature engineering decisions
  3_Model_Comparison.py              model performance and the consent trade-off
WHO_EDA.ipynb            exploratory analysis and modelling notebook
```

## Data

WHO life expectancy data covering 179 countries, 2000–2015. Country-level aggregates
only, so no personal data is involved.

## Note

A student project built against a simulated brief. Not affiliated with, endorsed by, or
an official product of the World Health Organization.
