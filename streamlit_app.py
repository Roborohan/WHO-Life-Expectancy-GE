import streamlit as st

from who_model import (
    predict_life_expectancy,
    predict_life_expectancy_min,
    VALID_REGIONS,
    FEATURE_RANGES,
    model_comparison,
)

st.set_page_config(page_title='WHO Life Expectancy Predictor', layout='centered')

st.title('Life Expectancy Predictor')
st.caption('WHO data, 179 countries, 2000–2015')


# ------------------------------------------------------------ Consent choice
st.subheader('Data consent')

st.write(
    'Do you consent to using advanced population data, which may include '
    'protected information, for better accuracy?'
)

consent = st.radio(
    'Select a model',
    ['Yes — use the advanced model', 'No — use the minimal model'],
    label_visibility='collapsed',
)

use_advanced = consent.startswith('Yes')

rmse = model_comparison.set_index('model')['rmse']
if use_advanced:
    st.success(
        f'Advanced model selected. Uses health statistics including mortality, '
        f'immunisation, HIV and BMI. Typical error ±{rmse.iloc[0]:.2f} years.'
    )
else:
    st.info(
        f'Minimal model selected. Uses only economic and demographic data — '
        f'no health statistics. Typical error ±{rmse.iloc[1]:.2f} years.'
    )

st.divider()


# ------------------------------------------------------------ Shared inputs
st.subheader('Population statistics')

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox('Region', VALID_REGIONS)
    year = st.slider('Year', 2000, 2015, 2015)
    schooling = st.slider('Average years of schooling',
                          *FEATURE_RANGES['Schooling'], 11.0)

with col2:
    gdp_per_capita = st.number_input('GDP per capita (USD)',
                                     min_value=1.0, value=9000.0, step=500.0)
    population_mln = st.number_input('Population (millions)',
                                     min_value=0.01, value=45.0, step=1.0)
    economy_status = st.selectbox('Economy status', ['Developing', 'Developed'])

economy_status_developed = 1 if economy_status == 'Developed' else 0


# ---------------------------------------------------- Advanced-only inputs
if use_advanced:
    st.divider()
    st.subheader('Health statistics')
    st.caption('These are the protected fields covered by the consent question above.')

    col3, col4 = st.columns(2)

    with col3:
        adult_mortality = st.slider('Adult mortality (per 1,000)',
                                    *FEATURE_RANGES['Adult_mortality'], 160.0)
        under_five_deaths = st.slider('Under-five deaths (per 1,000)',
                                      *FEATURE_RANGES['Under_five_deaths'], 25.0)
        bmi = st.slider('Average BMI', *FEATURE_RANGES['BMI'], 25.5)
        incidents_hiv = st.slider('HIV incidence (per 1,000)',
                                  *FEATURE_RANGES['Incidents_HIV'], 0.15)
        alcohol_consumption = st.slider('Alcohol consumption (litres)',
                                        *FEATURE_RANGES['Alcohol_consumption'], 4.0)

    with col4:
        hepatitis_b = st.slider('Hepatitis B coverage (%)',
                                *FEATURE_RANGES['Hepatitis_B'], 90.0)
        polio = st.slider('Polio coverage (%)', *FEATURE_RANGES['Polio'], 92.0)
        diphtheria = st.slider('Diphtheria coverage (%)',
                               *FEATURE_RANGES['Diphtheria'], 91.0)
        measles = st.slider('Measles coverage (%)', *FEATURE_RANGES['Measles'], 88.0)
        thinness_10_19 = st.slider('Thinness, ages 10–19 (%)',
                                   *FEATURE_RANGES['Thinness_ten_nineteen_years'], 8.0)
        thinness_5_9 = st.slider('Thinness, ages 5–9 (%)',
                                 *FEATURE_RANGES['Thinness_five_nine_years'], 8.5)


# ---------------------------------------------------------------- Predict
st.divider()

if st.button('Predict life expectancy', type='primary', use_container_width=True):

    if use_advanced:
        prediction = predict_life_expectancy(
            region=region, year=year,
            adult_mortality=adult_mortality, under_five_deaths=under_five_deaths,
            hepatitis_b=hepatitis_b, polio=polio, diphtheria=diphtheria,
            measles=measles, bmi=bmi, incidents_hiv=incidents_hiv,
            alcohol_consumption=alcohol_consumption,
            thinness_10_19=thinness_10_19, thinness_5_9=thinness_5_9,
            gdp_per_capita=gdp_per_capita, population_mln=population_mln,
            schooling=schooling, economy_status_developed=economy_status_developed,
        )
        error = rmse.iloc[0]
        label = 'Advanced model'
    else:
        prediction = predict_life_expectancy_min(
            region=region, year=year, gdp_per_capita=gdp_per_capita,
            population_mln=population_mln, schooling=schooling,
            economy_status_developed=economy_status_developed,
        )
        error = rmse.iloc[1]
        label = 'Minimal model'

    st.metric('Predicted life expectancy', f'{prediction:.1f} years',
              help=f'{label}, typical error ±{error:.2f} years')

    st.caption(
        f'Range allowing for typical model error: '
        f'{prediction - error:.1f} – {prediction + error:.1f} years.'
    )
