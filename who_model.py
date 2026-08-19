import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

CSV_PATH = 'Life Expectancy Data.csv'


# 1. EDA

df = pd.read_csv(CSV_PATH)

# df.head()
# df.shape
# df.info()
# df.dtypes
# df.describe()
# df['Measles'].unique()

# c = df.corr(numeric_only=True)
# mask = np.triu(np.ones(c.shape), k=1).astype(bool)
# pairs = c.where(mask).stack()
# pairs[pairs.abs() > 0.75].sort_values(key=abs, ascending=False)

# plt.figure(figsize=(12, 5))
# sns.boxplot(data=df, x='Region', y='Life_expectancy')
# plt.xticks(rotation=45, ha='right')
# plt.show()

# obj_cols = df.select_dtypes('object').columns
# for c in obj_cols:
#     print(f'{c:18s} {df[c].nunique():3d}  {sorted(df[c].unique())[:8]}')


# 2. Encoding

df_clean = df.copy()
df_clean = pd.get_dummies(df_clean, columns=['Region'], drop_first=True)


# 3. Feature Engineering

df_clean['Vaccination_coverage'] = df_clean[['Hepatitis_B', 'Polio',
                                             'Diphtheria', 'Measles']].mean(axis=1)
df_clean = df_clean.drop(columns=['Hepatitis_B', 'Polio', 'Diphtheria', 'Measles'])

df_clean = df_clean.drop(columns=['Infant_deaths'])
df_clean = df_clean.drop(columns=['Economy_status_Developing'])

df_clean['Thinness_avg'] = df_clean[['Thinness_ten_nineteen_years',
                                     'Thinness_five_nine_years']].mean(axis=1)
df_clean = df_clean.drop(columns=['Thinness_ten_nineteen_years',
                                  'Thinness_five_nine_years'])

df_clean['log_GDP_per_capita'] = np.log1p(df_clean['GDP_per_capita'])
df_clean = df_clean.drop(columns=['GDP_per_capita'])

df_clean['log_Incidents_HIV'] = np.log1p(df_clean['Incidents_HIV'])
df_clean = df_clean.drop(columns=['Incidents_HIV'])

# col = 'GDP_per_capita'
# plt.scatter(df[col], df['Life_expectancy'], s=5, alpha=0.3)
# plt.show()
# plt.scatter(np.log1p(df[col]), df['Life_expectancy'], s=5, alpha=0.3)
# plt.show()

# num_cols = df.select_dtypes('number').columns.drop(
#     ['Year', 'Economy_status_Developed', 'Economy_status_Developing'])
# fig, axes = plt.subplots(4, 4, figsize=(16, 14))
# for ax, col in zip(axes.flat, num_cols):
#     ax.scatter(df[col], df['Life_expectancy'], s=4, alpha=0.3)
#     ax.set_title(f'{col}  r={df[col].corr(df.Life_expectancy):.2f}', fontsize=9)
# plt.tight_layout()
# plt.show()

# print(f'Before: {df.shape} -> After: {df_clean.shape}')


# 4. Train/Test Split

y = df_clean['Life_expectancy']
X = df_clean.drop(columns=['Life_expectancy', 'Country'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_cols = X.columns.tolist()


# 5. Scaling

# scalers = {'unscaled': None, 'StandardScaler': StandardScaler(),
#            'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}
# rows = []
# for name, sc in scalers.items():
#     Xtr, Xte = X_train.copy().astype(float), X_test.copy().astype(float)
#     if sc is not None:
#         Xtr[X_cols] = sc.fit_transform(Xtr[X_cols])
#         Xte[X_cols] = sc.transform(Xte[X_cols])
#     lr = LinearRegression().fit(Xtr, y_train)
#     pred = lr.predict(Xte)
#     cond = sm.OLS(y_train, sm.add_constant(Xtr, has_constant='add')).fit().condition_number
#     rows.append([name, metrics.r2_score(y_test, pred),
#                  metrics.root_mean_squared_error(y_test, pred),
#                  metrics.mean_absolute_error(y_test, pred), cond])
# scaler_comparison = pd.DataFrame(rows, columns=['scaler', 'r2', 'rmse', 'mae', 'condition_no'])

scaler = StandardScaler()

X_train_s = X_train.copy().astype(float)
X_test_s = X_test.copy().astype(float)
X_train_s[X_cols] = scaler.fit_transform(X_train_s[X_cols])
X_test_s[X_cols] = scaler.transform(X_test_s[X_cols])


# 6. Fitting the Model

def fit_ols(Xtr, ytr, Xte):
    Xtr = sm.add_constant(Xtr.astype(float), has_constant='add')
    Xte = sm.add_constant(Xte.astype(float), has_constant='add')[Xtr.columns]
    res = sm.OLS(ytr, Xtr).fit()
    return res, res.predict(Xtr), res.predict(Xte)


res_full, ptr_full, pte_full = fit_ols(X_train_s, y_train, X_test_s)

REGION_COLS = [c for c in X_cols if c.startswith('Region_')]

MINIMAL = ['log_GDP_per_capita', 'Population_mln', 'Schooling',
           'Economy_status_Developed', 'Year'] + REGION_COLS

res_min, ptr_min, pte_min = fit_ols(X_train_s[MINIMAL], y_train, X_test_s[MINIMAL])

# design = sm.add_constant(X_train_s.astype(float), has_constant='add')
# print('columns:', design.shape[1], ' rank:', np.linalg.matrix_rank(design.values))

# Xv = sm.add_constant(X_train_s[X_cols].astype(float), has_constant='add')
# vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])],
#                 index=Xv.columns)
# vif.drop('const').sort_values(ascending=False).round(2)

# c = X_train.corr()
# pairs = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
# pairs[pairs.abs() > 0.7].sort_values(key=abs, ascending=False)


# 7. Metrics

def all_metrics(y_true, y_pred, label):
    return {
        'model': label,
        'rmse': metrics.root_mean_squared_error(y_true, y_pred),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'r2': metrics.r2_score(y_true, y_pred),
    }


model_comparison = pd.DataFrame([
    all_metrics(y_test, pte_full, 'Advanced (all features)'),
    all_metrics(y_test, pte_min, 'Minimal (no health data)'),
])

# res_log, ptr_log, pte_log = fit_ols(X_train_s, np.log(y_train), X_test_s)
# comparison = pd.DataFrame([
#     all_metrics(y_test, pte_full, 'OLS on LifeExp'),
#     all_metrics(y_test, np.exp(pte_log), 'OLS on log(LifeExp)'),
# ])

# res_log_min, _, pte_log_min = fit_ols(X_train_s[MINIMAL], np.log(y_train), X_test_s[MINIMAL])
# comparison_min = pd.DataFrame([
#     all_metrics(y_test, pte_min, 'Minimal on LifeExp'),
#     all_metrics(y_test, np.exp(pte_log_min), 'Minimal on log(LifeExp)'),
# ])

# base_rmse = metrics.root_mean_squared_error(y_test, pte_full)
# rows = []
# for col in X_cols:
#     _, _, pte = fit_ols(X_train_s.drop(columns=[col]), y_train, X_test_s.drop(columns=[col]))
#     rmse = metrics.root_mean_squared_error(y_test, pte)
#     rows.append([col, rmse, rmse - base_rmse])
# drop_test = pd.DataFrame(rows, columns=['dropped', 'rmse', 'change'])


# 8. Predictions and Minimal Model

VALID_REGIONS = sorted(df['Region'].unique())


def predict_life_expectancy(region, year, adult_mortality, under_five_deaths,
                            hepatitis_b, polio, diphtheria, measles,
                            bmi, incidents_hiv, alcohol_consumption,
                            thinness_10_19, thinness_5_9,
                            gdp_per_capita, population_mln,
                            schooling, economy_status_developed):
    if region not in VALID_REGIONS:
        raise ValueError(f"Unknown region '{region}'. Expected one of {VALID_REGIONS}")

    row = {
        'Year': year,
        'Adult_mortality': adult_mortality,
        'Under_five_deaths': under_five_deaths,
        'Alcohol_consumption': alcohol_consumption,
        'BMI': bmi,
        'log_Incidents_HIV': np.log1p(incidents_hiv),
        'Population_mln': population_mln,
        'Schooling': schooling,
        'Economy_status_Developed': economy_status_developed,
        'Vaccination_coverage': np.mean([hepatitis_b, polio, diphtheria, measles]),
        'Thinness_avg': np.mean([thinness_10_19, thinness_5_9]),
        'log_GDP_per_capita': np.log1p(gdp_per_capita),
    }
    row[f'Region_{region}'] = 1

    X_new = pd.DataFrame([row]).reindex(columns=X_cols, fill_value=0).astype(float)
    X_new[X_cols] = scaler.transform(X_new[X_cols])
    X_new = sm.add_constant(X_new, has_constant='add')

    return float(res_full.predict(X_new).iloc[0])


def predict_life_expectancy_min(region, year, gdp_per_capita, population_mln,
                                schooling, economy_status_developed):
    if region not in VALID_REGIONS:
        raise ValueError(f"Unknown region '{region}'. Expected one of {VALID_REGIONS}")

    row = {
        'Year': year,
        'Population_mln': population_mln,
        'Schooling': schooling,
        'Economy_status_Developed': economy_status_developed,
        'log_GDP_per_capita': np.log1p(gdp_per_capita),
    }
    row[f'Region_{region}'] = 1

    X_new = pd.DataFrame([row]).reindex(columns=X_cols, fill_value=0).astype(float)
    X_new[X_cols] = scaler.transform(X_new[X_cols])
    X_new = sm.add_constant(X_new[MINIMAL], has_constant='add')

    return float(res_min.predict(X_new).iloc[0])


FEATURE_RANGES = {
    'Year': (2000, 2015),
    'Adult_mortality': (float(df.Adult_mortality.min()), float(df.Adult_mortality.max())),
    'Under_five_deaths': (float(df.Under_five_deaths.min()), float(df.Under_five_deaths.max())),
    'Hepatitis_B': (float(df.Hepatitis_B.min()), float(df.Hepatitis_B.max())),
    'Polio': (float(df.Polio.min()), float(df.Polio.max())),
    'Diphtheria': (float(df.Diphtheria.min()), float(df.Diphtheria.max())),
    'Measles': (float(df.Measles.min()), float(df.Measles.max())),
    'BMI': (float(df.BMI.min()), float(df.BMI.max())),
    'Incidents_HIV': (float(df.Incidents_HIV.min()), float(df.Incidents_HIV.max())),
    'Alcohol_consumption': (float(df.Alcohol_consumption.min()), float(df.Alcohol_consumption.max())),
    'Thinness_ten_nineteen_years': (float(df.Thinness_ten_nineteen_years.min()),
                                    float(df.Thinness_ten_nineteen_years.max())),
    'Thinness_five_nine_years': (float(df.Thinness_five_nine_years.min()),
                                 float(df.Thinness_five_nine_years.max())),
    'GDP_per_capita': (float(df.GDP_per_capita.min()), float(df.GDP_per_capita.max())),
    'Population_mln': (float(df.Population_mln.min()), float(df.Population_mln.max())),
    'Schooling': (float(df.Schooling.min()), float(df.Schooling.max())),
}


if __name__ == '__main__':
    print(model_comparison.round(3).to_string(index=False))
    print()
    print('Advanced sample :', round(predict_life_expectancy(
        region='Asia', year=2015, adult_mortality=160, under_five_deaths=25,
        hepatitis_b=90, polio=92, diphtheria=91, measles=88, bmi=25.5,
        incidents_hiv=0.15, alcohol_consumption=4.0, thinness_10_19=8.0,
        thinness_5_9=8.5, gdp_per_capita=9000, population_mln=45.0,
        schooling=11.0, economy_status_developed=0), 2))
    print('Minimal sample  :', round(predict_life_expectancy_min(
        region='Asia', year=2015, gdp_per_capita=9000, population_mln=45.0,
        schooling=11.0, economy_status_developed=0), 2))