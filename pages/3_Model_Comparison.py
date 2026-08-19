import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

import style
from who_model import df, X_cols, X_train_s, MINIMAL, res_full, model_comparison

st.set_page_config(page_title='Comparison', layout='centered')

style.apply()

st.markdown("""<div class="mast">
<div class="eyebrow">Global Health Observatory &middot; Method</div>
<h1>Comparison</h1>
<div class="sub">Comparing the two models</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="lede">

</div>""", unsafe_allow_html=True)