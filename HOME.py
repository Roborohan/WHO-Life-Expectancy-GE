"""
Streamlit Complex Website Template
------------------------------------
A multi-section, styled Streamlit "website" template featuring a custom
theme, hero section, sidebar navigation, dashboard cards, charts, a
data table, a contact form, and a footer.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import style

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="WHO: Life Expectancy",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# GLOBAL CSS (from style.py)
# ------------------------------------------------------------------
style.apply()

# ------------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------------
with st.columns(3)[1]:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/WHO_logo.svg/3840px-WHO_logo.svg.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail")

st.markdown(
    """
    <div class="mast">
        <h1>Life Expectancy Measurements</h1>
        <div class="sub">Predicting and exploring life expectancy across countries and years.</div>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown('<div class="sect">In This Streamlit</div>', unsafe_allow_html=True)
st.markdown('       ')

st.page_link('pages/1_Life_Expectancy_Calculator.py', label='Estimate life expectancy using our two models', icon=':material/arrow_forward:')
st.page_link('pages/2_Feature_Choices.py', label='Features we kept, combined and dropped', icon=':material/arrow_forward:')
st.page_link('pages/3_Model_Comparison.py', label='Comparing the two models', icon=':material/arrow_forward:')


