import streamlit as st

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
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
.block-container { padding-top: 3rem; max-width: 940px; padding-left: 2rem; padding-right: 2rem; }
.lede, .body-text, .decision { max-width: 680px; }
.mast { border-bottom: 3px solid var(--who-blue); padding-bottom: 0.9rem; margin-bottom: 0.4rem; }
.mast .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--who-blue); margin-bottom: 0.5rem;
}
.mast h1 {
    font-size: 2.1rem; font-weight: 600; margin: 0; letter-spacing: -0.02em;
    color: inherit;
}
.mast .sub { font-size: 1rem; opacity: 0.7; margin-top: 0.45rem; }
.sect {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.86rem; font-weight: 500; letter-spacing: 0.13em;
    text-transform: uppercase; opacity: 0.78; margin: 2.2rem 0 0.3rem 0;
    border-top: 1px solid var(--who-rule); padding-top: 0.95rem;
}
.sect-note { font-size: 0.86rem; opacity: 0.65; margin-bottom: 0.7rem; }
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
    font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem;
    font-weight: 500; letter-spacing: 0.13em; text-transform: uppercase;
    opacity: 0.78;
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
.lede { font-size: 1rem; line-height: 1.65; margin: 1.2rem 0 0.4rem 0; }
.body-text { font-size: 0.92rem; line-height: 1.7; margin-bottom: 0.8rem; }
.decision {
    border-left: 3px solid var(--who-blue); padding: 0.1rem 0 0.1rem 0.9rem;
    margin: 0.9rem 0 1.2rem 0;
}
.decision .what { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem; }
.decision .why { font-size: 0.88rem; line-height: 1.65; opacity: 0.85; }
.featlist {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    line-height: 1.9; opacity: 0.85; word-spacing: 0.1em;
}
.featlist .drop { text-decoration: line-through; opacity: 0.45; }
.featlist .new { color: var(--who-blue); }
.stTable table, div[data-testid="stTable"] table {
    font-size: 0.84rem; border-collapse: collapse; width: 100%;
}
.stTable thead th, div[data-testid="stTable"] thead th {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7;
    text-align: left; border-bottom: 2px solid var(--who-rule);
    padding: 0.5rem 0.7rem; background: transparent;
}
.stTable tbody td, div[data-testid="stTable"] tbody td {
    border-bottom: 1px solid var(--who-rule); padding: 0.45rem 0.7rem;
    background: transparent;
}
.stTable tbody td:nth-child(n+2), div[data-testid="stTable"] tbody td:nth-child(n+2) {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}
.stTable tbody tr:hover td, div[data-testid="stTable"] tbody tr:hover td {
    background: rgba(128, 150, 162, 0.08);
}
div[data-testid="stPageLink"] a {
    border: 1px solid var(--who-rule); border-left: 3px solid var(--who-blue);
    border-radius: 0; padding: 0.7rem 1rem; width: 100%;
    transition: background 0.15s ease;
}
div[data-testid="stPageLink"] a:hover { background: rgba(128, 150, 162, 0.1); }
div[data-testid="stPageLink"] a p { font-size: 0.9rem !important; font-weight: 500; }
</style>"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)