import os

import streamlit as st
from utils import generate_data

all_columns = ["Apples", "Orange", "Pineapple"]
with st.container(border=True):
    columns = st.multiselect("Columns", all_columns, default=all_columns)

all_data = st.cache_data(generate_data)(columns=all_columns, seed=101)

data = all_data[columns]

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data, height=250)
tab2.dataframe(data, height=250, use_container_width=True)
st.write(f"Environment: {os.environ}")
