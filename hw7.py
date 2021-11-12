#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:34:23 2021

@author: tryfighter
"""

import numpy as np
import pandas as pd
import altair as alt
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import streamlit as st

st.title("K means cluster graph")
s = st.slider("select number of iterations", 0, 1000)

try:
    X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
    df = pd.DataFrame(X, columns = list("ab"))
    starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
    kmeans = KMeans(n_clusters = 5, max_iter=s, init=starting_points, n_init = 1)
    kmeans.fit(X);
    df["c"] = kmeans.predict(X)
    chart1 = alt.Chart(df).mark_circle().encode(
        x = "a",
        y = "b",
        color = "c:N"
    )
    
    df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))
    
    chart_centers = alt.Chart(df_centers).mark_point().encode(
        x = "a",
        y = "b",
        color = alt.value("black"),
        shape = alt.value("diamond"),
    )
    
    my_chart = chart1 + chart_centers
    st.altair_chart(my_chart)
except:
    st.write("Select number of iterations first")

st.write(np.__version__)
st.write(pd.__version__)
st.write(alt.__version__)
st.write(st.__version__)
