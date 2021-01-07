import pandas as pd
import numpy as np
import streamlit as st

chart_data = pd.DataFrame(
     np.random.randn(5, 1),
     columns=['a'])

chart = st.line_chart(chart_data)
chart.add_rows(np.array([[1.0]]))
chart.add_rows(np.array([[1.0]]))
chart.add_rows(np.array([[1.0]]))
chart.add_rows(np.array([[1.0]]))
chart.add_rows(np.array([[1.0]]))

