

#%%

import pandas as pd

filepath = "/home/lin/codebase/stock_analysis/WhatsgoodlyData-6.csv"
data = pd.read_csv(filepath)
# %%
data
# %%
data.dtypes

# %%
data.columns
# %%
data["Count"].sum()
# %%
data.head()
# %%
data.shape
# %%
data[(data["Segment Type"] == "Mobile") & (data["Answer"] == "Facebook")]
# %%
data.iloc[:, 0]
# %%
