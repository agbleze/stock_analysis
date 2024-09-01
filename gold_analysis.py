
#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


#%%
file_path = "/home/lin/codebase/stock_analysis/SIE.DE.csv"

data = pd.read_csv(file_path)



# %%
px.line(data_frame=data, y="Close", x="Date", template="plotly_dark")
