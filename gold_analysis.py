
#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


#%%
file_path = "/home/lin/codebase/stock_analysis/SIE.DE.csv"

data = pd.read_csv(file_path)



# %%
px.line(data_frame=data, y="Close", x="Date", template="plotly_dark")

#%%
data
# %%
data.info()
# %%
data["Date"] = pd.to_datetime(data["Date"])
# %%
cmb_filepath = "/home/lin/codebase/stock_analysis/CBK.DE.csv"

commerzebank_data = pd.read_csv(cmb_filepath)
# %%
px.line(data_frame=commerzebank_data, y="Low", x="Date", template="plotly_dark")
# %%
infineon_datapath = "/home/lin/codebase/stock_analysis/IFX.DE.csv"

infneone_data = pd.read_csv(infineon_datapath)
# %%
px.line(data_frame=infneone_data, y="Low", x="Date", template="plotly_dark",
        title="Infineion daily stocks")

# %%

bac_filepath = "/home/lin/codebase/stock_analysis/BAC.csv"

bac_data = pd.read_csv(bac_filepath)
px.line(data_frame=bac_data, y="Low", x="Date", template="plotly_dark",
        title="Bank of America daily stocks")
#%%
(110/100)*31
# %%
