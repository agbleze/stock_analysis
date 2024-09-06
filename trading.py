
#%%
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

#%%

# Download historical data for "WEAT" (Wheat ETF) from Yahoo Finance
ticker = "WEAT"
start_date = "2000-01-01"
end_date = "2023-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate 20-day and 50-day moving averages
data["20-day MA"] = data["Close"].rolling(window=20).mean()
data["50-day MA"] = data["Close"].rolling(window=50).mean()

# Create buy/sell signals
data["Buy Signal"] = np.where((data["20-day MA"] > data["50-day MA"]), 1, 0)
data["Sell Signal"] = np.where((data["20-day MA"] < data["50-day MA"]), -1, 0)

# Initialize balance and position
initial_balance = 100
balance = initial_balance
position = 0

# Simulate trading
for i in range(len(data)):
    if data["Buy Signal"][i] == 1:
        position += balance / data["Close"][i]
        balance = 0
    elif data["Sell Signal"][i] == -1:
        balance += position * data["Close"][i]
        position = 0

# Calculate final balance and profit
final_balance = balance + position * data["Close"][-1]
profit = final_balance - initial_balance

# Create an interactive line chart with moving averages and buy/sell signals
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Wheat Price"))
fig.add_trace(go.Scatter(x=data.index, y=data["20-day MA"], mode="lines", name="20-day MA", line=dict(color="green")))
fig.add_trace(go.Scatter(x=data.index, y=data["50-day MA"], mode="lines", name="50-day MA", line=dict(color="red")))
fig.add_trace(go.Scatter(x=data.index[data["Buy Signal"] == 1], y=data["Close"][data["Buy Signal"] == 1],
                         mode="markers", marker=dict(color="green", symbol="triangle-up", size=10), name="Buy Signal"))
fig.add_trace(go.Scatter(x=data.index[data["Sell Signal"] == -1], y=data["Close"][data["Sell Signal"] == -1],
                         mode="markers", marker=dict(color="red", symbol="triangle-down", size=10), name="Sell Signal"))

fig.update_layout(title="Wheat Trading System Backtesting",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  xaxis_rangeslider_visible=False,
                  )

# Print results
print(f"Initial balance: ${initial_balance:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")

# Show interactive plot
fig.show()

# %%
ticker = "ADN1.DE"

#%%
yf.download(tickers=ticker)
# %%
crwd_ticker = "45C.F"

crwdshare_data = yf.download(tickers=crwd_ticker)
# %%
crwdshare_asd = crwdshare_data.sort_values(by="Date", ascending=False)
# %%
crwdshare_asd.head(50)


# %%
crwdshare_asd.info()
# %%
crwdshare_asd.columns
# %%
crwdshare_asd.index.year
# %%
crwdshare_asd["month"] = crwdshare_asd.index.month
crwdshare_asd["day"] = crwdshare_asd.index.day
crwdshare_asd["weekday"] =crwdshare_asd.index.weekday
crwdshare_asd["day_name"] = crwdshare_asd.index.day_name()
crwdshare_asd["year"] = crwdshare_asd.index.year
crwdshare_asd["month_name"] = crwdshare_asd.index.month_name()
# %%

crwdshare_asd.groupby(by="day_name")[["Close"]].mean()

#%%

crwdshare_asd.month.unique()
# %%
crwdshare_asd[crwdshare_asd["month"]==9].groupby(by=["month","day_name"])[["Close"]].mean()
# %%
crwdshare_asd.groupby(by="month")[["Close"]].mean().sort_values(by="Close")
# %%
crwdshare_asd[["Close"]].diff(-1).min()
# %%
# develop the inverted triangle trading strategy
# start by getting the opening price
# monitor prices afterward to ensure they drop by a
# certain huge margin
# if this huge margin is meet, then trigger a buy 

## loss exit watch
# watch the price after buying, if they fall further 
# to a certain margin, exit with a loss

## watch profit exit
# monitor price after buying if monitored price
# is up to a certain margin above the buy price 
# then exit with profit
# design the algorithm in such a way that it triggers 
# not more than 1 in a week

#%%
crwdshare_asd.columns



# %%
px.line(data_frame=crwdshare_asd, y="Close")
# %%
import os
img_dir = "/home/lin/codebase/stock_analysis/images"
os.makedirs(img_dir, exist_ok=True)
for mth in crwdshare_asd.month_name.unique():
    data = crwdshare_asd[(crwdshare_asd.month_name == mth) & (crwdshare_asd.year==2023)]
    fig = px.line(data_frame=data, y="Close", 
                  template="plotly_dark",
                  title=f"{mth} Close price - crowdstrike",
                  #height=800, width=1800
                  )
    image = os.path.join(img_dir, f"{mth}.jpg")
    fig.write_image(image)
    
# %%
def create_monthly_graphs_per_year(data, month, year, images_dir):
    img_dir = os.path.join(images_dir, str(year))
    os.makedirs(img_dir, exist_ok=True)
    data = data[data.year==year]
    data = data[data.month_name == month]
    fig = px.line(data_frame=data, y="Close", 
                template="plotly_dark",
                title=f"{month} Close price - crowdstrike",
                #height=800, width=1800
                )
    image = os.path.join(img_dir, f"{month}.jpg")
    fig.write_image(image)

#%%
create_monthly_graphs_per_year(data=crwdshare_asd, months=crwdshare_asd.month_name.unique(), 
                               years=crwdshare_asd.year.unique(),
                               images_dir="images"
                               )



#%%

for yr in crwdshare_asd.year.unique():
    for mth in crwdshare_asd.month_name.unique():
        create_monthly_graphs_per_year(data=crwdshare_asd,
                                       month=mth, year=yr, images_dir="images"
                                       )
        


#%%

nvidia_data = yf.download(tickers="NVD.DE")   

#%%

nvidia_data    

nvidia_data["month"] = nvidia_data.index.month
nvidia_data["day"] = nvidia_data.index.day
nvidia_data["weekday"] =nvidia_data.index.weekday
nvidia_data["day_name"] = nvidia_data.index.day_name()
nvidia_data["year"] = nvidia_data.index.year
nvidia_data["month_name"] = nvidia_data.index.month_name()
#%%


for yr in nvidia_data.year.unique():
    for mth in nvidia_data.month_name.unique():
        create_monthly_graphs_per_year(data=nvidia_data,
                                       month=mth, year=yr, images_dir="nvidia_images"
                                       )
        

#%%
# schaeffler
ticker = "SHA.DE"

scaeffler_df = yf.download(tickers=ticker)

#%%

scaeffler_df

px.line(data_frame=scaeffler_df, y="Close", template="plotly_dark", title="Schaeffler stocks (close)")

#%%

scaeffler_df[["Close"]].describe()


#%%
def create_date_columns(data):
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["weekday"] = data.index.weekday
    data["day_name"] = data.index.day_name()
    data["year"] = data.index.year
    data["month_name"] = data.index.month_name()
    return data


scaeffler_df = create_date_columns(data=scaeffler_df)

#%%
scaeffler_df.sort_values(by="Date", ascending=False)


#%%
scaeffler_df[["Open"]].describe()

scaeffler_df[scaeffler_df.year==2024]



#%%

scaeffler_df.sort_values(by="Close").head(50).value_counts("month_name")

#%%

scaeffler_df[["Adj Close"]].max()


#%%

scaeffler_df.groupby(by="month_name")["Close"].mean().reset_index().sort_values("Close").describe()

#%%

scaeffler_df.describe()
#%%

scaeffler_df[["Close"]].min()

#%%

scaeffler_df.year.nunique()
#%%

# AI for trading, cv
# a model to predict whether to trade, if to trade then indicate entry and exit
# points. In production, the model will be given images of historical till current stock price charts
# then it will predict with mark on the chart, when to enter and another mark on,
# when to exit.
# approach 1 
# use GENAI to take a stock chart image  with historical data as imput and the model 
# generates an image with with both historical and future and prices and then mark enter and exit points 

## approach 2
# multimodal 
# -- give the model up to 2/3 of the month data (stock chart) and predicts whether it will 
# end on a higher value and amount on the last day of the month


# approach 3
# classification
# - give the model the stock chart price up to half of the month and let it predict 
# whether price with end very high, medium, low
# consider using the moving average to provide trends that can make the prediction easier

#### collect data
# visualize the stock price and save as images. For the
# training data, indicate enter arrows 