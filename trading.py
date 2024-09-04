
#%%
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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




