
#%%
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_seasonality
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import numpy as np
import itertools
from prophet.serialize import model_to_json, model_from_json
import json
import tensorflow as tf
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional, LSTM, Flatten,
                                     TimeDistributed, RepeatVector,
                                     Conv1D, MaxPool1D)
from keras.layers import LSTM, Dense, Bidirectional, MaxPool1D, Dropout
from statsmodels.tsa.seasonal import seasonal_decompose
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

scaeffler_df[scaeffler_df["year"].isin([2023, 2022, 2021, 2020])].describe()

#%%
for yr in scaeffler_df.year.unique():
    print(yr)
    print(scaeffler_df[scaeffler_df["year"].isin([yr])].describe())


#%%

#scaeffler_df[scaeffler_df["Close"]==17.1]#.index.values  
 
scaeffler_df[scaeffler_df["year"]==2015][["Close"]].max()


"""
describe the data for each year, based on how the mean, max and 25% percentile
is changing determine regime changes. Buy when the stock price is below the 
25percentile and exit at the predicted max or 75% percentile

"""

#%%


scaeffler_df[scaeffler_df["year"]==2023]
scaeffler_df["Date"] = scaeffler_df.index.values
def get_min_max(data, year, param="Close"):
    data = data[(data["year"]==year)]
    max_row = data[data[param]==data[param].max()]
    min_row = data[data[param]==data[param].min()]
    return min_row, max_row


#%%
scae2023_minrow, scae2023_maxrow = get_min_max(data=scaeffler_df, year=2023)

#%%
scae2023_minrow
#%%

scae2023_maxrow

#%%

px.line(data_frame=scaeffler_df[scaeffler_df["year"].isin([2024])], 
        y="Close",
        template="plotly_dark"
        )


#%%
(50/100)*36

(118/100)*11.9

#%%
def get_price(data, current_year):
    prev_yr = current_year - 1
    if prev_yr not in data.year.unique():
        print(f"Data not available for {current_year}")
        return 
    data = data[data["year"]==prev_yr]
    stats = data.describe()
    min_close_price = stats.loc["min"]["Close"]
    close_price_75per = stats.loc["75%"]["Close"]
    new_price = (close_price_75per/min_close_price) * 100 
    percent_diff = new_price - 100
    percent_to_add = (50/100) * percent_diff
    exit_price = ((100 + percent_to_add) /100) * min_close_price
    return min_close_price, exit_price

#%%
get_price(scaeffler_df, 2014)    


#%%
yr = 2022
enter_price, exit_price = get_price(scaeffler_df, yr)

print(enter_price, exit_price)
    
px.line(data_frame=scaeffler_df[scaeffler_df["year"].isin([yr])], 
        y="Close",
        template="plotly_dark"
        )
#%%

import numpy as np

np.mean([100, 50, 30, 10, 50])
#%%

sca_desc = scaeffler_df.describe()


#%%
(105/100)*7.05

#%%
(125/100)*5.82

#%%
'''TODO: Implement training profit where 70 0r 90% of profit is picked when 
when profit_rate = 50% is used
'''
def place_order(data, current_year, investment_amount=100, profit_rate=None,
                stop_loss=None
                ):
    buy_price, exit_price = get_price(data=data, current_year=current_year)
    # cal profit and loss for investment
    buy_setup = False
    sell_setup = False
    entered_market = False
    realized_amount = False
    buy_date, sell_date = None, None
    enter_price = None
    data = data.rename(columns={"Date": "_Date"})
    monitor_data = data[data["year"]==current_year]
    last_index = monitor_data.index[-1]
    #print(monitor_data)
    # monitor_data = monitor_data.sort_values(by="_Date", 
    #                                         ascending=True,
    #                                         inplace=True
    #                                         )
    for rowdata_index, rowdata in monitor_data.iterrows():
        if buy_setup and not entered_market:
            # TODO.: check if current close price has reduced beyond stop_loss and exit 
            enter_price = rowdata["Close"]
            # send buy order and get number of shares bought
            number_of_shares = investment_amount/enter_price
            entered_market = True
            if profit_rate:
                exit_price = ((100+profit_rate)/100) * enter_price
            buy_date = rowdata["_Date"]
        elif not buy_setup and not entered_market:
            close_price = rowdata["Close"]
            if close_price <= buy_price:
                buy_setup = True
            else:
                continue
        if entered_market:
            sell_price = rowdata["Close"]
            if sell_price >= exit_price:
                # send sell order with api 
                sell_date = rowdata["_Date"]
                realized_amount = sell_price * number_of_shares
                profit = realized_amount - investment_amount
                percent_increase = (realized_amount/investment_amount) * 100
                profit_percent = percent_increase - 100
                return {"enter_price": enter_price,
                        "buy_price": buy_price,
                        "buy_date": buy_date,
                        "sell_date": sell_date,
                        "exit_price": exit_price,
                        "sell_price": sell_price,
                        "number_of_shares": number_of_shares,
                        "profit_amount": profit,
                        "realized_amount": realized_amount,
                        "profit_lose_percent": profit_percent,
                        "exit_type": "profit" 
                        }
            elif rowdata_index != last_index:
                if stop_loss:
                    exit_stoploss_price = ((100 - stop_loss) / 100) * enter_price
                    if sell_price <= exit_stoploss_price:
                        # send sell order with api 
                        sell_date = rowdata["_Date"]
                        realized_amount = sell_price * number_of_shares
                        profit = realized_amount - investment_amount
                        percent_increase = (realized_amount/investment_amount) * 100
                        profit_percent = percent_increase - 100
                        return {"enter_price": enter_price,
                                "buy_price": buy_price,
                                "buy_date": buy_date,
                                "sell_date": sell_date,
                                "exit_price": exit_price,
                                "sell_price": sell_price,
                                "number_of_shares": number_of_shares,
                                "profit_amount": profit,
                                "realized_amount": realized_amount,
                                "profit_lose_percent": profit_percent,
                                "exit_type": "stop_loss",
                                "exit_stoploss_price": exit_stoploss_price
                                }
                    else:
                        continue                    
            elif last_index == rowdata_index:
                # send sell order with api on last trade day of the year
                sell_date = rowdata["_Date"]
                realized_amount = sell_price * number_of_shares
                profit = realized_amount - investment_amount
                percent_increase = (realized_amount/investment_amount) * 100
                profit_percent = percent_increase - 100
                print(f"No sell condition meet. Exited on last trade day of the {current_year}")
                return {"enter_price": enter_price,
                        "buy_price": buy_price,
                        "buy_date": buy_date,
                        "sell_date": sell_date,
                        "exit_price": exit_price,
                        "sell_price": sell_price,
                        "number_of_shares": number_of_shares,
                        "profit_amount": profit,
                        "realized_amount": realized_amount,
                        "profit_lose_percent": profit_percent,
                        "exit_type": "last_trade_day"
                        }
            
            else:
                continue
    print(f"No trigger in {current_year}")
                
#%%

place_order(data=scaeffler_df, current_year=2024, stop_loss=30)
               


#%% 
"""
TODO: reorganize your algo into a class that can be automatically be called
with a ticker and it does the backtesting

1 download the data of ticker
2. backtest it

"""

#%%
class YearOnYearStrategy(object):
    def __init__(self, ticker):
        self.ticker = ticker
        
    def download_data(self, ticker=None):
        if not ticker:
            ticker = self.ticker
        
        self.data = yf.download(tickers=ticker)
        
    def create_date_columns(self, data=None):
        if data is None:
            data = self.data
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["weekday"] = data.index.weekday
        data["day_name"] = data.index.day_name()
        data["year"] = data.index.year
        data["month_name"] = data.index.month_name()
        data["Date"] = data.index.values
        self.data = data
        return data
    
    def get_price(self, current_year, data=None,):
        if data is None:
            data = self.data
        prev_yr = current_year - 1
        if prev_yr not in data.year.unique():
            print(f"Data not available for {current_year}")
            return 
        data = data[data["year"]==prev_yr]
        stats = data.describe()
        min_close_price = stats.loc["min"]["Close"]
        close_price_75per = stats.loc["75%"]["Close"]
        new_price = (close_price_75per/min_close_price) * 100 
        percent_diff = new_price - 100
        percent_to_add = (50/100) * percent_diff
        exit_price = ((100 + percent_to_add) /100) * min_close_price
        return min_close_price, exit_price
    
    def place_order(self, data, current_year, 
                    investment_amount=100, 
                    profit_rate=None,
                    stop_loss=None,
                    monitor_duration=None,
                    ):
        if data is None:
            data = self.data
        buy_price, exit_price = self.get_price(data=data, current_year=current_year)
        # cal profit and loss for investment
        buy_setup = False
        sell_setup = False
        entered_market = False
        realized_amount = False
        buy_date, sell_date = None, None
        enter_price = None
        data = data.rename(columns={"Date": "_Date"})
        monitor_data = data[data["year"]==current_year]
        last_index = monitor_data.index[-1]
        monitor_duration_data = None
        for rowdata_index, rowdata in monitor_data.iterrows():
            if buy_setup and not entered_market:
                # TODO: check if current close price has reduced beyond stop_loss and exit 
                enter_price = rowdata["Close"]
                # send buy order and get number of shares bought
                number_of_shares = investment_amount/enter_price
                entered_market = True
                if profit_rate:
                    exit_price = ((100+profit_rate)/100) * enter_price
                buy_date = rowdata["_Date"]
                if monitor_duration:
                    monitor_datelimit = buy_date + DateOffset(months=monitor_duration)
                    monitor_duration_data = data[(data["_Date"] > buy_date) & (data["_Date"]<=monitor_datelimit)]
                    #print(f"buy date: {buy_date}")
                    #print(f"monitor_datelimit: {monitor_datelimit}")
                    #print(f"monitor_data. {monitor_data}")
            elif not buy_setup and not entered_market:
                close_price = rowdata["Close"]
                if close_price <= buy_price:
                    buy_setup = True
                else:
                    continue
            if entered_market:
                if not monitor_duration:
                    sell_price = rowdata["Close"]
                    if sell_price >= exit_price:
                        # send sell order with api 
                        sell_date = rowdata["_Date"]
                        realized_amount = sell_price * number_of_shares
                        profit = realized_amount - investment_amount
                        percent_increase = (realized_amount/investment_amount) * 100
                        profit_percent = percent_increase - 100
                        return {"enter_price": enter_price,
                                "buy_price": buy_price,
                                "buy_date": buy_date,
                                "sell_date": sell_date,
                                "exit_price": exit_price,
                                "sell_price": sell_price,
                                "number_of_shares": number_of_shares,
                                "profit_amount": profit,
                                "realized_amount": realized_amount,
                                "profit_lose_percent": profit_percent,
                                "exit_type": "profit",
                                "investment_amount": investment_amount,
                                "trade_duration": sell_date - buy_date
                                }
                    elif rowdata_index != last_index:
                        if stop_loss:
                            exit_stoploss_price = ((100 - stop_loss) / 100) * enter_price
                            if sell_price <= exit_stoploss_price:
                                # send sell order with api 
                                sell_date = rowdata["_Date"]
                                realized_amount = sell_price * number_of_shares
                                profit = realized_amount - investment_amount
                                percent_increase = (realized_amount/investment_amount) * 100
                                profit_percent = percent_increase - 100
                                return {"enter_price": enter_price,
                                        "buy_price": buy_price,
                                        "buy_date": buy_date,
                                        "sell_date": sell_date,
                                        "exit_price": exit_price,
                                        "sell_price": sell_price,
                                        "number_of_shares": number_of_shares,
                                        "profit_amount": profit,
                                        "realized_amount": realized_amount,
                                        "profit_lose_percent": profit_percent,
                                        "exit_type": "stop_loss",
                                        "exit_stoploss_price": exit_stoploss_price,
                                        "investment_amount": investment_amount,
                                        "trade_duration": sell_date - buy_date
                                        }
                            else:
                                continue                    
                    elif last_index == rowdata_index:
                        # send sell order with api on last trade day of the year
                        sell_date = rowdata["_Date"]
                        realized_amount = sell_price * number_of_shares
                        profit = realized_amount - investment_amount
                        percent_increase = (realized_amount/investment_amount) * 100
                        profit_percent = percent_increase - 100
                        print(f"No sell condition meet. Exited on last trade day of the {current_year}")
                        return {"enter_price": enter_price,
                                "buy_price": buy_price,
                                "buy_date": buy_date,
                                "sell_date": sell_date,
                                "exit_price": exit_price,
                                "sell_price": sell_price,
                                "number_of_shares": number_of_shares,
                                "profit_amount": profit,
                                "realized_amount": realized_amount,
                                "profit_lose_percent": profit_percent,
                                "exit_type": "last_trade_day",
                                "investment_amount": investment_amount,
                                "trade_duration": sell_date - buy_date
                                }
                    
                    else:
                        continue
                else:
                    last_index = monitor_duration_data.index[-1]
                    for rowdata_index, rowdata in monitor_duration_data.iterrows():
                        sell_price = rowdata["Close"]
                        if sell_price >= exit_price:
                            # send sell order with api 
                            sell_date = rowdata["_Date"]
                            realized_amount = sell_price * number_of_shares
                            profit = realized_amount - investment_amount
                            percent_increase = (realized_amount/investment_amount) * 100
                            profit_percent = percent_increase - 100
                            return {"enter_price": enter_price,
                                    "buy_price": buy_price,
                                    "buy_date": buy_date,
                                    "sell_date": sell_date,
                                    "exit_price": exit_price,
                                    "sell_price": sell_price,
                                    "number_of_shares": number_of_shares,
                                    "profit_amount": profit,
                                    "realized_amount": realized_amount,
                                    "profit_lose_percent": profit_percent,
                                    "exit_type": "profit",
                                    "investment_amount": investment_amount,
                                    "monitor_duration": monitor_duration,
                                    "trade_duration": sell_date - buy_date
                                    }
                        elif rowdata_index != last_index:
                            if stop_loss:
                                exit_stoploss_price = ((100 - stop_loss) / 100) * enter_price
                                if sell_price <= exit_stoploss_price:
                                    # send sell order with api 
                                    sell_date = rowdata["_Date"]
                                    realized_amount = sell_price * number_of_shares
                                    profit = realized_amount - investment_amount
                                    percent_increase = (realized_amount/investment_amount) * 100
                                    profit_percent = percent_increase - 100
                                    return {"enter_price": enter_price,
                                            "buy_price": buy_price,
                                            "buy_date": buy_date,
                                            "sell_date": sell_date,
                                            "exit_price": exit_price,
                                            "sell_price": sell_price,
                                            "number_of_shares": number_of_shares,
                                            "profit_amount": profit,
                                            "realized_amount": realized_amount,
                                            "profit_lose_percent": profit_percent,
                                            "exit_type": "stop_loss",
                                            "exit_stoploss_price": exit_stoploss_price,
                                            "investment_amount": investment_amount,
                                            "monitor_duration": monitor_duration,
                                            "trade_duration": sell_date - buy_date
                                            }
                                else:
                                    continue                    
                        elif last_index == rowdata_index:
                            # send sell order with api on last trade day of the year
                            sell_date = rowdata["_Date"]
                            realized_amount = sell_price * number_of_shares
                            profit = realized_amount - investment_amount
                            percent_increase = (realized_amount/investment_amount) * 100
                            profit_percent = percent_increase - 100
                            print(f"No sell condition meet. Exited on last trade day of the {current_year}")
                            return {"enter_price": enter_price,
                                    "buy_price": buy_price,
                                    "buy_date": buy_date,
                                    "sell_date": sell_date,
                                    "exit_price": exit_price,
                                    "sell_price": sell_price,
                                    "number_of_shares": number_of_shares,
                                    "profit_amount": profit,
                                    "realized_amount": realized_amount,
                                    "profit_lose_percent": profit_percent,
                                    "exit_type": "last_trade_day",
                                    "investment_amount": investment_amount,
                                    "monitor_duration": monitor_duration,
                                    "trade_duration": sell_date - buy_date
                                    }
                        
                        else:
                            continue          
        print(f"No trigger in {current_year}")
    def backtest(self, data: pd.DataFrame = None, 
                 investment_amount: int = 100, 
                profit_rate=None,
                stop_loss=None, 
                accumulated_investment: bool = False,
                monitor_duration=None,
                ):
        if data is None:
            if hasattr(self, "data"):
                data = self.data
            else:
                data = self.download_data(ticker=self.ticker)
        
        data = self.create_date_columns(data=data)
        years = data.sort_values(by="year")["year"].unique()
        if not accumulated_investment:
            backtested_results = [self.place_order(data=data, current_year=yr, 
                                                    investment_amount=investment_amount,
                                                    profit_rate=profit_rate,
                                                    stop_loss=stop_loss,
                                                    monitor_duration=monitor_duration,
                                                    ) 
                                    for yr in years[1:]
                                ]
        else:
            realized_amount = investment_amount
            backtested_results = []
            for yr in years[1:]:
                res = self.place_order(data=data, 
                                        current_year=yr,
                                        investment_amount=realized_amount,
                                        profit_rate=profit_rate,
                                        stop_loss=stop_loss, 
                                        monitor_duration=monitor_duration,
                                        )
                if isinstance(res, dict):
                    realized_amount = res["realized_amount"]
                    res["accumulated_investment"] = True
                backtested_results.append(res)
        return backtested_results

#%%
ticker = "TSLA"

stra_tester = YearOnYearStrategy(ticker=ticker)    
   
   
#%% TODO: tune duration of investment. change from 1 year to 18 months
# chnage duration of investment to 1 year from buying date
backtested_results = stra_tester.backtest(accumulated_investment=True, 
                                          profit_rate=None, 
                                          stop_loss=None,
                                          monitor_duration=18
                                          ) 
backtested_results  

#%%

#stra_tester.data["Date"] + DateOffset(months=18)
# for TSLA; NVDA, KO increasing the duration of investment eliminates some losses

#%%
ticker = "IFX.DE"

stra_tester = YearOnYearStrategy(ticker=ticker)
backtested_results = stra_tester.backtest(profit_rate=None, stop_loss=None,
                                          accumulated_investment=True, 
                                          monitor_duration=18
                                          ) 
print(ticker)
backtested_results 


#%%  #  AT&T  -- tuning duration can eliminate some losses
ticker = "T"

stra_tester = YearOnYearStrategy(ticker=ticker)
backtested_results = stra_tester.backtest(profit_rate=None, stop_loss=None,
                                          accumulated_investment=True, 
                                          monitor_duration=18
                                          ) 
print(ticker)
backtested_results 


#%%
#%%  #  verizon communication
ticker = "VZ"

stra_tester = YearOnYearStrategy(ticker=ticker)
backtested_results = stra_tester.backtest(profit_rate=None, 
                                          stop_loss=None,
                                          accumulated_investment=True, 
                                          monitor_duration=18
                                          ) 
print(ticker)
backtested_results 


#%%  #  META
ticker = "INTC"
# schaeffler
ticker = "SHA.DE"

ticker = "LHA.DE"
stra_tester = YearOnYearStrategy(ticker=ticker)
backtested_results = stra_tester.backtest(profit_rate=None, 
                                          stop_loss=30,
                                          accumulated_investment=True, 
                                          monitor_duration=18
                                          ) 
print(ticker)
backtested_results 

#%%

#stra_tester.data.index[0] - stra_tester.data.index[1]
#%%    
total_invested = 0
realized_amount = 0
investment_amount = 100
for res in backtested_results:
    if isinstance(res, dict):
        total_invested += investment_amount
        realized_amt = res["realized_amount"]
        realized_amount += realized_amt

[print(res) for res in backtested_results]

        
print(f"Total invested: {total_invested}")
print(f"Total realized amount: {realized_amount}")                            


#%%

((realized_amount/total_invested)) * 100


#%% 
"""

Rio tINCO -- set stop_loss to 20% and profit rate to None to get 11% profit 
on back test is the best.

Novo Nodick made 3% profit on default back test

PUMA made 8% profit on default back test and shows trend of loss a year before profit

Addeso made 15% loss on default back test and shows trend of losses in recent years.
It has the potential of lossing as much as 70% on a trade. Setting stop_loss of 10%
creates 21% profit on back test.


Siement makes 3.6% profit on default back test and 3.9% profit on 
on stop_loss of 30%

Allianz is unprofitable with default back test 
"""

#
#%%
"""
Implemented strategy 
1. previous year min for buy and sell at 75% percentile + 50% of diff b/t 75% and min

1. get the descriptive statistics of the immediate previous year of year to invest in
This is based on the assumption that near years are more related and previous lows and highs 
are likely to repeat immediately

"""


#%%

for index, r in scaeffler_df.iterrows():
    print(index)
    break


#%%

backtested_results = [place_order(data=scaeffler_df, current_year=yr, 
                                  investment_amount=1000,
                                  profit_rate=5,
                                  stop_loss=None
                                  ) 
                      for yr in range(2016, 2024)
                      ]

total_invested = 0
realized_amount = 0
for res in backtested_results:
    if isinstance(res, dict):
        total_invested += 1000
        realized_amt = res["realized_amount"]
        realized_amount += realized_amt

print(f"total_invested: {total_invested}")
print(f"realized_amount: {realized_amount}")
(realized_amount/total_invested)*100


[print(res) for res in backtested_results]
#%%
"""
The backtest results shows the best return is achieved by 
setting profit rate of 50% and stop loss of 30%. This 
produces 15.6% profit after backtest period. The only time 
50%+ profit was hit only once in an investment round and all 
other trades exited on the last trade day of the year

A lost of 30.4% was incurred once during the trading period for a year.
least profit was 5.9% 

2 years in the trading period did not trigger a trade


####  using 5% profit rate with 30% stop_loss or no stop loss
100% win rate is achieved with this but with only 6% profit

No trade went into last day of trading. There was no trigger in 2 years 
of the backtest periods

"""

#%% TODO
"""
Determine where and in which months the lowest price is found and highest 
and use that to improve the trading strategy
"""


# %%  TODO: time series forecasting and decompostion of stocks<



#%%
def decompose_timeseries(data, variable_to_decompose,
                         plot_width = 15, plot_height = 15,
                         period=1):
    decomposition = seasonal_decompose(data[[variable_to_decompose]], period=period)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig = plt.figure(figsize=(plot_width,plot_height))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    ax1.set_title(f'Original Time Series - {variable_to_decompose}')
    ax1.plot(data[variable_to_decompose])
    ax2.set_title('Trend')
    ax2.plot(trend)
    ax3.set_title('Seasonality')
    ax3.plot(seasonal)
    ax4.set_title('Residuals')
    ax4.plot(residual)
    plt.tight_layout()
    return plt.show()


#%%

scaeffler_df = stra_tester.data#.index#.info()

scaeffler_df.index = pd.to_datetime(scaeffler_df.index)

scaeffler_df.set_index(scaeffler_df["Date"], inplace=True)
#%%
decompose_timeseries(data=stra_tester.data, variable_to_decompose='Close',
                     period=5)


#%%

model = Prophet(seasonality_mode="multiplicative")

#%%
input_data = (scaeffler_df.rename(columns={"Date": "ds", "Close": "y"})
 [["ds", "y", "Volume", "day"]])

#%%

model.add_regressor(name="Volume")
model.fit(input_data)

#%%  validate model
model_cross_validate = cross_validation(model, horizon=30,
                                        parallel="processes",
                                        period=90
                                        )


#%%

perfmetric = performance_metrics(df=model_cross_validate, monthly=True)

#%%
plot_cross_validation_metric(model_cross_validate, metric="rmse")


#%%
#%%

param_grid_m = {"changepoint_prior_scale": [0.5, 0.1, 0.01, 0.001],
                "seasonality_prior_scale": [10.0, 1, 0.1, 0.01],
                "seasonality_mode": ["additive", "multiplicative"]
                }

all_params_m = [dict(zip(param_grid_m.keys(), value))
                for value in itertools.product(*param_grid_m.values())
                ]

#%%
rmse_values_m = []
for params in all_params_m:
    model = Prophet(**params)
    model.add_regressor(name="Volume")
    model.fit(input_data)
    df_cv = cross_validation(model, horizon="30 days")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmse_values_m.append(df_p["rmse"].values[0])
    

    
#%%   #####################                   ###################

dumy_model = Prophet()

dumy_model.fit(df=input_data)

#%%
future = dumy_model.make_future_dataframe(periods=1)

forecast = dumy_model.predict(future)

#%%

fig = dumy_model.plot(forecast, xlabel="Date", ylabel="stock price")
plt.title("Schaeffler stock price prediction")
plt.show()


#%% plot forecast components
fig2 = dumy_model.plot_components(forecast)
plt.show()

#%%

fig2
#%%

forecast.yearly

#%%
from prophet.plot import get_seasonality_plotly_props

#%%

yr_seasonality = get_seasonality_plotly_props(dumy_model, name="yearly")


#%%

scatter = yr_seasonality["traces"][0]
yval = scatter.y
xval = scatter.x

#%%

px.line(x=xval, y=yval)

#%%
input_data.ds.unique()

#%%

scae2017 = scaeffler_df[scaeffler_df["year"]==2017]

px.line(data_frame=scae2017, y="Close")

#%%
forecast["yearly"]

px.line(data_frame=forecast, y="yearly", template="plotly_dark")
#%%

scdf = scaeffler_df.groupby(["year", "month_name"])["Close"].agg(["min", "max"])

#%%
scdf.reset_index(inplace=True)#.columns#[scdf["month_name"].isin(["October", "January"])]



#%% # results shows potential of buying in october and selling in january to make profit
scdf[scdf["month_name"].isin(["October", "January"])]#["Close"].agg(["min", "max"])

#%%  approach to analysis 
"""
find seasonal plot and use it to determine key date trading

develop a trading system that
1. takes data and transform it to various components
2. Plot that components as graphs and seen it to a deep learning model
3. the model then describe an insight and recommend a trading strategy

"""

#%%

ticker = "VWSB.DE"
ticker = "RWE.DE"
ticker = "MSFT"
ticker = "AAG.HM"
ticker = "NVDA"
ticker = "LHA.DE"
#ticker = "SHA.DE"
vestas = YearOnYearStrategy(ticker=ticker)    
   
backtested_results = vestas.backtest(profit_rate=None, 
                                          stop_loss=30,
                                          accumulated_investment=True, 
                                          monitor_duration=18
                                          ) 
print(ticker)
backtested_results 
#%%

vestas_model = Prophet()
vestas_input_data = (vestas.data.rename(columns={"Date": "ds", "Close": "y"})
 [["ds", "y", "Volume", "day"]])
vestas_model.fit(df=vestas_input_data)
vestas_future = vestas_model.make_future_dataframe(periods=1)

vestas_forecast = vestas_model.predict(vestas_future)
fig = vestas_model.plot(vestas_forecast, xlabel="Date", ylabel="stock price")
plt.title(f"{ticker} stock price prediction")
plt.show()


#%% plot forecast components
fig2 = vestas_model.plot_components(vestas_forecast)
plt.show()

#%%

df_dp = scaeffler_df[["Close", "Volume", "month", "day", "weekday"]]

test_df = df_dp.tail(90)

train_df = df_dp.drop(test_df.index)

columns_to_scale = ["Volume"]

minmax_scaler = preprocessing.MinMaxScaler()

minmax_scaler.fit(train_df[columns_to_scale])

train_df[columns_to_scale] = minmax_scaler.transform(train_df[columns_to_scale])

test_df[columns_to_scale] = minmax_scaler.transform(test_df[columns_to_scale])


#%%
def horizon_style_data_splitter(predictors: pd.DataFrame,
                                target: pd.DataFrame, start: int,
                                end: int, window: int, horizon: int
                                ):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(predictors) - horizon

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(predictors.iloc[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target.iloc[indicey])
    return np.array(X), np.array(y)


dataX = train_df[train_df.columns[1:]]

dataY = train_df[['Close']]

hist_window_multi = 180
horizon_multi = 90
TRAIN_SPLIT = 4930

x_train_multi, y_train_multi = horizon_style_data_splitter(predictors=dataX, target=dataY,
                                                         start=0, end=TRAIN_SPLIT,
                                                         window=hist_window_multi,
                                                         horizon=horizon_multi
                                                         )

x_val_multi, y_val_multi = horizon_style_data_splitter(predictors=dataX, target=dataY,
                                                     start=TRAIN_SPLIT, end=None,
                                                     window=hist_window_multi,
                                                     horizon=horizon_multi
                                                     )

#%%
BATCH_SIZE = 64
BUFFER_SIZE = 100


train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%%
from sklearn.metrics import mean_absolute_percentage_error

def timeseries_evaluation_metrics(y_true, y_pred):
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')

#%%
def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.rcParams['figure.figsize'] = [16, 9]
    return plt.show()

#%%
tf.random.set_seed(2023)
np.random.seed(2023)

lstm_multi = tf.keras.models.Sequential()
lstm_multi.add(tf.keras.layers.LSTM(units=556, input_shape=x_train_multi.shape[-2:],
                                    return_sequences=True)
               )
lstm_multi.add(tf.keras.layers.LSTM(units=556, return_sequences=False))
lstm_multi.add(tf.keras.layers.Dense(256))
lstm_multi.add(tf.keras.layers.Dense(256))
lstm_multi.add(tf.keras.layers.Dense(units=horizon_multi))
lstm_multi.compile(optimizer='adam', loss='mse')


#%%
model_path_lstm_multi = 'model_store/lstm_multivariate.h5'

EVALUATION_INTERVAL = 50
EPOCHS = 100
history_multi = lstm_multi.fit(x=train_data_multi, epochs=EPOCHS,
                         steps_per_epoch=EVALUATION_INTERVAL,
                         validation_data=val_data_multi,
                         validation_steps=5, verbose=1,
                         callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(model_path_lstm_multi,
                                                                       monitor='val_loss',
                                                                       save_best_only=True,
                                                                       mode="min",
                                                                       verbose=0
                                                                       )
                                    ]
                         )
#%%
trained_model_multi = tf.keras.models.load_model(model_path_lstm_multi)

plot_loss_history(history_multi)

#%%
data_val = train_df[train_df.columns[1:]].tail(hist_window_multi)
val_rescaled = np.array(data_val).reshape(1, data_val.shape[0], data_val.shape[1])
predicted_results = trained_model_multi.predict(val_rescaled)

#%%
timeseries_evaluation_metrics(y_true=test_df['Close'],
                                   y_pred=predicted_results[0]
                                   )

def compare_forecast_actual_graph(forecast, actual, title="Actual vs Predicted",
                                  yaxsis_label="page turn count",
                                  xaxsis_label="horizon (hourly)",
                                  legend: set = ('Actual', 'Predicted')):
    plt.plot(list(actual['Close']))
    plt.plot(list(forecast[0]))
    plt.title(title)
    plt.ylabel(yaxsis_label)
    plt.xlabel(xaxsis_label)
    plt.legend(legend)
    return plt.show()
#%%
compare_forecast_actual_graph(forecast=predicted_results, actual=test_df)


#%%
bi_lstm_multi = tf.keras.models.Sequential()
bi_lstm_multi.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=556,
                                            input_shape=x_train_multi.shape[-2:],
                                            return_sequences=True))
               )
bi_lstm_multi.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=556, return_sequences=False))
                  )
bi_lstm_multi.add(tf.keras.layers.Dense(256))
bi_lstm_multi.add(tf.keras.layers.Dense(256))
bi_lstm_multi.add(tf.keras.layers.Dense(units=horizon_multi))
bi_lstm_multi.compile(optimizer='adam', loss='mse')

model_path_bilstm_multi = 'model_store/bi_lstm_multivariate.h5'

bi_lstm_history_multi = bi_lstm_multi.fit(x=train_data_multi, epochs=EPOCHS,
                         steps_per_epoch=EVALUATION_INTERVAL,
                         validation_data=val_data_multi,
                         validation_steps=5, verbose=1,
                         callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(model_path_bilstm_multi,
                                                                       monitor='val_loss',
                                                                       save_best_only=True,
                                                                       mode="min",
                                                                       verbose=0
                                                                       )
                                    ]
                         )

#%%

trained_model_bilstm_multi = tf.keras.models.load_model(model_path_bilstm_multi)

plot_loss_history(bi_lstm_history_multi)

#%%
bilstm_predicted_results = trained_model_bilstm_multi.predict(val_rescaled)
timeseries_evaluation_metrics(y_true=test_df['Close'],
                                   y_pred=bilstm_predicted_results[0]
                                   )

#%%
compare_forecast_actual_graph(forecast=bilstm_predicted_results, actual=test_df)

#%%
cnn_model_multi = Sequential()
cnn_model_multi.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                                       input_shape=(x_train_multi.shape[1],
                                                    x_train_multi.shape[2]
                                                    )
                                       )
                                )
cnn_model_multi.add(MaxPool1D(pool_size=2))
cnn_model_multi.add(tf.keras.layers.Dense(556))
cnn_model_multi.add(Flatten())
cnn_model_multi.add(Dense(556, activation='relu'))
cnn_model_multi.add(tf.keras.layers.Dense(256))
cnn_model_multi.add(tf.keras.layers.Dense(256))
cnn_model_multi.add(Dense(horizon_multi))
cnn_model_multi.compile(optimizer='adam', loss='mse')

model_path_cnn_multi = 'model_store/cnn_multivariate.h5'


cnn_history_multi = cnn_model_multi.fit(x=train_data_multi, epochs=EPOCHS,
                         steps_per_epoch=EVALUATION_INTERVAL,
                         validation_data=val_data_multi,
                         validation_steps=5, verbose=1,
                         callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(model_path_cnn_multi,
                                                                       monitor='val_loss',
                                                                       save_best_only=True,
                                                                       mode="min",
                                                                       verbose=0
                                                                       )
                                    ]
                         )

#%%
trained_model_cnn_multi = tf.keras.models.load_model(model_path_cnn_multi)

plot_loss_history(cnn_history_multi)

#%%
cnn_predicted_results = trained_model_cnn_multi.predict(val_rescaled)

timeseries_evaluation_metrics(y_true=test_df['Close'],
                                y_pred=cnn_predicted_results[0]
                            )

compare_forecast_actual_graph(forecast=cnn_predicted_results, 
                              actual=test_df,
                              yaxsis_label="Stock close price",
                              xaxsis_label="trading days")



#%%
px.line(data_frame=scaeffler_df, 
        y="Close",
        template="plotly_dark"
        )

#%%                
    
    # cal buy_price, exit and stop loss 

# adaptiopn to algo

# take 5% profit for each trade after 
# buying below or above previous year minimum
# after taking profit, check if 
# start checking afterwards again if 
# clossing price is about 50% below intial entried price 
# and enter trade again


## adaptation 2 
# adjust stop loss

#%%

scaeffler_df[scaeffler_df.year==2018]["Close"].is_monotonic_increasing
#%%
scaeffler_df["Close"]
#%%

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'num_posts': [4, 6, 3, 9, 1, 14, 2, 5, 7, 2],
    'date': ['2020-08-09', '2020-08-25', '2020-09-05', '2020-09-12', '2020-09-29', '2020-10-15', '2020-11-21', '2020-12-02', '2020-12-10', '2020-12-18']
})

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter rows with dates above a given date
given_date = '2020-09-15'
filtered_df_above = df.loc[df['date'] > given_date]

# Filter rows with dates below a given date
filtered_df_below = df.loc[df['date'] < given_date]

print("Dates above given date:\n", filtered_df_above)
print("Dates below given date:\n", filtered_df_below)

#%%
df.info()

#%%

scaeffler_df[scaeffler_df["Date"] > scae2023_minrow.Date.values[0]]

#%%

scaeffler_df[(scaeffler_df["year"]==scae2023_maxrow.year.values[0]) & (scaeffler_df["Date"] <= scae2023_maxrow.Date.values[0])]
#%% Is there a signle month that that explains well what will happen the next year?



    
    
#%%

px.line(data_frame=scaeffler_df, y="Close", facet_row="year",
        template="plotly_dark"
        )


#%%
px.line(data_frame=scaeffler_df, y="Close", facet_col_wrap="year",
        template="plotly_dark"
        )
#%%
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

#%%

import pandas as pd
from pandas.tseries.offsets import DateOffset

# Sample DataFrame
data = {'date_column': ['2023-01-01', '2023-02-15', '2023-03-20']}
df = pd.DataFrame(data)
df['date_column'] = pd.to_datetime(df['date_column'])

# Add months
months_to_add = 3  # Change this value to the number of months you want to add
df['new_date_column'] = df['date_column'] + DateOffset(months=months_to_add)

# View the updated DataFrame
print(df)



