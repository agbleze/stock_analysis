
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
ticker = "SHA"

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
                    stop_loss=None
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
    def backtest(self, data: pd.DataFrame = None, 
                 investment_amount: int = 100, 
                profit_rate=None,
                stop_loss=None
                ):
        if data is None:
            if hasattr(self, "data"):
                data = self.data
            else:
                data = self.download_data(ticker=self.ticker)
        
        data = self.create_date_columns(data=data)
        years = data.sort_values(by="year")["year"].unique()
        backtested_results = [self.place_order(data=data, current_year=yr, 
                                                investment_amount=investment_amount,
                                                profit_rate=profit_rate,
                                                stop_loss=stop_loss
                                                ) 
                                for yr in years[1:]
                            ]
        return backtested_results

#%%
ticker = "TSLA"

stra_tester = YearOnYearStrategy(ticker=ticker)    
   
   
#%% TODO: tune duration of investment. change from 1 year to 18 months
# chnage duration of investment to 1 year from buying date
backtested_results = stra_tester.backtest() 
backtested_results  

# for TSLA; NVDA, KO increasing the duration of investment eliminates some losses

#%%
ticker = "KO"

stra_tester = YearOnYearStrategy(ticker=ticker)
backtested_results = stra_tester.backtest(profit_rate=None) 
print(ticker)
backtested_results 
 
#%%    
total_invested = 0
realized_amount = 0
for res in backtested_results:
    if isinstance(res, dict):
        total_invested += investment_amount
        realized_amt = res["realized_amount"]
        realized_amount += realized_amt

[print(res) for res in backtested_results]

        
                            


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


#%%
px.line(data_frame=scaeffler_df, 
        y="Close",
        template="plotly_dark"
        )

#%%

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
scaeffler_df["Close"].mon
#%%
(5.82/7.05)*100

#%%
(70/100)*11.355
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