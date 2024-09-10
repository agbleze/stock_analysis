import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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
