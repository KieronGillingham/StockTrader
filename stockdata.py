# Threading
from PyQt5.QtCore import QThreadPool
from stockthreading import Worker

# Yahoo Finance
from yahoofinancials import YahooFinancials

# Data manipulation
import pandas as pd
from pandas import read_csv

# Date
from datetime import date




class StockData:
    def __init__(self, *args, **kwargs):
        # Threadpool
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.load_from_csv()

    # def recurring_timer(self):
    #     """Increment counter"""
    #
    #     self.counter += 1
    #     self.counter_label.setText("Counter: %d" % self.counter)

    def get_symbol(self, name):
        return self.stocksymbols[name]

    def progress_fn(self, n):
        print("%d%% done" % n)

    def load_stocks(self):
        stocksymbols = read_csv("data/stocksymbols.csv", header=0)
        self.stockdict = {}
        for row in stocksymbols.values:
            self.stockdict[row[1]] = row[0]
        return self.stockdict

    def load_data_from_yahoo_finance(self, progress_callback, start_date=None, end_date=None, stocksymbols=None, time_interval='monthly'):
        print("Load data from yahoo finance")
        if stocksymbols == None:
            stocks = self.load_stocks()
            stocknames = list(stocks.keys())[:5] # First 5 only
            stocksymbols = list(stocks.values())[:5]

        print("Stockdict =", stocks)
        print("Selected Stocks =", stocknames, stocksymbols)
        print(start_date, end_date, time_interval)

        yahoo_financials = YahooFinancials(stocksymbols)
        data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                          end_date=end_date,
                                                          time_interval=time_interval)

        self.prices_df = None
        for symbol in stocksymbols:
            stock_data = data[symbol] # Get stock data using the symbol for that stock
            prices = stock_data['prices'] # Get price data for given stock

            # Format price dict into dataframe with column headings: '[symbol]_[field]'
            df = pd.DataFrame.from_records(prices, index='date', exclude=['formatted_date'])
            df.columns = (symbol + "_" + field for field in df.columns)

            # Create or join to master dataframe
            if self.prices_df is None:
                self.prices_df = df
            else:
                self.prices_df = pd.concat([self.prices_df, df], axis=1)

        return "Done."

    def get_yahoo_finance_data(self, start_date=None, end_date=None, time_interval='monthly', stocksymbols=None, on_finish=None):
        # Pass the function to execute
        worker = Worker(self.load_data_from_yahoo_finance, start_date=start_date, end_date=end_date, time_interval=time_interval, stocksymbols=stocksymbols) # Any other args, kwargs are passed to the run function

        # Connect on_finish method to signal
        if on_finish is not None:
            worker.signals.finished.connect(on_finish)
        
        #worker.signals.result.connect(self.print_output)
        #worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def load_from_csv(self, progress_callback=None, localfile="data/localstorage.csv"):
        print("Loading from local file")
        try:
            self.prices_df = read_csv(localfile, index_col=0)
        except FileNotFoundError:
            print("Local storage file '%s' not found" % localfile)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    # def reloadData(self):
    #     # Pass the function to execute
    #     worker = Worker(self.load_from_yahoo_finance) # Any other args, kwargs are passed to the run function
    #     worker.signals.result.connect(self.print_output)
    #     worker.signals.finished.connect(self.data_loaded)
    #     worker.signals.progress.connect(self.progress_fn)
    #
    #     # Execute
    #     self.threadpool.start(worker)

    def calculate(self, stock=None, stock_count=0):
        stock = "TYT.L"
        stock_count = self.stock_invested.value()

        print(self.prices_df[stock][737364])

        buy_cost = self.prices_df[stock][737364] * stock_count
        sell_cost = self.prices_df[stock][737394] * stock_count
        profit = sell_cost - buy_cost

        print(buy_cost, sell_cost, profit)

    def save_to_csv(self):
        self.prices_df.to_csv("data/localstorage.csv")