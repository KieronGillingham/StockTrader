# General
import sys, datetime

# Threading
from PyQt5.QtCore import QThreadPool
from stockthreading import Worker

# Yahoo Finance
from yahoofinancials import YahooFinancials

# Data manipulation
import pandas as pd
import numpy as np
from pandas import read_csv

class StockData():
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

    def progress_fn(self, n):
        print("%d%% done" % n)

    def load_stocks(self):
        companies = read_csv("data/stocksymbols.csv", header=0)
        return companies

    def load_data_from_yahoo_finance(self, progress_callback, start_date=None, end_date=None, stocksymbols=None, time_interval='monthly'):
        print("Load data from yahoo finance")
        if stocksymbols == None:
            stocks = self.load_stocks()
            stocksymbols = stocks["Symbol"][:5]

        print(stocksymbols)
        print(start_date, end_date, time_interval)

        yahoo_financials = YahooFinancials(stocksymbols)
        data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                          end_date=end_date,
                                                          time_interval=time_interval)
        self.prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in stocksymbols
        })
        self.prices_df.columns = stocks["Name"][:5]

        self.save_to_csv()

        return "Done."

    def get_yahoo_finance_data(self, start_date=None, end_date=None, time_interval='monthly', stocksymbols=None):
        # Pass the function to execute
        worker = Worker(self.load_data_from_yahoo_finance, start_date=start_date, end_date=end_date, time_interval=time_interval, stocksymbols=stocksymbols) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

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

    def reloadData(self):
        # Pass the function to execute
        worker = Worker(self.load_from_yahoo_finance) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.data_loaded)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

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