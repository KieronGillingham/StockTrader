# Logging
import logging
_logger = logging.getLogger(__name__)

# Threading
from PyQt5.QtCore import QThreadPool
from stockthreading import Worker

# Yahoo Finance
from yahoofinancials import YahooFinancials

# Data manipulation
import pandas as pd
from pandas import read_csv

class StockData:
    """
    Class for accessing stock data from YahooFinance or a local CSV file.
    Threading is used to access data asynchronously.
    """
    def __init__(self, csv_path="data/localstorage.csv", symbols_file="data/stocksymbols.csv", *args, **kwargs):
        # Threading
        self.threadpool = QThreadPool()
        _logger.debug(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads.")

        # Attempt to load data from csv by default
        try:
            self.load_from_csv(csv_path=csv_path, symbols_file=symbols_file)
        except Exception as ex:
            _logger.error(ex)

    def get_symbol(self, name):
        return self.stockdict[name]

    def load_stocks(self, csv_path="data/stocksymbols.csv"):
        stocksymbols = read_csv(csv_path, header=0)
        self.stockdict = {}
        for row in stocksymbols.values:
            self.stockdict[row[1]] = row[0]

        self.stocknames = list(self.stockdict.keys())[:5]  # First 5 only

    def load_data_from_yahoo_finance(self, progress_callback, start_date=None, end_date=None, stocksymbols=None, time_interval='monthly'):
        _logger.debug("Load data from yahoo finance")
        if stocksymbols == None:
            stocks = self.stockdict
            self.stocknames = list(stocks.keys())[:5] # First 5 only
            stocksymbols = list(stocks.values())[:5]

        _logger.debug("Stockdict =", stocks)
        _logger.debug("Selected Stocks =", self.stocknames, stocksymbols)
        _logger.debug(start_date, end_date, time_interval)

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

    def load_from_csv(self, progress_callback=None, csv_path="data/localstorage.csv", symbols_file="data/stocksymbols.csv"):
        _logger.debug(f"Loading from local file {csv_path}")
        try:
            self.prices_df = read_csv(csv_path, index_col=0)
            self.describe_data()
        except FileNotFoundError as ex:
            _logger.warning(f"Local storage file '{csv_path}' not found: {ex}")
        except Exception as ex:
            _logger.error(ex)

        try:
            self.load_stocks(symbols_file)
        except FileNotFoundError as ex:
            _logger.error(f"Local storage file '{symbols_file}' not found: {ex}")
        except Exception as ex:
            _logger.error(ex)
    # def thread_complete(self):
    #     print("THREAD COMPLETE!")

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

        _logger.debug(self.prices_df[stock][737364])

        buy_cost = self.prices_df[stock][737364] * stock_count
        sell_cost = self.prices_df[stock][737394] * stock_count
        profit = sell_cost - buy_cost

        _logger.debug(buy_cost, sell_cost, profit)

    def save_to_csv(self, csv_path="data/localstorage.csv"):
        self.prices_df.to_csv(csv_path)

    def describe_data(self):
        if self.prices_df is not None:
            _logger.info(self.prices_df.describe())