# Logging
import logging

from sklearn.impute import SimpleImputer

_logger = logging.getLogger(__name__)

# Yahoo Finance
from yahoofinancials import YahooFinancials

# Data manipulation
import pandas as pd
from pandas import read_csv
import numpy as np

from matplotlib import pyplot as plt

class StockData:
    """
    Class for accessing stock data from YahooFinance or a local CSV file.
    """
    def __init__(self, csv_path="data/localstorage.csv", symbols_file="data/stocksymbols.csv", *args, **kwargs):
        """
        Initalise a StockData instance that can be used to access stock data from YahooFinance or a local CSV file.
        :param csv_path: Path to a local CSV file containing stock price data to load by default.
        :param symbols_file: Path to a CSV file containing stock names and symbols.
        """
        try:
            # Attempt to load data from csv by default
            self.load_from_csv(csv_path=csv_path, symbols_file=symbols_file)
            self.csv_path = csv_path
            self.symbols_file = symbols_file
        except FileNotFoundError as ex:
            _logger.error(f"Local storage file '{csv_path}' not found: {ex}")
        except Exception as ex:
            _logger.error(ex)

    def load_from_csv(self, progress_callback=None, csv_path="data/localstorage.csv", symbols_file="data/stocksymbols.csv"):
        _logger.debug(f"Loading stock data from local file {csv_path}.")

        # Load the symbols
        self.load_symbols(symbols_file)

        try:
            self.data = read_csv(csv_path, index_col=0)
            self.fill_missing_data()
            self.describe_data(True, False)
            # Convert all columns to weekly rolling averages
            # self.prices_df = self.prices_df.rolling(7).mean()
        except FileNotFoundError as ex:
            _logger.warning(f"Local storage file '{csv_path}' not found: {ex}")
        except Exception as ex:
            _logger.error(ex)

    def save_to_csv(self, csv_path="data/localstorage.csv"):
        try:
            self.data.to_csv(csv_path)
        except Exception as ex:
            _logger.error(ex)

    def get_symbol(self, name):
        """
        Get the symbol for a named stock.
        :param name: The name of an available stock.
        :return: The symbol for the named stock.
        """
        return self.stockdict[name]

    def load_symbols(self, symbol_csv="data/stocksymbols.csv"):
        try:
            stocksymbols = read_csv(symbol_csv, header=0)
            self.stockdict = {row[1]: row[0] for row in stocksymbols.values}
            _logger.debug(f"Loaded symbols: {self.stockdict}")
        except Exception as ex:
            _logger.error(ex)
        except FileNotFoundError as ex:
            _logger.error(f"Symbol file '{symbol_csv}' not found: {ex}")

        self.stocknames = list(self.stockdict.keys())

    def load_data_from_yahoo_finance(self, progress_callback=None, start_date=None, end_date=None, stocksymbols=None, time_interval='monthly'):
        _logger.debug("Load data from yahoo finance")
        if stocksymbols == None:
            stocks = self.stockdict
            self.stocknames = list(stocks.keys())
            stocksymbols = list(stocks.values())

        _logger.debug(f"Stockdict: {stocks}")
        _logger.debug(f"Selected Stocks: {self.stocknames}, {stocksymbols}")
        _logger.debug(f"{start_date}, {end_date}, {time_interval}")

        self.data = None

        for i in range(0,30):
            try:
                symbol = stocksymbols[i]
                print(symbol)
                yahoo_financials = YahooFinancials(symbol)
                data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                                  end_date=end_date,
                                                                  time_interval=time_interval)
                print(data)
                stock_data = data[symbol] # Get stock data using the symbol for that stock
                prices = stock_data['prices'] # Get price data for given stock

                # Format price dict into dataframe with column headings: '[symbol]_[field]'
                df = pd.DataFrame.from_records(prices, index='date', exclude=['formatted_date'])
                df.columns = (symbol + "_" + field for field in df.columns)
                print(df)
                # Create or join to master dataframe
                if self.data is None:
                    self.data = df
                else:
                    self.data = pd.concat([self.data, df], axis=1)
            except Exception as ex:
                _logger.error(f"Exception on loop {i}: {ex}")

        print(self.data.shape)
        self.fill_missing_data()

        return "Done."

    def describe_data(self, display=False, save=False):
        if self.data is not None:
            data_stats = self.data.describe()
            _logger.info(data_stats)
            _logger.info(self.data.corr())
            if display:

                # close_prices = self.data.filter(like="_close")
                # corr_close_prices = close_prices.corr()
                #
                # corr_fig = plt.figure()
                # axes = corr_fig.add_subplot(111)
                # axcorr = axes.matshow(corr_close_prices, vmin=-1, vmax=1)
                # corr_fig.colorbar(axcorr)
                # headings = list(self.stockdict.values())
                # ticks = np.arange(0, 10, 1)
                # axes.set_xticks(ticks)
                # axes.set_yticks(ticks)
                # axes.set_xticklabels(headings[:10], rotation=90)
                # axes.set_yticklabels(headings[:10])
                #plt.show()

                #plt.scatter(data_stats.loc["mean"], data_stats.loc["std"])
                #plt.show()
                if save:
                    plt.savefig("data/fig.png")

    def fill_missing_data(self):
        empty_data = self.data[self.data.isna().any(axis=1)]
        missing_rows = len(empty_data.index)
        if missing_rows > 0:
            _logger.warning(f"{missing_rows} rows missing. Imputing substitute values.")
            imputer = SimpleImputer(strategy="mean")
            imputer.fit(self.data.values)
            self.data = pd.DataFrame(data=imputer.transform(self.data), columns=self.data.columns, index=self.data.index)
        else:
            _logger.debug("No data missing.")
