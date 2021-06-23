# Logging
import logging

from sklearn.impute import SimpleImputer

_logger = logging.getLogger(__name__)

# Yahoo Finance
from yahoofinancials import YahooFinancials

# Data manipulation
import pandas as pd
from pandas import read_csv

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

        yahoo_financials = YahooFinancials(stocksymbols[:20])
        data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                          end_date=end_date,
                                                          time_interval=time_interval)

        self.data = None
        for symbol in stocksymbols:
            stock_data = data[symbol] # Get stock data using the symbol for that stock
            prices = stock_data['prices'] # Get price data for given stock

            # Format price dict into dataframe with column headings: '[symbol]_[field]'
            df = pd.DataFrame.from_records(prices, index='date', exclude=['formatted_date'])
            df.columns = (symbol + "_" + field for field in df.columns)

            # Create or join to master dataframe
            if self.data is None:
                self.data = df
            else:
                self.data = pd.concat([self.data, df], axis=1)

        print(self.data.shape())
        self.fill_missing_data()
        return "Done."

    def describe_data(self):
        if self.data is not None:
            _logger.info(self.data.describe())

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
