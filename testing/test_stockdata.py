import logging
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])
_logger = logging.getLogger(__name__)

from unittest import TestCase

from stockdata import StockData

import pandas as pd

import os.path as path

class TestStockData(TestCase):

    def setUp(self):
        _logger.info("Setting up test.")

        self.test_tradedata_file = path.abspath("test_tradedata.csv")
        self.test_symbols_file = path.abspath("test_symbols.csv")

        self.test_tradedata = pd.DataFrame([[0,1,2,3,4,5], [0,2,4,6,8,10], [0,1,4,9,16,25]])
        self.test_tradedata.to_csv(self.test_tradedata_file)

        test_symbolsdata = []
        for n in range(1, 10):
            test_symbolsdata.append([f"STK.{n}", f"Stock{n}"])
        for n in range(1, 10):
            test_symbolsdata.append([f"EQT.{n}", f"Equity{n}"])

        test_symbols = pd.DataFrame(test_symbolsdata, columns=["Symbol", "Name"])
        test_symbols.set_index("Symbol", inplace=True)
        test_symbols.to_csv(self.test_symbols_file)

        _logger.debug(f"Test data: {self.test_tradedata_file}")
        _logger.debug(f"Test symbols: {self.test_symbols_file}")

        self.stock_data = StockData(csv_path=self.test_tradedata_file, symbols_file=self.test_symbols_file)


    def test_get_symbol(self):
        _logger.info("Testing get_symbol()")
        self.stock_data.load_stocks(csv_path=self.test_symbols_file)

        for n in range(1,10):
            self.assertEqual(self.stock_data.get_symbol(f"Stock{n}"), f"STK.{n}")
            self.assertEqual(self.stock_data.get_symbol(f"Equity{n}"), f"EQT.{n}")

    def test_load_stocks(self):
        _logger.info("Testing load_stocks()")
        self.stock_data.stockdict = None
        self.assertEqual(self.stock_data.stockdict, None)

        self.stock_data.load_stocks(csv_path=self.test_symbols_file)
        self.assertNotEqual(self.stock_data.stockdict, None)

    def test_load_data_from_yahoo_finance(self):
        self.fail()

    def test_get_yahoo_finance_data(self):
        self.fail()

    def test_load_from_csv(self):
        self.fail()

    def test_calculate(self):
        self.fail()

    def test_get_stocknames(self):
        self.fail()

    def test_save_to_csv(self):
        self.fail()
