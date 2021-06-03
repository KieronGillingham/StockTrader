# General
import sys
from datetime import date
from typing import List

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, QComboBox
from PyQt5.QtCore import QTimer, QThreadPool

# Threading
from learningmodel import LearningModel
from stockthreading import Worker

# Yahoo Finance
from stockdata import StockData

# Plotting
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

# Data manipulation and machine learning
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

stock_data = StockData()
learning_model = LearningModel()

class MplCanvas(FigureCanvasQTAgg):
    """PyQt canvas for MatPlotLib graphs"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    """ Main window of application"""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Set window appearance
        self.setWindowTitle("Intelligent Stock Trader")
        self.setMinimumSize(1024, 512)

        # Initalise layouts
        self.vbox_main = QVBoxLayout()
        self.hbox_title = QHBoxLayout()
        self.hbox_main = QHBoxLayout()
        self.vbox_sidebar = QVBoxLayout()
        self.vbox_chartmenu = QVBoxLayout()
        self.vbox_prediction = QVBoxLayout()
        self.vbox_data = QVBoxLayout()

        # # Counter for debugging
        # self.counter = 0
        # self.counter_label = QLabel()
        # self.vbox_main.addWidget(self.counter_label)
        # # Timer for counter
        # self.timer = QTimer()
        # self.timer.setInterval(1000)
        # self.timer.timeout.connect(self.recurring_timer)
        # self.timer.start()

        wid = QLabel("Stock Trader")
        self.hbox_title.addWidget(wid)

        self.filter_combobox = QComboBox()
        self.filter_combobox.currentIndexChanged.connect(self.filter_changed)
        self.vbox_chartmenu.addWidget(self.filter_combobox)

        wid = QPushButton("Reload from Yahoo Finance")
        wid.pressed.connect(self.load_data_from_yf)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Save data to file")
        wid.pressed.connect(self.save_data_to_file)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Load data from file")
        wid.pressed.connect(self.load_data_from_file)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Clear Chart")
        wid.pressed.connect(self.clear_chart)
        self.vbox_data.addWidget(wid)

        wid = QLabel("Invest Amount (Stocks):")
        self.vbox_prediction.addWidget(wid)
        self.stock_invested = QSpinBox()
        self.vbox_prediction.addWidget(self.stock_invested)

        wid = QPushButton("Predict Profit")
        wid.pressed.connect(self.make_prediction)
        self.vbox_prediction.addWidget(wid)

        wid = QLabel("")
        self.vbox_prediction.addWidget(wid)

        # Main chart
        self.mainChart = MplCanvas(self)
        self.hbox_main.addWidget(self.mainChart, 5)
        self.prices_df = None

        # Set layout hierarchy
        self.vbox_main.addLayout(self.hbox_title, 1)
        self.vbox_main.addLayout(self.hbox_main, 5)
        self.hbox_main.addLayout(self.vbox_sidebar, 1)
        self.vbox_sidebar.addLayout(self.vbox_chartmenu, 1)
        self.vbox_sidebar.addLayout(self.vbox_prediction, 2)
        self.vbox_sidebar.addLayout(self.vbox_data, 1)

        # Display widgets
        widget = QWidget()
        widget.setLayout(self.vbox_main)
        self.setCentralWidget(widget)

        # Treadpool
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Display the main window
        self.show()

    def clear_chart(self):
        # Clear existing chart
        self.mainChart.axes.cla()
        self.mainChart.draw()

    def draw_single_stock(self, stocksymbol):

        self.clear_chart()
        data = stock_data.prices_df.filter(like=stocksymbol)

        self.mainChart.axes.plot(data)
        self.mainChart.axes.legend(data.columns.tolist(), loc='upper left')

        self.mainChart.axes.grid(True)
        self.mainChart.draw()

        labels = self.mainChart.axes.get_xticklabels()
        for t in labels:
            val = t.get_position()[0]  # Get raw numerical value of the label
            tdate = date.fromtimestamp(val)
            t.set_text(tdate.strftime("%d/%m"))
        self.mainChart.axes.set_xticklabels(labels)
        self.mainChart.draw()

    def data_loaded(self):
        print("Data loaded")
        self.clear_chart()
        self.filter_combobox.clear()
        data = stock_data.prices_df
        for n in stock_data.get_stocknames():
            self.filter_combobox.addItem(n, userData=stock_data.get_symbol(n))

        learning_model.set_data(data)

        self.draw_single_stock("TYT.L")

    def load_data_from_yf(self):
        # Pass the function to execute
        stock_data.get_yahoo_finance_data(start_date='2020-12-01', end_date='2021-03-01', time_interval='daily', on_finish=self.data_loaded)  # Any other args, kwargs are passed to the run function

    def load_data_from_file(self):
        # Pass the function to execute
        worker = Worker(stock_data.load_from_csv)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.data_loaded)

        # Execute
        return self.threadpool.start(worker)

    def filter_changed(self, value):
        print(f"{self.filter_combobox.itemText(value)} ({self.filter_combobox.itemData(value)}) selected.")
        if self.filter_combobox.itemData(value) is not None:
            self.draw_single_stock(self.filter_combobox.itemData(value))

    def calculate(self):
        # TODO: Fix implementation
        pass
        # stock_data.calculate()

    # def recurring_timer(self):
    #     """Increment counter"""
    #
    #     self.counter += 1
    #     self.counter_label.setText("Counter: %d" % self.counter)

    def draw_chart(self, data: List, prediction: List = None):
        self.clear_chart()

        p = self.mainChart.axes.plot(data)
        self.mainChart.axes.legend(data.columns.tolist())

        if prediction is not None:
            if len(data.columns) != len(prediction.columns):
                raise Exception('Historical and prediction lists are not equal length.')

            for i in range (0, len(prediction.columns)):
                self.mainChart.axes.plot(prediction.iloc(axis=1)[i], linestyle='--', color=p[i].get_color())

        self.mainChart.draw()


    def save_data_to_file(self):
        stock_data.save_to_csv()

    def make_prediction(self):
        learning_model.predict()
        # self.draw_single_stock(self.filter_combobox.itemData())

# Create application.
app = QApplication(sys.argv) # sys.argv are commandline arguments passed in when the program runs.

# Create and display the main window.
window = MainWindow()

# Start the main program loop.
print("Program starting.")
app.exec_()

# Program terminating.
print("Program termininating.")