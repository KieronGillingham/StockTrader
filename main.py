# General
import sys, datetime
from typing import List

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, QComboBox
from PyQt5.QtCore import QTimer, QThreadPool

# Threading
from stockthreading import Worker

# Yahoo Finance
from yahoofinancials import YahooFinancials

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

stocks = [("TYT.L", "Toyota"), ("ULVR.L","Unilever PLC"), ("BP-A.L","BP p.l.c.")]

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

        # Counter for debugging
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

        wid = QComboBox()
        wid.addItems(x[1] for x in stocks)
        self.vbox_chartmenu.addWidget(wid)

        wid = QPushButton("Reload from Yahoo Finance")
        wid.pressed.connect(self.reloadData)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Reload from file")
        wid.pressed.connect(self.reload_from_file)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Clear Chart")
        wid.pressed.connect(self.clear_chart)
        self.vbox_data.addWidget(wid)

        wid = QLabel("Invest Amount (Stocks):")
        self.vbox_prediction.addWidget(wid)
        self.stock_invested = QSpinBox()
        self.vbox_prediction.addWidget(self.stock_invested)

        wid = QPushButton("Predict Profit")
        wid.pressed.connect(self.calculate)
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

        self.reload_from_file()

        # Display the main window
        self.show()

    def clear_chart(self):
        # Clear existing chart
        self.mainChart.axes.cla()
        self.mainChart.draw()

    # def recurring_timer(self):
    #     """Increment counter"""
    #
    #     self.counter += 1
    #     self.counter_label.setText("Counter: %d" % self.counter)

    def progress_fn(self, n):
        print("%d%% done" % n)

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

    def load_from_yahoo_finance(self, progress_callback):
        self.clear_chart()

        companies = read_csv("data/stocksymbols.csv", header=0)#, quotechar='"')

        labels = companies['Symbol'].tolist()

        print(labels)

        return "Done."

        yahoo_financials = YahooFinancials(labels)
        data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
                                                          end_date='2019-12-31',
                                                          time_interval='monthly')
        self.prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in labels
        })

        self.prices_df.to_csv("data/localstorage.csv")

        # Draw new chart
        self.draw_chart(self.prices_df)

        return "Done."

    def load_from_csv(self, progress_callback, localfile="data/localstorage.csv"):
        print("Loading from local file")
        try:
            self.prices_df = read_csv(localfile, index_col=0)
        except FileNotFoundError:
            print("Local storage file '%s' not found" % localfile)
            return

        predictions = []

        x = self.prices_df.index.values
        for i in range(0, len(x)):
            x[i] = datetime.date.fromisoformat(x[i]).toordinal()

        x = x.reshape(-1,1)
        for i in range(0, len(self.prices_df.columns)):

            y = self.prices_df.iloc[:, i].values
            model = LinearRegression()
            model.fit(x, y)
            prediction = model.predict([[737424]])
            predictions.append(prediction[0])

        pred_df = pd.DataFrame(np.reshape(predictions, (1,-1)), columns=self.prices_df.columns, index=[737424])
        pred_df = pred_df.append(self.prices_df.tail(1))

        # Draw new chart
        self.draw_chart(self.prices_df, pred_df)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def reloadData(self):
        # Pass the function to execute
        worker = Worker(self.load_from_yahoo_finance) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def reload_from_file(self):
        # Pass the function to execute
        worker = Worker(self.load_from_csv)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.thread_complete)

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


# Create application.
app = QApplication(sys.argv) # sys.argv are commandline arguments passed in when the program runs.

# Create and display the main window.
window = MainWindow()

# Start the main program loop.
print("Program starting.")
app.exec_()

# Program terminating.
print("Program termininating.")