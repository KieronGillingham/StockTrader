# General
import sys, time, traceback

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt, QTimer, QRunnable, pyqtSlot, QThreadPool, pyqtSignal, QObject

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
from pandas import read_csv
from sklearn.linear_model import LinearRegression

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

# Encapsulate main window in a class
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Set window title
        self.setWindowTitle("Intelligent Stock Trader")
        self.setMinimumSize(1024, 512)

        # UI
        self.main_layout = QVBoxLayout()

        # Counter
        self.counter = 0
        self.label = QLabel()
        self.main_layout.addWidget(self.label)
        # Timer for counter
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        # Buttons
        buttons = QHBoxLayout()

        button = QPushButton("Reload from Yahoo Finance")
        button.pressed.connect(self.reloadData)
        buttons.addWidget(button)

        button = QPushButton("Reload from file")
        button.pressed.connect(self.reload_from_file)
        buttons.addWidget(button)

        button = QPushButton("Clear Chart")
        button.pressed.connect(self.clear_chart)
        buttons.addWidget(button)

        self.main_layout.addLayout(buttons)

        # Main chart
        self.mainChart = MplCanvas(self)
        self.main_layout.addWidget(self.mainChart)

        # Widget
        widget = QWidget()
        widget.setLayout(self.main_layout)
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

    def recurring_timer(self):
        self.counter += 1
        self.label.setText("Counter: %d" % self.counter)

    def progress_fn(self, n):
        print("%d%% done" % n)

    def load_from_yahoo_finance(self, progress_callback):
        self.clear_chart()

        companies = read_csv("data/stocksymbols.csv", header=0)

        labels = companies['Symbol'].head(10).tolist()

        yahoo_financials = YahooFinancials(labels)
        data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
                                                          end_date='2019-12-31',
                                                          time_interval='monthly')
        prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in labels
        })

        prices_df.to_csv("data/localstorage.csv")

        # Draw new chart
        self.mainChart.axes.plot(prices_df)
        self.mainChart.draw()

        return "Done."

    def load_from_csv(self, progress_callback):
        prices_df = read_csv("data/localstorage.csv", index_col=0)

        print(prices_df)

        # Draw new chart
        self.mainChart.axes.plot(prices_df)
        self.mainChart.draw()

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

# Create application.
app = QApplication(sys.argv) # sys.argv are commandline arguments passed in when the program runs.

# Create and display the main window.
window = MainWindow()

# Start the main program loop.
print("Program starting.")
app.exec_()

# Program terminating.
print("Program termininating.")