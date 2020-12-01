import sys, time, traceback, requests
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt, QTimer, QRunnable, pyqtSlot, QThreadPool, pyqtSignal, QObject
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from yahoofinancials import YahooFinancials
import pandas as pd
from pandas import read_csv

# Signals
class WorkerSignals(QObject):
    finished = pyqtSignal() # When completed
    error = pyqtSignal(tuple) # Tuple (exctype, value, traceback.format_exc() )
    result = pyqtSignal(object) # Data returned
    progress = pyqtSignal(int) # % progress

class Worker(QRunnable):
    ''' Worker thread
    # Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
        :param callback: The function callback to run on this worker thread. Supplied args and
                         kwargs will be passed through to the runner.
        :type callback: function
        :param args: Arguments to pass to the callback function
        :param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

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

    def execute_this_fn(self, progress_callback):

        # Clear existing chart
        self.mainChart.axes.cla()
        self.mainChart.draw()

        assets = ['TSLA', 'MSFT', 'III']
        yahoo_financials = YahooFinancials(assets)
        data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
                                                          end_date='2019-12-31',
                                                          time_interval='monthly')
        prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets
        })

        print(prices_df)

        prices_df.to_csv("data/localstorage.csv")

        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(n*100/4)

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
        worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)



        #self.mainChart.axes.draw()

        # Execute
        self.threadpool.start(worker)

    def reload_from_file(self):
        # Pass the function to execute
        worker = Worker(self.load_from_csv)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.thread_complete)

        # self.mainChart.axes.draw()

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