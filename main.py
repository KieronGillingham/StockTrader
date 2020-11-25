import sys, time, traceback, requests
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt, QTimer, QRunnable, pyqtSlot, QThreadPool, pyqtSignal, QObject
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from yahoofinancials import YahooFinancials
import pandas as pd

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
        layout = QVBoxLayout()

        """
        # Counter
        self.counter = 0
        self.label = QLabel()
        layout.addWidget(self.label)
        # Timer for counter
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        # Button
        button = QPushButton("DANGER!")
        button.pressed.connect(self.oh_no)
        layout.addWidget(button)

        # Widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        """

        assets = ['TSLA', 'MSFT', 'FB']
        yahoo_financials = YahooFinancials(assets)
        data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
                                                          end_date='2019-12-31',
                                                          time_interval='weekly')
        prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets
        })
        prices_df.plot()


        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot(prices_df)
        self.setCentralWidget(sc)

        # Treadpool
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Display the main window
        self.show()

    def recurring_timer(self):
        self.counter += 1
        self.label.setText("Counter: %d" % self.counter)

    def progress_fn(self, n):
        print("%d%% done" % n)

    def execute_this_fn(self, progress_callback):
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(n*100/4)

        return "Done."

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def oh_no(self):
        # Pass the function to execute
        worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)



        """
        assets = ['TSLA', 'MSFT', 'FB']
        yahoo_financials = YahooFinancials(assets)
        data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
                                                          end_date='2019-12-31',
                                                          time_interval='weekly')
        prices_df = pd.DataFrame({
            a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets
        })
        prices_df.plot()
        """


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