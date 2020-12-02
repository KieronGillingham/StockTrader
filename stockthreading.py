# General
import sys, traceback

# GUI
from PyQt5.QtCore import Qt, QTimer, QRunnable, pyqtSlot, QThreadPool, pyqtSignal, QObject

# Signals
class WorkerSignals(QObject):
    finished = pyqtSignal()  # When completed
    error = pyqtSignal(tuple)  # Tuple (exctype, value, traceback.format_exc() )
    result = pyqtSignal(object)  # Data returned
    progress = pyqtSignal(int) # % progress

class Worker(QRunnable):
    """ Worker thread
    # Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
        :param callback: The function callback to run on this worker thread. Supplied args and
                         kwargs will be passed through to the runner.
        :type callback: function
        :param args: Arguments to pass to the callback function
        :param kwargs: Keywords to pass to the callback function
    """

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
#
# # Encapsulate main window in a class
# class MainWindow(QMainWindow):
#     def __init__(self, *args, **kwargs):
#         # Treadpool
#         self.threadpool = QThreadPool()
#         print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
#
#     def progress_fn(self, n):
#         print("%d%% done" % n)
#
#     def load_from_yahoo_finance(self, progress_callback):
#         return "Done."
#
#     def load_from_csv(self, progress_callback):
#         return
#
#     def print_output(self, s):
#         print(s)
#
#     def thread_complete(self):
#         print("THREAD COMPLETE!")
#
#     def reloadData(self):
#         # Pass the function to execute
#         worker = Worker(self.load_from_yahoo_finance) # Any other args, kwargs are passed to the run function
#         worker.signals.result.connect(self.print_output)
#         worker.signals.finished.connect(self.thread_complete)
#         worker.signals.progress.connect(self.progress_fn)
#
#         self.threadpool.start(worker)
#
#     def reload_from_file(self):
#         # Pass the function to execute
#         worker = Worker(self.load_from_csv)  # Any other args, kwargs are passed to the run function
#         worker.signals.finished.connect(self.thread_complete)
#
#         self.threadpool.start(worker)