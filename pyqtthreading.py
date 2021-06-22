# Logging
import logging
_logger = logging.getLogger(__name__)

# General
import sys, traceback

# GUI
from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject

# Signals
class WorkerSignals(QObject):
    finished = pyqtSignal()  # When completed
    error = pyqtSignal(tuple)  # Tuple (exctype, value, traceback.format_exc() )
    result = pyqtSignal(object)  # Data returned
    progress = pyqtSignal(int) # Percentage progress

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
        """Initialise the runner function with passed args, kwargs."""

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
            _logger.debug("Worker finished")
            self.signals.finished.emit()  # Done


    def progress_fn(self, n):
        _logger.debug("%d%% done" % n)