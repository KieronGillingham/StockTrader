# Logging
import logging
_logger = logging.getLogger(__name__)

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, QComboBox, QDialog
from PyQt5.QtCore import QTimer, QThreadPool

class LandingWindow(QDialog):
    """ Main window of application"""

    def __init__(self, *args, **kwargs):
        super(LandingWindow, self).__init__(*args, **kwargs)

        # Set window appearance
        self.setWindowTitle("Sign In")
        self.setMinimumSize(512, 512)

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
        self.vbox_chartmenu.addWidget(self.filter_combobox)

        wid = QPushButton("Reload from Yahoo Finance")
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Save data to file")
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Load data from file")
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Clear Chart")
        self.vbox_data.addWidget(wid)

        wid = QLabel("Invest Amount (Stocks):")
        self.vbox_prediction.addWidget(wid)
        self.stock_invested = QSpinBox()
        self.vbox_prediction.addWidget(self.stock_invested)

        wid = QPushButton("Predict Profit")
        self.vbox_prediction.addWidget(wid)

        wid = QLabel("")
        self.vbox_prediction.addWidget(wid)

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

        # Display the window
        self.show()