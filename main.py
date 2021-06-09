# Logging
import logging
_logger = logging.getLogger(__name__)
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])

# General
import sys
from datetime import date
from typing import List

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, \
    QComboBox, QStackedWidget, QGroupBox, QFormLayout, QLineEdit, QTabWidget, QMessageBox, QBoxLayout
from PyQt5.QtCore import QTimer, QThreadPool

# Threading
from stockthreading import Worker

# Stock data
from stockdata import StockData

# Machine learning
from learningmodel import LearningModel

# Plotting
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

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

        # Program root widget for storing GUI pages
        self.root = QStackedWidget()

        # List pages and assign indicies
        self.pages = {
            "Login": 0,
            "Chart": 1,
            "Register": 2,
            "Forecast": 3,
            "ChangePass": 4,
            "Help": 5
        }

        # Setup page layouts
        self._setup_login_page()
        self._setup_register_page()
        self._setup_chart_page()
        self._setup_forecast_page()
        self._setup_changepass_page()
        self._setup_help_page()

        page = QWidget()
        page.setLayout(self.vbox_pagechart)
        self.root.insertWidget(self.pages["Chart"], page)

        # Center pages
        self.setCentralWidget(self.root)

        self.change_page("Login")

        # Treadpool
        self.threadpool = QThreadPool()
        _logger.debug("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Display the main window
        self.show()

    def show_not_available_dialog(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Registration Failure")
        dlg.setText("Registration cannot be completed at this time.")
        dlg.exec()

    def sign_in(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Login Failure")
        dlg.setText("Credentials not recognised.")
        dlg.exec()

    def _setup_chart_page(self):
        # Initalise layouts
        self.vbox_pagechart = QVBoxLayout()
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

        logout_button = QPushButton("Log Out")
        logout_button.released.connect(lambda: self.change_page("Login"))
        self.hbox_title.addWidget(logout_button)

        forecast_button = QPushButton("Forecast")
        forecast_button.released.connect(lambda: self.change_page("Forecast"))
        self.hbox_title.addWidget(forecast_button)

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
        self.vbox_pagechart.addLayout(self.hbox_title, 1)
        self.vbox_pagechart.addLayout(self.hbox_main, 5)
        self.hbox_main.addLayout(self.vbox_sidebar, 1)
        self.vbox_sidebar.addLayout(self.vbox_chartmenu, 1)
        self.vbox_sidebar.addLayout(self.vbox_prediction, 2)
        self.vbox_sidebar.addLayout(self.vbox_data, 1)

    def _setup_login_page(self):
        # Login page
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(380,100,380,200)
        page.setLayout(layout)
        self.root.insertWidget(self.pages["Login"], page)

        # Login form
        groupbox = QGroupBox("Login")
        formlayout = QFormLayout()
        # Username field
        formlayout.addRow(QLabel("Login"), QLineEdit())
        # Password field
        password_field = QLineEdit()
        password_field.setEchoMode(QLineEdit.Password)
        formlayout.addRow(QLabel("Password"), password_field)
        # Login button
        login_button = QPushButton("Login")
        login_button.released.connect(self.sign_in)
        formlayout.addRow(login_button)
        # Guest login button
        guest_login_button = QPushButton("Continue as Guest")
        guest_login_button.released.connect(lambda: self.change_page("Chart"))
        formlayout.addRow(guest_login_button)
        # Set form layout
        groupbox.setLayout(formlayout)
        layout.addWidget(groupbox)

        register_button = QPushButton("Register")
        register_button.released.connect(lambda: self.change_page("Register"))
        layout.addWidget(register_button)

    def _setup_register_page(self):
        # Register page
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(380, 100, 380, 200)
        page.setLayout(layout)
        self.root.insertWidget(self.pages["Register"], page)

        # Registration form
        groupbox = QGroupBox("Login")
        formlayout = QFormLayout()
        # Username field
        formlayout.addRow(QLabel("Email / Login"), QLineEdit())
        # Password field
        password_field = QLineEdit()
        password_field.setEchoMode(QLineEdit.Password)
        formlayout.addRow(QLabel("Password"), password_field)
        # Reenter password field
        reenter_password_field = QLineEdit()
        reenter_password_field.setEchoMode(QLineEdit.Password)
        formlayout.addRow(QLabel("Re-enter Password"), reenter_password_field)
        # Register button
        register_button = QPushButton("Register")
        register_button.released.connect(self.show_not_available_dialog)
        formlayout.addRow(register_button)
        # Set form layout
        groupbox.setLayout(formlayout)
        layout.addWidget(groupbox)

        # Back button
        back_button = QPushButton("Back")
        back_button.released.connect(lambda: self.change_page("Login"))
        layout.addWidget(back_button)

    def _setup_forecast_page(self):
        # Forecast page
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)
        self.root.insertWidget(self.pages["Forecast"], page)

        # Tabs
        tab_widget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab_widget.addTab(tab1, "Forecast")
        tab_widget.addTab(tab2, "Suggest")
        layout.addWidget(tab_widget)

        # Forecast tab
        forecast_tab_layout = QVBoxLayout()
        tab1.setLayout(forecast_tab_layout)

        forecast_tab_layout.addWidget(QComboBox())
        forecast_tab_layout.addWidget(QComboBox())
        forecast_tab_layout.addWidget(QComboBox())
        forecast_tab_layout.addWidget(QPushButton("+"))
        forecast_tab_layout.addWidget(QLabel("Calculation Result"))

        # Suggest tab
        suggest_tab_layout = QHBoxLayout()
        tab2.setLayout(suggest_tab_layout)
        # Suggest form
        groupbox = QGroupBox()
        formlayout = QFormLayout()
        # Profit field
        formlayout.addRow(QLabel("Desired profit"), QLineEdit())
        # Timeframe field
        formlayout.addRow(QLabel("Timeframe"), QLineEdit())
        # Calculate button
        button = QPushButton("Calculate")
        formlayout.addRow(button)
        # Set form layout
        groupbox.setLayout(formlayout)
        suggest_tab_layout.addWidget(groupbox, 1)

        resultlabel = QLabel("Calculation result")
        suggest_tab_layout.addWidget(resultlabel, 1)

        # Back button
        back_button = QPushButton("Back")
        back_button.released.connect(lambda: self.change_page("Chart"))
        layout.addWidget(back_button)

    def _setup_changepass_page(self):
        pass

    def _setup_help_page(self):
        pass

    def clear_chart(self):
        # Clear existing chart
        self.mainChart.axes.cla()
        self.mainChart.draw()

    def draw_single_stock(self, stocksymbol, prediction=None):

        self.clear_chart()

        data = stock_data.prices_df.filter(like=stocksymbol + "_close")

        self.mainChart.axes.plot(data)
        if prediction is not None:
            self.mainChart.axes.plot(prediction, linestyle='--')

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


        # self.clear_chart()
        #
        #
        # self.mainChart.axes.legend(data.columns.tolist())
        #
        # if prediction is not None:
        #     if len(data.columns) != len(prediction.columns):
        #         raise Exception('Historical and prediction lists are not equal length.')
        #
        #     for i in range(0, len(prediction.columns)):
        #         self.mainChart.axes.plot(prediction.iloc(axis=1)[i], linestyle='--', color=p[i].get_color())
        #
        # self.mainChart.draw()


    def data_loaded(self):
        _logger.debug("Data loaded")
        self.clear_chart()
        self.filter_combobox.clear()
        data = stock_data.prices_df

        if learning_model.set_data(data):
            _logger.debug("Data setup complete.")

        for n in stock_data.get_stocknames():
            self.filter_combobox.addItem(n, userData=stock_data.get_symbol(n))



        #self.draw_single_stock("TYT.L")

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
        _logger.debug(f"{self.filter_combobox.itemText(value)} ({self.filter_combobox.itemData(value)}) selected.")
        if self.filter_combobox.itemData(value) is not None:
            self.draw_single_stock(self.filter_combobox.itemData(value))
            self.make_prediction()

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
        predictions = learning_model.predict(self.filter_combobox.currentData())
        self.draw_single_stock(self.filter_combobox.currentData(), predictions)

    def change_page(self, page):
        try:
            if self.pages is False or self.pages is None:
                _logger.warning("Trying to change page, but no pages set.")
            elif isinstance(page, str):
                try:
                    index = self.pages[page]
                except IndexError as ex:
                    raise ValueError(f"Page not found: {page}")
            elif isinstance(page, int):
                index = page
            else:
                raise TypeError(f"Invalid page name or index: {page}")

            _logger.debug(f"Switching to {page} page (Index: {index}).")
            if self.root.widget(index) is None:
                _logger.warning(f"Page {page} has no root widget.")

            self.root.setCurrentIndex(index)

        except Exception as ex:
            _logger.error(f"Error changing page: {ex}")

if __name__ == '__main__':

    # Create application.
    app = QApplication(sys.argv) # sys.argv are commandline arguments passed in when the program runs.

    # Create and display the main window.
    window = MainWindow()

    # Start the main program loop.
    _logger.info("Program starting.")
    app.exec_()

    # Program terminating.
    _logger.info("Program termininating.")