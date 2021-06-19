# Logging
import calendar
import logging
_logger = logging.getLogger(__name__)
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])

# General
import sys
from datetime import date, datetime
from typing import List

# GUI
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, \
    QComboBox, QStackedWidget, QGroupBox, QFormLayout, QLineEdit, QTabWidget, QMessageBox, QBoxLayout, QDateEdit
from PyQt5.QtCore import QTimer, QThreadPool, QDateTime, QDate, pyqtSignal

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
            "User": 4,
            "Help": 5
        }
        self.user = None

        # Setup page layouts
        self._setup_login_page()
        self._setup_register_page()
        self._setup_chart_page()
        self._setup_forecast_page()
        self._setup_user_page()
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
        self.show_dialog("Registration Failure", "Registration cannot be completed at this time.")

    def set_user_label(self):
        if self.user is not None:
            self.user_label.setText(f"Signed in as: {self.user['username']} | Balance: £{self.user['balance']}")
        else:
            self.user_label.setText("No user found.")

    def sign_in(self, username, password):
        if username == "Guest":
            self.user = self.user_manager.guest_account()

        elif username in [None, False, ""]:
            self.show_dialog("Login Failure", "Please enter a valid username.")
            return

        elif password in [None, False, ""]:
            self.show_dialog("Login Failure", "Please enter a valid password.")
            return

        else:
            self.user = self.user_manager.sign_in(username, password)

        if self.user is None:
            self.show_dialog("Login Failure", "Credentials not recognised.")
        else:
            _logger.info(f"Signed in as {self.user['username']}")
            self.change_page("Chart")

    def log_out(self):
        _logger.info(f"Signing out {self.user['username']}")
        self.user = None
        self.change_page("Login")

    def _setup_chart_page(self):
        # Initalise layouts
        self.vbox_pagechart = QVBoxLayout()
        hbox_title = QHBoxLayout()
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

        self.user_label = QLabel()
        self.set_user_label()
        hbox_title.addWidget(self.user_label)

        logout_button = QPushButton("Log Out")
        logout_button.released.connect(self.log_out)
        hbox_title.addWidget(logout_button)

        forecast_button = QPushButton("Forecast")
        forecast_button.released.connect(lambda: [self.change_page("Forecast"), self.reset_transactions()])
        hbox_title.addWidget(forecast_button)

        self.filter_combobox = QComboBox()
        self.filter_combobox.currentIndexChanged.connect(self.filter_changed)
        self.vbox_chartmenu.addWidget(self.filter_combobox)

        wid = QPushButton("Reload from Yahoo Finance")
        wid.released.connect(self.load_data_from_yf)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Save data to file")
        wid.released.connect(stock_data.save_to_csv)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Load data from file")
        wid.released.connect(self.load_data_from_file)
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Train model")
        wid.released.connect(lambda: learning_model.train_model(persist_location="data/trainedmodel"))
        self.vbox_data.addWidget(wid)

        wid = QPushButton("Load model")
        wid.released.connect(lambda: learning_model.load_model(model_location="data/trainedmodel"))
        self.vbox_data.addWidget(wid)

        userpage_button = QPushButton("User")
        userpage_button.released.connect(lambda: self.change_page("User"))
        hbox_title.addWidget(userpage_button)

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
        self.vbox_pagechart.addLayout(hbox_title, 1)
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

        from users import UserManager
        self.user_manager = UserManager()

        # Login form
        groupbox = QGroupBox("Login")
        formlayout = QFormLayout()
        # Username field
        username_field = QLineEdit()
        formlayout.addRow(QLabel("Login"), username_field)
        # Password field
        password_field = QLineEdit()
        password_field.setEchoMode(QLineEdit.Password)
        formlayout.addRow(QLabel("Password"), password_field)
        # Login button
        login_button = QPushButton("Login")
        login_button.released.connect(lambda: self.sign_in(username_field.text(), password_field.text()))
        formlayout.addRow(login_button)
        # Guest login button
        guest_login_button = QPushButton("Continue as Guest")
        guest_login_button.released.connect(lambda: self.sign_in("Guest", None))
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
        groupbox = QGroupBox("Register")
        formlayout = QFormLayout()
        # Username field
        username_field = QLineEdit()
        formlayout.addRow(QLabel("Email / Login"), username_field)
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
        register_button.released.connect(lambda: self.register_form(username_field.text(), password_field.text(), reenter_password_field.text()))
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
        self.forecast_tab = QWidget()
        tab2 = QWidget()
        tab_widget.addTab(self.forecast_tab, "Forecast")
        tab_widget.addTab(tab2, "Suggest")
        layout.addWidget(tab_widget)

        # Forecast tab
        self.forecast_tab_layout = QVBoxLayout()
        self.forecast_tab.setLayout(self.forecast_tab_layout)

        add_button = QPushButton("+")
        add_button.released.connect(self.add_transaction)
        self.forecast_tab_layout.addWidget(add_button)

        self.forecast_tab_layout.addStretch()

        self.forecast_result = QLabel("Calculation Result")

        self.forecast_tab_layout.addWidget(self.forecast_result)

        self.reset_transactions()

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

    def _setup_user_page(self):
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

        for n in stock_data.stockdict:
            self.filter_combobox.addItem(n, userData=stock_data.get_symbol(n))

        #self.draw_single_stock("TYT.L")

    def load_data_from_yf(self):
        stock_data.get_yahoo_finance_data(start_date='2019-12-01', end_date='2021-03-01', time_interval='daily', on_finish=self.data_loaded)  # Any other args, kwargs are passed to the run function

    def load_data_from_file(self):
        # Pass the function to execute
        worker = Worker(stock_data.load_from_csv)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.data_loaded)

        # Execute
        return self.threadpool.start(worker)

    def filter_changed(self, value):
        _logger.debug(f"{self.filter_combobox.itemText(value)} ({self.filter_combobox.itemData(value)}) selected.")
        if self.filter_combobox.itemData(value) is not None:
            if learning_model.model is not None:
                self.draw_single_stock(self.filter_combobox.itemData(value))
                self.make_prediction()

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

    def make_prediction(self):
        predictions = learning_model.get_predictions(self.filter_combobox.currentData())
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
            self.set_user_label()

        except Exception as ex:
            _logger.error(f"Error changing page: {ex}")

    def show_dialog(self, title : str, message : str):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(message)
        dlg.exec()

    def add_transaction(self, count=1):
        for i in range(0,count):
            transaction = QTransaction(stock_data.stockdict)
            transaction.runupdate.connect(self.calculate_investments)
            self.forecast_tab_layout.insertWidget(self.forecast_tab_layout.count()-3, transaction, stretch=0)

    def get_transactions(self):
        return [widget for widget in self.forecast_tab.children() if isinstance(widget, QTransaction)]

    def reset_transactions(self):
        for transaction in self.get_transactions():
            transaction.remove_transaction()
        self.add_transaction(3)

    def calculate_investments(self):
        transactions = self.get_transactions()
        transaction_list = [t.total for t in transactions]
        # Create list of tuples of format:
        # (
        #   Stock symbol,
        #   Proposed number of shares purchased/sold,
        #   Proposed date of the transaction
        # )
        self.forecast_result.setText(f"Total: £{'%.2f' % sum(transaction_list)}")

    def register_form(self, username, password, reenter_password):
        """
        Process and validate inputs on the registration form. Shows a response message to the user.
        :param username: The input username.
        :param password: The input password.
        :param reenter_password: The input for re-enter password.
        :return: None.
        """
        # Confirm entered passwords match.
        if password != reenter_password:
            message = "Passwords do not match."
        else:
            # Attempt to register the user with the given username and password.
            message = self.user_manager.register_user(username, password)

        # Display a returned message to the user.
        self.show_dialog("Registration", message)


class QTransaction(QWidget):
    runupdate = pyqtSignal()
    def __init__(self, symbols, *args, **kwargs):
        super(QTransaction, self).__init__(*args, **kwargs)
        self.total = 0

        self.layout = QHBoxLayout()

        combobox_label = QLabel("Stock:")
        self.layout.addWidget(combobox_label, 1)

        self.combobox = QComboBox()
        self.combobox.currentIndexChanged.connect(self.update_price)
        self.layout.addWidget(self.combobox, 2)

        self.stocksymbol_label = QLabel()
        self.layout.addWidget(self.stocksymbol_label, 1)

        self.value = QSpinBox()
        self.value.setRange(-10000, 10000)
        self.value.valueChanged.connect(self.update_price)
        self.layout.addWidget(self.value, 1)

        self.sharesprice_label = QLabel()
        self.layout.addWidget(self.sharesprice_label, 1)

        self.date = QDateEdit()
        self.date.setDate(QDate.currentDate())
        self.date.dateChanged.connect(self.update_price)
        self.layout.addWidget(self.date, 3)

        self.total_label = QLabel()
        self.layout.addWidget(self.total_label, 1)

        self.remove = QPushButton("❌")
        self.remove.pressed.connect(self.remove_transaction)
        self.layout.addWidget(self.remove, 1)

        self.setLayout(self.layout)
        if symbols is not None:
            self.set_symbols(symbols)

    def update_price(self):

        stocksymbol = self.combobox.currentData()
        self.stocksymbol_label.setText(f"({stocksymbol})")
        value = self.value.value()
        transaction_date = self.date.date().toPyDate()

        prediction_date_stamp = calendar.timegm(transaction_date.timetuple())
        price = learning_model.get_value(stocksymbol, transaction_date)
        if price is not None:

            self.sharesprice_label.setText(f"shares at £{'%.2f' % price} /share")
            self.total = price * value
            self.total_label.setText(f"Total: £{'%.2f' % self.total}")

        self.runupdate.emit()

    def remove_transaction(self):
        self.setParent(None)

    def set_symbols(self, symbols):
        for n in symbols:
            self.combobox.addItem(n, userData=symbols[n])


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