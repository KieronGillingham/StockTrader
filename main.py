# Logging
import logging

from PyQt5.QtGui import QFont

_logger = logging.getLogger(__name__)
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])

# General
import sys
import calendar
from datetime import date, timedelta

# Table display
from tabulate import tabulate

# PyQt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSpinBox, \
    QComboBox, QStackedWidget, QGroupBox, QFormLayout, QLineEdit, QTabWidget, QMessageBox, QBoxLayout, QDateEdit, \
    QCheckBox
from PyQt5.QtCore import QThreadPool, QDate, pyqtSignal, Qt

# Threading
from pyqtthreading import Worker

# Stock data
from stockdata import StockData

from users import UserManager

# Plotting
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

# Machine learning
from learningmodel import LearningModel

stock_data = StockData()
learning_model = LearningModel()
user_manager = UserManager()
INTINF = 2**31 - 1

class MainWindow(QMainWindow):
    """ Main window of application"""
    # List pages and assign indicies
    pages = {
        "Login": 0,
        "Chart": 1,
        "Register": 2,
        "Forecast": 3,
        "User": 4,
    }

    testingranges = {
        "No testing": None,
        "Last 3 days": 3,
        "Last week": 7,
        "Last 2 weeks": 14,
        "Last month": 30,
        "Last 2 months": 60
    }

    # Current user
    user = None

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Threading
        self.threadpool = QThreadPool()
        _logger.debug("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Set window appearance
        self.setWindowTitle("Intelligent Stock Trader")
        self.setMinimumSize(1024, 512)

        # Root widget for storing GUI pages
        self.root = QStackedWidget()

        # Setup page layouts
        self._setup_login_page()
        self._setup_register_page()
        self._setup_chart_page()
        self._setup_forecast_page()
        self._setup_user_page()

        # Center pages
        self.setCentralWidget(self.root)

        # Start on login page
        self.change_page("Login")

        # Display the main window
        self.show()

    # Threading
    def thread(self, function, in_progress=None, on_finish=None, *args, **kwargs):
        # Pass the function to execute
        worker = Worker(function, *args, **kwargs) # Any other args, kwargs are passed to the run function

        # Connect on_finish method to signal
        if on_finish is not None:
            worker.signals.finished.connect(on_finish)

        # Execute
        self.threadpool.start(worker)


    # Page layout
    def _setup_chart_page(self):
        """Create the layout of the Chart page."""
        # Root widget
        page = QWidget()
        vbox_pagechart = QVBoxLayout()
        page.setLayout(vbox_pagechart)
        self.root.insertWidget(self.pages["Chart"], page)

        # Initalise layouts
        hbox_title = QHBoxLayout()
        hbox_main = QHBoxLayout()
        vbox_chart = QVBoxLayout()
        vbox_sidebar = QVBoxLayout()
        vbox_chartmenu = QVBoxLayout()
        vbox_prediction = QVBoxLayout()
        vbox_data = QVBoxLayout()

        # User label
        self.user_label = QLabel()
        self.set_user_label()
        hbox_title.addWidget(self.user_label)

        # Logout button
        logout_button = QPushButton("Log Out")
        logout_button.released.connect(self.log_out)
        hbox_title.addWidget(logout_button)

        # Forecast button
        forecast_button = QPushButton("Forecast")
        forecast_button.released.connect(lambda: [self.change_page("Forecast"), self.reset_transactions()])
        hbox_title.addWidget(forecast_button)

        # Stock filter combobox
        vbox_chartmenu.addWidget(QLabel("Stock:"))
        self.filter_combobox = QComboBox()
        self.filter_combobox.currentIndexChanged.connect(self.filter_changed)
        self.filter_combobox.setEnabled(False)
        vbox_chartmenu.addWidget(self.filter_combobox)

        # Data
        vbox_data.addWidget(QLabel("Data Start Date:"))
        self.data_start_date = QDateEdit()
        self.data_start_date.setCalendarPopup(True)
        self.data_start_date.setDate(QDate.currentDate().addYears(-1))
        vbox_data.addWidget(self.data_start_date)
        vbox_data.addWidget(QLabel("Data End Date:"))
        self.data_end_date = QDateEdit()
        self.data_end_date.setCalendarPopup(True)
        self.data_end_date.setDate(QDate.currentDate())
        self.data_end_date.setMaximumDate(QDate.currentDate())
        vbox_data.addWidget(self.data_end_date)
        wid = QPushButton("Reload from Yahoo Finance")
        wid.released.connect(lambda: self.load_data_from_yahoo_finance(start_date=self.data_start_date.date(),
                                                                       end_date=self.data_end_date.date()))
        vbox_data.addWidget(wid)
        wid = QPushButton("Save data to file")
        wid.released.connect(stock_data.save_to_csv)
        vbox_data.addWidget(wid)
        wid = QPushButton("Load data from file")
        wid.released.connect(lambda: self.load_data_from_file(start_date=self.data_start_date.date(),
                                                              end_date=self.data_end_date.date()))
        vbox_data.addWidget(wid)
        self.show_approx_checkbox = QCheckBox()
        self.show_approx_checkbox.setText("Show approximation")
        self.show_approx_checkbox.setToolTip("Show approximations used in accuracy calculations.")
        self.show_approx_checkbox.stateChanged.connect(self.show_approximations)
        vbox_data.addWidget(self.show_approx_checkbox)
        vbox_data.addSpacing(100)

        # Model
        vbox_data.addWidget(QLabel("Model name:"))
        model_name = QLineEdit()
        model_name.setToolTip("The name of the model that will be created or loaded. Leave blank for default: "
                              "'trainedmodel'.")
        vbox_data.addWidget(model_name)
        vbox_data.addWidget(QLabel("Model type:"))
        model_type = QComboBox()
        model_type.setToolTip("The type of the model to train.")
        for type in learning_model.models.keys():
            model_type.addItem(type, learning_model.models[type])
        vbox_data.addWidget(model_type)
        vbox_data.addWidget(QLabel("Testing data:"))
        test_range = QComboBox()
        test_range.setToolTip("The amount of data to withhold from training to test the model.")
        for type in self.testingranges.keys():
            test_range.addItem(type, self.testingranges[type])
        vbox_data.addWidget(test_range)
        wid = QPushButton("Create model")
        wid.released.connect(lambda: self.train_model(location=model_name.text(),
                                                      type=model_type.currentData(),
                                                      testingrange=test_range.currentData()))
        vbox_data.addWidget(wid)
        wid = QPushButton("Load model")
        wid.released.connect(lambda: self.load_model(location=model_name.text()))
        vbox_data.addWidget(wid)

        userpage_button = QPushButton("User")
        userpage_button.released.connect(lambda: self.change_page("User"))
        hbox_title.addWidget(userpage_button)

        wid = QLabel("")
        vbox_prediction.addWidget(wid)

        # Main chart
        self.mainChart = MplCanvas(self)
        vbox_chart.addWidget(self.mainChart, 8)
        self.status_label = QLabel()
        self.status_label.setFont(QFont('Consolas', 10))
        #self.status_label.setAlignment(Qt.AlignCenter)
        vbox_chart.addWidget(self.status_label, 1)

        # Set layout hierarchy
        vbox_pagechart.addLayout(hbox_title, 1)
        vbox_pagechart.addLayout(hbox_main, 5)
        hbox_main.addLayout(vbox_chart, 5)
        hbox_main.addLayout(vbox_sidebar, 1)
        vbox_sidebar.addLayout(vbox_chartmenu, 1)
        vbox_sidebar.addLayout(vbox_prediction, 2)
        vbox_sidebar.addLayout(vbox_data, 1)

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
        profit_spinbox = QSpinBox()
        profit_spinbox.setRange(0, INTINF)
        formlayout.addRow(QLabel("Desired profit"), profit_spinbox)
        # Timeframe field
        date_field = QDateEdit()
        date_field.setMinimumDate(QDate.currentDate())
        date_field.setCalendarPopup(True)
        formlayout.addRow(QLabel("Timeframe"), date_field)
        # Calculate button
        button = QPushButton("Calculate")

        formlayout.addRow(button)
        # Set form layout
        groupbox.setLayout(formlayout)
        suggest_tab_layout.addWidget(groupbox, 1)

        resultlabel = QLabel("Calculation result")
        suggest_tab_layout.addWidget(resultlabel, 1)
        button.released.connect(lambda: resultlabel.setText(self.investment_plan(self.user["balance"], date_field.date(), desired_profit=profit_spinbox.value())))

        # Back button
        back_button = QPushButton("Back")
        back_button.released.connect(lambda: self.change_page("Chart"))
        layout.addWidget(back_button)

    def _setup_user_page(self):
        # User page
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(380, 100, 380, 200)
        page.setLayout(layout)
        self.root.insertWidget(self.pages["User"], page)

        # User details form
        groupbox = QGroupBox("User")
        formlayout = QFormLayout()
        # Username field
        self.user_username_field = QLineEdit()
        formlayout.addRow(QLabel("Username"), self.user_username_field)
        # Password field
        password_field = QLineEdit()
        password_field.setEchoMode(QLineEdit.Password)
        formlayout.addRow(QLabel("Password"), password_field)
        # Balance field
        self.user_balance_field = QSpinBox()
        self.user_balance_field.setRange(-1 * INTINF, INTINF)
        formlayout.addRow(QLabel("Balance"), self.user_balance_field)

        # Update button
        update_button = QPushButton("Update")
        update_button.released.connect(lambda: self.update_user({
            "account": self.user["username"],
            "username": self.user_username_field.text(),
            "password": password_field.text(),
            "balance": self.user_balance_field.value()
        }))
        formlayout.addRow(update_button)

        # Set form layout
        groupbox.setLayout(formlayout)
        layout.addWidget(groupbox)

        back_button = QPushButton("Back")
        back_button.released.connect(lambda: self.change_page("Chart"))
        layout.addWidget(back_button)

    # Data loading
    def load_data_from_yahoo_finance(self, start_date=None, end_date=None, time_interval='daily', stocksymbols=None, on_finish=None):
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        if isinstance(start_date, QDate):
            start_date = start_date.toPyDate()
        if isinstance(end_date, QDate):
            end_date = end_date.toPyDate()
        self.status_label.setText("Loading...")
        self.thread(stock_data.load_data_from_yahoo_finance, start_date=start_date, end_date=end_date,
                    time_interval=time_interval, stocksymbols=stocksymbols, on_finish=self.data_loaded)

    def load_data_from_file(self, start_date=None, end_date=None):
        self.status_label.setText("Loading...")
        self.thread(stock_data.load_from_csv, on_finish=self.data_loaded)

    def data_loaded(self):
        _logger.debug("Data loaded")
        self.clear_chart()
        self.status_label.setText("")
        self.filter_combobox.clear()
        self.filter_combobox.setEnabled(False)
        data = stock_data.data

        if learning_model.set_data(data):
            _logger.debug("Data setup complete.")

        print(stock_data.stockdict.keys())

        for stockname in list(stock_data.stockdict.keys()):
            self.filter_combobox.addItem(stockname, userData=stock_data.get_symbol(stockname))

        if self.filter_combobox.count() > 0:
            self.filter_combobox.setEnabled(True)

    def load_model(self, location):
        if not isinstance(location, str):
            _logger.error(f"Model location ({location}) is invalid.")
            location = None
        if location is None or location == "":
            location = "trainedmodel"
        self.thread(learning_model.load_predictor(f"data/{location}"), on_finish=self.model_ready)

    # Chart drawing
    def clear_chart(self):
        # Clear existing chart
        self.mainChart.axes.cla()
        self.mainChart.axes.axis('off')
        self.mainChart.draw()

    def draw_chart(self, stocksymbol, prediction=None, approximation=None):

        colours = ['b', 'y', 'g']
        self.clear_chart()

        if stocksymbol + "_close" not in stock_data.data.columns:
            self.mainChart.axes.text(0,0,"No data available.")
            self.mainChart.draw()
            return

        data = stock_data.data[stocksymbol + "_close"]

        self.mainChart.axes.plot(data, color=colours[0])
        self.mainChart.axes.set_title(stocksymbol)
        self.mainChart.axes.set_ylabel("Close Price (£)", rotation="vertical")

        legend = ["Past values"]
        self.mainChart.axes.axis('on')
        if prediction is not None:
            self.mainChart.axes.plot(prediction, linestyle='--', color=colours[1])
            legend.append("Predictions")

        if approximation is not None:
            self.mainChart.axes.plot(approximation, linestyle='--', color=colours[2])
            legend.append("Approximation")

        self.mainChart.axes.legend(legend, loc='upper left')

        self.mainChart.axes.grid(True)
        self.mainChart.draw()

        labels = self.mainChart.axes.get_xticklabels()
        for t in labels:
            val = t.get_position()[0]  # Get raw numerical value of the label
            tdate = date.fromtimestamp(val)
            t.set_text(tdate.strftime("%d/%m"))
        self.mainChart.axes.set_xticklabels(labels)
        self.mainChart.draw()

    def filter_changed(self, value):
        _logger.debug(f"{self.filter_combobox.itemText(value)} ({self.filter_combobox.itemData(value)}) selected.")
        stocksymbol = self.filter_combobox.itemData(value)
        if stocksymbol is not None:
            self.show_stock(stocksymbol)

    def show_approximations(self):
        if stock_data.data is not None:
            self.show_stock(self.filter_combobox.currentData())

    def show_stock(self, stocksymbol):
        predictions = None
        if learning_model.predictor is not None:
            if learning_model.predictor.model is not None:
                #TODO: Reduce calls to get_predictions and check return value if stock not found
                predictions = learning_model.get_predictions()
                if f"{stocksymbol}_close" in predictions.columns:
                    predictions = predictions[f"{stocksymbol}_close"]
                else:
                    _logger.error("No predictions found.")
                    predictions = None

                test_scores = learning_model.test_scores
                if test_scores is not None:

                    stock_eval = test_scores[f"{stocksymbol}_close"]
                    model_eval = test_scores["Overall"]

                    testresults = [
                        ["Mean Squared Error", f"{stock_eval.loc['MSE']:.3f}", f"{model_eval.loc['MSE']:.3f}"],
                        ["Mean Absolute Error", f"{stock_eval.loc['MAE']:.3f}", f"{model_eval.loc['MAE']:.3f}"],
                        ["Explained Variance Score", f"{stock_eval.loc['EVS']:.2f}", f"{model_eval.loc['EVS']:.2f}"],
                        ["R2 Score", f"{stock_eval.loc['R2']:.2f}", f"{model_eval.loc['R2']:.2f}"]
                    ]

                    score_string = tabulate(testresults, headers=["Test Results", "Current Stock", "Overall"])

                    _logger.info(score_string)
                    self.status_label.setText(score_string)

        approximations = None
        if self.show_approx_checkbox.isChecked():
            approximations = learning_model.get_approximation(stocksymbol)

        self.draw_chart(stocksymbol, predictions, approximations)



    # Model training
    def train_model(self, location=None, type=None, testingrange=None):

        if not isinstance(location, str):
            _logger.error(f"Model location ({location}) is invalid.")
            location = None
        if location is None or location == "":
            location = "trainedmodel"

        if learning_model is not None:
            self.clear_chart()
            self.status_label.setText("Loading...")

            train_test_cutoff = None
            if testingrange is not None:
                end_date = self.data_end_date.date()
                end_date = end_date.toPyDate()
                train_test_cutoff_date = end_date - timedelta(days=testingrange)
                train_test_cutoff = calendar.timegm(train_test_cutoff_date.timetuple())
            self.thread(function=learning_model.train_model, train_test_cutoff=train_test_cutoff, model_type=type, on_finish=self.model_ready, persist_location=f"data/{location}")
        else:
            raise Exception("Learning model instance not initialised.")

    def model_ready(self):
        self.clear_chart()
        self.status_label.setText("")
        if learning_model.predictor is not None:
            if learning_model.predictor.model is not None:
                _logger.info("Model loaded.")

        stocksymbol = self.filter_combobox.currentData()
        if stocksymbol is not None:
            self.show_stock(stocksymbol)

    # User
    def sign_in(self, username, password):
        if username == "Guest":
            self.user = user_manager.guest_account()

        elif username in [None, False, ""]:
            self.show_dialog("Login Failure", "Please enter a valid username.")
            return

        elif password in [None, False, ""]:
            self.show_dialog("Login Failure", "Please enter a valid password.")
            return

        else:
            self.user = user_manager.sign_in(username, password)

        if self.user is None:
            self.show_dialog("Login Failure", "Credentials not recognised.")
        else:
            _logger.info(f"Signed in as {self.user['username']}")
            self.change_page("Chart")

    def log_out(self):
        _logger.info(f"Signing out {self.user['username']}")
        self.user = None
        self.change_page("Login")

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
            message = user_manager.register_user(username, password)

        # Display a returned message to the user.
        self.show_dialog("Registration", message)

    def update_user(self, values):
        message = user_manager.update_account(values)
        self.show_dialog("Account Update", message)
        self.set_user_label()

    def set_user_label(self):
        if self.user is not None:
            if hasattr(self, "user_label"):
                self.user_label.setText(f"Signed in as: {self.user['username']} | Balance: £{self.user['balance']}")
            if hasattr(self, "user_username_field"):
                self.user_username_field.setText(self.user['username'])
            if hasattr(self, "user_balance_field"):
                self.user_balance_field.setValue(self.user["balance"])
        else:
            if hasattr(self, "user_label"):
                self.user_label.setText("No user found.")
            if hasattr(self, "user_username_field"):
                self.user_username_field.setText(None)
            if hasattr(self, "user_balance_field"):
                self.user_balance_field.setValue(0)

    # Window
    def change_page(self, page):
        """
        Change the active view to a named page.
        :param page: The name of the new page. Must match a value in `self.pages`.
        :return: None.
        """
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
        """
        Show a QMessageBox with the provided title and message.
        :param title: The title of the QMessageBox.
        :param message: The message of the QMessageBox.
        :return: None.
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(message)
        dlg.exec()

    # Transactions
    def add_transaction(self, count=1):
        """
        Create new QTransactions and add them to the list of transactions.
        :param count: The number of new transaction items to create.
        :return: None.
        """
        for i in range(0, count):
            transaction = QTransaction(stock_data.stockdict)
            transaction.runupdate.connect(self.total_transactions)
            self.forecast_tab_layout.insertWidget(self.forecast_tab_layout.count()-3, transaction, stretch=0)

    def get_transactions(self):
        """
        Get a list of all active QTransactions.
        :return: A list of all QTransactions on the forecast tab.
        """
        return [widget for widget in self.forecast_tab.children() if isinstance(widget, QTransaction)]

    def reset_transactions(self):
        """
        Remove all QTransactions on the forecast tab and create three new defaults.
        :return: None.
        """
        for transaction in self.get_transactions():
            transaction.remove_transaction()
        self.add_transaction(3)

    def total_transactions(self):
        transactions = self.get_transactions()
        transaction_list = [t.total for t in transactions]
        # Create list of tuples of format:
        # (
        #   Stock symbol,
        #   Proposed number of shares purchased/sold,
        #   Proposed date of the transaction
        # )
        self.forecast_result.setText(f"Total: £{'%.2f' % sum(transaction_list)}")

    # Investments
    def investment_plan(self, starting_funds=0, end_date=date.today(), desired_profit=0):
        # Check desired profit is above 0
        if desired_profit <= 0:
            message = f"Desired profit {desired_profit} is invalid."
            _logger.warning(message)
            self.show_dialog("Investment", message)
            return ""

        # Get today date
        day = date.today()

        # Convert to PyDate
        if isinstance(end_date, QDate):
            end_date = end_date.toPyDate()

        # Check end_date is in the future
        if end_date <= day:
            return f"Invalid end date: {end_date}"

        if learning_model.data is None:
            return "No data loaded."

        if learning_model.predictor is None:
            return "No model loaded."

        # Get cheapest stock price
        # TODO: Include list of stocks included in predictions in learningmodel
        # Temp
        cols = [c[0] for c in learning_model.data.columns.str.split("_")]
        stocks = set(cols[:-6])
        prices = [learning_model.get_value(stock, date.today()) for stock in stocks]
        prices.sort()
        cheapest_stock = prices[0]

        # Check stocks are affordable
        if starting_funds < cheapest_stock:
            return "Insufficient funds to begin investing."

        current_funds = starting_funds
        stockreturn = []
        while day <= end_date:
            bestdeal = None
            for stock in stocks:
                todayprice = learning_model.get_value(stock, day)
                tomorrowprice = learning_model.get_value(stock, day + timedelta(days=1))
                if tomorrowprice <= todayprice:
                    continue

                stockcount = current_funds // todayprice
                if stockcount > 0:
                    profit = (stockcount * tomorrowprice) - (stockcount * todayprice)
                    if bestdeal is None or bestdeal[4] < profit:
                        bestdeal = (day, stock, stockcount, todayprice, tomorrowprice, profit)

            if bestdeal is None:
                break

            stockreturn.append(bestdeal)
            current_funds = current_funds + bestdeal[4]
            day = day + timedelta(days=1)

        result = ""
        _logger.debug(stockreturn)

        for a in stockreturn:
            result += f"{a[0]}: Buy {a[2]} stock(s) in {a[1]}. Sell next day for £{a[4]} profit.\n"
        _logger.debug(result)
        return str(result)

class QTransaction(QWidget):
    """
    A collection of QWidgets that represent a transaction (buying or selling) of a specified stock.
    """

    # Update signal that can be used to alert that a change has been made to this element.
    runupdate = pyqtSignal()

    def __init__(self, symbols, *args, **kwargs):
        super(QTransaction, self).__init__(*args, **kwargs)

        self.total = 0

        # Root layout
        self.layout = QHBoxLayout()

        # Combobox for selecting a stock to purchase/sell
        # Label
        combobox_label = QLabel("Stock:")
        self.layout.addWidget(combobox_label, 1)
        # Combobox
        self.combobox = QComboBox()
        self.combobox.currentIndexChanged.connect(self.update_price) # Update calculated price if selected stock changes
        self.layout.addWidget(self.combobox, 2)
        # Label to display the selected stock's symbol
        self.stocksymbol_label = QLabel()
        self.layout.addWidget(self.stocksymbol_label, 1)

        # Spinbox to select how many shares to buy/sell
        # Spinbox
        self.value = QSpinBox()
        self.value.setRange(-1000000, 1000000)
        self.value.valueChanged.connect(self.update_price) # Update calculated price if value changed
        self.layout.addWidget(self.value, 1)
        # Label to show individual share price
        self.sharesprice_label = QLabel()
        self.layout.addWidget(self.sharesprice_label, 1)

        # Select date for transaction
        self.date = QDateEdit()
        self.date.setDate(QDate.currentDate())
        self.date.setCalendarPopup(True)
        self.date.dateChanged.connect(self.update_price) # Update calculated price if date changed
        self.layout.addWidget(self.date, 3)

        # Label showing total cost/return of this transaction
        self.total_label = QLabel()
        self.layout.addWidget(self.total_label, 1)

        # Button to remove transaction
        self.remove = QPushButton("❌")
        self.remove.pressed.connect(self.remove_transaction)
        self.layout.addWidget(self.remove, 1)

        # Set the root widget's layout
        self.setLayout(self.layout)
        if symbols is not None:
            self.set_symbols(symbols)

    def update_price(self):
        """
        Update the calculated profit/loss for the investment described by this transaction.
        :return: None.
        """
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

        # Send update signal to trigger a global update of the total.
        self.runupdate.emit()

    def remove_transaction(self):
        """
        Delete this transaction.
        :return: None.
        """
        self.setParent(None)

    def set_symbols(self, symbols):
        """
        Set the selectable stocks for the transaction.
        :param symbols: Dict of stock names and their symbols.
        :return: None.
        """
        for n in symbols:
            self.combobox.addItem(n, userData=symbols[n])

class MplCanvas(FigureCanvasQTAgg):
    """PyQt canvas for MatPlotLib graphs"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')
        super(MplCanvas, self).__init__(fig)

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