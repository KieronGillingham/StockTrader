# Logging
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

_logger = logging.getLogger(__name__)

# General
from datetime import date
import calendar

# Data manipulation
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# Persistance
from joblib import dump, load

class LearningModel():

    periods = {
        "NEXTDAY": 86400,  # 60*60*24 (One day)
        "NEXTWEEK": 604800,  # 60*60*24 * 7
        "NEXTTWOWEEK": 1209600, # 60*60*24 * 14
        "NEXTMONTH": 2592000  # 60*60*24 * 30
    }

    models = {
        "Quick Multi-Layer Perceptron": "mlp-fast",
        "Full Multi-Layer Perceptron": "mlp",
        "Random Forest": "randfor"
    }

    # Seed for repeatability of random number generation
    seed = 98628

    # Data to train models with
    data = None
    test_data = None

    # Current trained predictor
    predictor = None

    # Column names for deconstructed dates.
    datecolumns = ['year', 'month', 'day', 'weekday', 'week', 'dayofyear']

    def __init__(self, data=None, *args, **kwargs):
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        """
        Set the DataFrame to be used for training a model.
        :param data: The DataFrame to use.
        :return: True if DataFrame is set successfully; False otherwise.
        """

        # Check data is DataFrame
        if isinstance(data, pd.DataFrame):
            # If data doesn't contain deconstructed date columns
            if not set(self.datecolumns).issubset(data.columns):
                # Turn data index into date columns
                data = data.join(self.deconstruct_date(data.index))
            # Set the data
            self.data = data
            return True
        else:
            return False

    def train_model(self, model_type="mlp", persist_location=None, data=None, train_test_cutoff=None, *args, **kwargs):
        """
        Train a machine learning model to forecast stock prices.
        :param data: The dataset to be used for training.
        :return: None.
        """
        if model_type not in self.models.values():
            _logger.error(f"Model type '{model_type}' not recognised.")
            return

        if self.predictor is None:
            self.predictor = Predictor()

        # Use object dataset if no alternative provided.
        if data is None:
            data = self.data

        # Stop if no dataset is available for training.
        if data is None:
            _logger.error("No dataset available for training.")
            return

        self.test_data = None

        # Sort the dataset chronologically.
        data.sort_index(inplace=True)

        # Get datestamps of earliest and latest records from the data.
        earliest_datestamp = data.index.min()
        if train_test_cutoff is not None:
            latest_datestamp = train_test_cutoff
            if latest_datestamp <= earliest_datestamp:
                _logger.warning("Insufficient data to train with. Ignoring testing.")
                latest_datestamp = data.index.max()
            else:
                self.test_data = data.loc[train_test_cutoff:]
        else:
            latest_datestamp = data.index.max()
        _logger.debug(f"Preparing to train model on data between {date.fromtimestamp(earliest_datestamp)} and "
                      f"{date.fromtimestamp(latest_datestamp)}.")

        train_data = data.loc[earliest_datestamp:latest_datestamp]

        # Get independent variables for model input (dates)
        x_train = train_data[self.datecolumns]
        _logger.debug(f"X train data size: {x_train.shape}")

        # Get dependent variables for model output (stock prices)
        y_train = train_data.drop(self.datecolumns, 1)
        _logger.debug(f"Y train data size: {y_train.shape}")

        # Create data scaler - Keep reference for later unscaling
        self.predictor.scaler = StandardScaler()

        # Scale y(s)
        self.predictor.scaler.fit(y_train)
        y_train = self.predictor.scaler.transform(y_train)

        try:
            if model_type == "mlp":
                self.train_mlp_model(x_train, y_train)
            elif model_type == "mlp-fast":
                self.train_mlp_model(x_train, y_train, fast=True)
            elif model_type == "randfor":
                self.train_randomforest_model(x_train, y_train)
            else:
                raise Exception(f"Model missing {model_type}.")
        except Exception as ex:
            _logger.error(ex)
            return

        if self.predictor.model is not None:
            _logger.debug("Model trained.")

            # If persist location is given, save the model out to a file
            if persist_location is not None:
                try:
                    self.predictor.save(location=persist_location)
                    _logger.info(f"Model persisted to {persist_location}.")
                except FileNotFoundError as ex:
                    _logger.error(f"Persist location for model could not be found: {ex}")
                    return
                except Exception as ex:
                    _logger.error(ex)
                    return

            if self.test_data is not None:
                self.test_model()
        else:
            _logger.error("Problem training model.")
        return

    def load_predictor(self, predictor_location):
        try:
            self.predictor = Predictor.load(predictor_location)
            if self.predictor is not None:
                if self.predictor.model is not None:
                    _logger.info(f"Model loaded: {self.predictor.model}")
                if self.predictor.scaler is not None:
                    _logger.info(f"Scaler loaded: {self.predictor.scaler}")
            else:
                raise Exception("Predictor could not be loaded.")
        except FileNotFoundError as ex:
            _logger.error(f"Persist location for model could not be found: {ex}")
        except Exception as ex:
            _logger.error(ex)

    def get_predictions(self, prediction_period='NEXTMONTH'):
        """
        Get predictions for a given period.
        :param prediction_period:
        :return:
        """
        if prediction_period not in self.periods:
            _logger.error(f"Prediction period '{prediction_period}' not recognised. Must be one of: {[*self.periods]}")
            return
        elif self.predictor is None or self.predictor.model is None:
            _logger.error("No prediction model loaded.")
            return
        elif not self._check_data():
            _logger.error("Error with prediction model.")
            return

        # Get prediction datestamps
        prediction_start = self.data.index.max() - self.periods[prediction_period]
        prediction_end = self.data.index.max() + self.periods[prediction_period]
        _logger.info(f"Predicting market between {date.fromtimestamp(prediction_start)} and "
                      f"{date.fromtimestamp(prediction_end)}")

        dates = [i for i in range(prediction_start, prediction_end, 86400)]

        predictions = self.predictor.model.predict(self.deconstruct_date(dates))

        if self.predictor.scaler is not None:
            _logger.info("Unscaling prediction values.")
            predictions = self.predictor.scaler.inverse_transform(predictions)
        _logger.debug(f"Predictions:\n{predictions}")

        return pd.DataFrame(data=predictions, index=dates, columns=self.data.columns[:-6])

    def get_value(self, stock: str, prediction_date: date):
        _logger.debug(f"Getting value of stock {stock} on date {prediction_date}.")

        if not isinstance(stock, str):
            raise TypeError("Stock must be a string of a stock symbol: I.E. GOOG")
        elif not self._check_data():
            _logger.error("Error with prediction model.")
            return

        prediction_date_stamp = calendar.timegm(prediction_date.timetuple())
        earliest_date_stamp = self.data.index.min()
        latest_date_stamp = self.data.index.max()

        _logger.debug(f"Latest date: {latest_date_stamp} / {date.fromtimestamp(latest_date_stamp)}")
        _logger.debug(f"Prediction date: {prediction_date_stamp} / {date.fromtimestamp(prediction_date_stamp)}")

        if prediction_date_stamp < earliest_date_stamp:
            _logger.warning("Prediction date earlier than historical data.")

        if prediction_date_stamp < latest_date_stamp:
            stock_data = self.data[f"{stock}_close"]
            print("stock_data:", stock_data)
            print("xp:", stock_data.index)
            print("fp:", stock_data.values)
            print("pred_date_stamp:", prediction_date_stamp)
            return np.interp(prediction_date_stamp, stock_data.index, stock_data.values)
        else:
            if self.predictor is None:
                self.train_mlp_model()
            predictions = self.predictor.model.predict(self.deconstruct_date([prediction_date_stamp]))
            predictions = self.predictor.scaler.inverse_transform(predictions)
            print(predictions)
            if len(predictions[0]) != (len(self.data.columns) - 6):
                _logger.error("Wrong length predictions.")
            prediction = pd.DataFrame(data=predictions, index=[prediction_date_stamp], columns=self.data.columns[:-6])
            return prediction[f"{stock}_close"].values[0]

    def train_randomforest_model(self, x, y):
        """
        Train a random forest model to forecast stock prices.
        :param data: The dataset to be used for training.
        :return: None.
        """
        _logger.debug("Beginning hyperparameter tuning and model training.")
        params = [
            {
                "random_state": [self.seed],
                "n_estimators": [10, 20, 50, 100],
                "criterion": ["mse"],
                "max_depth": [1, 2, 5, 10, 20],
                "verbose": [True]
            }
        ]

        regressor = GridSearchCV(RandomForestRegressor(), params)
        regressor.fit(x, y)

        _logger.debug(regressor.best_params_)

        self.predictor.model = regressor

    #def linear_model_prediction(self, stock, prediction_date_stamp):
    #     latest_date_stamp = self.data.index.max()
    #
    #     y = self.data[f"{stock}_close"].values
    #     x = np.array(self.data.index)
    #     _logger.debug(y)
    #     _logger.debug(x)
    #
    #     x = x.reshape(-1, 1)
    #     #for i in range(0, len(self.prices_df.columns)):
    #     #     y = self.prices_df.iloc[:, i].values
    #
    #     prediction = model.predict([[prediction_date_stamp]])
    #
    #     _logger.debug(f"{self.data[f'{stock}_close'][latest_date_stamp]} -> {prediction[0]}")
    #
    #     pred_df = pd.DataFrame([self.data[f"{stock}_close"][latest_date_stamp],prediction[0]], columns=[f"{stock}_close"], index=[latest_date_stamp, prediction_date_stamp])
    #
    #     return pred_df
    #
    # def naive_rolling_mlp_model_prediction(self, stock, prediction_date_stamp, latest_date_stamp=None):
    #
    #     print(latest_date_stamp)
    #     print(prediction_date_stamp)
    #
    #     self.data.sort_index(inplace=True)
    #
    #     if latest_date_stamp is None:
    #         latest_date_stamp = self.data.index.max()
    #
    #     earliest_date_stamp = self.data.index.min()
    #
    #     historical_data = self.data.loc[:latest_date_stamp]
    #     _logger.debug(f"Historical Data: {historical_data}")
    #
    #     test_data = self.data.loc[latest_date_stamp:]
    #     _logger.debug(f"Test Data: {test_data}")
    #
    #     # Scale x dates by subtracting the min
    #     x = np.array([x - earliest_date_stamp for x in historical_data.index])
    #     x = x.reshape(-1, 1)
    #     prediction_dates = [[x - earliest_date_stamp] for x in test_data.index]
    #     y = historical_data.values
    #     _logger.debug(y)
    #     _logger.debug(x)
    #
    #     # Scaling
    #     _logger.debug("Scaling (y)")
    #     scaler = StandardScaler()
    #     scaler.fit(y)
    #     y = scaler.transform(y)
    #     _logger.debug(y)
    #
    #     params = [
    #         {
    #             "random_state": [98629],
    #             "solver": ["adam"],
    #             "hidden_layer_sizes": [(10, 5), (10, 5, 10), (10, 5, 2)],
    #             "max_iter": [500],
    #             "verbose": [True]
    #         }
    #     ]
    #
    #     model = MLPRegressor()
    #     regressor = GridSearchCV(model, params)
    #
    #     predictions = []
    #     normed_latest_date_stamp = latest_date_stamp - earliest_date_stamp
    #
    #     prediction_dates.pop(0)
    #
    #
    #     for prediction_date in prediction_dates:
    #         if prediction_date[0] == normed_latest_date_stamp:
    #             continue
    #         regressor.fit(x, y)
    #
    #         self.model = regressor
    #
    #         print(regressor.best_params_)
    #
    #         # Single date prediction
    #         # prediction = model.predict([[prediction_date_stamp]])
    #         # _logger.debug(f"{self.data.loc[latest_date_stamp].values} -> {prediction[0]}")
    #         # pred_df = pd.DataFrame([self.data.loc[latest_date_stamp].values, prediction[0]],
    #         #                       columns=self.data.columns, index=[latest_date_stamp, prediction_date_stamp])
    #
    #         prediction = regressor.predict([prediction_date])
    #         predictions.append(prediction[0])
    #         #x = np.append(x, prediction_date, axis=0)
    #         x = np.concatenate([x, [prediction_date]], axis=0)
    #         y = np.concatenate([y, [prediction[0]]], axis=0)
    #
    #     predictions = scaler.inverse_transform(predictions)
    #     _logger.debug(f"Test Data: {test_data}")
    #     _logger.debug(f"Predictions: {predictions}")
    #     unscaled_prediction_dates = [(i[0] + earliest_date_stamp) for i in prediction_dates]
    #     pred_df = pd.DataFrame(predictions, columns=self.data.columns, index=unscaled_prediction_dates)
    #
    #     # Filter by stock
    #     pred_df = pred_df[f"{stock}_close"]
    #     _logger.debug(pred_df)
    #     # Using mean for naïve forecasting
    #     _logger.debug(f"Historical data mean: {historical_data[f'{stock}_close'].mean()}")
    #
    #     return pred_df

    def train_mlp_model(self, x, y, fast=False):
        _logger.debug("Beginning hyperparameter tuning and model training.")
        if fast:
            params = [
                {
                    "random_state": [self.seed],
                    "solver": ["lbfgs"],
                    "alpha": 1.0 ** -np.arange(1, 5),
                    "hidden_layer_sizes": [(5, 10), (20)],
                    "max_iter": [20]
                }
            ]
        else:
            params = [
                {
                    "random_state": [self.seed],
                    "solver": ["lbfgs"],  # ["lbfgs", "adam"],
                    "alpha": 1.0 ** -np.arange(1, 5),
                    "hidden_layer_sizes": [(5, 10), (20), (10, 20), (40, 40), (50, 100), (100, 200), (200, 300), (20, 40, 20), (20, 50, 20), (50, 100, 50), (20, 40, 40, 20), (10, 20, 20, 10), (10, 20, 20, 20, 10)],
                    "max_iter": [10, 20, 50, 100, 200, 500],  # 2000
                }
            ]

        regressor = GridSearchCV(MLPRegressor(), params)
        regressor.fit(x, y)

        print(regressor.best_params_)

        self.predictor.model = regressor

    def deconstruct_date(self, datestamps):
        """
        Split datestamps in a list into 6 integer columns: Year, Month, Day, Day of the week, Week of the year, and Day
        of the year.
        @:param datestamps: A list of datestamps to deconstruct.
        @:returns A DataFrame containing the deconstructed elements of the input datestamps.
        """
        # Convert datestamps into date objects
        dates = [date.fromtimestamp(datestamp) for datestamp in datestamps]

        # Deconstruct each date into it's components
        deconstructed_dates = [
            [
                dateobj.year,
                dateobj.month,
                dateobj.day,
                dateobj.weekday(),
                dateobj.isocalendar()[1], # Week number
                int(dateobj.strftime('%j')) # Day of year
            ]
            for dateobj in dates
        ]

        # Put dates into a DataFrame
        dates = pd.DataFrame(data=deconstructed_dates, index=datestamps, columns=self.datecolumns)

        # Return the result
        return dates

    def _check_data(self):
        if self.data is None:
            _logger.error("No dataset to train model on. Use `set_data()` to specify dataframe first.")
            return False
        elif not isinstance(self.data, pd.DataFrame):
            _logger.error("Dataset is not a dataframe. Please use `set_data()` to recreate the dataframe.")
            return False
        return True

    def test_model(self):
        if self.predictor is None or self.predictor.model is None:
            _logger.warning("No model to test.")

        if self.test_data is not None:
            x_test = self.test_data[self.datecolumns]
            _logger.debug(f"X test data size: {x_test.shape}")
            y_test = self.test_data.drop(self.datecolumns, 1)
            _logger.debug(f"Y test data size: {y_test.shape}")

            columns = y_test.columns
            dates = y_test.index

            if self.predictor.scaler is not None:
                y_test = self.predictor.scaler.transform(y_test)


            y_prediction = self.predictor.model.predict(x_test)


            mse = mean_squared_error(y_test, y_prediction, multioutput='raw_values')
            mae = mean_absolute_error(y_test, y_prediction, multioutput='raw_values')
            evs = explained_variance_score(y_test, y_prediction, multioutput='raw_values')
            r2 = r2_score(y_test, y_prediction, multioutput='raw_values')

            _logger.info(f"Test Scores:\nMSE: {mse}\nMAE: {mae}\nEVS: {evs},\nR2: {r2}")
            return pd.DataFrame(data=[mse, mae, evs, r2], columns=columns, index=["MSE", "MAE", "EVS", "R2"])

    # def calculate_return(self, investments : list):
    #     if self.predictor is None:
    #         # Must create or load a predictor first
    #         _logger.error("No predictor exists.")
    #         return
    #
    #     if self.predictor.model is None:
    #         # Must create or load model first
    #         _logger.error("Predictor is missing a model.")
    #         return
    #
    #     total = 0
    #
    #     for investment in investments:
    #         if not isinstance(investment, tuple):
    #             _logger.error(f"Investment format incorrect - Not a tuple: {investment}")
    #             return
    #         try:
    #             (stock, count, transaction_date) = investment
    #             print(investment)
    #             result = self.get_value(stock, transaction_date) * count
    #             print(f"{stock} investment returned: {result}.")
    #             total += result
    #
    #         except Exception as ex:
    #             _logger.error(ex)
    #
    #     return total

    def get_approximation(self, stock):
        earliest_date = self.data.index.min()
        latest_date = self.data.index.max()
        mean = self.data[f"{stock}_close"].mean()
        approx = pd.DataFrame([mean, mean], index=[earliest_date, latest_date])
        print(approx)
        return approx

class Predictor:

    model = None
    scaler = None
    location = None

    def __init__(self, model=None, scaler=None, location=None):
        self.model = model
        self.scaler = scaler
        self.location = location

    def save(self, model=None, scaler=None, location=None):
        if model is not None:
            _logger.debug(f"Setting model for {self} to {model}.")
            self.model = model

        if scaler is not None:
            _logger.debug(f"Setting scaler for {self} to {scaler}.")
            self.scaler = scaler

        if location is not None:
            _logger.debug(f"Setting location for {self} to {location}.")
            self.location = location

        if self.model is not None and self.location is not None:
            try:
                dump(self, self.location)
                _logger.info(f"Model persisted to {self.location}.")
            except FileNotFoundError as ex:
                _logger.error(f"Persist location for model could not be found: {ex}")
            except Exception as ex:
                _logger.error(ex)

    @staticmethod
    def load(location=None):

        if location is None:
            _logger.error("No location specified.")

        try:
            return load(location)
        except FileNotFoundError as ex:
            _logger.error(f"Location could not be found: {ex}")
        except Exception as ex:
            _logger.error(ex)