# Logging
import logging
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

# Persistance
from joblib import dump, load

class LearningModel():

    periods = {
        "NEXTDAY": 86400,  # 60*60*24 (One day)
        "NEXTWEEK": 604800,  # 60*60*24 * 7
        "NEXTTWOWEEK": 1209600, # 60*60*24 * 14
        "NEXTMONTH": 2592000  # 60*60*24 * 30
    }

    # Seed for repeatability of random number generation
    seed = 98628

    # Data to train models with
    data = None

    # Current trained predictor
    predictor = None

    # Column names for deconstructed dates.
    datecolumns = ['year', 'month', 'day', 'weekday', 'week', 'dayofyear']

    def __init__(self, data=None, *args, **kwargs):
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        if isinstance(data, pd.DataFrame):
            if not {"year", "month", "day", "weekday", "week", "dayofyear"}.issubset(data.columns):
                print(data.shape)
                data = data.join(self.deconstruct_date(data.index))
                print(data.shape)
                print(data)
            self.data = data
            return True
        else:
            return False

    def train_model(self, persist_location=None, *args, **kwargs):
        if self.predictor is None:
            self.predictor = Predictor()

        try:
            self.train_mlp_model(self.data)
        except Exception as ex:
            _logger.error(ex)

        if self.predictor.model is not None:
            _logger.debug("Model trained.")
            if persist_location is not None:
                try:
                    self.predictor.save(location=persist_location)
                    _logger.info(f"Model persisted to {persist_location}.")
                except FileNotFoundError as ex:
                    _logger.error(f"Persist location for model could not be found: {ex}")
                except Exception as ex:
                    _logger.error(ex)
        else:
            _logger.error("Problem training model.")

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

    def get_predictions(self, stock : str, prediction_period='NEXTMONTH'):

        _logger.debug(f"Loaded data shape: {self.data.shape}")

        if not isinstance(stock, str):
            raise TypeError("Stock must be a string of a stock symbol: I.E. GOOG")
        elif prediction_period not in self.periods:
            _logger.error(f"Prediction period '{prediction_period}' not recognised. Must be one of: {[*self.periods]}")
            return
        elif not self._check_data():
            _logger.error("Error with prediction model.")
            return

        latest_date_stamp = self.data.index.max()
        prediction_date_stamp = latest_date_stamp + self.periods[prediction_period]
        _logger.debug(f"Latest date: {latest_date_stamp} / {date.fromtimestamp(latest_date_stamp)}")
        _logger.debug(f"Prediction date ({prediction_period}): {prediction_date_stamp} / {date.fromtimestamp(prediction_date_stamp)}")

        prediction_date_stamp = latest_date_stamp
        latest_date_stamp = prediction_date_stamp - self.periods[prediction_period]
        if self.predictor.model is None:
            self.train_mlp_model()
        dates = [(latest_date_stamp + (i * 86400)) for i in range(0, 35)]
        predictions = self.predictor.model.predict(self.deconstruct_date(dates))
        if self.predictor.scaler is not None:
            _logger.info("Unscaling prediction values.")
            predictions = self.predictor.scaler.inverse_transform(predictions)
        print(predictions)

        return pd.DataFrame(data=predictions, index=dates, columns=self.data.columns[:-6])[f"{stock}_close"]

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
            if self.predictor.model is None:
                self.train_mlp_model()
            predictions = self.predictor.model.predict(self.deconstruct_date([prediction_date_stamp]))
            predictions = self.predictor.scaler.inverse_transform(predictions)
            print(predictions)
            if len(predictions[0]) != (len(self.data.columns) - 6):
                _logger.error("Wrong length predictions.")
            prediction = pd.DataFrame(data=predictions, index=[prediction_date_stamp], columns=self.data.columns[:-6])
            return prediction[f"{stock}_close"].values[0]
    # def linear_model_prediction(self, stock, prediction_date_stamp):
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
    #     model = LinearRegression()
    #     model.fit(x, y)
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

    def train_mlp_model(self, data=None):
        """
        Train a Multilayer Perceptron model to forecast stock prices. The model is
        :param data: The dataset to be used for training.
        :return: None.
        """
        # Use object dataset if no alternative provided.
        if data is None:
            data = self.data

        # Stop if no dataset is available for training.
        if data is None:
            _logger.error("No dataset available for training.")
            return

        # Sort the dataset chronologically.
        data.sort_index(inplace=True)

        # Get datestamps earliest and latest records from the data.
        earliest_datestamp = data.index.min()
        latest_datestamp = data.index.max()
        _logger.debug(f"Preparing to train model on data between {date.fromtimestamp(earliest_datestamp)} and "
                      f"{date.fromtimestamp(latest_datestamp)}.")

        # Get independent variables for model input (dates)
        x = data[self.datecolumns]
        _logger.debug(f"X data size: {x.shape}")

        # Get dependent variables for model output (stock prices)
        y = data.drop(self.datecolumns, 1)
        # y = y["TYT.L_close"]
        _logger.debug(f"Y data size: {y.shape}")

        _logger.debug(x)
        _logger.debug(y)

        # Create data scaler - Keep reference for later unscaling
        self.predictor.scaler = StandardScaler()

        # Scale y
        self.predictor.scaler.fit(y)
        y = self.predictor.scaler.transform(y)
        
        _logger.debug("Beginning hyperparameter tuning and model training.")
        params = [
            {
                "random_state": [self.seed],
                "solver": ["lbfgs"],#["lbfgs", "adam"],
                "hidden_layer_sizes": [(200, 500)],

                #"hidden_layer_sizes": [(2), (5), (5, 5), (5, 10), (20)],#, (10, 10), (10, 20), (40, 40), (50, 100), (100, 200), (200, 300), (20, 40, 20), (20, 50, 20), (50, 100, 50), (20, 40, 40, 20), (10, 20, 20, 10), (10, 20, 20, 20, 10)],
                "max_iter": [1000],
                "verbose": [True]
            }
        ]

        regressor = GridSearchCV(MLPRegressor(), params)
        regressor.fit(x, y)

        print(regressor.best_params_)

        self.predictor.model = regressor

    def deconstruct_date(self, datestamps):

        dates = [date.fromtimestamp(datestamp) for datestamp in datestamps]

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
        print(deconstructed_dates)

        dates = pd.DataFrame(data=deconstructed_dates, index=datestamps, columns=self.datecolumns)
        return dates

        # print(pd.to_datetime(input_dates, format="%d/%m/%Y"))
        # print(dates)

        # dates['Date'] = pd.to_datetime(input_dates)
        # dates['Date'] = dates['Date'].dt.strftime('%d.%m.%Y')
        # dates['year'] = pd.DatetimeIndex(dates['Date']).year
        # dates['month'] = pd.DatetimeIndex(dates['Date']).month
        # dates['day'] = pd.DatetimeIndex(dates['Date']).day
        # dates['dayofyear'] = pd.DatetimeIndex(dates['Date']).dayofyear
        # dates['weekofyear'] = pd.DatetimeIndex(dates['Date']).weekofyear
        # dates['weekday'] = pd.DatetimeIndex(dates['Date']).weekday
        # dates['quarter'] = pd.DatetimeIndex(dates['Date']).quarter
        # dates['is_month_start'] = pd.DatetimeIndex(dates['Date']).is_month_start
        # dates['is_month_end'] = pd.DatetimeIndex(dates['Date']).is_month_end
        #
        # print(dates)

        #input_date = date.fromtimestamp(input_date)
        #return [
        #    input_date.year,
        #    input_date.month,
        #    input_date.day,
        #    input_date.weekday()
        #]

    def _check_data(self):
        if self.data is None:
            _logger.error("No dataset to train model on. Use `set_data()` to specify dataframe first.")
            return False
        elif not isinstance(self.data, pd.DataFrame):
            _logger.error("Dataset is not a dataframe. Please use `set_data()` to recreate the dataframe.")
            return False
        return True
    #
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