# Logging
import logging

from sklearn.model_selection import GridSearchCV

_logger = logging.getLogger(__name__)

# General
from datetime import date

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class LearningModel():

    PERIODS = {
        "NEXTDAY": 86400,  # 60*60*24 (One day)
        "NEXTWEEK": 604800,  # 60*60*24 * 7
        "NEXTTWOWEEK": 1209600, # 60*60*24 * 14
        "NEXTMONTH": 2592000  # 60*60*24 * 30
    }

    def __init__(self, data=None, *args, **kwargs):
        self.data = data

    def predict(self, stock, prediction_period='NEXTMONTH'):

        _logger.debug(self.data)

        if not isinstance(stock, str):
            raise TypeError("Stock must be a string of a stock symbol: I.E. GOOG")
        elif prediction_period not in self.PERIODS:
            _logger.error(f"Prediction period '{prediction_period}' not recognised. Must be one of: {[*self.PERIODS]}")
            return
        elif not self._check_data():
            _logger.error("Error with prediction model.")
            return

        latest_date_stamp = self.data.index.max()
        prediction_date_stamp = latest_date_stamp + self.PERIODS[prediction_period]
        _logger.debug(f"Latest date: {latest_date_stamp} / {date.fromtimestamp(latest_date_stamp)}")
        _logger.debug(f"Prediction date ({prediction_period}): {prediction_date_stamp} / {date.fromtimestamp(prediction_date_stamp)}")



        #return self.linear_model_prediction(stock, prediction_date_stamp)
        #return self.mlp_model_prediction(stock, prediction_date_stamp)

        prediction_date_stamp = latest_date_stamp
        latest_date_stamp = prediction_date_stamp - self.PERIODS[prediction_period]
        return self.mlp_model_prediction(stock, prediction_date_stamp, latest_date_stamp)

    def linear_model_prediction(self, stock, prediction_date_stamp):
        latest_date_stamp = self.data.index.max()

        y = self.data[f"{stock}_close"].values
        x = np.array(self.data.index)
        _logger.debug(y)
        _logger.debug(x)

        x = x.reshape(-1, 1)
        #for i in range(0, len(self.prices_df.columns)):
        #     y = self.prices_df.iloc[:, i].values
        model = LinearRegression()
        model.fit(x, y)
        prediction = model.predict([[prediction_date_stamp]])

        _logger.debug(f"{self.data[f'{stock}_close'][latest_date_stamp]} -> {prediction[0]}")

        pred_df = pd.DataFrame([self.data[f"{stock}_close"][latest_date_stamp],prediction[0]], columns=[f"{stock}_close"], index=[latest_date_stamp, prediction_date_stamp])

        return pred_df

    def mlp_model_prediction(self, stock, prediction_date_stamp, latest_date_stamp=None):

        print(latest_date_stamp)
        print(prediction_date_stamp)

        self.data.sort_index(inplace=True)

        if latest_date_stamp is None:
            latest_date_stamp = self.data.index.max()

        earliest_date_stamp = self.data.index.min()

        historical_data = self.data.loc[:latest_date_stamp]
        _logger.debug(f"Historical Data: {historical_data}")

        test_data = self.data.loc[latest_date_stamp:]
        _logger.debug(f"Test Data: {test_data}")

        # Scale x dates by subtracting the min
        x = np.array([x - earliest_date_stamp for x in historical_data.index])
        x = x.reshape(-1, 1)
        prediction_dates = [[x - earliest_date_stamp] for x in test_data.index]
        y = historical_data.values
        _logger.debug(y)
        _logger.debug(x)

        # Scaling
        _logger.debug("Scaling (y)")
        scaler = StandardScaler()
        scaler.fit(y)
        y = scaler.transform(y)
        _logger.debug(y)

        params = [
            {
                "random_state": [98629],
                "solver": ["adam"],
                "hidden_layer_sizes": [(10, 5), (10, 5, 10), (10, 5, 2)],
                "max_iter": [500],
                "verbose": [True]
            }
        ]

        model = MLPRegressor()
        regressor = GridSearchCV(model, params)

        predictions = []
        normed_latest_date_stamp = latest_date_stamp - earliest_date_stamp

        prediction_dates.pop(0)


        for prediction_date in prediction_dates:
            if prediction_date[0] == normed_latest_date_stamp:
                continue
            regressor.fit(x, y)
            print(regressor.best_params_)

            # Single date prediction
            # prediction = model.predict([[prediction_date_stamp]])
            # _logger.debug(f"{self.data.loc[latest_date_stamp].values} -> {prediction[0]}")
            # pred_df = pd.DataFrame([self.data.loc[latest_date_stamp].values, prediction[0]],
            #                       columns=self.data.columns, index=[latest_date_stamp, prediction_date_stamp])

            prediction = regressor.predict([prediction_date])
            predictions.append(prediction[0])
            #x = np.append(x, prediction_date, axis=0)
            x = np.concatenate([x, [prediction_date]], axis=0)
            y = np.concatenate([y, [prediction[0]]], axis=0)

        predictions = scaler.inverse_transform(predictions)
        _logger.debug(f"Test Data: {test_data}")
        _logger.debug(f"Predictions: {predictions}")
        unscaled_prediction_dates = [(i[0] + earliest_date_stamp) for i in prediction_dates]
        pred_df = pd.DataFrame(predictions, columns=self.data.columns, index=unscaled_prediction_dates)

        # Filter by stock
        pred_df = pred_df[f"{stock}_close"]
        _logger.debug(pred_df)
        # Using mean for naïve forecasting
        _logger.debug(f"Historical data mean: {historical_data[f'{stock}_close'].mean()}")

        return pred_df


    def set_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
            return True
        else:
            return False

    def _check_data(self):
        if self.data is None:
            _logger.error("No dataset to train model on. Use `set_data()` to specify dataframe first.")
            return False
        elif not isinstance(self.data, pd.DataFrame):
            _logger.error("Dataset is not a dataframe. Please use `set_data()` to recreate the dataframe.")
            return False
        else:
            return True