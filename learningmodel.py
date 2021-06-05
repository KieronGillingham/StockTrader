# Logging
import logging
_logger = logging.getLogger(__name__)

# General
from datetime import date

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

class LearningModel():

    PERIODS = {
        "NEXTDAY": 86400,  # 60*60*24 (One day)
        "NEXTWEEK": 604800,  # 60*60*24 * 7
        "NEXTTWOWEEK": 1209600, # 60*60*24 * 14
        "NEXTMONTH": 2592000  # 60*60*24 * 30
    }

    def __init__(self, data=None, *args, **kwargs):
        self.data = data

    def predict(self, stock, prediction_period='NEXTWEEK'):

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

        _logger.debug(self.data)

        #return self.linear_model_prediction(stock, prediction_date_stamp)
        return self.mlp_model_prediction(stock, prediction_date_stamp)

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

    def mlp_model_prediction(self, stock, prediction_date_stamp):
        latest_date_stamp = self.data.index.max()

        x = np.array(self.data.index)
        y = self.data.values

        _logger.debug(y)
        _logger.debug(x)

        x = x.reshape(-1, 1)
        # for i in range(0, len(self.prices_df.columns)):
        #     y = self.prices_df.iloc[:, i].values
        model = MLPRegressor(random_state=98628, max_iter=500)
        model.fit(x, y)
        prediction = model.predict([[prediction_date_stamp]])

        _logger.debug(f"{self.data.loc[latest_date_stamp].values} -> {prediction[0]}")

        pred_df = pd.DataFrame([self.data.loc[latest_date_stamp].values, prediction[0]],
                               columns=self.data.columns, index=[latest_date_stamp, prediction_date_stamp])

        _logger.debug(pred_df)

        pred_df = pred_df[f"{stock}_close"]

        _logger.debug(pred_df)

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