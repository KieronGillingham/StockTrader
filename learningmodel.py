# General
import sys
from datetime import date, timedelta

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class LearningModel():

    PERIODS = {
        "NEXTDAY": 86400,  # 60*60*24 (One day)
        "NEXTWEEK": 604800,  # 60*60*24 * 7
        "NEXTTWOWEEK": 1209600, # 60*60*24 * 14
        "NEXTMONTH": 2592000  # 60*60*24 * 30
    }

    def __init__(self, data=None, *args, **kwargs):
        self.data = data

    def predict(self, prediction_period='NEXTWEEK'):
        predictions = []

        if prediction_period not in self.PERIODS:
            print(f"Prediction period '{prediction_period}' not recognised. Must be one of: {[*self.PERIODS]}")
            return
        elif not self._check_data():
            print("Error with prediction model.")
            return

        latest_date_stamp = self.data.index.max()
        prediction_date_stamp = latest_date_stamp + self.PERIODS[prediction_period]
        print(f"Latest date: {latest_date_stamp} / {date.fromtimestamp(latest_date_stamp)}")
        print(f"Prediction date ({prediction_period}): {prediction_date_stamp} / {date.fromtimestamp(prediction_date_stamp)}")

        y = self.data["TYT.L_close"].values
        x = np.array(self.data.index)
        print(y)
        print(x)

        x = x.reshape(-1, 1)
        #for i in range(0, len(self.prices_df.columns)):
        #     y = self.prices_df.iloc[:, i].values
        model = LinearRegression()
        model.fit(x, y)
        prediction = model.predict([[latest_date_stamp]])

        print(f"{self.data['TYT.L_close'][latest_date_stamp]} -> {prediction[0]}")

        pred_df = pd.DataFrame([self.data["TYT.L_close"][latest_date_stamp],prediction[0]], columns=["TYT.L_close"], index=[latest_date_stamp, prediction_date_stamp])

        return pred_df


    def set_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
            return True
        else:
            return False

    def _check_data(self):
        if self.data is None:
            print("No dataset to train model on. Use `set_data()` to specify dataframe first.")
            return False
        elif not isinstance(self.data, pd.DataFrame):
            print("Dataset is not a dataframe. Please use `set_data()` to recreate the dataframe.")
            return False
        else:
            return True