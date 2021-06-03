# General
import sys
from datetime import date, timedelta

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class LearningModel():
    def __init__(self, data=None, *args, **kwargs):
        self.data = data

    def predict(self, prediction_time='NEXTDAY'):
        predictions = []

        if not self._check_data():
            print("Error with prediction model.")
            return

        latest_date_stamp = self.data.index.max()
        # latest_date = date.fromtimestamp(latest_date_stamp)
        # prediction_date = latest_date + timedelta(days=1)
        prediction_date_stamp = latest_date_stamp + (60*60*24)
        print(f"Latest date: {latest_date_stamp} / {date.fromtimestamp(latest_date_stamp)}")
        print(f"Prediction date ({prediction_time}): {prediction_date_stamp} / {date.fromtimestamp(prediction_date_stamp)}")

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
        #     predictions.append(prediction[0])
        #
        # pred_df = pd.DataFrame(np.reshape(predictions, (1, -1)), columns=self.prices_df.columns, index=[737424])
        # pred_df = pred_df.append(self.prices_df.tail(1))

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