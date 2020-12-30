# General
import sys, datetime

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class LearningModel():
    def __init__(self, *args, **kwargs):
        print("")

    def predict(self, until):
        predictions = []

        x = self.prices_df.index.values
        for i in range(0, len(x)):
            x[i] = datetime.date.fromisoformat(x[i]).toordinal()

        x = x.reshape(-1, 1)
        for i in range(0, len(self.prices_df.columns)):
            y = self.prices_df.iloc[:, i].values
            model = LinearRegression()
            model.fit(x, y)
            prediction = model.predict([[until]])
            predictions.append(prediction[0])

        pred_df = pd.DataFrame(np.reshape(predictions, (1, -1)), columns=self.prices_df.columns, index=[737424])
        pred_df = pred_df.append(self.prices_df.tail(1))