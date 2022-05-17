#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from LucyMetric import LucyMetric


class LucyHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)

        y_fit = pd.DataFrame(self.model_1.predict(X_1),
                             index=X_1.index, columns=y.columns)
        # compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze()

        # fit self.model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            # predict with self.model_1
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns)
        y_pred = y_pred.stack().squeeze()  # wide to long

        # add self.model_2 predictions to y_pred
        y_pred += self.model_2.predict(X_2)
        return y_pred.unstack()  # long to wide

    def evaluate(self, y_train, y_fit, y_valid, y_pred):
        train_rmse = mean_squared_error(y_train, y_fit, squared=False)
        test_rmse = mean_squared_error(y_valid, y_pred, squared=False)

        train_mse = mean_squared_error(y_train, y_fit, squared=True)
        test_mse = mean_squared_error(y_valid, y_pred, squared=True)

        train_mae = mean_absolute_error(y_train, y_fit)
        test_mae = mean_absolute_error(y_valid, y_pred)

        metric = LucyMetric(train_rmse, test_rmse, train_mse, test_mse, train_mae, test_mae)
        return metric
