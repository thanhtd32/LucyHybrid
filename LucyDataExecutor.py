#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess


class LucyDataExecutor:
    dataset = 'train.csv'

    def load_data(self, datasetpath=dataset):
        self.rawdata = pd.read_csv(
            datasetpath,
            usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
            dtype={
                'store_nbr': 'category',
                'family': 'category',
                'sales': 'float32',
            },
            parse_dates=['date'],
            infer_datetime_format=True,
        )

    def normalize(self, time):
        self.rawdata['date'] = self.rawdata.date.dt.to_period('D')
        self.rawdata = self.rawdata.set_index(['store_nbr', 'family', 'date']).sort_index()

        self.normalize_data = (
            self.rawdata
                .groupby(['family', 'date'])
                .mean()
                .unstack('family')
                .loc[str(time)]
        )

    def setup_feature_target(self):
        # Target series
        self.y = self.normalize_data.loc[:, 'sales']

        # X_1: Features for Model 1
        dp = DeterministicProcess(index=self.y.index, order=1)
        self.X_1 = dp.in_sample()

        # X_2: Features for Model 2
        # onpromotion feature
        self.X_2 = self.normalize_data.drop('sales', axis=1).stack()

        # Label encoding for 'family'
        le = LabelEncoder()
        self.X_2 = self.X_2.reset_index('family')
        self.X_2['family'] = le.fit_transform(self.X_2['family'])

        # Label encoding for seasonality
        self.X_2["day"] = self.X_2.index.day  # values are day of the month

    def y_train_valid(self, timetrain, timevalid):
        return self.y[:timetrain], self.y[timevalid:]

    def X1_train_valid(self, timetrain, timevalid):
        return self.X_1[: timetrain], self.X_1[timevalid:]

    def X2_train_valid(self, timetrain, timevalid):
        return self.X_2.loc[:timetrain], self.X_2.loc[timevalid:]