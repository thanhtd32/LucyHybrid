#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-19/2022
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
import pandas as pd

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)


class LucyTrendForecast:
    def do_moving_average(self, dataset, columns, chart_title, days=365, showchart=True):
        self.moving_average = dataset[columns].rolling(
            window=days,  # 365-day window
            center=True,  # puts the average at the center of the window
            min_periods=183,  # choose about half the window size
        ).mean()  # compute the mean (could also do median, std, min, max, ...)
        if showchart:
            ax = self.moving_average.plot(title=chart_title, **plot_params, alpha=0.5, ylabel="items sold", linewidth=2)

    def do_trend(self, dataset, columns, chart_title, days=365, showchart=True):
        self.do_moving_average(dataset, columns, chart_title, days, False)
        dp = DeterministicProcess(
            index=self.moving_average.index,  # dates from the training data
            constant=True,  # dummy feature for the bias (y_intercept)
            order=3,  # the time dummy (trend)
            drop=True,  # drop terms if necessary to avoid collinearity
        )
        # `in_sample` creates features for the dates given in the `index` argument
        X = dp.in_sample()

        y = self.moving_average  # the target

        # The intercept is the same as the `const` feature from
        # DeterministicProcess. LinearRegression behaves badly with duplicated
        # features, so we need to be sure to exclude it here.
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        y_pred = pd.Series(model.predict(X), index=X.index)

        if showchart:
            ax = y.plot(**plot_params, alpha=0.5, ylabel="items sold", linewidth=2, title=chart_title)
            _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

    def do_trend_forecast(self, dataset, columns, chart_title, days=365, showchart=True):
        self.do_moving_average(dataset, columns, chart_title, days, False)
        y = self.moving_average

        # Instantiate `DeterministicProcess` with arguments appropriate for a cubic trend model
        dp = DeterministicProcess(index=y.index, order=3)

        # Create the feature set for the dates given in y.index
        X = dp.in_sample()

        # Create features for a 90-day forecast.
        X_fore = dp.out_of_sample(steps=90)

        # we can see the a plot of the result:
        model = LinearRegression()
        model.fit(X, y)

        y_pred = pd.Series(model.predict(X), index=X.index)
        y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
        if showchart:
            ax = y.plot(**plot_params, alpha=0.5, title=chart_title, ylabel="items sold")
            ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
            ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
            ax.legend()