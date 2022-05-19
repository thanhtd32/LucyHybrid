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
    def dotrend(self,dataset,features,target,polynomial_order,chart_title,showchart=True):
        self.dp = DeterministicProcess(
        index=dataset[features].index, # dates from the training data
        constant=True,# dummy feature for the bias (y_intercept)
        order=polynomial_order,# the time dummy (trend)
        drop=True,# drop terms if necessary to avoid collinearity
        )
        # `in_sample` creates features for the dates given in the `index` argument
        self.X = self.dp.in_sample()
        self.y = dataset[target]  # the target

        # The intercept is the same as the `const` feature from
        # DeterministicProcess. LinearRegression behaves badly with duplicated
        # features, so we need to be sure to exclude it here.
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X, self.y)

        self.y_pred = pd.Series(self.model.predict(self.X), index=self.X.index)
        if showchart:
            ax = dataset[features].plot(style=".", color="0.5", title=chart_title)
            _ = self.y_pred.plot(ax=ax, linewidth=3, label="Trend")
    def dotrendforecast(self,dataset,features,target,polynomial_order,chart_title,date,step=30,showchart=True):
        self.dotrend(dataset,features,target,polynomial_order,chart_title,False)

        self.X = self.dp.out_of_sample(steps=step)

        self.y_fore = pd.Series(self.model.predict(self.X), index=self.X.index)
        if showchart:
            ax = dataset[target][date:].plot(title=chart_title, **plot_params)
            ax = self.y_pred[date:].plot(ax=ax, linewidth=3, label="Trend")
            ax = self.y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
            _ = ax.legend()