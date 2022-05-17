#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
# Model 1 (trend)
from sklearn.linear_model import LinearRegression

# Model 2
from sklearn.neighbors import KNeighborsRegressor

# Create LinearRegression and KNeighborsRegressor hybrid for LucyHybrid object
from LucyDataExecutor import LucyDataExecutor
from LucyHybrid import LucyHybrid
from LucyUtil import LucyUtil

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

dataexec=LucyDataExecutor()
dataexec.load_data()
dataexec.normalize(2017)
dataexec.setup_feature_target()

model = LucyHybrid(
    model_1=LinearRegression(),
    model_2=KNeighborsRegressor(),
)
traindate="2017-07-01"
validdate="2017-07-02"
y_train, y_valid = dataexec.y_train_valid(traindate, validdate)
X1_train, X1_valid = dataexec.X1_train_valid(traindate,validdate)
X2_train, X2_valid = dataexec.X2_train_valid(traindate,validdate)
#call fit method
model.fit(X1_train, X2_train, y_train)
#call predict method
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

metric=model.evaluate(y_train,y_fit,y_valid,y_pred)
metric.printmetric()

LucyUtil.savemodel(model,"lucymodel.zip")

#test with 6 features in the dataset
families = dataexec.y.columns[0:10]
axs = dataexec.y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(10, 20), **plot_params, alpha=0.5,
)
#filter and plot the y_fit
y_fit.loc(axis=1)[families].plot(subplots=True, color='C0', ax=axs)
#filter and plot the y_pred
y_pred.loc(axis=1)[families].plot(subplots=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)