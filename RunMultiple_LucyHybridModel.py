#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
# Model 1 (trend)
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
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

model_on_trends=[LinearRegression(),ElasticNet(),Lasso(),Ridge()]
model_onresiduals=[ExtraTreesRegressor(),RandomForestRegressor(),
                    KNeighborsRegressor(),MLPRegressor(),XGBRegressor()
                   ]
for model1 in model_on_trends:
    for model2 in model_onresiduals:
        model = LucyHybrid(
            model_1=model1,
            model_2=model2,
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
        print("*"*50)
        hybridname=model1.__class__.__name__+"_"+model2.__class__.__name__
        print(hybridname)
        metric.printmetric()
        LucyUtil.savemodel(model, "models\\"+hybridname+".zip")