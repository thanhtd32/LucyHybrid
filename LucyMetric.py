#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
class LucyMetric:
    def __init__(self,rmse_train,rmse_test,mse_train,mse_test,mae_train,mae_test):
        self.rmse_train=rmse_train
        self.rmse_test=rmse_test
        self.mse_train=mse_train
        self.mse_test=mse_test
        self.mae_train=mae_train
        self.mae_test=mae_test
    def printmetric(self):
        print((f"Train RMSE: {self.rmse_train:.2f}\n" f"Test RMSE: {self.rmse_test:.2f}"))
        print((f"Train MSE: {self.mse_train:.2f}\n" f"Test MSE: {self.mse_test:.2f}"))
        print((f"Train MAE: {self.mae_train:.2f}\n" f"Test MAE: {self.mae_test:.2f}"))