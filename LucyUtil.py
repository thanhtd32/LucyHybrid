#Authors: Duy Thanh Tran, Prof. Jun-Ho Huh, Prof. Jae-Hwan Kim
#Data Science Lab - KMOU
#Department of Data Science, (National) Korea Maritime and Ocean University, Busan 49112, Republic of Korea.
#Created Date: May-17/2022
import pickle
class LucyUtil:
    @staticmethod
    def savemodel(model,filename):
        try:
            pickle.dump(model, open(filename, 'wb'))
            return True
        except:
            print("An exception occurred")
            return False
    @staticmethod
    def loadmodel(filename):
        try:
            model=pickle.load(open(filename, 'rb'))
            return model
        except:
            print("An exception occurred")
            return None