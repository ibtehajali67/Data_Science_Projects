import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pickle
class rock_predict:
    def __init__(self):
        self.df=pd.read_csv("dataset/sonar_data.csv", header=None)
        self.x=self.df.drop(columns=60,axis=1)
        self.y=self.df[60]
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.1,stratify=self.y,random_state=1)
        self.model= LogisticRegression()
        self.pkl_filename = "pickle_model.pkl"
        if not os.path.isfile(self.pkl_filename):
            self.train()
        with open(self.pkl_filename, 'rb') as file:
            print("file already exist")
            self.pickle_model = pickle.load(file)
    def save(self): 
    #if model is not exist then run 2 lines below:
        if not os.path.isfile(self.pkl_filename):
            with open(self.pkl_filename, 'wb') as file:
                pickle.dump(self.model, file)
    def train(self):
        self.model.fit(self.x_train,self.y_train)
        x_train_predict=self.model.predict(self.x_train)
        tr_accuracy=accuracy_score(x_train_predict,self.y_train)
        print("train accuracy==>", tr_accuracy)
        self.save()
    def test(self): 
#        without load prediction  
#        x_test_predict=self.model.predict(self.x_test)
#        te_accuracy=accuracy_score(x_test_predict,y_test)
#        print("test accuracy==>", te_accuracy)
         
        # Calculate the accuracy score and predict target values
        score = self.pickle_model.score(x_test, y_test)
        print("Test score: {0:.2f} %".format(100 * score))
        Ypredict = pickle_model.predict(x_test)
    def prediction(self,input_data):
        input_data_As_numpy= np.asarray(input_data)
        in_reshap=input_data_As_numpy.reshape(1,-1)
        prediction=self.pickle_model.predict(in_reshap)
        print("predicted value==>", prediction)
        return prediction


if __name__ == "__main__":
    o=rock_predict()
    input_data=(0.0762,0.0666,0.0481,0.0394,0.0590,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.5730,0.5399,0.3161,0.2285,0.6995,1.0000,0.7262,0.4724,
0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.2430,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.0230,0.0046,0.0156,0.0031,0.0054,0.0105,0.0110,0.0015,0.0072,0.0048,0.0107,0.0094)
    o.prediction(input_data)

