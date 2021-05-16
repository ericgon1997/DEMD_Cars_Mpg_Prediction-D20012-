import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

car_df=pd.read_csv('auto-mpg.csv')

car_df['hp']=car_df['hp'].replace('?',np.nan)

car_df['hp']=car_df['hp'].astype('float64')

car_df['hp']=car_df['hp'].fillna(car_df['hp'].median())

car_df=car_df.drop('accel',axis=1)
car_df.head()

car_df=car_df.drop('name',axis=1)
car_df.head()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics

y=car_df['mpg']
x=car_df.iloc[:,1:]

kf=KFold(n_splits=5,shuffle=True,random_state=2)
rmse=[]


for train,test in kf.split(x,y):
    LR=LinearRegression()
    #print(train)
    xtrain=x.iloc[train]
    xtest=x.iloc[test]
    ytrain=y.iloc[train]
    ytest=y.iloc[test]
    LR.fit(xtrain,ytrain)
    ypredict=LR.predict(xtest)
    
import pickle
pkl_file = open("car_mileage.pkl","wb")
pickle.dump(LR,pkl_file)
pkl_file.close()