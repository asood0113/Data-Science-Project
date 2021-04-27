import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv("hiring.csv")

df

X=df.drop('salary',axis=1)
y=df['salary']
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=150)

#Fitting model with trainig data
regressor.fit(X, y)

pickle.dump(regressor, open('salest2.pkl','wb'))