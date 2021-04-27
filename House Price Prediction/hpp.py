import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv("USA_Housing.csv")
df=df.drop('Address',axis=1)
X=df.drop('Price',axis=1)
y=df['Price']

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('housepp.pkl','wb'))