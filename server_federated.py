from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from federatedlearningiitp import *



def cleanData(df):
	return df

def preprocessing(X):
	labelencoder_X_1 = LabelEncoder()
	X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
	labelencoder_X_2 = LabelEncoder()
	X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
	onehotencoder = OneHotEncoder(categorical_features = [1])
	X = onehotencoder.fit_transform(X).toarray()
	X = X[:, 1:]
	sc = StandardScaler()
	X = sc.fit_transform(X)
	return X

def get_model():
	model = Sequential()
	model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
	model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
	model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return model

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
X = preprocessing(X)
createServer("", 5000, 5001, 10, 2, 2, 70, X, y, get_model, preprocessing, cleanData)