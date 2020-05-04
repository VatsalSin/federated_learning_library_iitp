import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from federatedlearningiitp import *



host = ""
port = 5000
pport = 5001
preprocessing = requestPreprocessing(host, port)
weights = requestWeights(host, port)
getModel = requestModelGen(host, port)
dataCleaner = requestDataCleaner(host,port)
dataset = dataCleaner(pd.read_csv('train.csv'))
X = dataset
y = dataset.iloc[:, 1]
X = preprocessing(X).iloc[:, 1:18].values # Selecting only input features
startFederatedLearning(host, port, pport, weights, getModel, X, y,200,batchsize=32)



