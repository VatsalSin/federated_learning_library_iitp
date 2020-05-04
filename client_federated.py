import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
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
X = dataset.iloc[:, 2:19].values
y = dataset.iloc[:, 1].values
X = preprocessing(X)
startFederatedLearning(host, port, pport, weights, getModel, X, y,200,batchsize=32)



