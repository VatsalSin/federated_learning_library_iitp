from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from federatedlearningiitp import *


def cleanData(df):
    return df


def preprocessing(df):
    df['Title'] = df.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace(['Mme', 'Lady', 'Ms'], 'Mrs')
    df.Title.loc[(df.Title != 'Master') & (df.Title != 'Mr') & (df.Title != 'Miss')
                 & (df.Title != 'Mrs')] = 'Others'
    df = pd.concat([df, pd.get_dummies(df['Title'])],
                   axis=1).drop(labels=['Name'], axis=1)

    # map the two genders to 0 and 1
    df.Sex = df.Sex.map({'male': 0, 'female': 1})

    # create a new feature "Family"
    df['Family'] = df['SibSp'] + df['Parch'] + 1

    df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

    df.Ticket = df.Ticket.map(lambda x: x[0])

    guess_Fare = df.Fare.loc[(df.Ticket == '3') & (
        df.Pclass == 3) & (df.Embarked == 'S')].median()
    df.Fare.fillna(guess_Fare, inplace=True)

    # inspect the mean Fare values for people who died and survived
    df[['Fare', 'Survived']].groupby(['Survived'], as_index=False).mean()

    # bin Fare into five intervals with equal amount of people
    df['Fare-bin'] = pd.qcut(df.Fare, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # notice that instead of using Title, we should use its corresponding dummy variables
    df_sub = df[['Age', 'Master', 'Miss', 'Mr',
                 'Mrs', 'Others', 'Fare-bin', 'SibSp']]

    X_train = df_sub.dropna().drop('Age', axis=1)
    y_train = df['Age'].dropna()
    X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

    regressor = RandomForestRegressor(n_estimators=300)
    regressor.fit(X_train, y_train)
    y_pred = np.round(regressor.predict(X_test), 1)
    df.Age.loc[df.Age.isnull()] = y_pred

    bins = [0, 4, 12, 18, 30, 50, 65, 100]  # This is somewhat arbitrary...
    age_index = (1, 2, 3, 4, 5, 6, 7)
    # ('baby','child','teenager','young','mid-age','over-50','senior')
    df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)

    df['Ticket'] = df['Ticket'].replace(
        ['A', 'W', 'F', 'L', '5', '6', '7', '8', '9'], '4')

    df = df.drop(labels=['Cabin'], axis=1)

    # fill the NAN
    df.Embarked.fillna('S', inplace=True)

    df = df.drop(labels='Embarked', axis=1)

    # dummy encoding
    df = pd.get_dummies(df, columns=['Ticket'])

    df = df.drop(labels=['SibSp','Parch','Age','Fare','Title','PassengerId'], axis=1)
    return df.copy()


def get_model():
    model = Sequential()

    # layers
    model.add(Dense(units=9, kernel_initializer='uniform',
                    activation='relu', input_dim=17))
    model.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


dataset = pd.read_csv('train.csv')
X = dataset
y = dataset.iloc[:, 1]
X = preprocessing(X).iloc[:, 1:18]
# X.apply(preprocessing)
createServer("", 5000, 5001, 10, 2, 2, 70, X, y,
             get_model, preprocessing, cleanData)
