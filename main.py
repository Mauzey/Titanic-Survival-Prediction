# import modules
from utils import preprocess, featureEngineer, transformData

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# import and store datasets to dataframes
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# assign names to the dataframes
train_data.name = "Training Dataset"
test_data.name = "Testing Dataset"

# preprocess the data and feature engineer before transforming the data
train_data, test_data = preprocess(train_data, test_data)
train_data, test_data = featureEngineer(train_data, test_data)
train_data, test_data = transformData(train_data, test_data)

# specify training/testing columns
train_columns = ['Title', 'Cabin', 'Deck', 'Embarked', 'FamilyName', 'FamilySize', 'GroupedFamilySize', 'Survived', 'Name',
               'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'TicketSurvivalRate',
               'FamilySurvivalRate', 'TicketSurvivalRateNA', 'FamilySurvivalRateNA']

test_columns = ['Title', 'Cabin', 'Deck', 'Embarked', 'FamilyName', 'FamilySize', 'GroupedFamilySize', 'Name',
               'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'TicketSurvivalRate',
               'FamilySurvivalRate', 'TicketSurvivalRateNA', 'FamilySurvivalRateNA']

# prepare training/testing data
X_train = StandardScaler().fit_transform(train_data.drop(columns=train_columns))
y_train = train_data['Survived'].values
X_test = StandardScaler().fit_transform(test_data.drop(columns=test_columns))

# initialise model
model = RandomForestClassifier(criterion='gini',
                              n_estimators=1100,
                              max_depth=5,
                              min_samples_split=4,
                              min_samples_leaf=5,
                              max_features='auto',
                              oob_score=True,
                              random_state=42,
                              n_jobs=-1,
                              verbose=1)

# fit model
model.fit(X_train, y_train)

# predict test data
predictions = model.predict(X_test).astype(int)

# save output
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)