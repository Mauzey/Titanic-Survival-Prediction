# import modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from preprocessing import preprocess
from preprocessing import featureEngineer

import pandas as pd
import numpy as np

# import training and testing data
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# use test data (for Kaggle submission), or validation data (for development)
USE_TEST_DATA = False

# select the prediction target
y = train_data['Survived']

# pre-process data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

#print(train_data.isnull().sum())

# feature engineer data
train_data = featureEngineer(train_data)
test_data = featureEngineer(test_data)

# select features on which to train the model
features = ['Pclass', 'Sex', 'FamilySize', 'Age']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# split training data into training and validation data
if not USE_TEST_DATA:
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# define and fit the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

if not USE_TEST_DATA:
    model.fit(train_X, train_y)
else:
    model.fit(X, y)

# make predictions
if not USE_TEST_DATA:
    val_predictions = model.predict(val_X)
    print("ROC Score:", roc_auc_score(val_y, val_predictions))
else:
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("Submission saved!")