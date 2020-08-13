# import modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

# import training data
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# select the prediction target
y = train_data['Survived']

# select and extract features on which to train the model
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features]) # 'get_dummies' quantifies the 'sex' data

# split training data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# define and fit the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X, train_y)

# make predictions on the validation data
val_predictions = model.predict(val_X)

print(roc_auc_score(val_y, val_predictions))