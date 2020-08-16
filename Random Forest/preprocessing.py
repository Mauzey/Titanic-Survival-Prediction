import pandas as pd
import numpy as np

def preprocess(dataset):
    """
    Preprocesses a given dataset:
        * Drops features that do not significantly contribute to the model
        * Creates a 'Title' feature, extracted from 'Name'
        * Converts 'Sex' from categorical to numeric
        * Fill missing 'Age' values with an estimate based on 'Pclass' and 'Gender'
            * 'Age' is then separated into bands:
                <= 16, >16 & <=32, >32 & <=48, >48 & <=64, >64
        * Creates an 'IsAlone' feature, based on 'FamilySize'
            * 'FamilySize' is a sum of 'Parch' and 'SibSp'
    """

    # DROP 'Ticket', 'Cabin', AND 'PassengerId' FEATURES
    dataset = dataset.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)



    # CREATE 'Title' FEATURE
    # extract title from 'Name'
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # categorise titles (less-frequent titles are grouped into a 'Rare' category)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                        'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert 'Title' from categorical to ordinal
    titleMap = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    dataset['Title'] = dataset['Title'].map(titleMap)
    dataset['Title'] = dataset['Title'].fillna(0)

    # drop the 'Name' feature
    dataset = dataset.drop(['Name'], axis=1)



    # CONVERT 'Sex' FROM CATEGORICAL TO NUMERIC
    dataset['Sex'] = dataset['Sex'].map({"female":1, "male":0}).astype(int)



    # FILL MISSING 'Age' VALUES
    estimated_ages = np.zeros((2, 3))

    # iterate over 'Sex' (0 and 1)
    for i in range(0, 2):
        # iterate over 'Pclass' (1, 2, and 3)
        for j in range(0, 3):
            # calculate median 'Age' value for given 'Sex' and 'Pclass'
            age_estimate_dataset = dataset[(dataset['Sex'] == i) &
                                (dataset['Pclass'] == j + 1)]['Age'].dropna()
            median_age = age_estimate_dataset.median()

            # round age to nearest 0.5 age
            estimated_ages[i, j] = int(median_age/0.5 + 0.5) * 0.5
    
    # iterate over 'Sex' (0 and 1)
    for i in range(0, 2):
        # iterate over 'Pclass' (1, 2, and 3)
        for j in range(0, 3):
            # replace missing values for given 'Sex' and 'Pclass'
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = estimated_ages[i,j]

    # typecast as int
    dataset['Age'] = dataset['Age'].astype(int)

    # separate 'Age' into ordinals
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



    # CREATE 'FamilySize' FEATURE
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # CREATE 'IsAlone' FEATURE
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # drop 'Parch', 'SibSp', and 'FamilySize'
    dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    return dataset