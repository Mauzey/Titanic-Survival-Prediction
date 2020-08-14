import pandas as pd
import numpy as np

def preprocess(dataset):
    """
    Preprocesses a given dataset:
        * Combines 'SibSp' and 'Parch' into new field: 'FamilySize'
        * Removes 'Embarked' and 'Cabin' fields as they serve little purpose (for now)
        * Fills missing values for 'Age' with the mean value within each 'Pclass'
    """

    # combine 'Parch' and 'SibSp' into new field: 'FamilySize'
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset = dataset.drop(['SibSp', 'Parch'], axis=1)


    # remove the 'embarked' and 'Cabin' features
    dataset = dataset.drop(['Embarked', 'Cabin'], axis=1)


    # fill missing values for 'Age' with the mean 'Age' for the appropriate 'Pclass'
    passengers = dataset[(dataset['Pclass'] == 1) & (dataset['Age'].notnull())]
    first_class_mean = passengers['Age'].mean()
    dataset['Age'] = np.where((pd.isnull(dataset['Age'])) & (dataset['Pclass'] == 1),
                            first_class_mean, dataset['Age'])

    passengers = dataset[(dataset['Pclass'] == 2) & (dataset['Age'].notnull())]
    second_class_mean = passengers['Age'].mean()
    dataset['Age'] = np.where((pd.isnull(dataset['Age'])) & (dataset['Pclass'] == 2),
                            second_class_mean, dataset['Age'])

    passengers = dataset[(dataset['Pclass'] == 3) & (dataset['Age'].notnull())]
    third_class_mean = passengers['Age'].mean()
    dataset['Age'] = np.where((pd.isnull(dataset['Age'])) & (dataset['Pclass'] == 3),
                            third_class_mean, dataset['Age'])

    return dataset

def featureEngineer(dataset):
    """
    Perform feature engineering for a given dataset:
        * Add 'Minor' feature (True if 'Age' < 16 or "Master" appears within the name)
    """

    # add 'Minor' tag (1 = True, 0 = False)
    dataset['Minor'] = 0
    dataset['Minor'] = np.where((dataset['Age'] < 16)
                                | (dataset['Name'].str.contains("Master", regex=False)),
                                1, dataset['Minor'])

    return dataset