import pandas as pd
import numpy as np

def preprocess(dataset):
    """
    Preprocesses a given dataset:
        * Drops features that do not significantly contribute to the model
        * Creates a 'Title' feature, extracted from 'Name'
        * Converts 'Sex' from categorical to numeric
    """

    # drop 'Ticket', 'Cabin', and 'PassengerId' features
    dataset = dataset.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)

    
    # create 'Title' feature
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


    # convert 'Sex' from categorical to numeric
    dataset['Sex'] = dataset['Sex'].map({"female":1, "male":0}).astype(int)

    return dataset