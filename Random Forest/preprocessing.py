def preprocess(dataset):
    """
    Preprocesses a given dataset


    """

    # combine 'Parch' and 'SibSp' into new field 'FamilySize'
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    # remove the two features, alongside 'Embarked' as it serves little purpose
    dataset = dataset.drop(['SibSp', 'Parch', 'Embarked'], axis=1)

    return dataset