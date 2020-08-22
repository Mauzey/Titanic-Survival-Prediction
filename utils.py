from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pandas as pd
import numpy as np
import string

def combine_dataframes(train_data, test_data):
    """ Returns a combined dataframe containing testing and training datasets """
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_dataframes(combined_data):
    """ Returns training and testing dataframes from the combined dataset """
    return combined_data.loc[:890], combined_data[891:].drop(['Survived'], axis=1)

def extractSurname(data):
    """ Extracts surnames from passengers using the 'Name' feature """

    # store family names
    families = []

    for i in range(len(data)):
        # acquire passenger name
        name = data.iloc[i]

        # remove brackets from name
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
        
        # the surname appears at the start of passenger names, followed by a comma
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        # remove punctuation from family name
        for c in string.punctuation:
            family = family.replace(c, '').strip()
        
        families.append(family)

    return families

def calcFamilySurvivalRates(train_data, test_data):
    """
    * Calculates the median survival rate for each family
    * Identifies whether a family is unique to the testing dataset
    """

    # get a list of family names that occur in both testing and training datasets
    non_unique_families = [x for x in train_data['FamilyName'].unique() if x in test_data['FamilyName'].unique()]

    # get each family's median survival rate in the training dataset
    family_survival_rate = train_data.groupby('FamilyName')['Survived', 'FamilyName', 'FamilySize'].median()

    # store the median survival rate for each family in the training dataset
    family_rates = {}

    # store the median survival rate of each family that has more than one member across both datasets
    for i in range(len(family_survival_rate)):
        if family_survival_rate.index[i] in non_unique_families and family_survival_rate.iloc[i, 1] > 1:
            family_rates[family_survival_rate.index[i]] = family_survival_rate.iloc[i, 0]

    # calculate the mean survival rate across all passengers in training set
    mean_survival_rate = np.mean(train_data['Survived'])

    # store family survival rates
    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    for i in range(len(train_data)):
        # if the passenger's family name occurs in 'family_rates' (and thefore the training set)
        if train_data['FamilyName'][i] in family_rates:
            # store the family survival rate
            train_family_survival_rate.append(family_rates[train_data['FamilyName'][i]])
            # mark that the family does exist in the training set
            train_family_survival_rate_NA.append(1)
        else:
            # store mean survival rate
            train_family_survival_rate.append(mean_survival_rate)
            # mark that the family doesn't exist in the training set
            train_family_survival_rate_NA.append(0)

    for i in range(len(test_data)):
        # if the passenger's family name occurs in 'family rates' (and therefore the training set)
        if test_data['FamilyName'].iloc[i] in family_rates:
            # store the family survival rate
            test_family_survival_rate.append(family_rates[test_data['FamilyName'].iloc[i]])
            # mark that the family does exist in the training set
            test_family_survival_rate_NA.append(1)
        else:
            # store mean survival rate
            test_family_survival_rate.append(mean_survival_rate)
            # mark that the family doesn't exist in the training set
            test_family_survival_rate_NA.append(0)

    # add these new features to the datasets
    train_data['FamilySurvivalRate'] = train_family_survival_rate
    train_data['FamilySurvivalRateNA'] = train_family_survival_rate_NA
    test_data['FamilySurvivalRate'] = test_family_survival_rate
    test_data['FamilySurvivalRateNA'] = test_family_survival_rate_NA

    return train_data, test_data

def calcTicketSurvivalRates(train_data, test_data):
    """
    * Calculates the survival rate for each ticket group
    * Identifies whether a ticket group is unique to the testing dataset
    """

    # get a list of tickets that occur in both training and testing datasets
    non_unique_tickets = [x for x in train_data['Ticket'].unique() if x in test_data['Ticket'].unique()]

    # get each ticket's median survival rate in the training dataset
    ticket_survival_rate = train_data.groupby('Ticket')['Survived', 'Ticket', 'TicketFreq'].median()

    # store the median survival rate for each ticket in the training dataset
    ticket_rates = {}

    # store the median survival rate of each ticket that has more than one member across both datasets
    for i in range(len(ticket_survival_rate)):
        if ticket_survival_rate.index[i] in non_unique_tickets and ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[ticket_survival_rate.index[i]] = ticket_survival_rate.iloc[i, 0]
    
    # calculate the mean survival rate across all passengers in training set
    mean_survival_rate = np.mean(train_data['Survived'])

    # store ticket survival rates
    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(train_data)):
        # if the passenger's ticket occurs in 'ticket_rates' (and thefore the training set)
        if train_data['Ticket'][i] in ticket_rates:
            # store the ticket survival rate
            train_ticket_survival_rate.append(ticket_rates[train_data['Ticket'][i]])
            # mark that the ticket does exist in the training set
            train_ticket_survival_rate_NA.append(1)
        else:
            # store mean survival rate
            train_ticket_survival_rate.append(mean_survival_rate)
            # mark that the ticket doesn't exist in the training set
            train_ticket_survival_rate_NA.append(0)

    for i in range(len(test_data)):
        # if the passenger's ticket occurs in 'ticket_rates' (and thefore the training set)
        if test_data['Ticket'].iloc[i] in ticket_rates:
            # store the ticket survival rate
            test_ticket_survival_rate.append(ticket_rates[test_data['Ticket'].iloc[i]])
            # mark that the ticket does exist in the training set
            test_ticket_survival_rate_NA.append(1)
        else:
            # store mean survival rate
            test_ticket_survival_rate.append(mean_survival_rate)
            # mark that the ticket doesn't exist in the training set
            test_ticket_survival_rate_NA.append(0)

    # add these new features to the datasets
    train_data['TicketSurvivalRate'] = train_ticket_survival_rate
    train_data['TicketSurvivalRateNA'] = train_ticket_survival_rate_NA
    test_data['TicketSurvivalRate'] = test_ticket_survival_rate
    test_data['TicketSurvivalRateNA'] = test_ticket_survival_rate_NA

    return train_data, test_data

def preprocess(train_data, test_data):
    """
    Preprocesses a given dataset:
        * Fill missing values for 'Age', 'Embarked', 'Fare', and 'Cabin'
    """

    dataset = combine_dataframes(train_data, test_data)

    # fill missing values for 'Age'
    dataset['Age'] = dataset.groupby(['Sex', 'Pclass'])['Age'].apply(
        lambda x: x.fillna(x.median()))

    # fill missing values for 'Embarked'
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # fill missing values for 'Fare'
    dataset['Fare'] = dataset['Fare'].fillna(
        dataset.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0])

    # fill missing values for 'Cabin'
    # create 'Deck' feature by extracting the first letter from the 'Cabin' feature
    dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

    # move passenger from deck 'T' to deck 'A'
    index = dataset[dataset['Deck'] == 'T'].index
    dataset.loc[index, 'Deck'] = 'A'

    # group decks based on correlations and similarities
    dataset['Deck'] = dataset['Deck'].replace(['A', 'B', 'C'], 'ABC')
    dataset['Deck'] = dataset['Deck'].replace(['D', 'E'], 'DE')
    dataset['Deck'] = dataset['Deck'].replace(['F', 'G'], 'FG')

    # drop 'Cabin' feature
    dataset.drop('Cabin', axis=1)

    train_data, test_data = divide_dataframes(dataset)

    return train_data, test_data

def featureEngineer(train_data, test_data):
    """
    Perform feature engineering for a given dataset:
        * Bin categorical features ('Fare' and 'Age')
        * Create 'FamilySize', 'GroupedFamilySize', 'TicketFreq', 'Title', and 'IsMarried' features
        * Create 'FamilyName' feature, which is the passenger's surname
        * Create 'TicketSurvivalRate' and 'TicketSurvivalRateNA' features
        * Create 'FamilySurvivalRate' and 'FamilySurvivalRateNA' features
        * Create 'SurvivalRate' and 'SurvivalRateNA' features
    """

    dataset = combine_dataframes(train_data, test_data)

    # bin 'Fare' and 'Age' features
    dataset['Fare'] = pd.qcut(dataset['Fare'], 13)
    dataset['Age' ] = pd.qcut(dataset[ 'Age'], 10)

    # create 'FamilySize' feature
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # create 'GroupedFamilySize' feature
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium',
        7: 'Large', 8: 'Large', 11: 'Large'}
    dataset['GroupedFamilySize'] = dataset['FamilySize'].map(family_map)

    # create 'TicketFreq' feature
    dataset['TicketFreq'] = dataset.groupby('Ticket')['Ticket'].transform('count')

    # create 'Title' feature
    dataset['Title'] = dataset['Name'].str.split(
        ', ', expand=True)[1].str.split('.', expand=True)[0]

    # replace female titles with 'Miss/Mrs/Ms'
    dataset['Title'] = dataset['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
    # replace unique titles with 'Dr/Military/Noble/Clergy
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


    # create 'IsMarried' feature
    dataset['IsMarried'] = 0
    dataset['IsMarried'].loc[dataset['Title'] == 'Mrs'] = 1

    # create 'FamilyName' feature
    dataset['FamilyName'] = extractSurname(dataset['Name'])

    # split dataframes
    train_data, test_data = divide_dataframes(dataset)

    # calculate the survival rates for each family and ticket group
    train_data, test_data = calcFamilySurvivalRates(train_data, test_data)
    train_data, test_data = calcTicketSurvivalRates(train_data, test_data)

    # calculate average survival rate of 'TicketSurvivalRate' and 'FamilySurvivalRate'
    for dataframe in [train_data, test_data]:
        dataframe['SurvivalRate'] = (dataframe['TicketSurvivalRate'] + dataframe['FamilySurvivalRate']) / 2
        dataframe['SurvivalRateNA'] = (dataframe['TicketSurvivalRateNA'] + dataframe['FamilySurvivalRateNA']) / 2

    return train_data, test_data

def transformData(train_data, test_data):
    """
    * Label encodes non-numerical features
    * One-Hot encodes categorical features
    """

    dataframes = [train_data, test_data]

    # label encode non-numerical features
    non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'GroupedFamilySize', 'Age', 'Fare']

    for dataframe in dataframes:
        for feature in non_numeric_features:
            dataframe[feature] = LabelEncoder().fit_transform(dataframe[feature])
    
    # one-hot encode categorical features
    categorical_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'GroupedFamilySize']
    encoded_features = []

    for dataframe in dataframes:
        for feature in categorical_features:
            encoded_feature = OneHotEncoder().fit_transform(dataframe[feature].values.reshape(-1, 1)).toarray()
            n = dataframe[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_dataframe = pd.DataFrame(encoded_feature, columns=cols)
            encoded_dataframe.index = dataframe.index
            encoded_features.append(encoded_dataframe)

    train_data = pd.concat([train_data, *encoded_features[:6]], axis=1)
    test_data = pd.concat([test_data, *encoded_features[6:]], axis=1)

    return train_data, test_data