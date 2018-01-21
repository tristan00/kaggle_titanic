from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
from sklearn import preprocessing


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def get_rf_predictions(testing_set, clf, min_max_scaler):
    test_x = [i[0] for _, i in testing_set.items()]
    #test_y = [i[1] for i in testing_set.item()]

    test_x = min_max_scaler.fit_transform(test_x)

    pred = clf.predict(test_x)

    output_df = []
    for i, j in zip(testing_set.keys(), pred):
        output_df.append({'PassengerId':i,'Survived':int(j)})

    output_df = pd.DataFrame.from_dict(output_df)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def get_ada_predictions(testing_set, clf, min_max_scaler):
    test_x = [i[0] for _, i in testing_set.items()]
    #test_y = [i[1] for i in testing_set.item()]

    test_x = min_max_scaler.fit_transform(test_x)

    pred = clf.predict(test_x)

    output_df = []
    for i, j in zip(testing_set.keys(), pred):
        output_df.append({'PassengerId':i,'Survived':int(j)})

    output_df = pd.DataFrame.from_dict(output_df)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def get_ada_model(training_set, min_max_scaler):
    clf = RandomForestClassifier(n_estimators=500)
    train_x = [i[0] for _, i in training_set.items()]
    train_y = [i[1] for _, i in training_set.items()]

    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    train_x = min_max_scaler.fit_transform(train_x)

    clf.fit(train_x, train_y)
    return clf


def get_rf_model(training_set, min_max_scaler):
    clf = RandomForestClassifier(n_estimators=1024)
    #clf = ExtraTreesClassifier(n_estimators=256)
    train_x = [i[0] for _, i in training_set.items()]
    train_y = [i[1] for _, i in training_set.items()]

    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    train_x = min_max_scaler.fit_transform(train_x)

    clf.fit(train_x, train_y)
    return clf


def tune_rf(training_set):
    clf = RandomForestClassifier()
    train_x = [i[0] for _, i in training_set.items()]
    train_y = [[i[1]] for _, i in training_set.items()]
    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    print(train_x.shape)
    print(train_y.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)


    param_grid = {"n_estimators": [64, 128],
                  "max_depth": [2, 3, 4, 5, None],
                  "max_features": [None, 'log2', 'sqrt'],
                  "min_samples_split": [2, 5, 10, 15],
                  "min_samples_leaf": [1, 2, 5, 10, 15],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time.time()
    grid_search.fit(train_x, train_y)


    print()
    print(grid_search.best_params_)
    report(grid_search.cv_results_)


def tune_gbc(training_set):
    clf = GradientBoostingClassifier()
    train_x = [i[0] for _, i in training_set.items()]
    train_y = [[i[1][0]] for _, i in training_set.items()]
    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    print(train_x.shape)
    print(train_y.shape)

    param_grid = {"loss": ['deviance', 'exponential'],
                  "learning_rate": [.05, .1, .2, .5],
                  "n_estimators": [100, 200, 500],
                  "max_depth": [2, 3, 5],
                  "min_samples_split":[2, 3, 5],
                  "min_samples_leaf": [1, 3, 5],
                  "max_features": ['sqrt', 'log2', None]}
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time.time()
    grid_search.fit(train_x, train_y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time.time() - start, len(grid_search.cv_results_['params'])))

    print()
    print(grid_search.best_params_)
    report(grid_search.cv_results_)
    print(grid_search.cv_results_)



def split_test_train(train_x, test_size):
    pass


def get_gb_predictions(testing_set, clf):
    test_x = [i[0] for _, i in testing_set.items()]
    #test_y = [i[1] for i in testing_set.item()]

    pred = clf.predict(test_x)

    output_df = []
    for i, j in zip(testing_set.keys(), pred):
        output_df.append({'PassengerId':i,'Survived':int(j[0])})

    output_df = pd.DataFrame.from_dict(output_df)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)

def get_gb_model(training_set):
    clf = RandomForestClassifier(n_estimators=64)

    train_x = [i[0] for _, i in training_set.items()]
    train_y = [[i[1][0]] for _, i in training_set.items()]

    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    clf.fit(train_x, train_y)
    return clf

def extract_features_from_row(row_tuple):
    _, row = row_tuple

    pj_class = [0 for _ in range(3)]
    pj_class[row['Pclass']-1] = 1

    name = row['Name']
    name_part_array = [0 for i in range(5)]
    if 'mr.' in name.lower():
        name_part_array[0] = 1
    elif 'mrs.' in name.lower():
        name_part_array[1] = 1
    elif 'miss' in name.lower():
        name_part_array[2] = 1
    else:
        name_part_array[3] = 1
    if '(' in name and ')' in name:
        name_part_array[4] = 1
    if row['Sex'] == 'male':
        sex = 1
    else:
        sex = 0
    sibsp = row['SibSp']
    parch = row['Parch']
    fare = row['Fare']
    name = row['Name']
    age = row['Age']
    is_age_there = int(row['is_age_there'])
    is_cabin_there = int(row['is_cabin_there'])

    embarked_part_array = [0 for i in range(4)]
    if 'S' in name.upper():
        embarked_part_array[0] = 1
    elif 'C' in name.upper():
        embarked_part_array[1] = 1
    elif 'Q' in name.upper():
        embarked_part_array[2] = 1
    else:
        embarked_part_array[3] = 1

    output_x = [sex, sibsp, parch, fare, age] + embarked_part_array + name_part_array + pj_class
    return output_x

def get_features(input_df):
    output = dict()

    input_df['is_age_there'] = pd.notnull(input_df['Age'])
    input_df['is_cabin_there'] = pd.notnull(input_df['Cabin'])
    input_df = input_df.fillna(input_df.median())
    max_list = []

    for i in input_df.iterrows():
        output_x = extract_features_from_row(i)
        if len(max_list) == 0:
            max_list = [0 for _ in output_x]

        for j in range(len(output_x)):
            max_list[j] = max(max_list[j], output_x[j])

        #output_x = [sex] + embarked_part_array
        try:
            output_y = [i[1]['Survived']]
        except:
            output_y = None
        output[i[1]['PassengerId']] = [output_x, output_y]


    return output

def read_dt(training_set):
    clf = tree.DecisionTreeClassifier()
    train_x = [i[0] for _, i in training_set.items()]
    train_y = [[i[1]] for _, i in training_set.items()]
    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    #min_max_scaler = preprocessing.MinMaxScaler([-1,1])
    min_max_scaler = preprocessing.QuantileTransformer(output_distribution = 'normal', random_state = 0)
    train_x = min_max_scaler.fit_transform(train_x)

    clf.fit(train_x, train_y)
    print([ i for i in clf.feature_importances_])
    return min_max_scaler


def main():
    train_df = pd.read_csv(r'C:\Users\trist\Documents\db_loc\titanic_data\train.csv')
    test_df = pd.read_csv(r'C:\Users\trist\Documents\db_loc\titanic_data\test.csv')

    train_features = get_features(train_df)
    min_max_scaler =read_dt(train_features)
    test_features = get_features(test_df)
    clf = get_rf_model(train_features, min_max_scaler)
    get_rf_predictions(test_features, clf, min_max_scaler)
    # clf = get_ada_model(train_features, min_max_scaler)
    # get_ada_predictions(test_features, clf, min_max_scaler)

if __name__ == '__main__':
    main()