from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing


def train_l1_gb(x_train, x_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=500)
    clf.fit(x_train, y_train)
    if y_test is not None:
        print('GradientBoostingClassifier:', clf.score(x_test, y_test))
    else:
        print('GradientBoostingClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]



def train_l1_et(x_train, x_test, y_train, y_test):
    clf = ExtraTreesClassifier(n_estimators=256, n_jobs=-1)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('ExtraTreesClassifier:', clf.score(x_test, y_test))
    else:
        print('ExtraTreesClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_ada(x_train, x_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=500)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('AdaBoostClassifier:', clf.score(x_test, y_test))
    else:
        print('AdaBoostClassifier:', clf.score(x_train, y_train))

    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_rf(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=256, n_jobs=-1)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('RandomForestClassifier:', clf.score(x_test, y_test))
    else:
        print('RandomForestClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_nn(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100,))
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('MLPClassifier:', clf.score(x_test, y_test))
    else:
        print('MLPClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_kmeans(x_train, x_test, y_train, y_test):
    clf = KNeighborsClassifier(n_jobs=-1)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('KNeighborsClassifier:', clf.score(x_test, y_test))
    else:
        print('KNeighborsClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_gaussian(x_train, x_test, y_train, y_test):
    clf = GaussianProcessClassifier(n_jobs=-1)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('GaussianProcessClassifier:', clf.score(x_test, y_test))
    else:
        print('GaussianProcessClassifier:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_svc(x_train, x_test, y_train, y_test):
    clf = SVC()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('SVC:', clf.score(x_test, y_test))
    else:
        print('SVC:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_gnb(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('GaussianNB:', clf.score(x_test, y_test))
    else:
        print('GaussianNB:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l1_qda(x_train, x_test, y_train, y_test):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('QuadraticDiscriminantAnalysis:', clf.score(x_test, y_test))
    else:
        print('QuadraticDiscriminantAnalysis:', clf.score(x_train, y_train))
    test_res = np.reshape(clf.predict(x_train), (-1,1))
    train_res =  np.reshape(clf.predict(x_test), (-1,1))
    return [test_res, train_res]


def train_l2_gb(x_train, x_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=500)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('GradientBoostingClassifier:', clf.score(x_test, y_test))
    else:
        print('GradientBoostingClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1,1))


def train_l2_et(x_train, x_test, y_train, y_test):
    clf = ExtraTreesClassifier(n_estimators=256)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('ExtraTreesClassifier:', clf.score(x_test, y_test))
    else:
        print('ExtraTreesClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))

def train_l2_ada(x_train, x_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=500)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('AdaBoostClassifier:', clf.score(x_test, y_test))
    else:
        print('AdaBoostClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))



def train_l2_rf(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=256)
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('RandomForestClassifier:', clf.score(x_test, y_test))
    else:
        print('RandomForestClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))



def train_l2_nn(x_train, x_test, y_train, y_test):
    clf = MLPClassifier()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('MLPClassifier:', clf.score(x_test, y_test))
    else:
        print('MLPClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))

def train_l2_kmeans(x_train, x_test, y_train, y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('KNeighborsClassifier:', clf.score(x_test, y_test))
    else:
        print('KNeighborsClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))


def train_l2_gaussian(x_train, x_test, y_train, y_test):
    clf = GaussianProcessClassifier()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('GaussianProcessClassifier:', clf.score(x_test, y_test))
    else:
        print('GaussianProcessClassifier:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))


def train_l2_gnb(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('GaussianNB:', clf.score(x_test, y_test))
    else:
        print('GaussianNB:', clf.score(x_train, y_train))
    return np.reshape(clf.predict(x_train), (-1, 1))


def train_l2_qda(x_train, x_test, y_train, y_test):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train, y_train)

    if y_test is not None:
        print('QuadraticDiscriminantAnalysis:', clf.score(x_test, y_test))
    return np.reshape(clf.predict(x_train), (-1, 1))


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
    if 'S' in name.lower():
        embarked_part_array[0] = 1
    elif 'C' in name.lower():
        embarked_part_array[1] = 1
    elif 'Q' in name.lower():
        embarked_part_array[2] = 1
    else:
        embarked_part_array[3] = 1

    output_x = np.array([sex, sibsp, parch, fare, age, is_age_there,
                is_cabin_there] + embarked_part_array + name_part_array + pj_class)
    output_x = np.array([sex, age, fare])
    return output_x

def get_features(input_df):
    output = dict()

    input_df['is_age_there'] = pd.notnull(input_df['Age'])
    input_df['is_cabin_there'] = pd.notnull(input_df['Cabin'])
    input_df = input_df.fillna(input_df.median())
    max_list = []
    x_list = []
    y_list = []
    p_ids = []

    for i in input_df.iterrows():
        x_list.append(extract_features_from_row(i))

        try:
            y_list.append(np.array([i[1]['Survived']]))
        except:
            y_list= None
        p_ids.append(i[1]['PassengerId'])
    x_list = np.vstack(x_list)
    try:
        y_list = np.vstack(y_list)
        y_list = np.ravel(y_list)
    except:
        y_list = None
    return x_list, y_list, p_ids


def min_max_scale_data(train_features):
    min_max_scaler= preprocessing.MinMaxScaler()
    min_max_scaler.fit(train_features)
    return min_max_scaler


def main():
    train_df = pd.read_csv(r'C:\Users\trist\Documents\db_loc\titanic_data\train.csv')
    test_df = pd.read_csv(r'C:\Users\trist\Documents\db_loc\titanic_data\test.csv')

    x_train, y_train, train_ids = get_features(train_df)
    x_test, y_test, test_ids = get_features(test_df)
    #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.1)

    min_max_scaler = min_max_scale_data(x_train)
    x_train = min_max_scaler.transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    res_list = []
    res_list.append(train_l1_gaussian(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_kmeans(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_nn(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_rf(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_gb(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_et(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_ada(x_train, x_test, y_train, y_test))
    res_list.append(train_l1_gnb(x_train, x_test, y_train, y_test))
    #res_list.append(train_l1_qda(x_train, x_test, y_train, y_test))
    test_res = np.hstack([i[1] for i in res_list])
    train_res = np.hstack([i[0] for i in res_list])
    print()

    res_list = []
    res_list.append(train_l2_rf(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_gb(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_et(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_ada(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_nn(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_kmeans(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_gaussian(train_res, test_res, y_train, y_test))
    res_list.append(train_l2_gnb(train_res, test_res, y_train, y_test))
    #res_list.append(train_l2_qda(train_res, test_res, y_train, y_test))
    test_res = np.hstack([i for i in res_list])


    prediction_array = []
    for i in test_res:
        prediction_array.append(int(np.around(np.average(i))))
        print(np.around(np.average(i)), np.average(i), i)

    output_df = []
    output_dict = {}
    passenger_id_list = []

    for i, j in zip(test_ids, prediction_array):
        output_df.append({'PassengerId':i,'Survived':int(j)})
        print()
        print({'PassengerId':i,'Survived':int(j)})
        print(test_df.loc[test_df['PassengerId'] == i].values)

    output_df = pd.DataFrame.from_dict(output_df)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)



if __name__ == '__main__':
    main()

