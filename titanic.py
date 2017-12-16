from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def get_predictions(testing_set, clf):
    test_x = [i[0] for _, i in testing_set.items()]
    #test_y = [i[1] for i in testing_set.item()]

    pred = clf.predict(test_x)

    output_df = []
    for i, j in zip(testing_set.keys(), pred):
        output_df.append({'PassengerId':i,'Survived':int(j[0])})

    output_df = pd.DataFrame.from_dict(output_df)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)

def get_model(training_set):
    clf = RandomForestClassifier(n_estimators=64)

    train_x = [i[0] for _, i in training_set.items()]
    train_y = [i[1] for _, i in training_set.items()]

    train_x = np.squeeze(np.array(train_x))
    train_y = np.squeeze(np.array(train_y))
    clf.fit(train_x, train_y)
    return clf

def get_features(input_df):
    output = dict()

    input_df = input_df.fillna(input_df.median())

    for i, j in input_df.iterrows():
        try:
            if j['Survived'] == 1:
                survived = [1, 0]
            if j['Survived'] == 0:
                survived = [0, 1]

        except:
            survived = None
        pclass = j['Pclass']
        name = j['Name']
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
        if j['Pclass'] == 'male':
            sex = 1
        else:
            sex = 0
        sibsp = j['SibSp']
        parch = j['Parch']
        fare = j['Fare']
        name = j['Name']

        embarked_part_array = [0 for i in range(4)]
        if 'S' in name.lower():
            embarked_part_array[0] = 1
        elif 'C' in name.lower():
            embarked_part_array[1] = 1
        elif 'Q' in name.lower():
            embarked_part_array[2] = 1
        else:
            embarked_part_array[3] = 1

        output_x = [pclass, sex, sibsp, parch, fare] + embarked_part_array + name_part_array
        output_y = [survived]
        output[j['PassengerId']] = [output_x, output_y]
    return output

def main():
    train_df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\titanic\train.csv')
    test_df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\titanic\test.csv')

    train_features = get_features(train_df)
    test_features = get_features(test_df)
    clf = get_model(train_features)
    get_predictions(test_features, clf)

if __name__ == '__main__':
    main()