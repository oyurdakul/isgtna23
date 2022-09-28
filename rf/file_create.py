import json
import pandas as pd  


def partition_locations(feature_score_file, locations):
    f = open(feature_score_file)
    data = json.load(f)
    for l in locations:
            with open('data/nyiso/locations/features_{}.txt'.format(l), "w") as outfile:
                outfile.write('')

    for i in data.keys():
        for l in locations:
            if i.find(l)!=-1:
                with open('data/nyiso/locations/features_{}.txt'.format(l), "a") as outfile:
                    outfile.write('{}: {}\n'.format(i, data[i]))
    f.close()

def create_one_hot_feature(feature_file, categorical_columns):
    feature_name = pd.read_csv('{}.csv'.format(feature_file))
    for column in categorical_columns:
        tempdf = pd.get_dummies(feature_name[column], prefix=column)
        feature_name = pd.merge(
            left=feature_name,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        feature_name = feature_name.drop(columns=column)
    feature_name.to_csv("{}.csv".format(feature_file), index=False)

def create_reduced_feature_file(feature_score_file, feature_file, threshold, categorical_columns):
    f = open(feature_score_file)
    data = json.load(f)
    feature_name = pd.read_csv('{}.csv'.format(feature_file))
    for i in data.keys():
        if data[i]<threshold:
            if i not in categorical_columns:
                feature_name.drop(i, inplace=True, axis=1)

    feature_name.to_csv("{}_red.csv".format(feature_file), index=False)
system = 'caiso'
feature_score_file = 'data/{}/abs_p_scores.json'.format(system)
locations = ['CENTRL', 'GENESE', 'CAPITL', 'MHK VL', 'WEST', 'HUD VL', 'MILLWD', 'N.Y.C.', 'DUNWOD', 'LONGIL', 'NORTH']
feature_file = 'data/{}/ts_data'.format(system)
categorical_columns = ['is_weekend_1.0', 'is_weekend_0.0', 'is_holiday_1.0', 'is_holiday_0.0']
threshold = 14.4

# create_one_hot_feature(feature_file, categorical_columns)

# create_reduced_feature_file(feature_score_file, feature_file, threshold, categorical_columns)
