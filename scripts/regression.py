#! /usr/bin/env python3

from math import floor, ceil
import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
import os
import warnings
import csv

input_fields = [
    'MISTAKE',
    'GRIPPER_A_PRE_X',
    'GRIPPER_A_PRE_Y',
    'GRIPPER_A_PRE_Z',
    'GRIPPER_B_PRE_X',
    'GRIPPER_B_PRE_Y',
    'GRIPPER_B_PRE_Z',
    'GRIPPER_A_POST_X',
    'GRIPPER_A_POST_Y',
    'GRIPPER_A_POST_Z',
    'GRIPPER_B_POST_X',
    'GRIPPER_B_POST_Y',
    'GRIPPER_B_POST_Z',
    'GRIPPER_DELTA_LENGTH_PRE',
    'GRIPPER_DELTA_LENGTH_POST',
    'MAX_BAND_LENGTH',
    'STARTING_BAND_LENGTH',
    'ENDING_DEFAULT_BAND_LENGTH',
    'SLICE_NUM_CONNECTED_COMPONENTS_PRE',
    'SLICE_NUM_CONNECTED_COMPONENTS_POST',
    'SLICE_NUM_CONNECTED_COMPONENTS_DELTA',
    'SLICE_NUM_FREE_CONNECTED_COMPONENTS_PRE',
    'SLICE_NUM_FREE_CONNECTED_COMPONENTS_POST',
    'SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA',
    'SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_PRE',
    'SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_POST',
    'SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA']
x_fields = [
    input_fields.index('GRIPPER_DELTA_LENGTH_PRE'),
    input_fields.index('GRIPPER_DELTA_LENGTH_POST'),
    input_fields.index('STARTING_BAND_LENGTH'),
    input_fields.index('ENDING_DEFAULT_BAND_LENGTH'),
    input_fields.index('SLICE_NUM_CONNECTED_COMPONENTS_DELTA'),
    input_fields.index('SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA'),
    input_fields.index('SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA'),
]

rootdir = '/mnt/big_narstie_data/dmcconac/transition_learning_data_generation/cannonical_straight_test'

aggregate_data = False

if aggregate_data:
    print("Aggregating data from csv files")
    all_data = []
    metadata = []
    for root, sub_folders, files in os.walk(rootdir):
        for file in files:
            if file.endswith('features.complete'):
                features_file = root + '/' + file[0:-8] + 'csv'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw_data = np.genfromtxt(features_file, delimiter=',')
                if len(raw_data) == 0:
                    print('Deleting', root[len(rootdir)+1:] + '/' + file, 'as the csv is empty')
                    os.remove(root + '/' + file)
                    os.remove(features_file)
                else:
                    all_data.append(raw_data)
                    metadata.extend([(features_file, idx) for idx in range(len(raw_data))])

    # Flatten the data into one large matrix of data
    data = np.vstack(all_data)

    np.savetxt(rootdir + '/' + 'aggregated_features.csv', data, delimiter=',')
    with open(rootdir + '/' + 'aggregated_features_metadata.csv', 'wb') as csvfile:
        csv_out = csv.writer(csvfile)
        for row in metadata:
            csv_out.writerow(row)
else:
    print("Loading aggregated data")
    data = np.genfromtxt(rootdir + '/' + 'aggregated_features.csv', delimiter=',')
    metadata = []
    with open(rootdir + '/' + 'aggregated_features_metadata.csv') as csvfile:
        csv_in = csv.reader(csvfile)
        for row in csv_in:
            metadata.append(tuple(row))

num_data_points = len(data)
permutation = np.random.permutation(num_data_points)
permuted_X_data = data[permutation, :][:, x_fields]
permuted_Y_data = data[permutation, input_fields.index('MISTAKE')]
permuted_metadata = [metadata[idx] for idx in permutation]

train_data_size = int(floor(num_data_points * 0.7))
test_data_size = num_data_points - train_data_size
X_train = permuted_X_data[:train_data_size, :]
X_test = permuted_X_data[-test_data_size:, :]
scalar = preprocessing.StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)
Y_train = permuted_Y_data[:train_data_size]
Y_test = permuted_Y_data[-test_data_size:]


param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

print("Fitting data and predicting on held out samples")
# classifier = svm.SVC(gamma='scale')
# classifier = svm.SVC(class_weight={0: 1, 1: 10})
classifier = svm.SVC(class_weight='balanced')
classifier.fit(X_train_scaled, Y_train)
Y_test_predict = classifier.predict(X_test_scaled)
confusion_matrix(Y_test, Y_test_predict)

IPython.embed()

# dist_est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#                                      max_depth=1, random_state=0, loss='ls').fit(X_train, Y_train[:, 0])
# print "Mean squared error: ", mean_squared_error(Y_test[:, 0], dist_est.predict(X_test))
# print "Feature importance: ", dist_est.feature_importances_
#
# foh_est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#                                     max_depth=1, random_state=0, loss='ls').fit(X_train, Y_train[:, 1])
# print "Mean squared error: ", mean_squared_error(Y_test[:, 1], foh_est.predict(X_test))
# print "Feature importance: ", foh_est.feature_importances_
#
#
# dist_predict = dist_est.predict(X_test)
# plt.scatter(dist_predict, Y_data[:, 0])
# plt.xlabel('Predicted dist')
# plt.ylabel('True dist')
# plt.show()
