#! /usr/bin/env python

from math import floor, ceil
import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

datafile = '/mnt/big_narstie_data/dmcconac/transition_learning_data_generation/cannonical_straight_test/dists_etc_valid_only.csv'
raw_data = np.genfromtxt(datafile, delimiter=',')
num_data_points = len(raw_data)
permutation = np.random.permutation(num_data_points)
raw_data = raw_data[permutation, :]

input_fields = [
    "TEMPLATE_MISALIGNMENT_EUCLIDEAN",
    "DEFAULT_VS_ADAPTATION_FOH",
    "DEFAULT_VS_ADAPTATION_EUCLIDEAN",
    "BAND_TIGHTEN_DELTA",
    "SOURCE_NUM_FOH_CHANGES",
    "RESULT_NUM_FOH_CHANGES",
    "TRUE_VS_DEFAULT_FOH",
    "TRUE_VS_DEFAULT_EUCLIDEAN",
    "TRUE_VS_ADAPATION_FOH",
    "TRUE_VS_ADAPATION_EUCLIDEAN"]

template_misalignment_dist = raw_data[:, 0]
default_vs_adapt_foh = raw_data[:, 1]
default_vs_adapt_dist = raw_data[:, 2]
band_tighten_delta = raw_data[:, 3]
source_foh_changes = raw_data[:, 4]
adapt_foh_changes = raw_data[:, 5]
true_vs_default_foh = raw_data[:, 6]
true_vs_default_dist = raw_data[:, 7]
true_vs_adapt_foh = raw_data[:, 8]
true_vs_adapt_dist = raw_data[:, 9]

X_data = np.vstack((template_misalignment_dist, band_tighten_delta, abs(source_foh_changes - adapt_foh_changes))).transpose()
Y_data = np.vstack((true_vs_adapt_dist, 2 * true_vs_adapt_foh - 1)).transpose()

train_data_size = int(floor(num_data_points * 1))
test_data_size = num_data_points - train_data_size
X_train = X_data[:train_data_size, :]
X_test = X_data[-test_data_size:, :]
Y_train = Y_data[:train_data_size, :]
Y_test = Y_data[-test_data_size:, :]



dist_est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                     max_depth=1, random_state=0, loss='ls').fit(X_train, Y_train[:, 0])
print "Mean squared error: ", mean_squared_error(Y_test[:, 0], dist_est.predict(X_test))
print "Feature importance: ", dist_est.feature_importances_

foh_est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls').fit(X_train, Y_train[:, 1])
print "Mean squared error: ", mean_squared_error(Y_test[:, 1], foh_est.predict(X_test))
print "Feature importance: ", foh_est.feature_importances_


dist_predict = dist_est.predict(X_test)
plt.scatter(dist_predict, Y_data[:, 0])
plt.xlabel('Predicted dist')
plt.ylabel('True dist')
plt.show()
