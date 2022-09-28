from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import datetime
import csv
import time
import os
import json
import shutil, errno
from matplotlib import pyplot
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

def select_features_f(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_MI(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_anova(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=f_classif, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_p(X_train, y_train):
	# configure to select all features
	p_vals= SelectKBest(score_func=r_regression, k='all').fit(X_train, y_train)
	p_vals.transform(X_train)
	return p_vals

def select_features_rfe(X_train, y_train, feat_num, feature_names):
	rf = RandomForestRegressor(n_estimators=100,
								max_depth=6,
								max_features=0.4)
	rfe = RFE(estimator=rf, n_features_to_select=feat_num, step = 1)
	rfe = rfe.fit(X_train, y_train)
	return rfe.get_feature_names_out(feature_names)

def rfe_experiment(feature_file, feat_num, X_train, y_train, feature_names):
	maintained = select_features_rfe(X_train, y_train, feat_num, feature_names)
	feature_name = pd.read_csv('{}.csv'.format(feature_file))
	print(maintained)
	for i in feature_name.columns:
		if i not in maintained:
			if i!='Datetime':
				feature_name.drop(i, inplace=True, axis=1)
	feature_name.to_csv("{}_red.csv".format(feature_file), index=False)


def f_experiment(X_train, y_train, system):
	fs = select_features_f(X_train, y_train[:,0])
	p_values = np.nan_to_num(fs.pvalues_)
	f_scores = np.nan_to_num(fs.scores_)
	for i in range(1,output_dim):
		p_values += abs(np.nan_to_num(select_features_f(X_train, y_train[:,i]).pvalues_))
		f_scores += (np.nan_to_num(select_features_f(X_train, y_train[:,i]).scores_.replace(np.nan, 0)))

	index = np.argsort(p_values, axis=-1, kind='quicksort', order=None)
	dict = {}

	for i in reversed(index):
		dict[ts_data.columns[i]] = p_values[i]

	with open("data/{}/p_vals_for_f.json".format(system), "w") as outfile:
		json.dump(dict, outfile)

	index2 = np.argsort(f_scores, axis=-1, kind='quicksort', order=None)
	dict2 = {}

	for i in reversed(index2):
		if np.isnan(f_scores[i]):
			dict2[ts_data.columns[i]] = 0
		else:
			dict2[ts_data.columns[i]] = f_scores[i]

	with open("data/{}/f_features.json".format(system), "w") as outfile:
		json.dump(dict2, outfile)

def p_experiment(X_train, y_train, system):
	p_s = select_features_p(X_train, y_train[:,0])
	p_scores = np.nan_to_num(p_s.scores_)
	for i in range(1,output_dim):
		p_scores += (np.nan_to_num(select_features_p(X_train, y_train[:,i]).scores_))
	

	index_p = np.argsort(p_scores, axis=-1, kind='quicksort', order=None)
	dict_p = {}

	for i in reversed(index_p):
		dict_p[ts_data.columns[i]] = p_scores[i]

	with open("data/{}/p_scores.json".format(system), "w") as outfile:
		json.dump(dict_p, outfile)

	abs_p_scores = abs(p_scores)
	index_abs_p = np.argsort(abs_p_scores, axis=-1, kind='quicksort', order=None)
	dict_abs_p = {}
	for i in reversed(index_abs_p):
		dict_abs_p[ts_data.columns[i]] = abs_p_scores[i]

	with open("data/{}/abs_p_scores.json".format(system), "w") as outfile:
		json.dump(dict_abs_p, outfile)

def mi_experiment(X_train, y_train, system):
	mi_s = select_features_MI(X_train, y_train[:,0])
	mi_scores = mi_s.scores_
	for i in range(1,output_dim):
		mi_scores += (select_features_MI(X_train, y_train[:,i]).scores_)

	index3 = np.argsort(mi_scores, axis=-1, kind='quicksort', order=None)
	dict3 = {}

	for i in reversed(index3):
		dict3[ts_data.columns[i]] = mi_scores[i]

	with open("data/{}/mi_features.json".format(system), "w") as outfile:
	    json.dump(dict3, outfile)

system = 'caiso'
if system == 'caiso':
	output_dim = 24
else:
	output_dim = 264
ts_data = pd.read_csv('data/{}/ts_data.csv'.format(system), index_col = 0, parse_dates=['Datetime'])
target_data = pd.read_csv('data/{}/target_data.csv'.format(system), index_col = 0, parse_dates=['Datetime'])
rfe_feature_file = 'data/{}/ts_data_red'.format(system)
ts_data_red = pd.read_csv('{}.csv'.format(rfe_feature_file), index_col = 0, parse_dates=['Datetime'])

date_range = ts_data.index.date
header = ts_data.head()

n_train_days = 365
n_oos = 30
feat_num = 25

end_test = str(date_range[-1]) + ' 23:55:00'
start_test = str(date_range[-(n_oos+1)]) + ' 23:55:00'
end_train = str(date_range[-(n_oos+1)]) + ' 23:55:00'
start_train = str(date_range[-n_train_days-(n_oos+1)]) + ' 23:55:00'

X_train = ts_data.loc[start_train:end_train, :].values
X_train_rfe = ts_data_red.loc[start_train:end_train, :].values
y_train = target_data.loc[start_train:end_train, :].values
y_train_rfe = target_data.loc[start_train:end_train, :].values

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaler_red = MinMaxScaler()
y_scaler_red = MinMaxScaler()
X_scaler = X_scaler.fit(X_train)
y_scaler = y_scaler.fit(y_train)
X_scaler_red = X_scaler_red.fit(X_train_rfe)
y_scaler_red = y_scaler_red.fit(y_train_rfe)
X_train = X_scaler.transform(X_train)
y_train = y_scaler.transform(y_train)
X_train_rfe = X_scaler_red.transform(X_train_rfe)
y_train_rfe = y_scaler_red.transform(y_train_rfe)

p_experiment(X_train, y_train, system)

# rfe_experiment(rfe_feature_file, feat_num, X_train_rfe, y_train_rfe, ts_data_red.columns)