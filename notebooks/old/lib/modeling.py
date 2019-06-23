import numpy as np
import pandas as pd
import psycopg2 as pg
import datetime as dt
import pickle
import ast
import os
import time

# modeling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold



def train_binary_level(train_data, cluster, model, parameters, num_round = 10, verbose=False):					
	
	# num_round for xgboost only
		
	if verbose == True:
		print('Training cluster {}:'.format(str(cluster)))
		
	time_start = time.time()
	metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1score': [], 'roc_auc': []}

	cluster_train = train_data.loc[train_data['cluster'] == cluster]
	skf = StratifiedKFold(n_splits=5, shuffle=False)
	X = cluster_train.drop(columns=['level_binary','level_max','cluster'])
	y = cluster_train['level_binary']
	skf.get_n_splits(X, y)
	
	i = 1
	
	for train_index, val_index in skf.split(X, y):
		
		X_train, X_val = X.iloc[train_index], X.iloc[val_index]
		y_train, y_val = y.iloc[train_index], y.iloc[val_index]
		
		if model == 'LogisticRegression':
			clf = LogisticRegression(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'SVM':
			clf = SVC(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'RandomForest':
			clf = RandomForest(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'XGBoost':
			dtrain = xgb.DMatrix(X_train,label=y_train)
			dtest = xgb.DMatrix(X_val,label=y_val)
			evallist = [(dtest, 'eval'), (dtrain, 'train')]
			parameters['objective'] = 'binary:logistic'
			parameters['silent'] = 1
			bst = xgb.train(parameters,dtrain,num_round,evallist,verbose_eval=False)
			predictions = np.round(bst.predict(dtest))
		else:
			raise Exception('Model not defined')


		metrics['accuracy'].append(accuracy_score(y_val,predictions))
		metrics['precision'].append(precision_score(y_val,predictions,average='binary'))
		metrics['recall'].append(recall_score(y_val,predictions,average='binary'))
		metrics['f1score'].append(f1_score(y_val,predictions,average='binary'))
		metrics['roc_auc'].append(roc_auc_score(y_val,predictions))
		
		if verbose == True:
			print('Cross validation set {} complete'.format(str(i)))
			
		i+=1
   
	time_end = time.time()
	time_taken = time_end-time_start
	
	metrics_aggregated = {'model': model,
				'pos_neg_ratio': train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)].shape[0]/train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'parameters': str(parameters),
				'step': 'Binary',
				'cluster': cluster,
				'accuracy': np.mean(metrics['accuracy']),
				'precision': np.mean(metrics['precision']),
				'recall': np.mean(metrics['recall']),
				'f1score': np.mean(metrics['f1score']),
				'roc_auc': np.mean(metrics['roc_auc']),
				'time_taken': time_taken
				}
	
	if verbose == True:
		print('Cluster {} complete'.format(str(cluster)))
			
	return metrics_aggregated


	

def train_multi_level(train_data, cluster, model, parameters, num_round = 10, verbose=False):					
	
	# num_round for xgboost only
		
	if verbose == True:
		print('Training cluster {}:'.format(str(cluster)))
		
	time_start = time.time()
	metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1score': [], 'roc_auc': []}

	cluster_train = train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)]
	skf = StratifiedKFold(n_splits=5, shuffle=False)
	X = traffic_only_train.drop(columns=['level_binary','level_max','cluster'])
	y = traffic_only_train['level_max']
	skf.get_n_splits(X, y)
	
	i = 1
	
	for train_index, val_index in skf.split(X, y):
		
		X_train, X_val = X.iloc[train_index], X.iloc[val_index]
		y_train, y_val = y.iloc[train_index], y.iloc[val_index]
		
		if model == 'LogisticRegression':
			clf = LogisticRegression(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'SVM':
			clf = SVC(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'RandomForest':
			clf = RandomForest(**parameters)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_val)
		elif model == 'XGBoost':
			dtrain = xgb.DMatrix(X_train,label=y_train)
			dtest = xgb.DMatrix(X_val,label=y_val)
			evallist = [(dtest, 'eval'), (dtrain, 'train')]
			parameters['objective'] = 'multi:softmax'
			parameters['silent'] = 1
			bst = xgb.train(parameters,dtrain,num_round,evallist,verbose_eval=False)
			predictions = np.round(bst.predict(dtest))
		else:
			raise Exception('Model not defined')


		metrics['accuracy'].append(accuracy_score(y_val,predictions))
		metrics['precision'].append(precision_score(y_val,predictions,average='weighted'))
		metrics['recall'].append(recall_score(y_val,predictions,average='weighted'))
		metrics['f1score'].append(f1_score(y_val,predictions,average='weighted'))
		metrics['roc_auc'].append(roc_auc_score(y_val,predictions))
		
		if verbose == True:
			print('Cross validation set {} complete'.format(str(i)))
			
		i+=1
   
	time_end = time.time()
	time_taken = time_end-time_start
	
	metrics_aggregated = {'model': model,
				'pos_neg_ratio': train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)].shape[0]/train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'train_size': train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'parameters': str(parameters),
				'step': 'Multi',
				'cluster': cluster,
				'accuracy': np.mean(metrics['accuracy']),
				'precision': np.mean(metrics['precision']),
				'recall': np.mean(metrics['recall']),
				'f1score': np.mean(metrics['f1score']),
				'roc_auc': np.mean(metrics['roc_auc']),
				'time_taken': time_taken
				}
	
	if verbose == True:
		print('Cluster {} complete'.format(str(cluster)))
			
	return metrics_aggregated
	
	
	
def test_binary_level(train_data, test_data, cluster, param_inputs, num_round = 10):

	print('Starting predictions for cluster {} and parameters {}:'.format(str(cluster),str(param_inputs)))
	time_start = time.time()
	
	cluster_train = train_data.loc[train_data['cluster'] == cluster]
	cluster_test = test_data.loc[test_data['cluster'] == cluster]
	X_train = cluster_train.drop(columns=['level_binary','level_max','cluster'])
	y_train = cluster_train['level_max']
	X_test = cluster_test.drop(columns=['level_binary','level_max','cluster'])
	y_test = cluster_test['level_max']

	model = param_inputs['model']
	param = param_inputs['param']
			
	if model == 'LogisticRegression':
		clf = LogisticRegression(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'SVM':
		clf = SVC(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'RandomForest':
		clf = RandomForest(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'XGBoost':
		dtrain = xgb.DMatrix(X_train,label=y_train)
		dtest = xgb.DMatrix(X_test,label=y_val)
		param['objective'] = 'multi:softmax'
		param['silent'] = 1
		bst = xgb.train(param,dtrain,num_round,verbose_eval=False)
		predictions = bst.predict(dtest)
	else:
		raise Exception('Model not defined')
		
   
	time_end = time.time()
	time_taken = time_end-time_start
	
	metrics = {'model': model,
				'pos_neg_ratio': train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)].shape[0]/train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'train_size': train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'parameters': str(param),
				'step': 'Binary',
				'cluster': cluster,
				'accuracy': accuracy_score(y_test,predictions),
				'precision': precision_score(y_test,predictions,average='binary'),
				'recall': recall_score(y_test,predictions,average='binary'),
				'f1score': f1_score(y_test,predictions,average='binary'),
				'roc_auc': roc_auc_score(y_test,predictions,average='binary'),
				'time_taken': time_taken
				}
	
	print('Training and testing complete')
	
	return predictions, metrics

	

def test_multi_level(train_data, test_data, cluster, param_inputs, num_round = 10):

	print('Starting predictions for cluster {} and parameters {}:'.format(str(cluster),str(param_inputs)))
	time_start = time.time()
	
	cluster_train = train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)]
	cluster_test = test_data.loc[(test_data['cluster'] == cluster) & (test_data['level_binary_predictions'] == 1)]
	X_train = cluster_train.drop(columns=['level_binary','level_max','cluster'])
	y_train = cluster_train['level_max']
	X_test = cluster_test.drop(columns=['level_binary','level_max','cluster'])
	y_test = cluster_test['level_max']

	model = param_inputs['model']
	param = param_inputs['param']
			
	if model == 'LogisticRegression':
		clf = LogisticRegression(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'SVM':
		clf = SVC(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'RandomForest':
		clf = RandomForest(**param)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
	elif model == 'XGBoost':
		dtrain = xgb.DMatrix(X_train,label=y_train)
		dtest = xgb.DMatrix(X_test,label=y_val)
		param['objective'] = 'multi:softmax'
		param['silent'] = 1
		bst = xgb.train(param,dtrain,num_round,verbose_eval=False)
		predictions = bst.predict(dtest)
	else:
		raise Exception('Model not defined')
		
   
	time_end = time.time()
	time_taken = time_end-time_start
	
	metrics = {'model': model,
				'pos_neg_ratio': train_data.loc[(train_data['cluster'] == cluster) & (train_data['level_binary'] == 1)].shape[0]/train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'train_size': train_data.loc[(train_data['cluster'] == cluster)].shape[0],
				'parameters': str(param),
				'step': 'Multi',
				'cluster': cluster,
				'accuracy': accuracy_score(y_test,predictions),
				'precision': precision_score(y_test,predictions,average='weighted'),
				'recall': recall_score(y_test,predictions,average='weighted'),
				'f1score': f1_score(y_test,predictions,average='weighted'),
				'roc_auc': roc_auc_score(y_test,predictions,average='weighted'),
				'time_taken': time_taken
				}
	
	print('Training and testing complete')
	
	return predictions, metrics
