import numpy as np
import pandas as pd
import psycopg2 as pg
import datetime as dt
import os
import string
import re
import pickle
import math
import time
import yaml
import warnings
import logging
from sqlalchemy import create_engine

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

import experiments

class prediction:

    def __init__(self, conn, file_args, df):
        self.pg_conn = conn['pg_conn']
        self.sqlalchemy_conn = conn['sqlalchemy_conn']
        self.cur = conn['pg_conn'].cursor()
        self.file_args = file_args
        self.binary_validation_metric = file_args['binary_validation_metric']
        self.multi_validation_metric = file_args['multi_validation_metric']
        self.write_path = file_args['write_path']
        self.df = df
        self.seed = file_args['seed']
        self.version = experiments.version()
        self.cross_validation_folds = file_args['cross_validation_folds']
        self.train_experts = file_args['train_experts']
        self.train_test_ratio_clst = file_args['train_test_ratio_clf']
        self.val_metrics_df = pd.DataFrame(columns=['version','experiment','model','parameters','step','cluster',
                                                    'accuracy','precision','recall','f1score','roc_auc',
                                                    'confusion_matrix','rmse','mae','r2',
                                                    'time_taken','test_set','train_size','pos_neg_ratio'])
        self.test_metrics_df = pd.DataFrame(columns=['version','experiment','model','parameters','step','cluster',
                                                    'accuracy','precision','recall','f1score','roc_auc',
                                                    'confusion_matrix','rmse','mae','r2',
                                                    'time_taken','test_set','train_size','pos_neg_ratio'])
    
    def check_experiment(self):
        cur = self.pg_conn.cursor()
        cur.execute("SELECT * FROM information_schema.tables WHERE table_name='test_metrics'")
        if bool(cur.rowcount):
            cur.execute("SELECT max(experiment) FROM test_metrics WHERE version = {}".format(str(self.version)))
            if cur.fetchone()[0] != None:
                cur.execute("SELECT max(experiment) FROM test_metrics WHERE version = {}".format(str(self.version)))
                experiment_number = cur.fetchone()
                self.experiment = experiment_number[0]+1
            else:
                self.experiment = 0
        else:
            self.experiment = 0
        logger.info('Running version {}, experiment {}'.format(str(self.version), str(self.experiment)))

    def prepare_data(self):
        logger.info('Preparing data set by dropping non-train columns.')
        prepared_df = self.df.drop(columns=['index','date','time','datetime','segment_id','street',
                                         'level_min','level_mean','level_count'])
        return prepared_df

    def split_train_test(self, df):
        if self.train_experts == False:
            if 'set' in df.columns:
                df = df.drop(columns=['set'])
            train_data, test_data = train_test_split(df, random_state=self.seed, test_size=self.train_test_ratio_clst)
        else:
            train_data = df.loc[df['set'] == 'train'].drop(columns=['set'])
            test_data = df.loc[df['set'] == 'test'].drop(columns=['set'])
        return train_data, test_data

    def validate_binary(self, df, params, cluster=None):
        time_start = time.time()
        logger.info('Starting validation for cluster: {}, parameters: {}, step: binary.'.format(str(cluster),str(params)))
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1score': [], 'roc_auc': [] }
        skf = StratifiedKFold(n_splits=self.cross_validation_folds, shuffle=True)
        X = df.drop(columns=['level_binary','level_max','cluster'])
        y = df['level_binary']
        skf.get_n_splits(X, y)
        i = 1
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            predictions, model_name = experiments.binary_clf(X_train, X_val, y_train, y_val, params)
            metrics['accuracy'].append(accuracy_score(y_val,predictions))
            metrics['precision'].append(precision_score(y_val,predictions,average='binary'))
            metrics['recall'].append(recall_score(y_val,predictions,average='binary'))
            metrics['f1score'].append(f1_score(y_val,predictions,average='binary'))
            metrics['roc_auc'].append(roc_auc_score(y_val,predictions))
            logger.debug('Cross validation set {} complete.'.format(str(i)))
            i+=1
        time_end = time.time()
        time_taken = time_end-time_start
        logger.info('Finished validation.')
        if cluster is None:
            train_size = df.shape[0]
        else:
            train_size = df.loc[(df['cluster'] == cluster)].shape[0]
        metrics_aggregated = {
                    'version': self.version,
                    'experiment': self.experiment,
                    'model': model_name,
                    'pos_neg_ratio': df.loc[df['level_binary'] == 1].shape[0]/df.shape[0],
                    'train_size': train_size,
                    'parameters': str(params),
                    'step': 'Binary',
                    'cluster': cluster,
                    'accuracy': np.mean(metrics['accuracy']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1score': np.mean(metrics['f1score']),
                    'roc_auc': np.mean(metrics['roc_auc']),
                    'confusion_matrix': None,
                    'rmse': None,
                    'mae': None,
                    'r2': None,
                    'test_set': 'validation',
                    'time_taken': time_taken
                    }
        self.val_metrics_df = self.val_metrics_df.append(pd.DataFrame(metrics_aggregated, index=[0]), ignore_index=True)

    def validate_multi(self, df, params, cluster=None):
        time_start = time.time()
        logger.info('Starting validation for cluster: {}, parameters: {}, step: multi.'.format(str(cluster),str(params)))
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1score': [], 'rmse': [], 'mae': [], 'r2': [] }
        traffic_only_train = df.loc[df['level_binary'] == 1]
        skf = StratifiedKFold(n_splits=self.cross_validation_folds, shuffle=True)
        X = traffic_only_train.drop(columns=['level_binary','level_max','cluster'])
        y = traffic_only_train['level_max']
        skf.get_n_splits(X, y)
        i = 1
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            predictions, model_name = experiments.multi_clf(X_train, X_val, y_train, y_val, params)
            metrics['accuracy'].append(accuracy_score(y_val,predictions))
            metrics['precision'].append(precision_score(y_val,predictions, average='weighted'))
            metrics['recall'].append(recall_score(y_val,predictions, average='weighted'))
            metrics['f1score'].append(f1_score(y_val,predictions, average='weighted'))
            metrics['rmse'].append(math.sqrt(mean_squared_error(y_val,predictions)))
            metrics['mae'].append(mean_absolute_error(y_val,predictions))
            metrics['r2'].append(r2_score(y_val,predictions))
            logger.debug('Cross validation set {} complete: {}'.format(str(i), str(time.time())))
            i+=1
        time_end = time.time()
        time_taken = time_end-time_start
        logger.info('Finishing validation.')
        if cluster is None:
            train_size = df.shape[0]
        else:
            train_size = df.loc[(df['cluster'] == cluster)].shape[0]
        metrics_aggregated = {
                    'version': self.version,
                    'experiment': self.experiment,
                    'model': model_name,
                    'pos_neg_ratio': df.loc[df['level_binary'] == 1].shape[0]/df.shape[0],
                    'train_size': train_size,
                    'parameters': str(params),
                    'step': 'Multi',
                    'cluster': cluster,
                    'accuracy': np.mean(metrics['accuracy']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1score': np.mean(metrics['f1score']),
                    'roc_auc': None,
                    'confusion_matrix': None,
                    'rmse': np.mean(metrics['rmse']),
                    'mae': np.mean(metrics['mae']),
                    'r2': np.mean(metrics['r2']),
                    'test_set': 'validation',
                    'time_taken': time_taken
                    }
        self.val_metrics_df = self.val_metrics_df.append(pd.DataFrame(metrics_aggregated, index=[0]), ignore_index=True)

    def append_validation_metrics_sql(self):
        self.val_metrics_df.to_sql(name='validation_metrics', con=self.sqlalchemy_conn, if_exists='append')
        return None

    def pick_best_params(self, cluster):
        if self.train_experts == False:
            binary_query = "SELECT parameters \
                            FROM validation_metrics \
                            WHERE step = 'Binary' \
                                AND experiment = {} \
                                AND version = {} \
                            ORDER BY {} DESC \
                            LIMIT 1 \
                            ".format(str(self.experiment), str(self.version), str(self.binary_validation_metric))
            self.cur.execute(binary_query)
            binary_parameters = eval(self.cur.fetchone()[0])
            multi_query = "SELECT parameters \
                            FROM validation_metrics \
                            WHERE step = 'Multi' \
                                AND experiment = {} \
                                AND version = {} \
                            ORDER BY {} DESC \
                            LIMIT 1 \
                            ".format(str(self.experiment), str(self.version), str(self.multi_validation_metric))
            self.cur.execute(multi_query)
            multi_parameters = eval(self.cur.fetchone()[0])
        if self.train_experts == True:
            binary_query = "SELECT parameters \
                                FROM validation_metrics \
                                WHERE step = 'Binary' \
                                    AND cluster = {} \
                                    AND experiment = {} \
                                    AND version = {} \
                                ORDER BY {} DESC \
                                LIMIT 1 \
                                ".format(str(cluster), str(self.experiment), str(self.version), str(self.binary_validation_metric))
            self.cur.execute(binary_query)
            binary_parameters = eval(self.cur.fetchone()[0])
            multi_query = "SELECT parameters \
                                FROM validation_metrics \
                                WHERE step = 'Multi' \
                                    AND cluster = {} \
                                    AND experiment = {} \
                                    AND version = {} \
                                ORDER BY {} DESC \
                                LIMIT 1 \
                                ".format(str(cluster), str(self.experiment), str(self.version), str(self.multi_validation_metric))
            self.cur.execute(multi_query)
            multi_parameters = eval(self.cur.fetchone()[0])
        return binary_parameters, multi_parameters

    def test_binary(self, train_df, test_df, cluster=None):
        #binary_parameters, multi_parameters = self.pick_best_params(cluster)
        binary_parameters = experiments.binary_clf_parameters()[0]
        time_start = time.time()
        logger.info('Starting final test for cluster: {}  with parameters: {} for step: binary.'.format(str(cluster), str(binary_parameters)))
        X_train = train_df.drop(columns=['level_binary','level_max','cluster'])
        y_train = train_df['level_binary']
        X_test = test_df.drop(columns=['level_binary','level_max','cluster'])
        y_test = test_df['level_binary']
        predictions, model_name, clf = experiments.binary_clf(X_train, X_test, y_train, y_test, binary_parameters)
        time_end = time.time()
        time_taken = time_end-time_start
        logger.info('Finished testing for {} data set.')
        if cluster is None:
            train_size = train_df.shape[0]
        else:
            train_size = train_df.loc[(train_df['cluster'] == cluster)].shape[0]
        metrics_aggregated = {
                    'version': self.version,
                    'experiment': self.experiment,
                    'model': model_name,
                    'pos_neg_ratio': train_df.loc[train_df['level_binary'] == 1].shape[0]/train_df.shape[0],
                    'train_size': train_size,
                    'parameters': str(binary_parameters),
                    'step': 'Binary',
                    'cluster': cluster,
                    'accuracy': accuracy_score(y_test,predictions),
                    'precision': precision_score(y_test,predictions,average='binary'),
                    'recall': recall_score(y_test,predictions,average='binary'),
                    'f1score': f1_score(y_test,predictions,average='binary'),
                    'roc_auc': roc_auc_score(y_test,predictions),
                    'confusion_matrix': np.array2string(confusion_matrix(y_test,predictions),separator=','),
                    'rmse': None,
                    'mae': None,
                    'r2': None,
                    'test_set': 'test',
                    'time_taken': time_taken
                    }
        self.test_metrics_df = self.test_metrics_df.append(pd.DataFrame(metrics_aggregated, index=[0]), ignore_index=True)
        indices_traffic1 = np.squeeze(np.argwhere(predictions == 1))
        pickle.dump(predictions, open('binary_predictions_cluster_{}.pkl'.format(str(cluster)),'wb'))
        pickle.dump(clf, open('binary_model_cluster_{}.pkl'.format(str(cluster)),'wb'))
        return indices_traffic1

    def test_multi(self, train_df, test_df, indices_traffic1, cluster=None):
        binary_parameters, multi_parameters = self.pick_best_params(cluster)
        multi_parameters = experiments.multi_clf_parameters()[0]
        time_start = time.time()
        logger.info('Starting final test for cluster: {} with parameters: {} for step: multi.'.format(str(cluster), str(multi_parameters)))
        train_df_traffic1 = train_df.loc[train_df['level_binary'] == 1]
        test_df_traffic1 = test_df.iloc[indices_traffic1]
        X_train = train_df_traffic1.drop(columns=['level_binary','level_max','cluster'])
        y_train = train_df_traffic1['level_max']
        X_test = test_df_traffic1.drop(columns=['level_binary','level_max','cluster'])
        y_test = test_df_traffic1['level_max']
        predictions, model_name, clf = experiments.multi_clf(X_train, X_test, y_train, y_test, multi_parameters)
        time_end = time.time()
        time_taken = time_end-time_start
        logger.info('Finished testing for data set.')
        if cluster is None:
            train_size = train_df.shape[0]
        else:
            train_size = train_df.loc[(train_df['cluster'] == cluster)].shape[0]
        metrics_aggregated = {
                    'version': self.version,
                    'experiment': self.experiment,
                    'model': model_name,
                    'pos_neg_ratio': train_df_traffic1.loc[train_df_traffic1['level_binary'] == 1].shape[0]/train_df_traffic1.shape[0],
                    'train_size': train_size,
                    'parameters': str(multi_parameters),
                    'step': 'Multi',
                    'cluster': cluster,
                    'accuracy': accuracy_score(y_test,predictions),
                    'precision': precision_score(y_test,predictions,average='weighted'),
                    'recall': recall_score(y_test,predictions,average='weighted'),
                    'f1score': f1_score(y_test,predictions,average='weighted'),
                    'roc_auc': None,
                    'confusion_matrix': np.array2string(confusion_matrix(y_test,predictions),separator=','),
                    'rmse': math.sqrt(mean_squared_error(y_test,predictions)),
                    'mae': mean_absolute_error(y_test,predictions),
                    'r2': r2_score(y_test,predictions),
                    'test_set': 'test',
                    'time_taken': time_taken
                    }
        self.test_metrics_df = self.test_metrics_df.append(pd.DataFrame(metrics_aggregated, index=[0]), ignore_index=True)
        pickle.dump(predictions, open('multi_predictions_cluster_{}.pkl'.format(str(cluster)),'wb'))
        pickle.dump(clf, open('multi_model_cluster_{}.pkl'.format(str(cluster)),'wb'))

    def append_test_metrics_sql(self):
        self.test_metrics_df.to_sql(name='test_metrics', con=self.sqlalchemy_conn, if_exists='append')
        return None

    def run(self):
        logger.info('Starting model testing.')
        self.check_experiment()
        prepared_df = self.prepare_data()
        train_data, test_data = self.split_train_test(prepared_df)
        #validation
        if self.train_experts == False:
            for params in experiments.binary_clf_parameters():
                self.validate_binary(train_data, params, cluster=None)
            for params in experiments.multi_clf_parameters():
                self.validate_multi(train_data, params, cluster=None)
        if self.train_experts == True:
            for cluster in np.sort(train_data['cluster'].unique()):
                clustered_train_data = train_data.loc[train_data['cluster'] == cluster]
                clustered_train_data.reset_index(inplace=True)
                for params in experiments.binary_clf_parameters():
                    self.validate_binary(clustered_train_data, params, cluster=cluster)
                for params in experiments.multi_clf_parameters():
                    self.validate_multi(clustered_train_data, params, cluster=cluster)
        self.append_validation_metrics_sql()
        # test
        if self.train_experts == False:
            indices_traffic1 = self.test_binary(train_data, test_data, cluster=None)
            self.test_multi(train_data, test_data, indices_traffic1, cluster=None)
        if self.train_experts == True:
            for cluster in np.sort(train_data['cluster'].unique()):
                clustered_train_data = train_data.loc[train_data['cluster'] == cluster]
                clustered_train_data.reset_index(inplace=True)
                clustered_test_data = test_data.loc[test_data['cluster'] == cluster]
                clustered_test_data.reset_index(inplace=True)
                indices_traffic1 = self.test_binary(clustered_train_data, clustered_test_data, cluster=cluster)
                self.test_multi(clustered_train_data, clustered_test_data, indices_traffic1, cluster=cluster)
                del clustered_train_data
                del clustered_test_data
        self.append_test_metrics_sql()
        logger.info('Model testing complete. Experiment metrics added to database.')
