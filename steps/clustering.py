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
from sklearn.model_selection import train_test_split

import experiments


class clustering:

    def __init__(self, conn, file_args, df):
        self.sqlalchemy_conn = conn['sqlalchemy_conn']
        self.file_args = file_args
        self.write_path = self.file_args['write_path']
        self.train_test_ratio_clst = file_args['train_test_ratio_clst']
        self.sql_chunksize = self.file_args['sql_chunksize']
        self.df = df
        self.seed = file_args['seed']
        
    def prepare_data(self):
        """ 
        the preprocessing stage involves only three steps:
                1) drop all features that will not be used. this includes all non-numeric features
                2) scale features so clustering algorithm is not weighed towards any feature with larger range
                3) train test split the data (4:1 split used)
        """
        cluster_data_df = self.df.drop(columns=['index','date','time','datetime','segment_id','street',
                                                'level_min','level_max','level_mean','level_count','level_binary'])
        cluster_data_df = pd.DataFrame(preprocessing.scale(cluster_data_df),columns=cluster_data_df.columns)
        train_data, test_data = train_test_split(cluster_data_df, random_state=self.seed, test_size=self.train_test_ratio_clst)
        return train_data, test_data

    def cluster_and_predict(self, train_data, test_data):
        pred_train, pred_test = experiments.cluster(train_data, test_data)
        return pred_train, pred_test
        
    def add_cluster_pred_df(self, pred_train, pred_test, train_data, test_data):
        train_data['set'] = 'train'
        test_data['set'] = 'test'
        predictions_all = np.concatenate((pred_train,pred_test))
        df_test = train_data.append(test_data)
        df_test2 = df_test.merge(pd.DataFrame(predictions_all, columns=['cluster'], 
                                                        index = np.concatenate((train_data.index,test_data.index))),
                                                        left_index=True, right_index=True)
        sampled_df_clustered = self.df.merge(df_test2[['set','cluster']], right_index=True, left_index=True)
        return sampled_df_clustered

    def run(self):
        """ 
        clustering steps as originally done in notebook 
        """
        logger.info('Starting clustering.')
        train_data, test_data = self.prepare_data()
        pred_train, pred_test = self.cluster_and_predict(train_data, test_data)
        sampled_df_clustered = self.add_cluster_pred_df(pred_train, pred_test, train_data, test_data)
        if self.file_args['write_data_sql'] == True:
            sampled_df_clustered.to_sql(name='clustered_training_data', con=self.sqlalchemy_conn, if_exists='replace', index=False, chunksize=self.sql_chunksize)
            logger.info('Saved processed_training_data to SQL database.')
        if self.file_args['write_data_pkl'] == True:
            pickle.dump(sampled_df_clustered, open(self.write_path+'clustered_training_data.pkl', 'wb'), protocol=4)
            logger.info('Saved pkl file "clustered_training_data.pkl" in current directory.')
        logger.info('Clustering complete.')
        return sampled_df_clustered