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
#from shapely.geometry import LineString, shape
#from shapely.wkb import dumps, loads
#from shapely.wkt import dumps, loads
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import xgboost as xgb
import keras

from steps.segments_times_levels import segments
from steps.segments_times_levels import times
from steps.segments_times_levels import segments_times_levels
from steps.process_data import process_data
from steps.clustering import clustering
from steps.prediction import prediction
import steps.sqlqueries
import steps.experiments

# Create Logger
logger = logging.getLogger('main_log')
logger.setLevel(logging.INFO)
# Log File
fh = logging.FileHandler('logfile.log')
fh.setLevel(logging.INFO)
# Stream Handler (print to console)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# Add Handlers
logger.addHandler(fh)
logger.addHandler(ch)


def load_config(config_str_file):
    path = open(config_str_file, 'r')
    file_args = yaml.load(path)
    return file_args

def create_pg_connection(conn_str_file):
    pg_conn_str = open(conn_str_file, 'r').read()
    pg_conn = pg.connect(pg_conn_str)
    return pg_conn

def create_sqlalchemy_connection(conn_str_file):
    sqlalchemy_conn_str = open(conn_str_file,'r').read()
    sqlalchemy_conn = create_engine(sqlalchemy_conn_str)
    return sqlalchemy_conn
    
def main():
    """ 
    Initialize parameters and run model. 
    Set steps in configuration file.
    The application runs in up to 4 steps:
        1) Build segment_times_levels_selected table. Selection parameters found in config file. Can be skipped if want to keep previous data set.
        2) Clean and build training data for clustering and modeling later. Can also be skipped if want to keep previous data set.
        3) Cluster data to create 'experts'. The 'experts' should improve training accuracy as the algorithm deals with chunks that individually contain less noise. Can be skipped.
        4) Model and predict traffic levels. If step 3 skipped, train full data set. Otherwise, train each cluster individually. This is done in two steps:
            a) Levels_binary (traffic or no traffic) 
            b) Levels_multi (if there is traffic, how much traffic will there be)
    """
    logger.info('Running main.py.')
    conn = {
            'pg_conn': create_pg_connection('./db_conn_str.txt'),
            'sqlalchemy_conn': create_sqlalchemy_connection('./sqlalchemy_conn_str.txt')
            }
    file_args = load_config('./pipeline_args.yml')
    if file_args['skip_data_selection'] == False:
        part1 = segments_times_levels(conn, file_args)
        part1.run()
    if file_args['skip_data_processing'] == False:
        part2 = process_data(conn, file_args)
        sampled_df = part2.run()
        if file_args['skip_clustering'] == False:
            part3 = clustering(conn, file_args, sampled_df)
            sampled_df_clustered = part3.run()
            part4 = prediction(conn, file_args, sampled_df_clustered)
            part4.run()
        elif file_args['skip_clustering'] == True:
            sampled_df['cluster'] = None
            part4 = prediction(conn, file_args, sampled_df)
            part4.run()
    elif file_args['skip_data_processing'] == True:
        if file_args['skip_clustering'] == False:
            if file_args['load_data_sql_or_pkl'] == 'sql':
                sampled_df = pd.read_sql("SELECT * FROM processed_training_data", con=conn['pg_conn'])
            elif file_args['load_data_sql_or_pkl'] == 'pkl':
                sampled_df = pickle.load(open(file_args['write_path']+'processed_training_data.pkl', 'rb'))
            part3 = clustering(conn, file_args, sampled_df)
            sampled_df_clustered = part3.run()
            del sampled_df
            part4 = prediction(conn, file_args, sampled_df_clustered)
            part4.run()
        elif file_args['skip_clustering'] == True:
            if file_args['train_experts'] == True:
                if file_args['load_data_sql_or_pkl'] == 'sql':
                    sampled_df_clustered = pd.read_sql("SELECT * FROM clustered_training_data", con=conn['pg_conn'])
                elif file_args['load_data_sql_or_pkl'] == 'pkl':
                    sampled_df_clustered = pickle.load(open(file_args['write_path']+'clustered_training_data.pkl', 'rb'))
                part4 = prediction(conn, file_args, sampled_df_clustered)
                part4.run()
            elif file_args['train_experts'] == False:
                if file_args['load_data_sql_or_pkl'] == 'sql':
                    sampled_df_clustered = pd.read_sql("SELECT * FROM processed_training_data", con=conn['pg_conn'])
                elif file_args['load_data_sql_or_pkl'] == 'pkl':
                    sampled_df_clustered = pickle.load(open(file_args['write_path']+'processed_training_data.pkl', 'rb'))
                sampled_df_clustered['cluster'] = None
                part4 = prediction(conn, file_args, sampled_df_clustered)
                part4.run()

if __name__ == '__main__':
    main()
