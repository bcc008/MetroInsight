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

class process_data:

    def __init__(self, conn, file_args):
        self.file_args = file_args
        self.pg_conn = conn['pg_conn']
        self.sqlalchemy_conn = conn['sqlalchemy_conn']
        self.cur = conn['pg_conn'].cursor()
        self.negative_positive_ratio = self.file_args['negative_positive_ratio']
        self.level_max_average = self.file_args['level_max_average']
        self.highway_features_two_way = self.file_args['highway_features_two_way']
        self.event_attendance_threshold = self.file_args['event_attendance_threshold']
        self.event_distance_threshold = self.file_args['event_distance_threshold']
        self.event_start_window_before = self.file_args['event_start_window_before']
        self.event_start_window_after = self.file_args['event_start_window_after']
        self.event_end_window_before = self.file_args['event_end_window_before']
        self.event_end_window_after = self.file_args['event_end_window_after']
        self.sql_chunksize = self.file_args['sql_chunksize']
        self.streets = self.file_args['streets']
        self.skip_sampling = self.file_args['skip_sampling']
        self.load_data_sql_or_pkl = self.file_args['load_data_sql_or_pkl']
        self.write_path = self.file_args['write_path']
        
    """ Functions to help data processing steps start here """

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Written by Jon Anderson on StackOverFlow.
        
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        All args must be of equal length.    
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        mi = (3958.756 * c) + 0.0001
        return mi
        
    def build_highway_features_two_way(self,df):
        """ 
        Function to build highway features. 
        Used if setting direction of streets as just 1 vs -1 (N/E) vs (S/W).
        """
        # Create highway/freeway features and set to 0
        for highway in [i[0:-2] for i in self.streets]:
            df[highway] = 0
        # Fill North and East with value 1
        for highway in [i for i in self.streets if i[-1] in ['N','E']]:
            df[highway[0:-2]].loc[df['street'].str.contains(highway)] = 1
        # Fill South and West with value -1
        for highway in [i for i in self.streets if i[-1] in ['S','W']]:
            df[highway[0:-2]].loc[df['street'].str.contains(highway)] = -1
        return df

    def build_highway_features_one_way(self,df):
        """ 
        Function to build highway features. 
        Used if setting to set direction of street to own feature.
        """
        # Create highway/freeway features and set to 0
        for highway in self.streets:
            df[highway] = 0
        # Fill North and East with value 1
        for highway in [i for i in self.streets if i[-1] in ['N','E']]:
            df[highway].loc[df['street'].str.contains(highway)] = 1
        # Fill South and West with value -1
        for highway in [i for i in self.streets if i[-1] in ['S','W']]:
            df[highway].loc[df['street'].str.contains(highway)] = 1
        return df
        
    def clean_event_title(self, e):
        e_clean = e.translate(str.maketrans("","",string.punctuation)).replace(' ','_')
        return e_clean
        
    """ Data process steps start here """
        
    def build_sampled_df(self):
        # if subsampling time_level_segments
        if self.negative_positive_ratio > 0:
            self.cur.execute("""
                        SELECT count(*) FROM segments_time_level_selected WHERE NOT level_mean IS NULL
                        """)
            results = self.cur.fetchone()
            positive_count = results[0]
            logger.info('Number of segments selected: '+str(positive_count))
            df = pd.read_sql("""(SELECT * FROM segments_time_level_selected WHERE NOT level_mean IS NULL) UNION 
                                    (SELECT * FROM segments_time_level_selected WHERE level_mean IS NULL ORDER BY RANDOM() LIMIT """ + 
                                    str(positive_count*self.negative_positive_ratio) + ');',
                                    con=self.pg_conn)
        # take all from time_level_segments
        elif self.negative_positive_ratio == 'all':
            df = pd.read_sql("""SELECT * FROM segments_time_level_selected;""" ,con=self.pg_conn)
        else: 
            raise("""negative_positive_ratio must be positive number or all'""")
        if self.file_args['write_data_sql'] == True:
            df.to_sql(name='sampled_data', con=self.sqlalchemy_conn, if_exists='replace', index=False, chunksize=self.sql_chunksize)
        if self.file_args['write_data_pkl'] == True:
            pickle.dump(df, open(self.write_path+'sampled_data.pkl', 'wb'), protocol=4)
        logger.info('Building sampled_df complete.')
        return df
    
    def clean_data(self, df):
        # replace na values with zeros for assumption of no congestion
            level_cols = [c for c in df.columns if c.startswith('level')]
            for c in level_cols:
                df[c].fillna(0, inplace=True)
        # create 'target' column and set it to appropriate value based on input
            logger.info('Creating level_binary column.')
            df['level_binary'] = 0
            df['level_binary'][df['level_mean'] != 0] = 1
        # add date_idx for number of days since earliest date
            logger.info('Adding date_idx , for number of days since earliest date.')
            td = pd.to_datetime(df['date']) - pd.to_datetime(df.date.min())
            date_idx_vals = (td / np.timedelta64(1, 'D')).astype(int)
            df['date_idx'] = date_idx_vals
        # add time_idx for number of minutes since midnight
            logger.info('Adding time_idx for number of minutes since midnight.')
            time_idx_vals = list(map(lambda t: t.hour*60 + t.minute, df['time'].values))
            df['time_idx'] = time_idx_vals
        # encode categorical data using label encoder - do not encode date and time
            logger.info('Encoding categorical columns as numeric.')
            le = preprocessing.LabelEncoder()
            for col in ['day_of_week','month']:
                logger.info('Processing {} column.'.format(col))
                df[col] = le.fit_transform(df[col]) 
        # one hot encode the road_type column. this is necessary because they are not necessarily numeric columns
            df = pd.concat([df,pd.get_dummies(df['road_type'],prefix='road_type')],axis=1)
            df.drop(columns=['road_type'], inplace=True)
            logger.info('Clean data complete.')
        # add datetime field
            df['datetime'] = df[['date','time']].apply(lambda row: dt.datetime.combine(row['date'], row['time']), axis=1)
            return df
            
    def create_features_for_major_highways(self, df):
        if self.highway_features_two_way == True:
            df = self.build_highway_features_two_way(df)
            logger.info('Creating features for major highways (two directions per feature) complete.')
        if self.highway_features_two_way == False:
            df = self.build_highway_features_one_way(df)
            logger.info('Creating features for major highways (one direction per feature) complete.')
        return df

    def convert_datetime(self,x):
        date = str(x['COLLISION_DATE'])[0:4]+'/'+str(x['COLLISION_DATE'])[4:6]+'/'+str(x['COLLISION_DATE'])[6:8]
        time = str(x['COLLISION_TIME'])[0:2]+':'+str(x['COLLISION_TIME'])[2:4]+':'+'00'
        datetime = pd.to_datetime(date+' '+time)
        return datetime
        
    def create_segment_average(self, df):
        df = df.sort_values(['segment_id','day_of_week','time_idx','date']).copy(deep=True)
        df = df.reset_index()
        if self.level_max_average == 'time series':
            df['level_max_average'] = df.groupby(['segment_id','day_of_week','time_idx'])['level_max'].cumsum()/(np.floor(df['date_idx']/7.0)+1)
            logger.info('Create level max average complete (time series).')
        if self.level_max_average == 'all':
            ndow = df.groupby(['day_of_week'])['date'].nunique()
            overall_level_max_average = df.groupby(['segment_id','day_of_week','time_idx'])['level_max'].sum()
            overall_level_max_average = overall_level_max_average.reset_index()
            overall_level_max_average['level_max_average'] = overall_level_max_average.apply(lambda x: x['level_max']/ndow[x['day_of_week']], axis=1)
            overall_level_max_average.drop(['level_max'], axis=1, inplace=True)
            df = pd.merge(df, overall_level_max_average, how = 'left', on=['segment_id','day_of_week','time_idx'])
            logger.info('Create level max average complete (all).')
        return df

    def add_non_padres_events(self, df):
        # get events from dataframe
            events_df = pd.read_sql('SELECT event_type,event_start,event_end,exp_attendance,latitude,longitude FROM events', con=self.pg_conn)
            events_df['event_start'] = pd.to_datetime(events_df['event_start'])
            events_df['event_end'] = pd.to_datetime(events_df['event_end'])
        # modify str
            events_df['exp_attendance'] = events_df['exp_attendance'].map(lambda x: re.sub('[^0-9]','',x))
            events_df['exp_attendance'].loc[events_df['exp_attendance'] == ''] = np.NaN
            events_df['exp_attendance'] = events_df['exp_attendance'].astype('float64')
        # subset to events larger than event_attendance_threshold
            events_of_interest = events_df[(events_df['exp_attendance']>=self.event_attendance_threshold) & 
                                            events_df['event_type'].isin(['ATHLETIC','FESTIVAL','CONCERTS'])]
        # add columns for events
            df['event_festival'] = 0
            df['event_athletic'] = 0
            df['event_concerts'] = 0
            df['distance_festival'] = 0.0
            df['distance_athletic'] = 0.0
            df['distance_concerts'] = 0.0
            # df['exp_attendance'] = 0
        # set values for padres_event column to 1 if padres game was occurring
            for index, row in events_of_interest.iterrows():    
                if math.isnan(row['latitude']) == False & math.isnan(row['longitude']) == False:
                # set active before/after game start time
                    start = row['event_start'] - dt.timedelta(hours=self.event_start_window_before)
                    end = row['event_start'] + dt.timedelta(hours=self.event_start_window_after)
                    events_sampled_df = df.loc[(df['datetime']>=start) & (df['datetime']<=end)]
                    df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'distance_'+row['event_type'].lower()] = \
                        events_sampled_df.apply(lambda x: max(1/self.haversine(row['latitude'],row['longitude'],x['lat1'],x['lon1']), x['distance_'+row['event_type'].lower()]), axis=1)
                    df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df['distance_'+row['event_type'].lower()]>=self.event_distance_threshold), 'event_'+row['event_type'].lower()] = 1
                # set active before/after game end time
                    start = row['event_end'] - dt.timedelta(hours=self.event_end_window_before)
                    end = row['event_end'] + dt.timedelta(hours=self.event_end_window_after)
                    events_sampled_df = df.loc[(df['datetime']>=start) & (df['datetime']<=end)]
                    df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'distance_'+row['event_type'].lower()] = \
                        events_sampled_df.apply(lambda x: max(1/self.haversine(row['latitude'],row['longitude'],x['lat1'],x['lon1']), x['distance_'+row['event_type'].lower()]), axis=1)
                    df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df['distance_'+row['event_type'].lower()]>=self.event_distance_threshold), 'event_'+row['event_type'].lower()] = 1
        # drop added datetime column
            logger.info('Add non-Padres events complete. Distance threshold (inverse): {}'.format(self.event_distance_threshold))
            return df
            
    def add_padres_events(self, df):
        # get padres from database
            padres_df = pd.read_sql('SELECT * FROM padres_games', con=self.pg_conn)
            padres_df['game_start'] = pd.to_datetime(padres_df['game_start'])
            padres_df['game_end'] = pd.to_datetime(padres_df['game_end'])
        # add padres_game column to data
            df['event_padres'] = 0
            df['distance_padres'] = 0.0
        # set values for padres_event column to 1 if padres game was occurring
            petco_park_coordinates = (32.7076,-117.1570)
            for index, row in padres_df.iterrows():
        # set active before/after game start time
                start = row['game_start'] - dt.timedelta(hours=self.event_start_window_before)
                end = row['game_start'] + dt.timedelta(hours=self.event_start_window_after)
                padres_sampled_df = df.loc[(df['datetime']>=start) & (df['datetime']<=end)]
                df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'event_padres'] = 1
                df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'distance_padres'] = padres_sampled_df.apply(lambda x: 1/self.haversine(32.707,-117.1570,x['lat1'],x['lon1']), axis=1)
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df['distance_padres']>=self.event_distance_threshold), 'event_padres'] = 1
        # set active before/after game end time
                start = row['game_end'] - dt.timedelta(hours=self.event_end_window_before)
                end = row['game_end'] + dt.timedelta(hours=self.event_end_window_after)
                padres_sampled_df = df.loc[(df['datetime']>=start) & (df['datetime']<=end)]
                df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'event_padres'] = 1
                df.loc[(df['datetime']>=start) & (df['datetime']<=end), 'distance_padres'] = padres_sampled_df.apply(lambda x: 1/self.haversine(32.707,-117.1570,x['lat1'],x['lon1']), axis=1)
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df['distance_padres']>=self.event_distance_threshold), 'event_padres'] = 1
        # drop added datetime column
            logger.info('Add Padres events complete. Distance threshold (inverse): {}'.format(self.event_distance_threshold))
            return df

    def add_traffic_events(self,df):
        streets_collisions =     {
                                 'I-5 N': ['I-5 N','INTERSTATE 5 N','INTERSTATE 5 (N'],
                                 'I-8 E': ['I-8 E','INTERSTATE 8 E','INTERSTATE 8 (E'],
                                 'I-15 N': ['I-15 N','INTERSTATE 15 N','INTERSTATE 15 (N'],
                                 'I-805 N': ['I-805 N','INTERSTATE 805 N','INTERSTATE 805 (N'],
                                 'SR-15 N': ['SR-15 N','STATE ROUTE 15 N','STATE ROUTE 15 (N'],
                                 'SR-52 E': ['SR-52 E','STATE ROUTE 52 E','STATE ROUTE 52 (E'],
                                 'SR-54 E': ['SR-54 E','STATE ROUTE 54 E','STATE ROUTE 54 (E'],
                                 'SR-56 E': ['SR-56 E','STATE ROUTE 56 E','STATE ROUTE 56 (E'],
                                 'SR-67 N': ['SR-67 N','STATE ROUTE 67 N','STATE ROUTE 67 (N'],
                                 'SR-75 N': ['SR-75 N','STATE ROUTE 75 N','STATE ROUTE 75 (N'],
                                 'SR-94 E': ['SR-94 E','STATE ROUTE 94 E','STATE ROUTE 94 (E'],
                                 'SR-125 N': ['SR-125 N','STATE ROUTE 125 N','STATE ROUTE 125 (N'],
                                 'SR-163 N': ['SR-163 N','STATE ROUTE 163 N','STATE ROUTE 163 (N'],
                                 'SR-905 E': ['SR-905 E','STATE ROUTE 905 E','STATE ROUTE 905 (E'],
                                 'I-5 S': ['I-5 S','INTERSTATE 5 S','INTERSTATE 5 (S'],
                                 'I-8 W': ['I-8 W','INTERSTATE 8 W','INTERSTATE 8 (W'],
                                 'I-15 S': ['I-15 S','INTERSTATE 15 S','INTERSTATE 15 (S'],
                                 'I-805 S': ['I-805 S','INTERSTATE 805 S','INTERSTATE 805 (S'],
                                 'SR-15 S': ['SR-15 S','STATE ROUTE 15 S','STATE ROUTE 15 (S'],
                                 'SR-52 W': ['SR-52 W','STATE ROUTE 52 W','STATE ROUTE 52 (W'],
                                 'SR-54 W': ['SR-54 W','STATE ROUTE 54 W','STATE ROUTE 54 (W'],
                                 'SR-56 W': ['SR-56 W','STATE ROUTE 56 W','STATE ROUTE 56 (W'],
                                 'SR-67 S': ['SR-67 S','STATE ROUTE 67 S','STATE ROUTE 67 (S'],
                                 'SR-75 S': ['SR-75 S','STATE ROUTE 75 S','STATE ROUTE 75 (S'],
                                 'SR-94 W': ['SR-94 W','STATE ROUTE 94 W','STATE ROUTE 94 (W'],
                                 'SR-125 S': ['SR-125 S','STATE ROUTE 125 S','STATE ROUTE 125 (S'],
                                 'SR-163 S': ['SR-163 S','STATE ROUTE 163 S','STATE ROUTE 163 (S'],
                                 'SR-905 W': ['SR-905 W','STATE ROUTE 905 W','STATE ROUTE 905 (W']
                                }
        # load collision data
        collisions_df = pd.read_csv('../CollisionRecords.txt',sep=',',dtype={'COLLISION_TIME': np.str})
        collisions_df = collisions_df.loc[(collisions_df['LATITUDE'] < 32.9) & (collisions_df['LONGITUDE'] > 116.9)]
        collisions_df = collisions_df.loc[(collisions_df['COLLISION_DATE'] > 20170131) & (collisions_df['COLLISION_DATE'] < 20170701)]
        collisions_df = collisions_df.loc[collisions_df['COLLISION_TIME'] != '2500']
        collisions_df.loc[collisions_df['COLLISION_SEVERITY'] == 0, 'COLLISION_SEVERITY'] = 5
        collisions_df['COLLISION_SEVERITY_FLIP'] = collisions_df.apply(lambda x: 1/x['COLLISION_SEVERITY'], axis = 1)
        # add collision to data
        df['event_accident'] = 0
        df['distance_accident'] = 0.0
        df['timesince_accident'] = 0.0
        df['severity_accident'] = 0.0
        accidents_max_hours = 2
        for k,v in streets_collisions.items():
            highway_accidents_df = collisions_df.iloc[0:0]
            for name in v:
                highway_subset_df = collisions_df[collisions_df['PRIMARY_RD'].str.startswith(name)]
                highway_accidents_df = highway_accidents_df.append(highway_subset_df)
            for index, row in highway_accidents_df.iterrows():
                start = self.convert_datetime(row)
                end = self.convert_datetime(row)+dt.timedelta(hours=accidents_max_hours)
                highway_sampled_df = df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df[k] == 1) & (df['severity_accident'] <= row['COLLISION_SEVERITY_FLIP'])]
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df[k] == 1) & (df['severity_accident'] <= row['COLLISION_SEVERITY_FLIP']), 
                               'event_accident'] = 1
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df[k] == 1) & (df['severity_accident'] <= row['COLLISION_SEVERITY_FLIP']), 
                               'distance_accident'] = highway_sampled_df.apply(lambda x: self.haversine(row['LATITUDE'],-row['LONGITUDE'],x['lat1'],x['lon1']), axis=1)
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df[k] == 1) & (df['severity_accident'] <= row['COLLISION_SEVERITY_FLIP']), 
                               'timesince_accident'] = highway_sampled_df.apply(lambda x: (x['datetime']-convert_datetime(row)).seconds, axis=1)
                df.loc[(df['datetime']>=start) & (df['datetime']<=end) & (df[k] == 1) & (df['severity_accident'] <= row['COLLISION_SEVERITY_FLIP']), 
                               'severity_accident'] = row['COLLISION_SEVERITY_FLIP']
        logger.info('Add collision event completed')
        return df
            
    def run(self):
        """ 
        data processing steps as originally done in notebook 
        """
        logger.info('Starting data processing.')
        if self.skip_sampling == False:
            sampled_df = self.build_sampled_df()
        elif self.skip_sampling == True:
            if self.load_data_sql_or_pkl == 'sql':
                sampled_df = pd.read_sql('SELECT * FROM sampled_data', con=self.pg_conn)
            if self.load_data_sql_or_pkl == 'pkl':
                sampled_df = pickle.load(open(self.file_args['write_path']+'sampled_data.pkl','rb'))
        sampled_df = self.clean_data(sampled_df)
        sampled_df = self.create_features_for_major_highways(sampled_df)
        sampled_df = self.create_segment_average(sampled_df)
        sampled_df = self.add_non_padres_events(sampled_df)
        sampled_df = self.add_padres_events(sampled_df)
        sampled_df = self.add_traffic_events(sampled_df)
        sampled_df = sampled_df.loc[sampled_df['date_idx'] > 6]
        if self.file_args['write_data_sql'] == True:
            sampled_df.to_sql(name='processed_training_data', con=self.sqlalchemy_conn, if_exists='replace', index=False, chunksize=self.sql_chunksize)
            logger.info('Saved processed_training_data to SQL database.')
        if self.file_args['write_data_pkl'] == True:
            pickle.dump(sampled_df, open(self.write_path+'processed_training_data.pkl', 'wb'), protocol=4)
            logger.info('Saved pkl file "processed_training_data.pkl" in current directory.')
        logger.info('Data processing complete.')
        return sampled_df