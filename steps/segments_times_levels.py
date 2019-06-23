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


class segments:
    """
    Data and queries to set relevant segments for various traffic analysis.
    This class is used to sample all available segments by 1 or more methods and store the segments
    in a table for use by other objects.

    Available methods:
    radius:         Select all segments within a given radius of a provided point.
    bounding_box:   Select all segments with a given bounding box.
    sample:         Select a random sample of the provided percent from all segments.
    street:         Select all segments on a given street name.
    road_type:      Select all segments of a given road type.
    ignore:         Select all segments with an Ignore flag of True.
    cum_seg_pct:    Select all segments such that the provided percent of positive traffic points are retained.

    This class will be used in the Pipeline notebook for exploration purposes and in the Production
    python code used to perform the actual clustering and modeling of traffic data.
    """
    def __init__(self, conn, queries, args):
        """Constructor for Segment object.
        :param conn: database connection object.
        :param queries: list of queries to be run against the main segments table.
        :param args: dict of arguments read from the arguments file.
        :return: Segment instance.
        """
        self.conn = conn
        self.args = args
        self.segments_filter = 'segment_id'
        self.segments_table = 'segments'
        self.segments_selected_table = 'segments_selected'
        self.time_table = 'time_' + str(self.args['time_resolution'])
        self.matrix_table = 'matrix_' + str(self.args['time_resolution'])
        self.selected_segments = []
        self.queries = queries
        self.query_map = {
            'radius': self.create_radius_sql,
            'bounding_box': self.create_boundingbox_sql,
            'sample': self.create_sample_sql,
            'street': self.create_street_sql,
            'road_type': self.create_road_type_sql,
            'ignore': self.create_ignore_sql,
            'cum_seg_pct': self.create_cum_seg_pct_sql
        }

    def run_queries(self):
        """
        Run all queries to select desired segments for clustering and modelling.
        :return: None
        """
        queries = []
        cur = self.conn.cursor()
        for idx, query in enumerate(self.queries):
            logger.info(idx, query)
            if idx == 0:
                table = self.segments_table
                sql_truncate = 'TRUNCATE {}; \n\n'.format(self.segments_selected_table)
                sql = self.query_map[query](table)
                sql_with = 'WITH segments_to_keep AS ( ' + sql + ' ) \n'
                sql_select_inner = 'SELECT segment_id from segments_to_keep'
                sql_where = ' WHERE segment_id IN ({})'.format(sql_select_inner)
                sql_select_outer = 'SELECT lat1, lon1, lat2, lon2, segment_id, street, city, road_type, geom, direction, seg_length, cum_seg_pct FROM {}{}'.format(table, sql_where)
                sql_insert = 'INSERT INTO {} ({})'.format(self.segments_selected_table, sql_select_outer)
                final_sql = '{}{}{} \n'.format(sql_truncate, sql_with, sql_insert)
                logger.info(final_sql)
                cur.execute(final_sql)
                self.conn.commit()
            else:
                table = self.segments_selected_table
                sql = self.query_map[query](table)
                sql_with = 'with segments_to_keep as ( ' + sql + ' ) \n'
                sql_delete = 'DELETE from ' + self.segments_selected_table
                sql_select = 'SELECT segment_id from segments_to_keep'
                sql_where = ' WHERE segment_id NOT IN ({});'.format(sql_select)
                final_sql = '{}{}{} \n'.format(sql_with, sql_delete, sql_where)
                logger.info(final_sql)
                cur.execute(final_sql)
                self.conn.commit()
        cur.close()

    def create_cum_seg_pct_sql(self, table):
        """Create SQL to create seg_cum_pct table.
        The seg_cum_pct table is used to allow the selection of segments that will retain a given percentage
        of positive traffic data point.
        :param table: name of table to select segments from.
        :return: complete sql string to select all segments to retain a given percentage of positive traffic data points.
        """
        # create seg_cum_pct table
        self.create_cum_seg_pct_table(self.conn, self.args['time_resolution'], table)
        final_sql = 'SELECT segment_id FROM seg_cum_pct WHERE cum_pos_pct <= {}'.format(str(self.args['segment_queries']['cum_seg_pct']))
        return final_sql

    def create_with(self, query):
        """DEPRECATED! 
        Create with clause for queries.
        :param query: query to add with clause to.
        :return:
        """
        sql_with = 'with segments_to_keep as ( ' + query + ' ) \n'
        return sql_with

    def create_ignore_sql(self, table):
        """Create query to select only segments with the ignore flag set to TRUE
        :param table: table to select segments from.
        :return: SQL query string
        """
        # assumes ignore filter is enabled in args
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE s.ignore = TRUE'
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def mile_to_meter(self):
        """Convert miles to meters for use in geom queries.
        :return: meters as a string.
        """
        meter = int(round(self.args['segment_queries']['radius']['input_radius'] * 1609.344))
        return str(meter)

    def sql_radius(self):
        """Create geom radius part of a query.
        :return: string to be used as where clause in segments radius query.
        """
        lat_lon = self.args['segment_queries']['radius']['input_poi'].replace(" N", "").replace(" W", "")
        lat_lon = lat_lon.split(', ')
        return "ST_DWithin(geom, ST_MakePoint(" + "-" + lat_lon[1] + "," + lat_lon[
            0] + ")::geography," + self.mile_to_meter() + ')'

    def create_radius_sql(self, table):
        """Create SQL to select all segments within a given radius (set in args).
        :param table: name of table to select segments from.
        :return: complete sql string to select all segments within a given radius.
        """
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE {}'.format(self.sql_radius())
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def bounding_box(self):
        """Create geom bounding box part of a query.
        :return: string to be used in where clause of bounding box query.
        """
        nw = self.args['segment_queries']['bounding_box']['input_nw_corner'].replace(" N", "").replace(" W", "")
        nw = nw.split(', ')
        se = self.args['segment_queries']['bounding_box']['input_se_corner'].replace(" N", "").replace(" W", "")
        se = se.split(', ')
        return "geom @ ST_MakeEnvelope (-{}, {}, -{}, {}) and ST_Length(geom) > 0".format(nw[1], nw[0], se[1], se[0])

    def create_boundingbox_sql(self, table):
        """Create SQL to select all segments within a given bounding box (set in args).
        :param table: name of table to select segments from.
        :return: complete sql string to select all segments within a given bounding box.
        """
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE {}'.format(self.bounding_box())
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def create_sample_sql(self, table):
        """Create SQL to select random sample of a given percentage (set in args) of segments.
        :param table: name of table to select segments from.
        :return: complete sql string to select a random sample of a given percentage) of all segments
        """
        # tablesample needs to go after alias s but before join ...
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_sample = ' TABLESAMPLE SYSTEM ({}) REPEATABLE ({})'.format(
            self.args['segment_queries']['sample']['segments_sample'],
            self.args['seed'])
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_sample)
        return final_sql

    def street(self):
        """Create where clause of query to select all segments on a given street.
        :return: string to be used in where clause of query to find segments on a given street.
        """
        return 'street = {}'.format('\'' + self.args['segment_queries']['street']['input_street'] + '\'')

    def create_street_sql(self, table):
        """Create SQL to select all segments on a given street (set in args).
        :param table: name of table to select segments from.
        :return: complete sql string to select all segments on a given street.
        """
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE {}'.format(self.street())
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def road_type(self):
        """Create where clause of query to select all segments of a given road type.
        :return: string to be used in where clause of query to find segments of a given road type.
        """
        return 'road_type = {}'.format(self.args['segment_queries']['road_type'])

    def create_road_type_sql(self, table):
        """Create SQL to select all segments of a given road type (set in args).
        :param table: name of table to select segments from.
        :return: complete sql string to select all segments of a given road type.
        """
        sql_select = 'SELECT {}'.format(self.segments_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE {}'.format(self.road_type())
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def create_cum_seg_pct_table(self, conn, time_resolution, segments_table):
        """Create table showing the cumulative percentage of segments and positive traffic data points.

        Create a table showing the cumulative percentage of segments and positive data points.
        This table allows the user to select segments such that a given percentage of positive traffic data points
        are retained in the data. This was done because there are many segments that have a small number
        of positive traffic data points across the data range. Removing such segments allows one to
        make the data set much smaller while still retaining most of the information contained in the data
        (similar to PCA).

        :param conn: database connection object.
        :param time_resolution: time window resolution in minutes of traffic data points.
        :param segments_table: table name containing segment data
        :return: None
        """

        # drop table
        sql_drop_seg_cum_pct_table = 'drop table if exists seg_cum_pct'

        # create table
        sql_create_seg_cum_pct_table = '''
        create table seg_cum_pct as
        (with seg_counts as
        (select s.segment_id, count(distinct m.time_id) as num_timestamps
        from matrix_''' + str(time_resolution) + ''' m, time_''' + str(time_resolution) + ' t,' + segments_table + ''' s
        where m.time_id = t.time_id and s.segment_id = m.segment_id
        group by s.segment_id),

        seg_ts_counts as
        (select num_timestamps, count(*) as seg_count
        from seg_counts
        group by num_timestamps),

        cum_seg_count as
        (select num_timestamps, seg_count, sum(seg_count)
        over (order by num_timestamps desc)
        from seg_ts_counts
        order by sum asc),

        seg_count_total as
        (select sum(seg_count) as total_segments from cum_seg_count),

        cum_seg_pct_table as
        (select csc.num_timestamps, csc.seg_count, csc.sum, csc.sum / sct.total_segments as cum_seg_pct
        from cum_seg_count as csc, seg_count_total as sct),

        pos_counts as
        (select num_timestamps, seg_count, num_timestamps*seg_count as pos_count
        from seg_ts_counts),

        cum_pos_count as
        (select num_timestamps, pos_count, sum(pos_count)
        over (order by num_timestamps desc)
        from pos_counts
        order by sum asc),

        pos_count_total as
        (select sum(pos_count) as total_positives from cum_pos_count),

        cum_pos_pct_table as
        (select cpc.num_timestamps, cpc.pos_count, cpc.sum, cpc.sum / pct.total_positives as cum_pos_pct
        from cum_pos_count as cpc, pos_count_total as pct)

        select seg_counts.segment_id, round(cum_seg_pct_table.cum_seg_pct*100, 4) as cum_seg_pct,
        round(cum_pos_pct_table.cum_pos_pct*100, 4) as cum_pos_pct
        from seg_counts, cum_seg_pct_table, cum_pos_pct_table
        where seg_counts.num_timestamps = cum_seg_pct_table.num_timestamps
        and seg_counts.num_timestamps = cum_pos_pct_table.num_timestamps
        order by cum_seg_pct asc)
        '''
        cur = conn.cursor()
        cur.execute(sql_drop_seg_cum_pct_table)
        conn.commit()
        cur.execute(sql_create_seg_cum_pct_table)
        conn.commit()
        cur.close()
        return None




class times:
    """
    Data and queries to set relevant days and times for traffic analysis.
    This class is used to sample all available time buckets (30 minute window) by 1 or more methods
    and store the time data in a table for use by other objects.

    Available methods:
    time_window:    Select all time buckets within 1 or more given time windows.
    day_of_week:    Select all time buckets with a given day of the week.
    sample:         Select a random sample of the provided percent from all time buckets.
    exclude_dates:  Select all time buckets that do not fall on a given set of dates.
    cum_ts_pct:     Select all time buckets such that the provided percent of positive traffic points are retained.

    This class will be used in the Pipeline notebook for exploration purposes and in the Production
    python code used to perform the actual clustering and modeling of traffic data.
    """

    def __init__(self, conn, queries, args):
        """
        Constructor for Time object.
        :param conn: database connection object.
        :param queries: list of queries to be run against the main times table.
        :param args: dict of arguments read from the arguments file.
        :return: Time instance.
        """
        self.conn = conn
        self.args = args
        self.times_filter = 'time_id'
        self.times_table = 'time_' + str(self.args['time_resolution']) + ' t'
        self.time_table = 'time_' + str(self.args['time_resolution'])
        self.matrix_table = 'matrix_' + str(self.args['time_resolution'])
        self.times_selected_table = 'times_selected'
        self.selected_times = []
        self.queries = queries
        if 'time_window_min' in self.args and 'time_window_max' in self.args:
            self.args['time_window_min'] = args['time_window_min']
            self.args['time_window_max'] = args['time_window_max']
        self.query_map = {
            'time_window': self.create_timewindow_sql,
            'day_of_week': self.create_day_of_week_sql,
            'exclude_dates': self.create_exclude_dates_sql,
            'cum_ts_pct': self.create_cum_ts_pct_sql,
            'sample': self.create_sample_sql
        }

    def run_queries(self):
        """
        Run all queries to select desired time buckets for clustering and modelling.
        :return: None
        """
        queries = []
        cur = self.conn.cursor()
        for idx, query in enumerate(self.queries):
            logger.info(idx, query)
            if idx == 0:
                table = self.time_table
                sql_truncate = 'TRUNCATE {}; \n\n'.format(self.times_selected_table)
                sql = self.query_map[query](table)
                sql_with = 'WITH times_to_keep AS ( ' + sql + ' ) \n'
                sql_select_inner = 'SELECT time_id from times_to_keep'
                sql_where = ' WHERE time_id IN ({})'.format(sql_select_inner)
                sql_select_outer = 'SELECT * FROM {}{}'.format(table, sql_where)
                sql_insert = 'INSERT INTO {} ({})'.format(self.times_selected_table, sql_select_outer)
                final_sql = '{}{}{} \n'.format(sql_truncate, sql_with, sql_insert)
                logger.info(final_sql)
                cur.execute(final_sql)
                self.conn.commit()
            else:
                table = self.times_selected_table
                sql = self.query_map[query](table)
                sql_with = 'with times_to_keep as ( ' + sql + ' ) \n'
                sql_delete = 'DELETE from ' + self.times_selected_table
                sql_select = 'SELECT time_id from times_to_keep'
                sql_where = ' WHERE time_id NOT IN ({});'.format(sql_select)
                final_sql = '{}{}{} \n'.format(sql_with, sql_delete, sql_where)
                logger.info(final_sql)
                cur.execute(final_sql)
                self.conn.commit()
        cur.close()

    def create_cum_ts_pct_sql(self, table):
        """
        Create SQL to create ts_cum_pct table.
        The ts_cum_pct table is used to allow the selection of time buckets that will retain a given percentage
        of positive traffic data point.
        :param table: name of table to select times from.
        :return: complete sql string to select all time buckets to retain a given percentage of positive traffic data points.
        """
        self.create_cum_ts_pct_table(self.conn, self.args['time_resolution'], table)
        final_sql = 'SELECT time_id FROM ts_cum_pct WHERE cum_pos_pct <= {}'.format(str(self.args['time_queries']['cum_ts_pct']))
        return final_sql

    def create_exclude_dates_sql(self, table):
        """
        Select time buckets that do not fall within a given list of excluded dates.
        :param table: name of table to select times from.
        :return: complete sql string to select all time buckets that are not in a given list of excluded dates.
        """
        exclude_dates = ['\'' + x + '\'' for x in self.args['time_queries']['exclude_dates']]
        exclude_dates = ','.join(exclude_dates)
        sql_select = 'SELECT {}'.format(self.times_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE date not in ({})'.format(exclude_dates)
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def create_timewindow_sql(self, table):
        """
        Select time buckets that fall within a given list of time windows.
        :param table: name of table to select times from.
        :return: complete sql string to select all time buckets that are within the given list of time windows.
        """
        sql_select = 'SELECT {}'.format(self.times_filter)
        sql_from = '\n FROM {}'.format(table)
        # if both time_window_min AND max are set
        # select fields from times table where times between window min and max
        sql_where = "\n WHERE (time >= '{}' AND time <= '{}')".format(
            self.args['time_queries']['time_window']['time_window_include'][0][0],
            self.args['time_queries']['time_window']['time_window_include'][0][1])
        for idx, twin in enumerate(self.args['time_queries']['time_window']['time_window_include'][1:]):
            sql_where += "\n OR (time >= '{}' AND time <= '{}')".format(
                self.args['time_queries']['time_window']['time_window_include'][idx + 1][0],
                self.args['time_queries']['time_window']['time_window_include'][idx + 1][1])
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def create_sample_sql(self, table):
        """
        Create SQL to select random sample of a given percentage (set in args) of timestamps.
        :param table: name of table to select timestamps from.
        :return: complete sql string to select a random sample of a given percentage) of all timestamps.
        """
        # tablesample needs to go after alias s but before join ...
        sql_select = 'SELECT {}'.format(self.times_filter)
        sql_from = ' FROM {}'.format(table)
        sql_sample = ' TABLESAMPLE SYSTEM ({}) REPEATABLE ({})'.format(
            self.args['time_queries']['sample']['time_sample'],
            self.args['seed'])
        final_sql = '{}{}{}'.format(sql_select, sql_from, sql_sample)
        return final_sql
    
    def day_of_week(self):
        """
        Create day of week filter part of a query.
        :return: string to be used as where clause in day of week query.
        """
        return 'day_of_week = ' + '\'' + str(self.args['time_queries']['day_of_week']['input_dow']) + '\''

    def create_day_of_week_sql(self, table):
        """
        Create SQL to select all time buckets that fall on a given day of the week (set in args).
        :param table: name of table to select times from.
        :return: complete sql string to select all time buckets that fall on a given day of the week.
        """
        sql_select = 'SELECT {}'.format(self.times_filter)
        sql_from = ' FROM {}'.format(table)
        sql_where = ' WHERE {}'.format(str(self.day_of_week()))
        final_sql = '{}{}{}{}{}'.format(sql_select, sql_from, sql_where)
        return final_sql

    def create_cum_ts_pct_table(self, conn, time_resolution, time_table):
        """Create table showing the cumulative percentage of times and positive traffic data points.

        Create a table showing the cumulative percentage of time buckets and positive data points.
        This table allows the user to select times such that a given percentage of positive traffic data points
        are retained in the data. This was done because there are many times that have a small number
        of positive traffic data points across the data range. Removing such times allows one to
        make the data set much smaller while still retaining most of the information contained in the data
        (similar to PCA).

        :param conn: database connection object.
        :param time_resolution: time window resolution in minutes of traffic data points.
        :param time_table: table name containing time data
        :return: None
        """

        # drop table
        sql_drop_ts_cum_pct_table = '''
        drop table if exists ts_cum_pct
        '''

        # create table
        sql_create_ts_cum_pct_table = '''
        create table ts_cum_pct as
        (with ts_counts as
        (select t.time_id, count(distinct s.segment_id) as num_segments
            from matrix_''' + str(time_resolution) + ''' m, ''' + time_table + ''' t, segments_selected s

        where m.time_id = t.time_id and s.segment_id = m.segment_id
        group by t.time_id),

        ts_seg_counts as
        (select num_segments, count(*) as ts_count
        from ts_counts
        group by num_segments),

        cum_ts_count as
        (select num_segments, ts_count, sum(ts_count)
        over (order by num_segments desc)
        from ts_seg_counts
        order by sum asc),

        ts_count_total as
        (select sum(ts_count) as total_ts from cum_ts_count),

        cum_ts_pct_table as
        (select csc.num_segments, csc.ts_count, csc.sum, csc.sum / sct.total_ts as cum_ts_pct
        from cum_ts_count as csc, ts_count_total as sct),

        pos_counts as
        (select num_segments, ts_count, num_segments*ts_count as pos_count
        from ts_seg_counts),

        cum_pos_count as
        (select num_segments, pos_count, sum(pos_count)
        over (order by num_segments desc)
        from pos_counts
        order by sum asc),

        pos_count_total as
        (select sum(pos_count) as total_positives from cum_pos_count),

        cum_pos_pct_table as
        (select cpc.num_segments, cpc.pos_count, cpc.sum, cpc.sum / pct.total_positives as cum_pos_pct
        from cum_pos_count as cpc, pos_count_total as pct)

        select ts_counts.time_id, round(cum_ts_pct_table.cum_ts_pct*100, 4) as cum_ts_pct,
        round(cum_pos_pct_table.cum_pos_pct*100, 4) as cum_pos_pct
        from ts_counts, cum_ts_pct_table, cum_pos_pct_table
        where ts_counts.num_segments = cum_ts_pct_table.num_segments
        and ts_counts.num_segments = cum_pos_pct_table.num_segments
        order by cum_ts_pct asc)
        '''
        cur = conn.cursor()
        cur.execute(sql_drop_ts_cum_pct_table)
        conn.commit()
        cur.execute(sql_create_ts_cum_pct_table)
        conn.commit()
        cur.close()
        return None

    


class segments_times_levels:

    def __init__(self, conn, file_args):
        self.file_args = file_args
        self.filepath = file_args['data_dir']
        self.segment_queries_to_run = file_args['segment_queries_to_run']
        self.time_queries_to_run = file_args['time_queries_to_run']
        self.time_bucket = str(self.file_args['time_resolution'])
        self.cum_ts = 100/self.file_args['time_queries']['cum_ts_pct']
        self.cum_seg = 100/self.file_args['segment_queries']['cum_seg_pct']
        self.pg_conn = conn['pg_conn']
        self.cur = conn['pg_conn'].cursor()

    def build_intermediary_tables(self):
        logger.info('Start intermediary tables.')
        self.cur.execute(sqlqueries.SQL_drop_indexes)
        self.cur.execute(sqlqueries.SQL_update_segments)
        #self.cur.execute(sqlqueries.SQL_create_events_table)
        self.cur.execute(sqlqueries.SQL_time_bucketing)
        self.cur.execute(sqlqueries.SQL_create_segments_times_selected)
        self.cur.execute(sqlqueries.SQL_pct_segments)
        self.cur.execute(sqlqueries.SQL_pct_time)
        self.cur.execute(sqlqueries.SQL_unique_segments)
        self.cur.execute(sqlqueries.SQL_drop_indexes)
        self.cur.execute('COMMIT')
        logger.info('Finish intermediary tables.')
        self.pg_conn.commit()

    def create_selected_segments(self):
        logger.info('Selecting segments.')
        segments_obj = segments(self.pg_conn, self.segment_queries_to_run, self.file_args)
        segments_obj.run_queries()

    def created_selected_times(self):
        logger.info('Selecting times.')
        times_obj = times(self.pg_conn, self.time_queries_to_run, self.file_args)
        times_obj.run_queries()

    def build_segment_times_levels(self):
        logger.info('Start creating segments/times/level selected table.')
        self.cur.execute('DROP TABLE IF EXISTS segments_time_selected')
        self.cur.execute(sqlqueries.segments_time_selected)
        self.cur.execute('DROP TABLE IF EXISTS levels_selected')
        self.cur.execute(sqlqueries.levels_selected)
        self.cur.execute('DROP TABLE IF EXISTS segments_time_level_selected')
        self.cur.execute(sqlqueries.segments_time_level_selected)
        self.cur.execute('COMMIT')
        logger.info('Done creating segments/times/level selected table.')

    def run(self):
        #self.build_intermediary_tables()
        self.create_selected_segments()
        self.created_selected_times()
        self.build_segment_times_levels()

