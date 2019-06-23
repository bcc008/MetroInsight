class Times:
    
    """Data and queries to set relevant days and times for traffic analysis.

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
        """Constructor for Time object.

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
        """Run all queries to select desired time buckets for clustering and modelling.

        :return: None
        """

        queries = []
        cur = self.conn.cursor()
        for idx, query in enumerate(self.queries):
            print(idx, query)
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
                print(final_sql)
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
                print(final_sql)
                cur.execute(final_sql)
                self.conn.commit()
        cur.close()

    def create_cum_ts_pct_sql(self, table):
        """Create SQL to create ts_cum_pct table.

        The ts_cum_pct table is used to allow the selection of time buckets that will retain a given percentage
        of positive traffic data point.

        :param table: name of table to select times from.
        :return: complete sql string to select all time buckets to retain a given percentage of positive traffic data points.
        """

        # create seg_cum_pct table
        self.create_cum_ts_pct_table(self.conn, self.args['time_resolution'], table)
        final_sql = 'SELECT time_id FROM ts_cum_pct WHERE cum_pos_pct <= {}'.format(str(self.args['time_queries']['cum_ts_pct']))
        return final_sql

    #    def create_cum_ts_pct_sql(self, table):
    #        final_sql = create_cum_ts_pct_table(conn, self.args['time_resolution'], table, self.args['time_pct'])
    #        return final_sql

    def create_exclude_dates_sql(self, table):
        """Select time buckets that do not fall within a given list of excluded dates.

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
        """Select time buckets that fall within a given list of time windows.

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
        """Create SQL to select random sample of a given percentage (set in args) of timestamps.

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
        """Create day of week filter part of a query.

        :return: string to be used as where clause in day of week query.
        """

        return 'day_of_week = ' + '\'' + str(self.args['time_queries']['day_of_week']['input_dow']) + '\''

    def create_day_of_week_sql(self, table):
        """Create SQL to select all time buckets that fall on a given day of the week (set in args).

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