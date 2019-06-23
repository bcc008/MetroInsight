class Segments:

    
    """Data and queries to set relevant segments for various traffic analysis.

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
        """Run all queries to select desired segments for clustering and modelling.

        :return: None
        """

        queries = []
        cur = self.conn.cursor()
        for idx, query in enumerate(self.queries):
            print(idx, query)
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
                print(final_sql)
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
                print(final_sql)
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