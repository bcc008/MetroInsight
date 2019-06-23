SQL_create_events_table = """
							DROP TABLE IF EXISTS events;
                            DROP TABLE IF EXISTS events_dictionary;

							CREATE TABLE public.events
							(
    							id serial NOT NULL,
    							event_title character varying(50) COLLATE pg_catalog."default",
    							event_id integer,
    							event_subtitle character varying(25) COLLATE pg_catalog."default",
    							event_type character varying(10) COLLATE pg_catalog."default",
    							event_desc character varying(512) COLLATE pg_catalog."default",
    							event_loc character varying(512) COLLATE pg_catalog."default",
    							event_start character varying(25) COLLATE pg_catalog."default",
    							event_end character varying(25) COLLATE pg_catalog."default",
    							exp_attendance character varying(25) COLLATE pg_catalog."default",
    							exp_participants character varying(10) COLLATE pg_catalog."default",
    							event_host character varying(50) COLLATE pg_catalog."default",
    							event_url character varying(128) COLLATE pg_catalog."default",
    							event_address character varying(50) COLLATE pg_catalog."default",
    							latitude double precision,
    							longitude double precision
							)
							WITH (
    							OIDS = FALSE
							)
							TABLESPACE pg_default;

							ALTER TABLE public.events
    							OWNER to postgres;
                            

                            CREATE TABLE public.events_dictionary
                            (
                            id serial NOT NULL,
                            field character varying(25),
                            description character varying(50),
                            possible_values character varying(50)
                            );

                            COPY events (event_title,event_id,event_subtitle,event_type,event_desc,event_loc,event_start,event_end,exp_attendance,exp_participants,event_host,event_url,event_address,latitude,longitude)
                            FROM '{0}special_events_list_datasd.csv' DELIMITER ',' CSV HEADER;

                            COPY events_dictionary (field,description,possible_values)
                            FROM '{0}special_events_listings_dictionary.csv' DELIMITER ',' CSV HEADER; 
							
                            ALTER TABLE events DROP COLUMN IF EXISTS geom;
                            ALTER TABLE events ADD COLUMN geom geometry;
                            UPDATE events SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
							""".format(filepath)


SQL_update_segments = """
						ALTER TABLE segments_all DROP COLUMN IF EXISTS geom;
                        ALTER TABLE segments_all DROP COLUMN IF EXISTS seg_length;
                        ALTER TABLE segments_all DROP COLUMN IF EXISTS direction;
                        ALTER TABLE segments_all ADD COLUMN geom geometry;
                        ALTER TABLE segments_all ADD COLUMN seg_length double precision;
                        ALTER TABLE segments_all ADD COLUMN direction integer;
                        UPDATE segments_all SET geom = st_makeline(st_makepoint(lon1,lat1),st_makepoint(lon2,lat2));
                        UPDATE segments_all SET seg_length = ST_Length(geom);
                        
                        UPDATE segments_all SET direction = 1 WHERE degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) > 315 AND degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) <= 45;
                        UPDATE segments_all SET direction = 2 WHERE degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) > 45 AND degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) <= 135;
                        UPDATE segments_all SET direction = 3 WHERE degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) > 135 AND degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) <= 225;
                        UPDATE segments_all SET direction = 4 WHERE degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) > 225 AND degrees(st_azimuth(st_point(lon1,lat1),st_point(lon2,lat2))) <= 315;
						
						"""



SQL_Time_Bucketing = """
						DROP TABLE IF EXISTS time_{0};

						CREATE TABLE public.time_{0} AS 
                        (
                        SELECT DISTINCT 
                            date, 
                            day_of_week, 
                            month, 
                            is_weekend, 
                            is_holiday, 
                            is_rushhour,
                            (date_trunc('hour', time) + date_part('minute', time)::int / {0} * interval '{0} min')::time AS time,
                            (date_trunc('hour', timestamp_round) + date_part('minute', timestamp_round)::int / {0} * interval '{0} min')::timestamp AS timestamp_round
                        FROM time
                        );

                        ALTER TABLE time_{0} ADD COLUMN time_id SERIAL;

                        ALTER TABLE time_{0} ADD PRIMARY KEY (time_id);


                        DROP VIEW IF EXISTS v_time_{0};

                        CREATE VIEW v_time_{0} AS 
                        (
                            SELECT  date,
                            		day_of_week,
                            		month,
                            		is_weekend, 
                                    is_holiday, 
                                    is_rushhour,
                                    time,
                            		timestamp_round,
                                    time_id
                            FROM time_{0}
                        );

                        CREATE TABLE time_{0}a AS 
                        (
                            SELECT  date,
                            		day_of_week,
                            		month,
                            		is_weekend, 
                                    is_holiday, 
                                    is_rushhour,
                                    time,
                            		timestamp_round,
                                    time_id
                            FROM    v_time_{0}
                        );


                        DROP VIEW v_time_{0};

                        DROP TABLE time_{0};

                        ALTER TABLE time_{0}a RENAME TO time_{0};
                        
                        
                        DROP TABLE IF EXISTS matrix_{0};
                            CREATE TABLE matrix_{0} AS (
                                WITH time_id_map AS (
                                    SELECT  time_{0}.time_id AS new_time_id, 
                                        time.time_id AS orig_time_id 
                                    FROM    time_{0}, time
                                    WHERE   time_{0}.timestamp_round = (
                                        date_trunc('hour', time.timestamp_round) + 
                                        date_part('minute', time.timestamp_round)::int / {0} * interval '{0} min' )::timestamp
                            )

                            SELECT  m.uuid_instance_id, m.path, m.segment_id, time_id_map.new_time_id AS time_id
                            FROM    matrix m, time_id_map
                            WHERE   m.time_id = time_id_map.orig_time_id)
					""".format(time_bucket)
                            
SQL_create_segments_times_selected = """
										DROP TABLE IF EXISTS public.segments_selected;
										CREATE TABLE public.segments_selected
										(
    										lat1 double precision,
    										lon1 double precision,
    										lat2 double precision,
    										lon2 double precision,
    										segment_id integer,
    										street character varying(50) COLLATE pg_catalog."default",
    										city character varying(50) COLLATE pg_catalog."default",
    										road_type integer,
    										geom geometry,
    										direction integer,
    										seg_length double precision,
											cum_seg_pct numeric
											)
										WITH (
    										OIDS = FALSE
										)
										TABLESPACE pg_default;

										ALTER TABLE public.segments_selected
    										OWNER to postgres;
    							
									DROP TABLE IF EXISTS public.times_selected;

									CREATE TABLE public.times_selected
									(
    									date date,
    									day_of_week character varying COLLATE pg_catalog."default",
    									month character varying COLLATE pg_catalog."default",
    									is_weekend boolean,
    									is_holiday boolean,
    									is_rushhour boolean,
    									"time" time without time zone,
    									timestamp_round timestamp without time zone,
    									time_id integer,
										cum_seg_pct numeric
										)
										WITH (
    										OIDS = FALSE
										)
										TABLESPACE pg_default;

										ALTER TABLE public.times_selected
    										OWNER to postgres;
									"""

SQL_drop_indexes = """
					ALTER TABLE segments_all DROP COLUMN IF EXISTS index;
					ALTER TABLE time DROP COLUMN IF EXISTS index;
					ALTER TABLE uuid DROP COLUMN IF EXISTS index;
					ALTER TABLE matrix DROP COLUMN IF EXISTS index;
					"""
					
					
SQL_pct_segments = """
					DROP TABLE IF EXISTS seg_cum_pct;
					
					CREATE TABLE seg_cum_pct AS
					(WITH all_segments AS
					(SELECT s.segment_id, COUNT(m.time_id) AS num_timestamps
					FROM matrix m, time t, segments_all s
					WHERE m.time_id = t.time_id AND s.segment_id = m.segment_id
					GROUP BY s.segment_id
					ORDER BY num_timestamps DESC),

					seg_counts AS
					(SELECT num_timestamps, COUNT(*) AS seg_count
					FROM all_segments
					GROUP BY num_timestamps),

					cum_seg_count AS
					(SELECT num_timestamps, seg_count, SUM(seg_count)
					OVER (ORDER BY seg_count DESC)
					FROM seg_counts
					ORDER BY sum ASC),

					seg_count_total AS
					(SELECT SUM(seg_count) AS total_segments FROM cum_seg_count),

					cum_seg_pct_table AS
					(SELECT csc.num_timestamps, csc.seg_count, csc.sum, csc.sum / sct.total_segments AS cum_seg_pct
					FROM cum_seg_count AS csc, seg_count_total AS sct)

					SELECT all_segments.segment_id, ROUND(cum_seg_pct_table.cum_seg_pct, 3) AS cum_seg_pct
					FROM all_segments, cum_seg_pct_table
					WHERE all_segments.num_timestamps = cum_seg_pct_table.num_timestamps
					ORDER BY cum_seg_pct ASC);
					
					ALTER TABLE segments_all
					ADD COLUMN IF NOT EXISTS cum_seg_pct numeric;
					
					UPDATE segments_all
					SET cum_seg_pct = seg_cum_pct.cum_seg_pct
					FROM seg_cum_pct
					WHERE seg_cum_pct.segment_id = segments_all.segment_id;
					""".format(cum_seg)

SQL_pct_time = """
				DROP TABLE IF EXISTS ts_cum_pct;

				CREATE TABLE ts_cum_pct AS
				(WITH all_ts AS
				(SELECT t.time_id, count(s.segment_id) AS num_segments
				FROM matrix m, time t, segments_all s
				WHERE m.time_id = t.time_id and s.segment_id = m.segment_id
				GROUP BY t.time_id
				ORDER BY num_segments DESC),

				ts_counts AS
				(SELECT num_segments, count(*) AS ts_count
				FROM all_ts
				GROUP BY num_segments),

				cum_ts_count AS
				(SELECT num_segments, ts_count, SUM(ts_count)
				over (ORDER BY ts_count DESC)
				FROM ts_counts
				ORDER BY sum ASC),

				ts_count_total AS
				(SELECT SUM(ts_count) AS total_ts FROM cum_ts_count),

				cum_ts_pct_table AS
				(SELECT csc.num_segments, csc.ts_count, csc.sum, csc.sum / sct.total_ts AS cum_ts_pct
				FROM cum_ts_count AS csc, ts_count_total AS sct)

				SELECT all_ts.time_id, ROUND(cum_ts_pct_table.cum_ts_pct*100, 3) AS cum_ts_pct
				FROM all_ts, cum_ts_pct_table
				WHERE all_ts.num_segments = cum_ts_pct_table.num_segments
				ORDER BY cum_ts_pct ASC);
				
				ALTER TABLE time
				ADD COLUMN IF NOT EXISTS cum_ts_pct numeric;
				
				UPDATE time
				SET cum_ts_pct = ts_cum_pct.cum_ts_pct
				FROM ts_cum_pct
				WHERE ts_cum_pct.time_id = time.time_id;
				
				UPDATE time
				SET cum_ts_pct = 0
				WHERE cum_ts_pct is null;
				""".format(cum_ts)

SQL_unique_segments = """
						DROP TABLE IF EXISTS segments_roadtype;
						DROP TABLE IF EXISTS segments_stname;
						DROP TABLE IF EXISTS segments_city;
						DROP TABLE IF EXISTS segments;
						
						SELECT * INTO segments_roadtype FROM 
						(SELECT lat1, lon1, lat2, lon2, segment_id, road_type, cum_seg_pct, geom, seg_length, direction, ROW_NUMBER() OVER(PARTITION BY segment_id ORDER BY count(road_type)) 
						FROM segments_all
						GROUP BY lat1, lon1, lat2, lon2, segment_id, road_type, cum_seg_pct, geom, seg_length, direction ORDER BY segment_id) subquery
						WHERE row_number = 1;
						
						SELECT * INTO segments_stname FROM 
						(SELECT segment_id, street, ROW_NUMBER() OVER(PARTITION BY segment_id ORDER BY count(street)) 
						FROM segments_all
						GROUP BY segment_id, street ORDER BY segment_id) subquery
						WHERE row_number = 1;
						
						SELECT * INTO segments_city FROM 
						(SELECT segment_id, city, ROW_NUMBER() OVER(PARTITION BY segment_id ORDER BY count(city)) 
						FROM segments_all
						GROUP BY segment_id, city ORDER BY segment_id) subquery
						WHERE row_number = 1;
						
						SELECT segments_roadtype.lat1, segments_roadtype.lon1, segments_roadtype.lat2, segments_roadtype.lon2, 
								segments_roadtype.segment_id, segments_roadtype.road_type, segments_roadtype.cum_seg_pct, 
								segments_roadtype.geom, segments_roadtype.seg_length, segments_roadtype.direction,
								segments_stname.street, segments_city.city
						INTO segments
						FROM segments_roadtype JOIN segments_stname ON segments_roadtype.segment_id = segments_stname.segment_id
						JOIN segments_city ON segments_roadtype.segment_id = segments_city.segment_id;
						
						DROP TABLE IF EXISTS segments_roadtype;
						DROP TABLE IF EXISTS segments_stname;
						DROP TABLE IF EXISTS segments_city;
					"""