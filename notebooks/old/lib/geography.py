import pandas as pd
import numpy as np
import math
import os


def haversine(lon1, lat1, lon2, lat2):
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

	
def build_highway_features_two_way(df,highways):
	# Create highway/freeway features and set to 0
	for highway in [i[0:-2] for i in highways]:
		df[highway] = 0
	# Fill North and East with value 1
	for highway in [i for i in highways if i[-1] in ['N','E']]:
		sampled_df[highway[0:-2]].loc[sampled_df['street'].str.contains(highway)] = 1
	# Fill South and West with value -1
	for highway in [i for i in highways if i[-1] in ['S','W']]:
		sampled_df[highway[0:-2]].loc[sampled_df['street'].str.contains(highway)] = -1
	return df


def build_highway_features_one_way(df,highways):
	# Create highway/freeway features and set to 0
	for highway in highways:
		df[highway] = 0
	# Fill North and East with value 1
	for highway in [i for i in highways if i[-1] in ['N','E']]:
		sampled_df[highway].loc[sampled_df['street'].str.contains(highway)] = 1
	# Fill South and West with value -1
	for highway in [i for i in highways if i[-1] in ['S','W']]:
		sampled_df[highway].loc[sampled_df['street'].str.contains(highway)] = 1
		
	return df

	
	