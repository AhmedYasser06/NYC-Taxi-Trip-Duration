import os
import datetime
import pandas as pd
import numpy as np
from geopy import distance
from geopy.point import Point
from scipy import stats
from sklearn.model_selection import train_test_split
import math
from data_evaluation import *


def haversine_distance(row):
    pick = Point(row['pickup_latitude'], row['pickup_longitude'])
    drop = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    return distance.geodesic(pick, drop).km

def calculate_direction(row):
    pickup_coordinates = Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = (math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) -
         math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) *
         math.cos(math.radians(delta_longitude)))
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def manhattan_distance(row):
    lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111
    lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))
    return lat_distance + lon_distance
