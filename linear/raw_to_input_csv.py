from scipy.stats import zscore
from datetime import datetime as dt
import numpy as np
import pandas as pd

import concurrent.futures
import time

TEST_PERCENTAGE = 0.2

RAW_DATA_PATH = "raw/raw_data.csv"
CYCLE_AMOUNT_PATH = "raw/cycle_amount.csv"

TRAIN_DATA_PATH = "input/train_data.csv"
TEST_DATA_PATH = "input/test_data.csv"

def read_raw_data():
  print("Reading weather and cycle amount data...")
  # Adding 0 - 3 numbers as header names.
  raw_data_df = pd.read_csv(RAW_DATA_PATH, header=None, names=np.arange(4))
  cycle_amount_df = pd.read_csv(CYCLE_AMOUNT_PATH, header=None)

  return raw_data_df, cycle_amount_df

def store_split_datetime(raw_data_df, weather_df):
  print("Splitting datetime to date and hour...")
  # index 1, 2, 3 is used later
  weather_df = raw_data_df[0].apply(lambda x: pd.Series(x.split(" "), index=[0,4]))
  return weather_df
  
def store_real_values(raw_data_df, weather_df):
  print("Normalizing real values...")
  # Normalize real_value columns
  for j in [ 1, 2, 3 ]:
    weather_df[j] = zscore(raw_data_df[j])

  return weather_df

def pivot_date_x_hour(weather_df):
  print("Pivoting columns date x hour...")
  # Pivot data to date x hour
  return weather_df.pivot(index=0, columns=4)

def store_categolized_values(weather_df):
  print("Appending categolized values...")
  # Append oter inputs and labels after pivot
  for l in weather_df.index:
    date = dt.strptime(l, "%Y/%m/%d")
    weather_df.loc[l, 5] = date.month
    weather_df.loc[l, 6] = date.weekday()

  return weather_df

def store_label_values(weather_df, cycle_amount_df):
  print("Appending label values...")
  # Reset indexes of weather_df as default interger, to match index of two dataframes
  weather_df.reset_index(drop=True, inplace=True)
  weather_df[7] = cycle_amount_df[0]

  return weather_df

def configure_weather_df(raw_data_df, cycle_amount_df):
  weather_df = pd.DataFrame()
  weather_df = store_split_datetime(raw_data_df, weather_df)
  weather_df = store_real_values(raw_data_df, weather_df)
  weather_df = pivot_date_x_hour(weather_df)
  weather_df = store_categolized_values(weather_df)
  weather_df = store_label_values(weather_df, cycle_amount_df)

  return weather_df

def make_train_test_data(weather_df):
  print("Split train and test data by TEST_PERCENTAGE...")
  # Select random columns from whole input data with directed percentage
  test_df = weather_df.sample(frac=TEST_PERCENTAGE)
  train_df = weather_df.drop(test_df.index.values)

  return train_df, test_df

def save_train_test_data(train_df, test_df):
  print("Save train and test data as CSV files...")
  # Save results as csv files
  train_df.to_csv(TRAIN_DATA_PATH, header=None)
  test_df.to_csv(TEST_DATA_PATH, header=None)

def raw_to_input():
  raw_data_df, cycle_amount_df = read_raw_data()
  weather_df = configure_weather_df(raw_data_df, cycle_amount_df)
  train_df, test_df = make_train_test_data(weather_df)
  save_train_test_data(train_df, test_df)

def main():
  raw_to_input()

main()
