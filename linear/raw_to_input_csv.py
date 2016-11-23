from scipy.stats import zscore
from datetime import datetime as dt
import numpy as np
import pandas as pd

TEST_PERCENTAGE = 0.2

RAW_DATA_PATH = "raw/raw_data.csv"
CYCLE_AMOUNT_PATH = "raw/cycle_amount.csv"

TRAIN_DATA_PATH = "input/train_data.csv"
TEST_DATA_PATH = "input/test_data.csv"

class WeatherDataGenerator:

  def __init__(self, raw_data, amount_data):
    self.weather_data = pd.DataFrame()
    self.raw_data = raw_data
    self.amount_data = amount_data

  def configure(self):
    self.__store_split_datetime()
    self.__store_real_values()
    self.__pivot_date_x_hour()
    self.__store_categolized_values()
    self.__store_label_values()
  
  def get_data(self):
    return self.weather_data

  def __store_split_datetime(self):
    print("Splitting datetime to date and hour...")
    # index 1, 2, 3 is used later
    self.weather_data = self.raw_data[0].apply(lambda x: pd.Series(x.split(" "), index=[0,4]))
    
  def __store_real_values(self):
    print("Normalizing real values...")
    # Normalize real_value columns
    for j in [ 1, 2, 3 ]:
      self.weather_data[j] = zscore(self.raw_data[j])
  
  def __pivot_date_x_hour(self):
    print("Pivoting columns date x hour...")
    # Pivot data to date x hour
    self.weather_data = self.weather_data.pivot(index=0, columns=4)
  
  def __store_categolized_values(self):
    print("Appending categolized values...")
    # Append oter weathers and labels after pivot
    for l in self.weather_data.index:
      date = dt.strptime(l, "%Y/%m/%d")
      self.weather_data.loc[l, 5] = date.month
      self.weather_data.loc[l, 6] = date.weekday()
  
  def __store_label_values(self):
    print("Appending label values...")
    # Reset indexes of self.weather_data as default interger, to match index of two dataframes
    self.weather_data.reset_index(drop=True, inplace=True)
    self.weather_data[7] = self.amount_data[0]

def read_raw_data():
  print("Reading weather and cycle amount data...")
  # Adding 0 - 3 numbers as header names.
  raw_data_df = pd.read_csv(RAW_DATA_PATH, header=None, names=np.arange(4))
  amount_data_df = pd.read_csv(CYCLE_AMOUNT_PATH, header=None)

  return raw_data_df, amount_data_df

def make_train_test_data(weather_df):
  print("Make train and test data by TEST_PERCENTAGE...")
  # Select random columns from whole weather data with directed percentage
  test_df = weather_df.sample(frac=TEST_PERCENTAGE)
  train_df = weather_df.drop(test_df.index.values)

  return train_df, test_df

def save_train_test_data(train_df, test_df):
  print("Save train and test data as CSV files...")
  # Save results as csv files
  train_df.to_csv(TRAIN_DATA_PATH, header=None)
  test_df.to_csv(TEST_DATA_PATH, header=None)

def raw_to_weather():
  raw_data_df, amount_data_df = read_raw_data()
  weather_data_generator = WeatherDataGenerator(raw_data_df, amount_data_df)
  weather_data_generator.configure()
  train_df, test_df = make_train_test_data(weather_data_generator.get_data())
  save_train_test_data(train_df, test_df)

def run():
  raw_to_weather()

if __name__ == "__main__":
  run()
