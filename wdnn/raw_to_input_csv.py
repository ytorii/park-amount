from scipy.stats import zscore
from datetime import datetime as dt
import numpy as np
import pandas as pd

TEST_PERCENTAGE = 0.1

RAW_DIR = "raw/"
RAW_TRAIN_PATH = RAW_DIR + "raw_train_data.csv"
RAW_PREDICT_PATH = RAW_DIR + "raw_predict_data.csv"
CYCLE_AMOUNT_PATH = RAW_DIR + "cycle_amount.csv"

INPUT_DIR = "input/"
TRAIN_DATA_PATH = INPUT_DIR + "train_data.csv"
TEST_DATA_PATH = INPUT_DIR + "test_data.csv"
PREDICT_DATA_PATH = INPUT_DIR + "predict_data.csv"

class WeatherDataGenerator:

  CLOSED_HOURS = [ "22:00", "23:00", "0:00", "1:00", "2:00", "3:00", "4:00", "5:00" ]

  def __init__(self, raw_data=None, amount_data=None):
    self.weather_data = pd.DataFrame()
    self.raw_data = raw_data
    self.amount_data = amount_data

  def generate_data(self):
    self.__store_split_datetime()
    self.__store_real_values()
    self.__drop_closed_hours()
    self.__pivot_date_x_hour()
    self.__store_categolized_values()
    self.__store_label_values()

  def get_data(self):
    return self.weather_data

  def __store_split_datetime(self):
    print("Splitting datetime to date and hour...")
    # index 1, 2, 3 is used later
    self.weather_data = self.raw_data[0].apply(lambda datehour: pd.Series(datehour.split(" "), index=[0,4]))
    
  def __drop_closed_hours(self):
    print("Dropping closed hours columns...")
    drop_rows = self.weather_data.loc[self.weather_data[4].isin(CLOSED_HOURS)]
    self.weather_data.drop(drop_rows.index, inplace=True)

  def __store_real_values(self):
    print("Storing temprature and precipiation and wind speed...")
    for j in [ 1, 2, 3 ]:
    #for j in [ 1, 3 ]: # Passing wind speed
      self.weather_data[j] = self.raw_data[j]

  def __normalize_real_values(self):
    print("Normalizing real values...")
    # Normalize real_value columns
    for j in [ 1, 2, 3 ]:
    #for j in [ 1, 3 ]: # Passing wind speed
      # Regression problems doesn't need to be normalized?
      self.weather_data[j] = zscore(self.weather_data[j], axis=0)
  
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
    # Reset indexes of self.weather_data as default interger, to match index of two dataframes
    self.weather_data.reset_index(drop=True, inplace=True)
 
    if self.amount_data is None:
      print("Skipping appending label values...")
    else:
      print("Appending label values...")
      self.weather_data[7] = self.amount_data[0]

def read_raw_data():
  print("Reading weather and cycle amount data...")
  # Adding 0 - 3 numbers as header names.
  raw_train_data_df = pd.read_csv(RAW_TRAIN_PATH, header=None, names=np.arange(4))
  raw_predict_data_df = pd.read_csv(RAW_PREDICT_PATH, header=None, names=np.arange(4))
  amount_data_df = pd.read_csv(CYCLE_AMOUNT_PATH, header=None)

  return raw_train_data_df, raw_predict_data_df, amount_data_df

def make_train_test_data(weather_df):
  print("Make train and test data by TEST_PERCENTAGE...")
  # Select random columns from whole weather data with directed percentage
  test_df = weather_df.sample(frac=TEST_PERCENTAGE)
  train_df = weather_df.drop(test_df.index.values)

  return train_df, test_df

def raw_to_weather():
  print('Generating train and test data...')
  raw_train_data_df, raw_predict_data_df, amount_data_df = read_raw_data()
  train_data_generator = WeatherDataGenerator(raw_train_data_df, amount_data_df)
  train_data_generator.generate_data()
  train_df, test_df = make_train_test_data(train_data_generator.get_data())

  print('Saving train and test data...')
  train_df.to_csv(TRAIN_DATA_PATH, header=None)
  test_df.to_csv(TEST_DATA_PATH, header=None)

  print('Generating predict data...')
  predict_data_generator = WeatherDataGenerator(raw_predict_data_df)
  predict_data_generator.generate_data()
  predict_df = predict_data_generator.get_data()

  print('Saving predict data...')
  predict_df.to_csv(PREDICT_DATA_PATH, header=None)

def run():
  raw_to_weather()

if __name__ == "__main__":
  run()
