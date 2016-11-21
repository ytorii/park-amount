from scipy.stats import zscore
from datetime import datetime as dt
import numpy as np
import pandas as pd

TEST_PERCENTAGE = 0.2

RAW_DATA_PATH = "raw/raw_data.csv"
CYCLE_AMOUNT_PATH = "raw/cycle_amount.csv"

TRAIN_DATA_PATH = "input/train_data.csv"
TEST_DATA_PATH = "input/test_data.csv"

# Adding 0 - 10 numbers as header names.
raw_df = pd.read_csv(RAW_DATA_PATH, header=None, names=np.arange(6))
cycle_amount_df = pd.read_csv(CYCLE_AMOUNT_PATH, header=None)
weather_df = pd.DataFrame()

print("Splitting to datetime to date and hour...")
for i in range(raw_df.shape[0]):
  weather_df.loc[i, 0], weather_df.loc[i, 4] = raw_df.loc[i, 0].split(" ")

print("Normalizing real_value columns...")
# Normalize real_value columns
for j in [ 1, 2, 3 ]:
  weather_df[j] = zscore(raw_df[j])

# Pivot data to date x hour
weather_df = weather_df.pivot(index=0, columns=4)

m = 0

print("Appending categolized value and  label...")
# Append oter inputs and labels after pivit
for l in weather_df.index:
  date = dt.strptime(l, "%Y/%m/%d")
  weather_df.loc[l, 5] = date.month
  weather_df.loc[l, 6] = date.weekday()
  # Append label data
  weather_df.loc[l, 7] = cycle_amount_df[0][m]
  m += 1

# Select random columns from whole input data with directed percentage
test_df = weather_df.sample(frac=TEST_PERCENTAGE)
train_df = weather_df.drop(test_df.index.values)

# Save results as csv files
train_df.to_csv(TRAIN_DATA_PATH, header=None)
test_df.to_csv(TEST_DATA_PATH, header=None)
