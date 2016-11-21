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

for i in range(raw_df.shape[0]):
  # Parse string to date format
  date = dt.strptime(raw_df.loc[i, 0], "%Y/%m/%d %H:%M")
  # Selecting 'm/d' form 'y/m/d' format
  weather_df.loc[i, 0] = str(date.strftime('%Y/%m/%d'))
  # Selecting hour, month and weekday form 'YYYY/MM/DD' format
  weather_df.loc[i, 1] = date.hour

# Normalize real_value columns
# Temprature, Wind-speed, Precipiation
for j, k in [[2, 1], [3, 2], [4, 3]]:
  weather_df[j] = zscore(raw_df[k])

# Pivot data to date x hour
weather_df = weather_df.pivot(index=0, columns=1)

#for l in range(len(weather_df.index)):
amount_iter = cycle_amount_df.iterrows()
m = 0
for l in weather_df.index:
  date = dt.strptime(l, "%Y/%m/%d")
  weather_df.loc[l, 5] = date.month
  weather_df.loc[l, 6] = date.weekday()
  # Append label data
  weather_df.loc[l, 7] = cycle_amount_df[0][m]
  m += 1


# Select random columns from whole input data with directed percentage.
test_df = weather_df.sample(frac=TEST_PERCENTAGE)
train_df = weather_df.drop(test_df.index.values)

train_df.to_csv(TRAIN_DATA_PATH, header=None)
test_df.to_csv(TEST_DATA_PATH, header=None)
