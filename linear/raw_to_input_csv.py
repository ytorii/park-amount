import numpy as np
import pandas as pd
import re

DAYS = ["日", "月", "火", "水", "木", "金", "土"]
NUM_PATTERN = r"[0-9]+"

TEST_PERCENTAGE = 0.1

# Adding 0 - 10 numbers as header names.
weather_df = pd.read_csv("raw/input_data.csv", header=None, names=np.arange(5))
cycle_amount_df = pd.read_csv("raw/cycle_amount.csv", header=None)

# Removing unused coloumns, if caution flags for wheater data are set
#weather_df = pd.read_csv("raw/input_data.csv", header=None, names=np.arange(11))
#tmp_df = weather_df.drop([3,4,6,7,9,10], axis=1)

for i in range(weather_df.shape[0]):
  # Selecting month form 'YYYY/MM/DD' format
  date = weather_df.loc[i, 0]
  weather_df.loc[i, 0] = re.findall(NUM_PATTERN, date)[1]
  # Transform days to index
  days = weather_df.loc[i, 1]
  weather_df.loc[i, 1] = DAYS.index(days)

# Append label data at the last of columns
weather_df[weather_df.shape[0] + 1] = cycle_amount_df

# Select random columns from whole input data with directed percentage.
test_df = weather_df.sample(frac=TEST_PERCENTAGE)
train_df = weather_df.drop(test_df.index.values)

train_df.to_csv('input/train_data.csv', header=None)
test_df.to_csv('input/test_data.csv', header=None)
