from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

INPUTDATA_PATH = "input/"
TRAIN_FILENAME = INPUTDATA_PATH + "train_data.csv"
TEST_FILENAME = INPUTDATA_PATH + "test_data.csv"

LABEL_COLUMN = 'label'
# precipitation = 降水量
INPUT_COLUMNS = [ "month", "days",  "temprature", "precipitation", "wind_speed", LABEL_COLUMN ]

CATEGORICAL_COLUMNS = [ "month", "days" ]
CONTINUOUS_COLUMNS = [ "temprature", "precipitation", "wind_speed" ]
MONTH_KEYS = [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" ]
DAYS_KEYS = [ "0", "1", "2", "3", "4", "5", "6" ]

def build_estimator():
  
  # Inputs with catagolized values(like November, Monday)
  month = tf.contrib.layers.sparse_column_with_keys(column_name="month", keys=MONTH_KEYS)
  days = tf.contrib.layers.sparse_column_with_keys(column_name="days", keys=DAYS_KEYS)

  # Inputs with real values(like 1.03, 2.3)
  temprature = tf.contrib.layers.real_valued_column("temprature")
  precipitation = tf.contrib.layers.real_valued_column("precipitation")
  wind_speed = tf.contrib.layers.real_valued_column("wind_speed")

  # Relevance among inputs, this shuold be derived from catagolized values.
  t_buckets = tf.contrib.layers.bucketized_column(temprature, boundaries=[0, 5, 10, 15, 20, 25, 30, 35, 40])
  p_buckets = tf.contrib.layers.bucketized_column(precipitation, boundaries=[0, 0.5, 1.0, 2.0, 3.0, 4.0])
  w_buckets = tf.contrib.layers.bucketized_column(wind_speed, boundaries=[0, 0.3, 0.5, 1.0, 2.0, 3.0])
  t_x_p_x_w = tf.contrib.layers.crossed_column([t_buckets, p_buckets, w_buckets], hash_bucket_size=int(1e6))

  feature_columns = [month, days, temprature, precipitation, wind_speed, t_x_p_x_w]
  #feature_columns = [month, days, temprature, precipitation, wind_speed]

  m = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns)

  return m

def input_fn(df):
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i,0] for i in range(df[k].size)],
    values=df[k].values.astype(str),
    shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}

  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  #feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  label = tf.constant(df[LABEL_COLUMN].values)

  return feature_cols, label

def train_and_eval():
  df_train = pd.read_csv(tf.gfile.Open(TRAIN_FILENAME),
    names=INPUT_COLUMNS,
    skipinitialspace=True,
    engine="python")

  df_test = pd.read_csv(tf.gfile.Open(TEST_FILENAME),
    names=INPUT_COLUMNS,
    skipinitialspace=True,
    engine="python")

  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  m = build_estimator()
  m.fit(input_fn=lambda: input_fn(df_train), steps=20000)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()

if __name__ == "__main__":
  tf.app.run()
