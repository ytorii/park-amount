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
  
  # Inputs with catagorical values(like November, Monday)
  month = tf.contrib.layers.sparse_column_with_keys(column_name="month", keys=MONTH_KEYS)
  days = tf.contrib.layers.sparse_column_with_keys(column_name="days", keys=DAYS_KEYS)

  # Catagorical values are converted into real values so that concatinating with real value columns
  m_embed = tf.contrib.layers.embedding_column(month, dimension=8)
  d_embed = tf.contrib.layers.embedding_column(days, dimension=8)

  # Inputs with real values(like 1.03, 2.3)
  temprature = tf.contrib.layers.real_valued_column("temprature")
  precipitation = tf.contrib.layers.real_valued_column("precipitation")
  wind_speed = tf.contrib.layers.real_valued_column("wind_speed")

  feature_columns = [m_embed, d_embed, temprature, precipitation, wind_speed]

  m = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns, hidden_units=[10, 20, 10])

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
  m.fit(input_fn=lambda: input_fn(df_train), steps=2000)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()

if __name__ == "__main__":
  tf.app.run()
