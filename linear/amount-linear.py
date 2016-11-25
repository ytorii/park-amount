import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

INPUTDATA_PATH = "input/"
TRAIN_FILENAME = INPUTDATA_PATH + "train_data.csv"
TEST_FILENAME = INPUTDATA_PATH + "test_data.csv"

def make_columns_list(name_list, num_range):
  columns_list = []
  for name in name_list:
    for num in range(num_range):
      columns_list.append('%s%s' % (name, num))

  return columns_list
      
# precipitation = 降水量
CONTINUOUS_COLUMNS_NAME = [ "temp", "prec", "wind" ]
CONTINUOUS_COLUMNS_NUM = 24
CONTINUOUS_COLUMNS = make_columns_list(CONTINUOUS_COLUMNS_NAME, CONTINUOUS_COLUMNS_NUM)

CATEGORICAL_COLUMNS = [ "month", "days" ]

LABEL_COLUMN = [ "label" ]

INPUT_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMN

MONTH_KEYS = [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" ]
DAYS_KEYS = [ "0", "1", "2", "3", "4", "5", "6" ]

def build_estimator():
  
  # Inputs with catagolized values(like November, Monday)
  month = tf.contrib.layers.sparse_column_with_keys(column_name="month", keys=MONTH_KEYS)
  days = tf.contrib.layers.sparse_column_with_keys(column_name="days", keys=DAYS_KEYS)

  feature_columns = [ month, days ]

  # Inputs with real values(like 1.03, 2.3)
  for col in CONTINUOUS_COLUMNS:
    feature_columns.append(tf.contrib.layers.real_valued_column(col))

  
  # Relevance among inputs, this shuold be derived from catagolized values.
  for i in range(CONTINUOUS_COLUMNS_NUM):
    temp = 'temp%s' % (i)
    prec = 'prec%s' % (i)
    wind = 'wind%s' % (i)

    feature_columns.append(
      tf.contrib.layers.crossed_column([
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(temp), boundaries=[0, 5, 10, 15, 20, 25, 30, 35, 40]),
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(prec), boundaries=[0, 0.5, 1.0, 2.0, 3.0, 4.0]),
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(wind), boundaries=[0, 0.3, 0.5, 1.0, 2.0, 3.0])
      ], hash_bucket_size=int(1e6))
    )

  grad_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  adagrad_opt = tf.train.AdagradOptimizer(learning_rate=0.1)
  adam_opt = tf.train.AdamOptimizer(learning_rate=0.1)
  m = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns, optimizer=adam_opt, enable_centered_bias=True)
  #m = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns, optimizer=adagrad_opt, model_dir="/tmp/park_amount")
  #m = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns)

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

  model = build_estimator()
  #validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn=lambda: input_fn(df_test), every_n_steps=50)
  #model.fit(input_fn=lambda: input_fn(df_train), steps=10000, monitors=[validation_monitor])
  model.fit(input_fn=lambda: input_fn(df_train), steps=10000)
  results = model.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()

if __name__ == "__main__":
  tf.app.run()
