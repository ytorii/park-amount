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
CONTINUOUS_COLUMNS_NAME = [ "temp", "wind", "prec" ]
#CONTINUOUS_COLUMNS_NAME = [ "temp", "prec" ]
CONTINUOUS_COLUMNS_NUM = 10
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

  wide_columns = [ month, days ]

  # Catagorical values are converted into real values so that concatinating with real value columns
  #m_embed = tf.contrib.layers.embedding_column(month, dimension=4)
  #d_embed = tf.contrib.layers.embedding_column(days, dimension=3)
  m_embed = tf.contrib.layers.one_hot_column(month)
  d_embed = tf.contrib.layers.one_hot_column(days)

  deep_columns = [ m_embed, d_embed ]

  # Inputs with real values(like 1.03, 2.3)
  for col in CONTINUOUS_COLUMNS:
    deep_columns.append(tf.contrib.layers.real_valued_column(col))
    #deep_columns.append(tf.contrib.layers.real_valued_column(col, dimension=2))

  # Relevance among inputs, this shuold be derived from catagolized values.
  for i in range(CONTINUOUS_COLUMNS_NUM):
    temp = 'temp%s' % (i)
    wind = 'wind%s' % (i)
    prec = 'prec%s' % (i)

    wide_columns.append(
      tf.contrib.layers.crossed_column([
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(temp), boundaries=[0, 10, 20, 25, 35, 40]),
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(wind), boundaries=[0, 0.3, 0.5, 1.0, 2.0, 3.0]),
        tf.contrib.layers.bucketized_column(tf.contrib.layers.real_valued_column(prec), boundaries=[0, 0.5, 1.0, 2.0, 3.0, 4.0])
      ], hash_bucket_size=int(1e6),
      hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY))

  # Optimizers
  grad_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  adagrad_opt = tf.train.AdagradOptimizer(learning_rate=0.02)
  adam_opt = tf.train.AdamOptimizer()
  padagrad_opt = tf.train.ProximalAdagradOptimizer( learning_rate=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.001)

  #m = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns, optimizer=adam_opt, dropout=0.15, hidden_units=[ 50, 20 ])
  m = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_optimizer=adam_opt,
        #dnn_hidden_units=[ 30, 50, 1 ])
        dnn_hidden_units=[ 256, 256, 1 ])

  return m

def input_fn(df):
  # Tensor rank should be 2 (e.g. [[1.], [2]]), so put shape as [size, 1]
  df_size = df[CONTINUOUS_COLUMNS[0]].shape[0]
  continuous_cols = {k: tf.constant(df[k].values, shape=[df_size,1]) for k in CONTINUOUS_COLUMNS}
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i,0] for i in range(df[k].size)],
    values=df[k].values.astype(str),
    shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}

  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)

  # dtype should be epilicitly declarated as float when all labels are inetger
  label = tf.constant(df[LABEL_COLUMN].values, dtype=tf.float64)

  return feature_cols, label

def train_and_eval():
  df_train = pd.read_csv(tf.gfile.Open(TRAIN_FILENAME),
    names=INPUT_COLUMNS,
    #dtype=tf.float64
    skipinitialspace=True,
    engine="python")

  df_test = pd.read_csv(tf.gfile.Open(TEST_FILENAME),
    names=INPUT_COLUMNS,
    #dtype=tf.float64,
    skipinitialspace=True,
    engine="python")

  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  model = build_estimator()
  #model.fit(input_fn=lambda: input_fn(df_train), steps=10000)
  model.fit(input_fn=lambda: input_fn(df_train), steps=10000)
  test_results = model.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(test_results):
    print("%s: %s" % (key, test_results[key]))

def main(_):
  train_and_eval()

if __name__ == "__main__":
  tf.app.run()
