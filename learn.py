import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import os
from datetime import datetime, date, time

flags = tf.app.flags
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
FLAGS = flags.FLAGS

def fetch_yahoo_data(stock_sym):
	
	df = pdr.get_data_yahoo("^N225", start="2000-01-01", end="2018-12-30")
		
	col_list = df.columns.tolist()
	print(col_list)
	col_list.remove('Open')
	col_list.remove('High')
	col_list.remove('Low') 
	col_list.remove('Adj Close') 
	col_list.remove('Volume')
	
	# print(col_list)
	df = df[col_list] #.set_index('Date')
	# print(df.head())
	return df
	
	
def fetch_data(stock_sym):
	filename = 'data/'+stock_sym+'_dailyadj.csv'
	if os.path.isfile(filename):
		df = pd.read_csv(filename)
	else:
		ts = TimeSeries(key='XNL4', output_format='pandas')
		df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')
		df.to_csv(filename)
		df = pd.read_csv(filename)
		
	col_list = df.columns.tolist()
	col_list.remove('close')
	col_list.remove('dividend amount')
	col_list.remove('split coefficient') 
	col_list.remove('adjusted close') 
	col_list.remove('open') 
	col_list.remove('high') 
	col_list.remove('low') 
	col_list.remove('volume') 
	col_list.append('close')
	
	df = df[col_list[1:]].set_index('date')
	df = df['2010-01-01':'2017-10-01']
	df = df.loc[df['close'] != 0]
	# df = df[(datetime.strptime(df['date'], '%Y-%m-%d') > datetime.date(2008, 6, 24)) & (datetime.strptime(df['date'], '%Y-%m-%d') < datetime.date(2010, 6, 24))]
	# print(df.head())
	return df
	
closing_data = pd.DataFrame()

snp = fetch_data('^GSPC')
nikkei = fetch_data('^N225')
dax = fetch_data('^GDAXI')
djai = fetch_data('^DJI')
nyse = fetch_data('^NYA')
ftse_100 = fetch_data('^FTSE')
hangseng = fetch_data('^HSI')
allord = fetch_data('^AORD')

closing_data['snp_close'] = snp['close']
closing_data['nyse_close'] = nyse['close']
closing_data['djia_close'] = djai['close']
closing_data['nikkei_close'] = nikkei['close']
closing_data['hangseng_close'] = hangseng['close']
closing_data['ftse_close'] = ftse_100['close']
closing_data['dax_close'] = dax['close']
closing_data['aord_close'] = allord['close']

# Pandas includes a very convenient function for filling gaps in the data.
closing_data = closing_data.fillna(method='ffill')

# print(closing_data.head())
print(closing_data.describe())	

closing_data['snp_close_scaled'] = closing_data['snp_close'] / max(closing_data['snp_close'])
closing_data['nyse_close_scaled'] = closing_data['nyse_close'] / max(closing_data['nyse_close'])
closing_data['djia_close_scaled'] = closing_data['djia_close'] / max(closing_data['djia_close'])
closing_data['nikkei_close_scaled'] = closing_data['nikkei_close'] / max(closing_data['nikkei_close'])
closing_data['hangseng_close_scaled'] = closing_data['hangseng_close'] / max(closing_data['hangseng_close'])
closing_data['ftse_close_scaled'] = closing_data['ftse_close'] / max(closing_data['ftse_close'])
closing_data['dax_close_scaled'] = closing_data['dax_close'] / max(closing_data['dax_close'])
closing_data['aord_close_scaled'] = closing_data['aord_close'] / max(closing_data['aord_close'])

# _ = pd.concat([closing_data['snp_close_scaled'],
# closing_data['nyse_close_scaled'],
# closing_data['djia_close_scaled'],
# closing_data['nikkei_close_scaled'],
# closing_data['hangseng_close_scaled'],
# closing_data['ftse_close_scaled'],
# closing_data['dax_close_scaled'],
# closing_data['aord_close_scaled']], axis=1).plot(figsize=(20, 15))

  
# fig = plt.figure()
# fig.set_figwidth(20)
# fig.set_figheight(15)

# _ = autocorrelation_plot(closing_data['snp_close'], label='snp_close')
# _ = autocorrelation_plot(closing_data['nyse_close'], label='nyse_close')
# _ = autocorrelation_plot(closing_data['djia_close'], label='djia_close')
# _ = autocorrelation_plot(closing_data['nikkei_close'], label='nikkei_close')
# _ = autocorrelation_plot(closing_data['hangseng_close'], label='hangseng_close')
# _ = autocorrelation_plot(closing_data['ftse_close'], label='ftse_close')
# _ = autocorrelation_plot(closing_data['dax_close'], label='dax_close')
# _ = autocorrelation_plot(closing_data['aord_close'], label='aord_close')

# _ = plt.legend(loc='upper right')	
# plt.show()

# _ = scatter_matrix(pd.concat([closing_data['snp_close_scaled'],
  # closing_data['nyse_close_scaled'],
  # closing_data['djia_close_scaled'],
  # closing_data['nikkei_close_scaled'],
  # closing_data['hangseng_close_scaled'],
  # closing_data['ftse_close_scaled'],
  # closing_data['dax_close_scaled'],
  # closing_data['aord_close_scaled']], axis=1), figsize=(20, 20), diagonal='kde')
# plt.show()

# Stationary : https://people.duke.edu/~rnau/411diff.htm

log_return_data = pd.DataFrame()

log_return_data['snp_log_return'] = np.log(closing_data['snp_close']/closing_data['snp_close'].shift())
log_return_data['nyse_log_return'] = np.log(closing_data['nyse_close']/closing_data['nyse_close'].shift())
log_return_data['djia_log_return'] = np.log(closing_data['djia_close']/closing_data['djia_close'].shift())
log_return_data['nikkei_log_return'] = np.log(closing_data['nikkei_close']/closing_data['nikkei_close'].shift())
log_return_data['hangseng_log_return'] = np.log(closing_data['hangseng_close']/closing_data['hangseng_close'].shift())
log_return_data['ftse_log_return'] = np.log(closing_data['ftse_close']/closing_data['ftse_close'].shift())
log_return_data['dax_log_return'] = np.log(closing_data['dax_close']/closing_data['dax_close'].shift())
log_return_data['aord_log_return'] = np.log(closing_data['aord_close']/closing_data['aord_close'].shift())

# print(log_return_data.describe())


# _ = pd.concat([log_return_data['snp_log_return'],
  # log_return_data['nyse_log_return'],
  # log_return_data['djia_log_return'],
  # log_return_data['nikkei_log_return'],
  # log_return_data['hangseng_log_return'],
  # log_return_data['ftse_log_return'],
  # log_return_data['dax_log_return'],
  # log_return_data['aord_log_return']], axis=1).plot(figsize=(20, 15))
  
# fig = plt.figure()
# fig.set_figwidth(20)
# fig.set_figheight(15)

# _ = autocorrelation_plot(log_return_data['snp_log_return'], label='snp_log_return')
# _ = autocorrelation_plot(log_return_data['nyse_log_return'], label='nyse_log_return')
# _ = autocorrelation_plot(log_return_data['djia_log_return'], label='djia_log_return')
# _ = autocorrelation_plot(log_return_data['nikkei_log_return'], label='nikkei_log_return')
# _ = autocorrelation_plot(log_return_data['hangseng_log_return'], label='hangseng_log_return')
# _ = autocorrelation_plot(log_return_data['ftse_log_return'], label='ftse_log_return')
# _ = autocorrelation_plot(log_return_data['dax_log_return'], label='dax_log_return')
# _ = autocorrelation_plot(log_return_data['aord_log_return'], label='aord_log_return')

# _ = plt.legend(loc='upper right')

# plt.show()

# Prepare data

log_return_data['snp_log_return_positive'] = 0
log_return_data.ix[log_return_data['snp_log_return'] >= 0, 'snp_log_return_positive'] = 1
log_return_data['snp_log_return_negative'] = 0
log_return_data.ix[log_return_data['snp_log_return'] < 0, 'snp_log_return_negative'] = 1

training_test_data = pd.DataFrame(
  columns=[
    'snp_log_return_positive', 'snp_log_return_negative',
    'snp_log_return_1', 'snp_log_return_2', 'snp_log_return_3',
    'nyse_log_return_1', 'nyse_log_return_2', 'nyse_log_return_3',
    'djia_log_return_1', 'djia_log_return_2', 'djia_log_return_3',
    'nikkei_log_return_0', 'nikkei_log_return_1', 'nikkei_log_return_2',
    'hangseng_log_return_0', 'hangseng_log_return_1', 'hangseng_log_return_2',
	# 'ftse_log_return_0',
	'ftse_log_return_1', 'ftse_log_return_2',
    # 'dax_log_return_0',
	'dax_log_return_1', 'dax_log_return_2',
    'aord_log_return_0', 'aord_log_return_1', 'aord_log_return_2'])

for i in range(7, len(log_return_data)):
  snp_log_return_positive = log_return_data['snp_log_return_positive'].ix[i]
  snp_log_return_negative = log_return_data['snp_log_return_negative'].ix[i]
  snp_log_return_1 = log_return_data['snp_log_return'].ix[i-1]
  snp_log_return_2 = log_return_data['snp_log_return'].ix[i-2]
  snp_log_return_3 = log_return_data['snp_log_return'].ix[i-3]
  nyse_log_return_1 = log_return_data['nyse_log_return'].ix[i-1]
  nyse_log_return_2 = log_return_data['nyse_log_return'].ix[i-2]
  nyse_log_return_3 = log_return_data['nyse_log_return'].ix[i-3]
  djia_log_return_1 = log_return_data['djia_log_return'].ix[i-1]
  djia_log_return_2 = log_return_data['djia_log_return'].ix[i-2]
  djia_log_return_3 = log_return_data['djia_log_return'].ix[i-3]
  nikkei_log_return_0 = log_return_data['nikkei_log_return'].ix[i]
  nikkei_log_return_1 = log_return_data['nikkei_log_return'].ix[i-1]
  nikkei_log_return_2 = log_return_data['nikkei_log_return'].ix[i-2]
  hangseng_log_return_0 = log_return_data['hangseng_log_return'].ix[i]
  hangseng_log_return_1 = log_return_data['hangseng_log_return'].ix[i-1]
  hangseng_log_return_2 = log_return_data['hangseng_log_return'].ix[i-2]
  # ftse_log_return_0 = log_return_data['ftse_log_return'].ix[i]
  ftse_log_return_1 = log_return_data['ftse_log_return'].ix[i-1]
  ftse_log_return_2 = log_return_data['ftse_log_return'].ix[i-2]
  # dax_log_return_0 = log_return_data['dax_log_return'].ix[i]
  dax_log_return_1 = log_return_data['dax_log_return'].ix[i-1]
  dax_log_return_2 = log_return_data['dax_log_return'].ix[i-2]
  aord_log_return_0 = log_return_data['aord_log_return'].ix[i]
  aord_log_return_1 = log_return_data['aord_log_return'].ix[i-1]
  aord_log_return_2 = log_return_data['aord_log_return'].ix[i-2]
  training_test_data = training_test_data.append(
    {'snp_log_return_positive':snp_log_return_positive,
    'snp_log_return_negative':snp_log_return_negative,
    'snp_log_return_1':snp_log_return_1,
    'snp_log_return_2':snp_log_return_2,
    'snp_log_return_3':snp_log_return_3,
    'nyse_log_return_1':nyse_log_return_1,
    'nyse_log_return_2':nyse_log_return_2,
    'nyse_log_return_3':nyse_log_return_3,
    'djia_log_return_1':djia_log_return_1,
    'djia_log_return_2':djia_log_return_2,
    'djia_log_return_3':djia_log_return_3,
    'nikkei_log_return_0':nikkei_log_return_0,
    'nikkei_log_return_1':nikkei_log_return_1,
    'nikkei_log_return_2':nikkei_log_return_2,
    'hangseng_log_return_0':hangseng_log_return_0,
    'hangseng_log_return_1':hangseng_log_return_1,
    'hangseng_log_return_2':hangseng_log_return_2,
    # 'ftse_log_return_0':ftse_log_return_0,
    'ftse_log_return_1':ftse_log_return_1,
    'ftse_log_return_2':ftse_log_return_2,
    # 'dax_log_return_0':dax_log_return_0,
    'dax_log_return_1':dax_log_return_1,
    'dax_log_return_2':dax_log_return_2,
    'aord_log_return_0':aord_log_return_0,
    'aord_log_return_1':aord_log_return_1,
    'aord_log_return_2':aord_log_return_2},
    ignore_index=True)
  
print(training_test_data.describe())  
# Separate in training and test data
predictors_tf = training_test_data[training_test_data.columns[2:]]

classes_tf = training_test_data[training_test_data.columns[:2]]

training_set_size = int(len(training_test_data) * 0.8)
test_set_size = len(training_test_data) - training_set_size

training_predictors_tf = predictors_tf[:training_set_size]
training_classes_tf = classes_tf[:training_set_size]
test_predictors_tf = predictors_tf[training_set_size:]
test_classes_tf = classes_tf[training_set_size:]

# training_predictors_tf.describe()

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict
    )

  tpfn = float(tp) + float(fn)
  tpr = 0 if tpfn == 0 else float(tp)/tpfn
  fpr = 0 if tpfn == 0 else float(fp)/tpfn

  total = float(tp) + float(fp) + float(fn) + float(tn)
  accuracy = 0 if total == 0 else (float(tp) + float(tn))/total

  recall = tpr
  tpfp = float(tp) + float(fp)
  precision = 0 if tpfp == 0 else float(tp)/tpfp
  
  f1_score = 0 if recall == 0 else (2 * (precision * recall)) / (precision + recall)
  
  print('Precision = ', precision)
  print('Recall = ', recall)
  print('F1 Score = ', f1_score)
  print('Accuracy = ', accuracy)
  
  
 

sess1 = tf.Session()

num_predictors = len(training_predictors_tf.columns)
num_classes = len(training_classes_tf.columns)

feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, 2])

weights1 = tf.Variable(tf.truncated_normal([len(training_test_data.columns)-2, 50], stddev=0.0001))
biases1 = tf.Variable(tf.ones([50]))

weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
biases2 = tf.Variable(tf.ones([25]))
                     
weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
biases3 = tf.Variable(tf.ones([2]))

hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

cost = -tf.reduce_sum(actual_classes*tf.log(model))

train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.global_variables_initializer()
sess1.run(init)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 15001):
  sess1.run(
    train_op1, 
    feed_dict={
      feature_data: training_predictors_tf.values, 
      actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
    }
  )
  if i%5000 == 0:
    print(i, sess1.run(
      accuracy,
      feed_dict={
        feature_data: training_predictors_tf.values, 
        actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
      }
    ))
	
feed_dict= {
  feature_data: test_predictors_tf.values,
  actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)
}

tf_confusion_metrics(model, actual_classes, sess1, feed_dict)