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
from collections import OrderedDict

flags = tf.app.flags
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
FLAGS = flags.FLAGS

class Stock:

	def __init__(self, symbol, name, past_idx):
		self.symbol = symbol
		self.name = name
		self.past_idx = past_idx
		
	def __str__(self):
		return self.name + '[' + self.symbol + '] -> idx used: ' +  str(self.past_idx)


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
	

stocks = [Stock('^GSPC', 'snp', list(range(1,4))), Stock('^NYA', 'nyse', list(range(1,4))), Stock('^DJI', 'djia', list(range(1,4))), Stock('^N225', 'nikkei', list(range(0,3))),
			Stock('^HSI', 'hangseng', list(range(0,3))), Stock('^FTSE', 'ftse', list(range(1,3))), Stock('^GDAXI', 'dax', list(range(1,3))), Stock('^AORD', 'aord', list(range(0,3)))]

def build_data(stocks, pred_data):
	closing_data = pd.DataFrame()
	log_return_data = pd.DataFrame()
	
	for stock in stocks:
		d = fetch_data(stock.symbol)
		closing_data[stock.name] = d['close']
		
	closing_data = closing_data.fillna(method='ffill')
	
	for stock in stocks:
		log_return_data[stock.name] = np.log(closing_data[stock.name]/closing_data[stock.name].shift())
		
	output_pos = pred_data+'_log_return_positive'
	output_neg = pred_data+'_log_return_negative'
	log_return_data[output_pos] = 0
	log_return_data.ix[log_return_data[pred_data] >= 0, output_pos] = 1
	log_return_data[output_neg] = 0
	log_return_data.ix[log_return_data[pred_data] < 0, output_neg] = 1
	
	# build data column
	columns = [output_pos, output_neg]
	for stock in stocks:
		for idx in stock.past_idx:
			columns.append(stock.name + '_' + str(idx))
	print(columns)
	
	training_test_data = pd.DataFrame(columns)
	print(training_test_data.describe())
	
	for i in range(7, len(log_return_data)):
		val = OrderedDict()
		val[output_pos] = log_return_data[output_pos].ix[i]
		val[output_neg] = log_return_data[output_neg].ix[i]
		for stock in stocks:
			for idx in stock.past_idx:
				val[stock.name + '_' + str(idx)] = log_return_data[stock.name].ix[i-idx]
		# print(val)
		training_test_data = training_test_data.append(val,ignore_index=True)
		print(training_test_data.head())
	
	training_test_data = training_test_data[columns]
	print(training_test_data.head())
	return training_test_data

training_test_data = build_data(stocks,'snp')
  
# Separate in training and test data
predictors_tf = training_test_data[training_test_data.columns[2:]]

classes_tf = training_test_data[training_test_data.columns[:2]]

print(classes_tf)

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