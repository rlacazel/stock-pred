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

#flags = tf.app.flags
#flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
#FLAGS = flags.FLAGS

class Stock:

	def __init__(self, symbol, name, past_idx, data=['close']):
		self.symbol = symbol
		self.name = name
		self.past_idx = past_idx
		self.data = data
		
	def __str__(self):
		return self.name + '[' + self.symbol + '] -> idx used: ' +  str(self.past_idx)

class Indic:

	def __init__(self, symbol, name, indics):
		self.symbol = symbol
		self.name = name
		self.indics = indics
		
	def __str__(self):
		return self.name + '[' + self.symbol + '] -> idx used: ' +  str(self.past_idx)

#def fetch_yahoo_data(stock_sym):
#	
#	df = pdr.get_data_yahoo("^N225", start="2000-01-01", end="2018-12-30")
#		
#	col_list = df.columns.tolist()
#	print(col_list)
#	col_list.remove('Open')
#	col_list.remove('High')
#	col_list.remove('Low') 
#	col_list.remove('Adj Close') 
#	col_list.remove('Volume')
#	
#	# print(col_list)
#	df = df[col_list] #.set_index('Date')
#	# print(df.head())
#	return df
	
def fetch_indic(stock_sym, indic):	
	filename = 'data/'+stock_sym+'_' + indic + '.csv'
	if not os.path.isfile(filename):
		ti = TechIndicators(key='XNL4', output_format='pandas')
		if indic == 'sma': 
			df, meta_data = ti.get_sma(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'rsi': 
			df, meta_data = ti.get_rsi(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'trix': 
			df, meta_data = ti.get_trix(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'adx': 
			df, meta_data = ti.get_adx(symbol=stock_sym, interval='daily', time_period=20)
		elif indic == 'adxr': 
			df, meta_data = ti.get_adxr(symbol=stock_sym, interval='daily', time_period=20)
		elif indic == 'apo': 
			df, meta_data = ti.get_apo(symbol=stock_sym, interval='daily', series_type='close')
		elif indic == 'aroon': 
			df, meta_data = ti.get_aroon(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'atr': 
			df, meta_data = ti.get_atr(symbol=stock_sym, interval='daily', time_period=20)
		elif indic == 'bbands': 
			df, meta_data = ti.get_bbands(symbol=stock_sym, interval='daily', time_period=20)
		elif indic == 'cci': 
			df, meta_data = ti.get_cci(symbol=stock_sym, interval='daily', time_period=20)
		elif indic == 'cmo': 
			df, meta_data = ti.get_cmo(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'dx': 
			df, meta_data = ti.get_dx(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'ema': 
			df, meta_data = ti.get_ema(symbol=stock_sym, interval='daily', time_period=20, series_type='close')
		elif indic == 'macd': 
			df, meta_data = ti.get_macd(symbol=stock_sym, interval='daily', series_type='close')
		df.to_csv(filename)
	df = pd.read_csv(filename)
	
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	df = df.set_index('date')
	df = df['2010-01-01':'2017-10-01']
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
	
	df = df[col_list[1:]].set_index('date')
	# df = df[[type]]
	df = df['2010-01-01':'2017-10-01']
	df = df.loc[df['close'] != 0]
	# print(df.head())
	return df
	

_stocks_snp = [Stock('^GSPC', 'snp', list(range(1,4))), Stock('^NYA', 'nyse', list(range(1,4))), Stock('^DJI', 'djia', list(range(1,4))), Stock('^N225', 'nikkei', list(range(0,3))),
			Stock('^HSI', 'hangseng', list(range(0,3))), Stock('^FTSE', 'ftse', list(range(1,3))), Stock('^GDAXI', 'dax', list(range(1,3))), Stock('^AORD', 'aord', list(range(0,3)))]
_indics_snp = [Indic('^GSPC','snp',['sma'])]

_stocks_msft = [Stock('MSFT', 'msft', list(range(1,4)), ['close','open'])]
_indics_msft = [Indic('MSFT','msft',['sma','rsi','trix','adx','adxr','apo','atr','cci','cmo','dx','ema','macd'])] # 'aroon','bbands'

def build_data(stocks, indics, pred_data):
	closing_data = pd.DataFrame()
	log_return_data = pd.DataFrame()
	
	# collect data
	for stock in stocks:
		d = fetch_data(stock.symbol)
		for dta in stock.data:
			closing_data[stock.name+'_'+dta] = d[dta]
		
	for indic in indics:
		for indic_name in indic.indics:	
			d = fetch_indic(indic.symbol, indic_name)
			closing_data[indic.name+'_'+indic_name] = d[indic_name.upper()]
		
	closing_data = closing_data.fillna(method='ffill')
	
	# normalize data
	for stock in stocks:
		for dta in stock.data:
			log_return_data[stock.name+'_'+dta] = np.log(closing_data[stock.name+'_'+dta]/closing_data[stock.name+'_'+dta].shift())
		
	for indic in indics:
		for indic_name in indic.indics:	
			# Z-Score normalisation
			# log_return_data[indic.name+'_'+indic_name] = (closing_data[indic.name+'_'+indic_name] - closing_data[indic.name+'_'+indic_name].mean()) / closing_data[indic.name+'_'+indic_name].std()
			# tan estimator
			log_return_data[indic.name+'_'+indic_name] = 0.5*(np.tanh(0.01*(closing_data[indic.name+'_'+indic_name] - closing_data[indic.name+'_'+indic_name].mean())/closing_data[indic.name+'_'+indic_name].std())+1)
			# print(log_return_data[indic.name+'_'+indic_name])
			
	# build output
	output_pos = pred_data+'_log_return_positive'
	output_neg = pred_data+'_log_return_negative'
	log_return_data[output_pos] = 0
	log_return_data.loc[log_return_data[pred_data] >= 0, output_pos] = 1
	log_return_data[output_neg] = 0
	log_return_data.loc[log_return_data[pred_data] < 0, output_neg] = 1
	
	# build data column
	columns = [output_pos, output_neg]
	for stock in stocks:
		for dta in stock.data:
			for idx in stock.past_idx:
				columns.append(stock.name + '_' + dta  + '_' + str(idx))
	for indic in indics:
		for indic_name in indic.indics:	
			columns.append(indic.name+'_'+indic_name)
	
	training_test_data = pd.DataFrame(columns)
	# print(training_test_data.describe())
	
	for i in range(7, len(log_return_data)):
		training_test_data.loc[i-7,output_pos] = log_return_data[output_pos].iloc[i]
		training_test_data.loc[i-7,output_neg] = log_return_data[output_neg].iloc[i]
		for stock in stocks:
			for dta in stock.data:
				for idx in stock.past_idx:
					training_test_data.loc[i-7,stock.name + '_' + dta  + '_' + str(idx)] = log_return_data[stock.name + '_' + dta].iloc[i-idx]
		for indic in indics:
			for indic_name in indic.indics:	
				training_test_data.loc[i-7,indic.name+'_'+indic_name] = log_return_data[indic.name+'_'+indic_name].iloc[i-1]
			
	training_test_data.reset_index()
	training_test_data = training_test_data[columns]
	print(training_test_data.describe())
	return training_test_data

training_test_data = build_data(_stocks_msft, _indics_msft, 'msft_close')
# training_test_data = build_data(_stocks_snp, _indics_snp, 'snp_close')
  
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

weights1 = tf.Variable(tf.truncated_normal([num_predictors, num_predictors*2], stddev=0.0001))
biases1 = tf.Variable(tf.ones([num_predictors*2]))

weights2 = tf.Variable(tf.truncated_normal([num_predictors*2, num_predictors], stddev=0.0001))
biases2 = tf.Variable(tf.ones([num_predictors]))
                     
weights3 = tf.Variable(tf.truncated_normal([num_predictors, 2], stddev=0.0001))
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