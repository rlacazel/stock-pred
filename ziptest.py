from zipline.api import order, record, symbol
from zipline.algorithm import TradingAlgorithm
import os
import pandas as pd
# from pandas.plotting import autocorrelation_plot
# from pandas.plotting import scatter_matrix
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import os
from datetime import datetime, date, time
from collections import OrderedDict
from six import iteritems
import pytz

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
	df = df['2010-01-01':'2017-10-01']
	df = df.loc[df['close'] != 0]
	# print(df.head())
	return df
	
# fetch data
# data = fetch_data('^GSPC')

def fetch_data2(stock_sym):
	# ts = TimeSeries(key='XNL4', output_format='pandas')
	# df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')
	
	# df.to_csv('{}_D1.csv'.format(symbol))
	# return pd.read_csv('{}_D1.csv'.format(symbol),
	return pd.read_csv('data/^GSPC_dailyadj.csv'.format(symbol),
		parse_dates=['date'],
		index_col='date',
		usecols=["close", "date"],
		squeeze=True,  # squeeze tells pandas to make this a Series
					   # instead of a 1-column DataFrame
	).sort_index().tz_localize('UTC').pct_change(1).iloc[1:]

dd = dict()
dd['^GSPC'] = fetch_data2('^GSPC')
# dd['^GSPC'] = dd['^GSPC'][4000:]
# print(dd['^GSPC'])
df = pd.DataFrame({key: d for key, d in iteritems(dd)})
# df.index = df.index.tz_localize(pytz.utc)
# print(df.head())	

# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
	# pass
	context.security = symbol('^GSPC')

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
	print(data)
	# Implement your algorithm logic here.

	MA1 = data[context.security].mavg(50)
	MA2 = data[context.security].mavg(200)
	
	current_price = data[context.security].price
	current_positions = context.portfolio.positions[symbol('^GSPC')].amount
	cash = context.portfolio.cash
	
	if (MA1 > MA2) and current_positions == 0:
		number_of_shares = int(cash/current_price)
		order(context.security, number_of_shares)
		log.info("Buying shares")
	elif (MA1 < MA2) and current_positions != 0:
		order_target(context.security, 0)
		log.info("Selling shares")
		
	record(MA1 = MA1, MA2 = MA2, Price=current_price)
	

algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data)

print(df)
# panel = pd.Panel({'^GSPC': df})
	
# panel = pd.Panel(data)
# panel.minor_axis = ['close']
# data.major_axis = data.major_axis.tz_localize(pytz.utc)

perf_manual = algo_obj.run(df)