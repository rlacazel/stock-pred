import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import os
from datetime import datetime, date, time
# from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf

# yf.pdr_override() # <== that's all it takes :-)

# download dataframe
# data = pdr.get_data_yahoo("^N225", start="2017-01-01", end="2018-12-30")
# print(data.tail())

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
	df = df['2010-01-01':'2015-10-01']
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
closing_data['djai_close'] = djai['close']
closing_data['nikkei_close'] = nikkei['close']
closing_data['hangseng_close'] = hangseng['close']
closing_data['ftse_100_close'] = ftse_100['close']
closing_data['dax_close'] = dax['close']
closing_data['allord_close'] = allord['close']

# Pandas includes a very convenient function for filling gaps in the data.
closing_data = closing_data.fillna(method='ffill')

print(closing_data.head())
print(closing_data.describe())	