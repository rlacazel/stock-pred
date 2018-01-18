#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Dual Moving Average Crossover algorithm.
This algorithm buys apple once its short moving average crosses
its long moving average (indicating upwards momentum) and sells
its shares once the averages cross again (indicating downwards
momentum).
"""

# Import exponential moving average from talib wrapper
# from talib import EMA
from toolz import merge
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd

from zipline.data.bundles import register, ingest
from zipline.data.bundles.csvdir import csvdir_equities
from zipline import run_algorithm
from zipline.api import order, record, symbol
from zipline.finance import commission


def initialize(context):
	context.asset = symbol('AAPL')

	# To keep track of whether we invested in the stock or not
	context.invested = False

	# Explicitly set the commission to the "old" value until we can
	# rebuild example data.
	# github.com/quantopian/zipline/blob/master/tests/resources/
	# rebuild_example_data#L105
	context.set_commission(commission.PerShare(cost=.0075, min_trade_cost=1.0))


def handle_data(context, data):
	trailing_window = data.history(context.asset, 'price', 40, '1d')
	if trailing_window.isnull().values.any():
		return
	
	print(data)
	# short_ema = EMA(trailing_window.values, timeperiod=20)
	# long_ema = EMA(trailing_window.values, timeperiod=40)

	buy = False
	sell = False

	if (short_ema[-1] > long_ema[-1]) and not context.invested:
		order(context.asset, 100)
		context.invested = True
		buy = True
	elif (short_ema[-1] < long_ema[-1]) and context.invested:
		order(context.asset, -100)
		context.invested = False
		sell = True

	record(AAPL=data.current(context.asset, "price"),
		   short_ema=short_ema[-1],
		   long_ema=long_ema[-1],
		   buy=buy,
		   sell=sell)


# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
	import matplotlib.pyplot as plt
	import logbook
	logbook.StderrHandler().push_application()
	log = logbook.Logger('Algorithm')

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	results.portfolio_value.plot(ax=ax1)
	ax1.set_ylabel('Portfolio value (USD)')

	ax2 = fig.add_subplot(212)
	ax2.set_ylabel('Price (USD)')

	# If data has been record()ed, then plot it.
	# Otherwise, log the fact that no data has been recorded.
	if 'AAPL' in results and 'short_ema' in results and 'long_ema' in results:
		results[['AAPL', 'short_ema', 'long_ema']].plot(ax=ax2)

		ax2.plot(results.ix[results.buy].index, results.short_ema[results.buy],
				 '^', markersize=10, color='m')
		ax2.plot(results.ix[results.sell].index,
				 results.short_ema[results.sell],
				 'v', markersize=10, color='k')
		plt.legend(loc=0)
		plt.gcf().set_size_inches(18, 8)
	else:
		msg = 'AAPL, short_ema and long_ema data not captured using record().'
		ax2.annotate(msg, xy=(0.1, 0.5))
		log.info(msg)

	plt.show()

def fetch_data_to_csv(stock_sym):

	filename = 'data/'+stock_sym+'.csv'
	
	if os.path.isfile(filename):
		df = pd.read_csv(filename)
	else:
		ts = TimeSeries(key='XNL4', output_format='pandas')
		df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')

		col_list = df.columns.tolist()
		col_list.remove('5. adjusted close') 
		df = df[col_list]
		df = df.rename(index=str, columns={'1. open': "open", '2. high': "high", '3. low': "low", '4. close': "close", '6. volume': "volume", '7. dividend amount': "dividend", '8. split coefficient': "split"})
		print(df.head())
		df.to_csv(filename)
	
def _test_args():
	"""Extra arguments to use when zipline's automated tests run this example.
	"""
	import pandas as pd

	return {
		'start': pd.Timestamp('2014-01-01', tz='utc'),
		'end': pd.Timestamp('2014-11-01', tz='utc'),
	}

# bundles: https://github.com/quantopian/zipline/blob/master/docs/source/bundles.rst
# run_algorithm(
        # initialize=initialize,
        # handle_data=handle_data,
        # before_trading_start=None,
        # analyze=analyze,
        # bundle='test',
        # Provide a default capital base, but allow the test to override.
        # **merge({'capital_base': 1e7}, _test_args())
    # )
	
fetch_data_to_csv('^GSPC')
start_session = pd.Timestamp('2016-1-3', tz='utc')
end_session = pd.Timestamp('2018-1-1', tz='utc')
print('register bundle')
register(
    'csvbundle',
    csvdir_equities(
        ['daily'],
        '/',
    ),
    # calendar_name='NYSE', # US equities
    # start_session=start_session,
    # end_session=end_session
)

ingest("csvbundle", show_progress=True)