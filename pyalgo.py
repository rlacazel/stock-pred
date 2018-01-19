from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.stratanalyzer import returns
from toolz import merge
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
from pyalgotrade.technical import ma
import strategies

# TUTO : http://gbeced.github.io/pyalgotrade/docs/v0.18/html/tutorial.html

def fetch_data(stock_sym):

	filename = 'data/'+stock_sym+'.csv'
	
	# if os.path.isfile(filename):
		# df = pd.read_csv(filename)
	# else:
	if not os.path.isfile(filename):
		ts = TimeSeries(key='XNL4', output_format='pandas')
		df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')

		col_list = df.columns.tolist()
		col_list.remove('7. dividend amount') 
		col_list.remove('8. split coefficient') 
		# print(col_list)
		df = df[col_list]
		df = df.reset_index()
		df = df.rename(columns={'date': "Date", '1. open': "Open", '2. high': "High", '3. low': "Low", '4. close': "Close", '6. volume': "Volume", '5. adjusted close': "Adj Close"})
		df = df.set_index('Date')
		df = df['2016-01-01':'2018-10-01']
		# print(df.head())
		df.to_csv(filename)
	return filename

def run(strategy):

	initial_capital = strategy.getResult()
	
	# Attach a returns analyzers to the strategy.
	returnsAnalyzer = returns.Returns()
	strategy.attachAnalyzer(returnsAnalyzer)

	# Attach the plotter to the strategy.
	plt = plotter.StrategyPlotter(strategy)
	# Include the SMA in the instrument's subplot to get it displayed along with the closing prices.
	# plt.getInstrumentSubplot("data").addDataSeries("SMA", strategy.getSMA())
	
	# Plot the simple returns on each bar.
	plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())

	# Run the strategy.
	strategy.run()
	percent_return = (strategy.getResult() - initial_capital) / initial_capital * 100
	strategy.info("Final portfolio value: $%.2f (tot return: %.2f%%) (daily return: %.3f%%)" % (strategy.getResult(), percent_return, percent_return / strategy._count))
	print(strategy._count)
	
	# Plot the strategy.
	plt.plot()
	
feed = yahoofeed.Feed()
file = fetch_data('MSFT')
feed.addBarsFromCSV("data", file)
	
# Evaluate the strategy with the feed's bars.
sma = strategies.SMACrossOver(feed, "data", 20)

feed = yahoofeed.Feed()
feed.addBarsFromCSV("data", file)

rsi2 = strategies.RSI2(feed, "data", 154, 5, 2, 91, 18)

run(sma)
# run(rsi2)