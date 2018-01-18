from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.stratanalyzer import returns
from toolz import merge
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
from pyalgotrade.technical import ma
import smacrossover

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
		
class MyStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, smaPeriod):
		super(MyStrategy, self).__init__(feed, 1000000)
		self.__position = None
		self.__instrument = instrument
		# We'll use adjusted close values instead of regular close values.
		self.setUseAdjustedValues(True)
		self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)

	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("BUY at $%.2f" % (execInfo.getPrice()))

	def onEnterCanceled(self, position):
		self.__position = None

	def onExitOk(self, position):
		execInfo = position.getExitOrder().getExecutionInfo()
		self.info("SELL at $%.2f" % (execInfo.getPrice()))
		self.__position = None

	def onExitCanceled(self, position):
		# If the exit was canceled, re-submit it.
		self.__position.exitMarket()

	def onBars(self, bars):
		# Wait for enough bars to be available to calculate a SMA.
		if self.__sma[-1] is None:
			return

		bar = bars[self.__instrument]
		# If a position was not opened, check if we should enter a long position.
		if self.__position is None:
			if bar.getPrice() > self.__sma[-1]:
				# Enter a buy market order for 10 shares. The order is good till canceled.
				self.__position = self.enterLong(self.__instrument, 10, True)
		# Check if we have to exit the position.
		elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
			self.__position.exitMarket()

def run_strategy(smaPeriod):
	# Load the yahoo feed from the CSV file
	feed = yahoofeed.Feed()
	file = fetch_data('^GSPC')
	feed.addBarsFromCSV("data", file)

	# Evaluate the strategy with the feed.
	myStrategy = MyStrategy(feed, "data", smaPeriod)
	myStrategy.run()
	print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())

# run_strategy(15)

feed = yahoofeed.Feed()
file = fetch_data('MSFT')
feed.addBarsFromCSV("data", file)
	
# Evaluate the strategy with the feed's bars.
myStrategy = smacrossover.SMACrossOver(feed, "data", 20)

# Attach a returns analyzers to the strategy.
returnsAnalyzer = returns.Returns()
myStrategy.attachAnalyzer(returnsAnalyzer)

# Attach the plotter to the strategy.
plt = plotter.StrategyPlotter(myStrategy)
# Include the SMA in the instrument's subplot to get it displayed along with the closing prices.
plt.getInstrumentSubplot("data").addDataSeries("SMA", myStrategy.getSMA())
# Plot the simple returns on each bar.
plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())

# Run the strategy.
myStrategy.run()
myStrategy.info("Final portfolio value: $%.2f" % myStrategy.getResult())

# Plot the strategy.
plt.plot()