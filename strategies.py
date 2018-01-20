from pyalgotrade import strategy, broker
from pyalgotrade.barfeed import yahoofeed
from toolz import merge
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.technical import cross
import datetime

def getStrOrder(order):
	if order == 1:
		return 'buy'
	elif order == 2:
		return 'buy_to_cover'
	if order == 3:
		return 'sell'
	elif order == 4:
		return 'sell_to_cover'
	else:
		return 'undefined'


class SMACrossOver(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, smaPeriod, lastDayOrder = False, has_a_position = False):
		super(SMACrossOver, self).__init__(feed)
		self.__instrument = instrument
		self.__position = None if not has_a_position else self.enterLong(self.__instrument, 1, True)
		# We'll use adjusted close values instead of regular close values.
		self.setUseAdjustedValues(True)
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__sma = ma.SMA(self.__prices, smaPeriod)
		self._count = 0
		self._lastDayOrder = lastDayOrder
		self._init_price = None
		self._last_price = None

	def getSMA(self):
		return self.__sma

	def onEnterCanceled(self, position):
		self.__position = None

	def onExitOk(self, position):
		self.__position = None

	def onExitCanceled(self, position):
		# If the exit was canceled, re-submit it.
		self.__position.exitMarket()

	def getChangePrice(self):
		return (self._last_price - self._init_price) / self._init_price * 100
		
	def onBars(self, bars):
		if self.__sma[-1] is None or (self._lastDayOrder and not self.getFeed().eof()):
			return
		if self._init_price is None:
			self._init_price = bars[self.__instrument].getPrice()
		elif self.getFeed().eof():
			self._last_price = bars[self.__instrument].getPrice()
			
		self._count = self._count + 1
		# If a position was not opened, check if we should enter a long position.
		if self.__position is None:
			if cross.cross_above(self.__prices, self.__sma) > 0:
				shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
				# Enter a buy market order. The order is good till canceled.
				self.__position = self.enterLong(self.__instrument, shares, True)
				# print("BUY on %s" % bars[self.__instrument].getDateTime())		
		# Check if we have to exit the position.
		elif not self.__position.exitActive() and cross.cross_below(self.__prices, self.__sma) > 0:
			self.__position.exitMarket()
			# print("SELL on %s" % bars[self.__instrument].getDateTime())		
			
		if self.__position is not None and self._lastDayOrder and self.getFeed().eof():
			for order in self.__position.getActiveOrders():
				print(str(bars.getDateTime().date()) + '->' + order.getInstrument() + ': ' + getStrOrder(order.getAction()))

			
class RSI2(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold):
		super(RSI2, self).__init__(feed)
		self.__instrument = instrument
		# We'll use adjusted close values, if available, instead of regular close values.
		if feed.barsHaveAdjClose():
			self.setUseAdjustedValues(True)
		self.__priceDS = feed[instrument].getPriceDataSeries()
		self.__entrySMA = ma.SMA(self.__priceDS, entrySMA)
		self.__exitSMA = ma.SMA(self.__priceDS, exitSMA)
		self.__rsi = rsi.RSI(self.__priceDS, rsiPeriod)
		self.__overBoughtThreshold = overBoughtThreshold
		self.__overSoldThreshold = overSoldThreshold
		self.__longPos = None
		self.__shortPos = None
		self._count = 0

	def getEntrySMA(self):
		return self.__entrySMA

	def getExitSMA(self):
		return self.__exitSMA

	def getRSI(self):
		return self.__rsi

	def onEnterCanceled(self, position):
		if self.__longPos == position:
			self.__longPos = None
		elif self.__shortPos == position:
			self.__shortPos = None
		else:
			assert(False)

	def onExitOk(self, position):
		if self.__longPos == position:
			self.__longPos = None
		elif self.__shortPos == position:
			self.__shortPos = None
		else:
			assert(False)

	def onExitCanceled(self, position):
		# If the exit was canceled, re-submit it.
		position.exitMarket()

	def onBars(self, bars):
		# Wait for enough bars to be available to calculate SMA and RSI.
		if self.__exitSMA[-1] is None or self.__entrySMA[-1] is None or self.__rsi[-1] is None:
			return

		self._count = self._count + 1
		bar = bars[self.__instrument]
		if self.__longPos is not None:
			if self.exitLongSignal():
				self.__longPos.exitMarket()
		elif self.__shortPos is not None:
			if self.exitShortSignal():
				self.__shortPos.exitMarket()
		else:
			if self.enterLongSignal(bar):
				shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
				self.__longPos = self.enterLong(self.__instrument, shares, True)
			elif self.enterShortSignal(bar):
				shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
				self.__shortPos = self.enterShort(self.__instrument, shares, True)

	def enterLongSignal(self, bar):
		return bar.getPrice() > self.__entrySMA[-1] and self.__rsi[-1] <= self.__overSoldThreshold

	def exitLongSignal(self):
		return cross.cross_above(self.__priceDS, self.__exitSMA) and not self.__longPos.exitActive()

	def enterShortSignal(self, bar):
		return bar.getPrice() < self.__entrySMA[-1] and self.__rsi[-1] >= self.__overBoughtThreshold

	def exitShortSignal(self):
		return cross.cross_below(self.__priceDS, self.__exitSMA) and not self.__shortPos.exitActive()