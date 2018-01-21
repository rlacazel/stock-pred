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
import enum

class EntryType(enum.Enum):
    Nothing = 0
    Short = 1
    Long = 2
	
def getStrOrder(order):
	if order == 1:
		return 'buy'
	elif order == 2:
		return 'buy_to_cover'
	if order == 3:
		return 'sell'
	elif order == 4:
		return 'sell_short'
	else:
		return 'undefined'

class BaseStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, lastDayOrder):
		super(BaseStrategy, self).__init__(feed)
		self._init_price = None
		self._last_price = None
		self._instrument = instrument
		self._lastDayOrder = lastDayOrder
		self._count = 0
		if feed.barsHaveAdjClose():
			self.setUseAdjustedValues(True)
	
	def updateVarContext(self, bars):
		self._count = self._count + 1
		if self._init_price is None:
			self._init_price = bars[self._instrument].getPrice()
		elif self.getFeed().eof():
			self._last_price = bars[self._instrument].getPrice()
	
	def displayActionIfLastDay(self, bars):
		if self._lastDayOrder and self.getFeed().eof():
			for pos in filter(lambda x: x is not None, self.getPositions()):
				for order in pos.getActiveOrders():
					print(str(bars.getDateTime().date()) + '->' + order.getInstrument() + ': ' + getStrOrder(order.getAction()))
			
	def getPositions(self):
		raise NotImplementedError("Must override getPositions")
		
	def getChangePrice(self):
		return (self._last_price - self._init_price) / self._init_price * 100	

	def strategies(self, bars):
		raise NotImplementedError("Must override strategies")
	
	def onBars(self, bars):
		# Wait for enough bars to be available to calculate SMA and RSI.
		if not(self.availableData()) or (self._lastDayOrder and not self.getFeed().eof()): return
		self.updateVarContext(bars)

		self.strategies(bars)
		
		self.displayActionIfLastDay(bars)
	
class SMACrossOver(BaseStrategy):
	def __init__(self, feed, instrument, smaPeriod, lastDayOrder = False, entryType = EntryType.Nothing):
		super(SMACrossOver, self).__init__(feed, instrument, lastDayOrder)
		self.__position = self.enterLong(instrument, 1, True) if entryType == EntryType.Long else None
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__sma = ma.SMA(self.__prices, smaPeriod)

	def getSMA(self):
		return self.__sma

	def onEnterCanceled(self, position):
		self._position = None

	def onExitOk(self, position):
		self.__position = None

	def onExitCanceled(self, position):
		# If the exit was canceled, re-submit it.
		self.__position.exitMarket()

	def availableData(self):
		return self.__sma[-1] is not None
		
	def getPositions(self):
		return [self.__position]
		
	def strategies(self, bars):
		# If a position was not opened, check if we should enter a long position.
		if self.__position is None:
			if cross.cross_above(self.__prices, self.__sma) > 0:
				shares = int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
				# Enter a buy market order. The order is good till canceled.
				self.__position = self.enterLong(self._instrument, shares, True)
				# print("BUY on %s" % bars[self._instrument].getDateTime())		
		# Check if we have to exit the position.
		elif not self.__position.exitActive() and cross.cross_below(self.__prices, self.__sma) > 0:
			self.__position.exitMarket()
			# print("SELL on %s" % bars[self._instrument].getDateTime())		

			
class RSI2(BaseStrategy):
	def __init__(self, feed, instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold, lastDayOrder = False, entryType = EntryType.Nothing):
		super(RSI2, self).__init__(feed, instrument, lastDayOrder)
		self.__priceDS = feed[instrument].getPriceDataSeries()
		self.__entrySMA = ma.SMA(self.__priceDS, entrySMA)
		self.__exitSMA = ma.SMA(self.__priceDS, exitSMA)
		self.__rsi = rsi.RSI(self.__priceDS, rsiPeriod)
		self.__overBoughtThreshold = overBoughtThreshold
		self.__overSoldThreshold = overSoldThreshold
		self.__longPos = self.enterLong(instrument, 1, True) if entryType == EntryType.Long else None
		self.__shortPos = self.enterShort(instrument, 1, True) if entryType == EntryType.Short else None

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
		
	def availableData(self):
		return self.__exitSMA[-1] is not None and self.__entrySMA[-1] is not None and self.__rsi[-1] is not None
		
	def getPositions(self):
		return [self.__shortPos, self.__longPos]
				
	def strategies(self, bars):
		bar = bars[self._instrument]
		if self.__longPos is not None:
			if self.exitLongSignal():
				self.__longPos.exitMarket()
		elif self.__shortPos is not None:
			if self.exitShortSignal():
				self.__shortPos.exitMarket()
		else:
			if self.enterLongSignal(bar):
				shares = int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
				self.__longPos = self.enterLong(self._instrument, shares, True)
			elif self.enterShortSignal(bar):
				shares = int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
				self.__shortPos = self.enterShort(self._instrument, shares, True)
		
	def enterLongSignal(self, bar):
		return bar.getPrice() > self.__entrySMA[-1] and self.__rsi[-1] <= self.__overSoldThreshold

	def exitLongSignal(self):
		return cross.cross_above(self.__priceDS, self.__exitSMA) and not self.__longPos.exitActive()

	def enterShortSignal(self, bar):
		return bar.getPrice() < self.__entrySMA[-1] and self.__rsi[-1] >= self.__overBoughtThreshold

	def exitShortSignal(self):
		return cross.cross_below(self.__priceDS, self.__exitSMA) and not self.__shortPos.exitActive()