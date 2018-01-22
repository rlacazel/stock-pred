from pyalgotrade import strategy, broker
from pyalgotrade.barfeed import yahoofeed
from toolz import merge
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.technical import cross
from pyalgotrade.technical import bollinger
# from pyalgotrade.stratanalyzer import sharpe
import datetime
import enum

class EntryType(enum.Enum):
	Nothing = 0
	Short = 1
	Long = 2

def codeFromOrder(order, shares):
	if order == 1 and shares == 0:
		return 1
	elif order == 3 and shares == 0:
		return -1
	if order == 1 or order == 3:
		return 0
	
def getStrOrder(order, shares):
	if order == 1 and shares == 0:
		return 'enter_long'
	elif order == 1 and shares < 0:
		return 'sell_short'
	elif order == 3 and shares > 0:
		return 'sell_long'
	if order == 3 and shares == 0:
		return 'enter_short'
	else:
		return 'undefined'

class BaseStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, lastDayOrder = False, entryType = EntryType.Nothing):
		super(BaseStrategy, self).__init__(feed)
		self._init_price = None
		self._last_price = None
		self._instrument = instrument
		self._lastDayOrder = lastDayOrder
		self._count = 0
		if entryType != EntryType.Nothing:
			self.marketOrder(self._instrument, 1 if EntryType.Long else -1)
		# self._longPos = self.enterLong(instrument, 1, True) if entryType == EntryType.Long else None
		# self._shortPos = self.enterShort(instrument, 1, True) if entryType == EntryType.Short else None
		d =  pd.DataFrame(columns=['date'], dtype=datetime.date)
		v =  pd.DataFrame(columns=[self.__class__.__name__], dtype=int)
		self._actions = pd.concat([d, v], axis=1)
		if feed.barsHaveAdjClose():
			self.setUseAdjustedValues(True)
	
	def onOrderUpdated(self, position):
		shares = self.getBroker().getShares(self._instrument)
		print(getStrOrder(position.getAction(), shares))

	def onExitCanceled(self, position):
		# If the exit was canceled, re-submit it.
		position.exitMarket()
		
	def updateVarContext(self, bars):
		self._count = self._count + 1
		if self._init_price is None:
			self._init_price = bars[self._instrument].getPrice()
		elif self.getFeed().eof():
			self._last_price = bars[self._instrument].getPrice()
	
	def onFinish(self, bars):
		print("FINISH: " + str(bars.getDateTime().date()))
	
	def displayActionIfLastDay(self, bars):
		has_order_update = False
		shares = self.getBroker().getShares(self._instrument)
		# for pos in filter(lambda x: x is not None, [self._longPos, self._shortPos]):
		for order in self.getBroker().getActiveOrders(self._instrument):
			# print(getStrOrder(order.getAction(), shares))
			if self._lastDayOrder and self.getFeed().eof():
				print(str(bars.getDateTime().date()) + '->' + order.getInstrument() + ': ' + getStrOrder(order.getAction(), shares))	
			has_order_update = True
			self._actions.loc[len(self._actions)] = [bars.getDateTime().date(),codeFromOrder(order.getAction(), shares)]  
		if not has_order_update:
			self._actions.loc[len(self._actions)] = [bars.getDateTime().date(), 0 if self._actions.empty else self._actions.loc[len(self._actions)-1][1]]
		
	def getChangePrice(self):
		return (self._last_price - self._init_price) / self._init_price * 100	

	def strategies(self, bars):
		raise NotImplementedError("Must override strategies")
	
	def getNbSharesToTake(self, bars):
		return int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
	
	def onBars(self, bars):
		# Wait for enough bars to be available to calculate SMA and RSI.
		if not(self.availableData()) or (self._lastDayOrder and not self.getFeed().eof()): return
		self.updateVarContext(bars)

		self.strategies(bars)
		
		self.displayActionIfLastDay(bars)

class SMAandRSI2(BaseStrategy):
	def __init__(self, feed, instrument, orders, lastDayOrder = False, entryType = EntryType.Nothing):
		super(SMAandRSI2, self).__init__(feed, instrument, lastDayOrder, entryType)
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__orders = orders

	def availableData(self):
		return True

	def strategies(self, bars):
		# If a position was not opened, check if we should enter a long position.
		d = bars.getDateTime().date()
		if d in self.__orders.index:
			row = self.__orders.loc[d].tolist()
			nb_buy_signal = sum([1 if r == 1 else 0 for r in row])
			nb_sell_signal = sum([1 if r == -1 else 0 for r in row])
			if self._longPos is not None:
				if nb_buy_signal <= 0 and not self._longPos.exitActive() :
					self._longPos.exitMarket()
					# print("LEAVE on %s" % bars[self._instrument].getDateTime())		
			elif self._shortPos is not None:
				if nb_sell_signal <= 0 and not self._shortPos.exitActive() :
					self._shortPos.exitMarket()
			else:
				if nb_buy_signal > 1:
					self._longPos = self.enterLong(self._instrument, self.getNbSharesToTake(bars), True)
					# print("BUY on %s" % bars[self._instrument].getDateTime())		
				elif nb_sell_signal > 1:
					self._shortPos = self.enterShort(self._instrument, self.getNbSharesToTake(bars), True)

class BBands(BaseStrategy):
	def __init__(self, feed, instrument, bBandsPeriod, lastDayOrder = False, entryType = EntryType.Nothing):
		super(BBands, self).__init__(feed, instrument, lastDayOrder, entryType)
		self.__bbands = bollinger.BollingerBands(feed[instrument].getCloseDataSeries(), bBandsPeriod, 2)

	def availableData(self):
		return self.__bbands.getUpperBand()[-1] is not None
		
	def strategies(self, bars):
		lower = self.__bbands.getLowerBand()[-1]
		upper = self.__bbands.getUpperBand()[-1]
		
		shares = self.getBroker().getShares(self._instrument)
		bar = bars[self._instrument]
		if shares == 0 and bar.getClose() < lower:
			sharesToBuy = int(self.getBroker().getCash(False) / bar.getClose())
			self.marketOrder(self._instrument, sharesToBuy)
		elif shares > 0 and bar.getClose() > upper:
			self.marketOrder(self._instrument, -1*shares)	
			
			
class SMACrossOver(BaseStrategy):
	def __init__(self, feed, instrument, smaPeriod, lastDayOrder = False, entryType = EntryType.Nothing):
		super(SMACrossOver, self).__init__(feed, instrument, lastDayOrder, entryType)
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__sma = ma.SMA(self.__prices, smaPeriod)

	def getSMA(self):
		return self.__sma

	def availableData(self):
		return self.__sma[-1] is not None
		
	def strategies(self, bars):
		# If a position was not opened, check if we should enter a long position.
		# if self._longPos is None:
			# if cross.cross_above(self.__prices, self.__sma) > 0:
				# self._longPos = self.enterLong(self._instrument, self.getNbSharesToTake(bars), True)
				# print("BUY on %s" % bars[self._instrument].getDateTime())		
		# elif not self._longPos.exitActive() and cross.cross_below(self.__prices, self.__sma) > 0:
			# self._longPos.exitMarket()
			# print("SELL on %s" % bars[self._instrument].getDateTime())	
		shares = self.getBroker().getShares(self._instrument)
		if shares == 0 and cross.cross_above(self.__prices, self.__sma) > 0:
			self.marketOrder(self._instrument, self.getNbSharesToTake(bars))
		elif shares > 0 and cross.cross_below(self.__prices, self.__sma) > 0:
			self.marketOrder(self._instrument, -1*shares)

			
class RSI2(BaseStrategy):
	def __init__(self, feed, instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold, lastDayOrder = False, entryType = EntryType.Nothing):
		super(RSI2, self).__init__(feed, instrument, lastDayOrder, entryType)
		self.__priceDS = feed[instrument].getPriceDataSeries()
		self.__entrySMA = ma.SMA(self.__priceDS, entrySMA)
		self.__exitSMA = ma.SMA(self.__priceDS, exitSMA)
		self.__rsi = rsi.RSI(self.__priceDS, rsiPeriod)
		self.__overBoughtThreshold = overBoughtThreshold
		self.__overSoldThreshold = overSoldThreshold

	def getEntrySMA(self):
		return self.__entrySMA

	def getExitSMA(self):
		return self.__exitSMA

	def getRSI(self):
		return self.__rsi
		
	def availableData(self):
		return self.__exitSMA[-1] is not None and self.__entrySMA[-1] is not None and self.__rsi[-1] is not None
				
	def strategies(self, bars):
		bar = bars[self._instrument]
		# if self._longPos is not None:
			# if self.exitLongSignal():
				# self._longPos.exitMarket()
				# print("Sell on %s" % bars[self._instrument].getDateTime())	
		# elif self._shortPos is not None:
			# if self.exitShortSignal():
				# self._shortPos.exitMarket()
				# print("Sell short on %s" % bars[self._instrument].getDateTime())	
		# else:
			# if self.enterLongSignal(bar):
				# self._longPos = self.enterLong(self._instrument, self.getNbSharesToTake(bars), True)
				# print("BUY on %s" % bars[self._instrument].getDateTime())	
			# elif self.enterShortSignal(bar):
				# self._shortPos = self.enterShort(self._instrument, self.getNbSharesToTake(bars), True)
				# print("BUY Short on %s" % bars[self._instrument].getDateTime())	
	
		shares = self.getBroker().getShares(self._instrument)
		if shares > 0 and self.exitLongSignal():
			self.marketOrder(self._instrument, -shares)
			# print("Sell on %s" % bars[self._instrument].getDateTime())
		elif shares < 0 and self.exitShortSignal():
			self.marketOrder(self._instrument, -shares) # need pos value but its minus by minus equal plus
			# print("Sell short on %s" % bars[self._instrument].getDateTime())
		elif shares == 0:
			if self.enterLongSignal(bar):
				self.marketOrder(self._instrument, self.getNbSharesToTake(bars))
				# print("BUY on %s" % bars[self._instrument].getDateTime())
			elif self.enterShortSignal(bar):
				self.marketOrder(self._instrument, -self.getNbSharesToTake(bars))
				# print("BUY Short %s" % bars[self._instrument].getDateTime())
		
	def enterLongSignal(self, bar):
		return bar.getPrice() > self.__entrySMA[-1] and self.__rsi[-1] <= self.__overSoldThreshold

	def exitLongSignal(self):
		return cross.cross_above(self.__priceDS, self.__exitSMA)#and not self._longPos.exitActive()# and self.getBroker().getShares(self._instrument) > 0 # not self._longPos.exitActive()

	def enterShortSignal(self, bar):
		return bar.getPrice() < self.__entrySMA[-1] and self.__rsi[-1] >= self.__overBoughtThreshold

	def exitShortSignal(self):
		return cross.cross_below(self.__priceDS, self.__exitSMA)# and not self._shortPos.exitActive()# and self.getBroker().getShares(self._instrument) > 0 #not self._shortPos.exitActive()