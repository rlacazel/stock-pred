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

class PredictionFromType(enum.Enum):
	NoPred = 0
	Nothing = 1
	Short = 2
	Long = 3

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
	def __init__(self, feed, instrument, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BaseStrategy, self).__init__(feed)
		if feed.barsHaveAdjClose():
			self.setUseAdjustedValues(True)
		self._init_price = None
		self._last_price = None
		self._instrument = instrument
		self._count = 0
		self.entryTypeWhenAskPrediction = entryTypeWhenAskPrediction
		if entryTypeWhenAskPrediction == PredictionFromType.Short:
			self.marketOrder(self._instrument, -1)
		elif entryTypeWhenAskPrediction == PredictionFromType.Long:
			self.marketOrder(self._instrument, 1)
		d =  pd.DataFrame(columns=['date'], dtype=datetime.date)
		v =  pd.DataFrame(columns=[self.__class__.__name__], dtype=int)
		# self._actions = pd.concat([d, v], axis=1)
		self._actions = dict()
		self._onGoingAcceptedOrder = dict()
	
	def getName(self):
		return self.__class__.__name__
	
	def onOrderUpdated(self, position):
		# print(position.getId())
		shares = self.getBroker().getShares(self._instrument)
		date = position.getSubmitDateTime().date() if position.getSubmitDateTime() is not None else None
		if position.isAccepted() and date is not None:
			for order in self.getBroker().getActiveOrders(self._instrument):
				if order.getSubmitDateTime().date() < date:
					self._onGoingAcceptedOrder[order.getId()] = str(date)
					self.getBroker().cancelOrder(order)
			self._onGoingAcceptedOrder[position.getId()] = str(date) + ' -> ' + getStrOrder(position.getAction(), shares)
			self.debug('Accepted  [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
		elif position.isFilled() and position.getId() in self._onGoingAcceptedOrder:
			self.debug('Activated [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
			self._actions[date] = 1 if shares > 0 else -1 if shares < 0 else 0
		elif position.isCanceled() and position.getId() in self._onGoingAcceptedOrder:
			self.debug('Cancelled [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
			# raise Exception('MarketOrder cancelled')

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
		shares = self.getBroker().getShares(self._instrument)
		if self.entryTypeWhenAskPrediction != PredictionFromType.NoPred:
			for order in self.getBroker().getActiveOrders(self._instrument):
				print(str(bars.getDateTime().date()) + '->' + order.getInstrument() + ': ' + getStrOrder(order.getAction(), shares))	

	def getIndicators(self):
		return []
	
	def getChangePrice(self):
		return (self._last_price - self._init_price) / self._init_price * 100	

	def strategies(self, bars):
		raise NotImplementedError("Must override strategies")
	
	def getNbSharesToTake(self, bars):
		return int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
	
	def onBars(self, bars):
		shares = self.getBroker().getShares(self._instrument)
		self._actions[bars.getDateTime().date()] = 1 if shares > 0 else -1 if shares < 0 else 0
		
		if not(self.availableData()) or (self.entryTypeWhenAskPrediction != PredictionFromType.NoPred and not self.getFeed().eof()): return
		self.updateVarContext(bars)
		self.strategies(bars)

class AllStrat(BaseStrategy):
	def __init__(self, feed, instrument, orders, weights, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(AllStrat, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.__orders = orders
		self._weights = weights

	def getName(self):
		return '_'.join([k for k in self._weights.keys() if self._weights[k] > 0])
			
	def availableData(self):
		return True

	def getThreshold(self, val):
		score = 0
		tot_weight = sum(self._weights.values())
		for strat, weight in self._weights.items():
			score += weight*val[strat]
		return score/tot_weight
		
	def strategies(self, bars):
		# If a position was not opened, check if we should enter a long position.
		d = bars.getDateTime().date()
		bar = bars[self._instrument]
		if d in self.__orders:
			row = self.__orders[d]
			threshold = self.getThreshold(row)
			# print(threshold)
	
			shares = self.getBroker().getShares(self._instrument)
			if shares > 0 and threshold < 1:
				self.marketOrder(self._instrument, -shares)
			elif shares < 0 and threshold > -1:
				self.marketOrder(self._instrument, -shares) # need pos value but its minus by minus equal plus
			elif shares == 0:
				if threshold == 1:
					self.marketOrder(self._instrument, self.getNbSharesToTake(bars))
				elif threshold == -1:
					self.marketOrder(self._instrument, -self.getNbSharesToTake(bars))

class BBands(BaseStrategy): # params = [bBandsPeriod]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BBands, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.setUseAdjustedValues(True) # Raise an error is no adj value
		self.__bbands = bollinger.BollingerBands(feed[instrument].getAdjCloseDataSeries(), params[0], 2)

	def getIndicators(self):
		return [['upper', self.__bbands.getUpperBand()], ['middle', self.__bbands.getMiddleBand()], ['lower', self.__bbands.getLowerBand()]]
		
	def availableData(self):
		return self.__bbands.getUpperBand()[-1] is not None
		
	def strategies(self, bars):
		lower = self.__bbands.getLowerBand()[-1]
		upper = self.__bbands.getUpperBand()[-1]
		
		shares = self.getBroker().getShares(self._instrument)
		bar = bars[self._instrument]
		if shares == 0 and bar.getAdjClose() < lower:
			shareToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, shareToBuy)
			self.stopOrder(self._instrument, bar.getAdjClose()-2, -shareToBuy, True)
		elif shares > 0 and bar.getAdjClose() > upper:
			self.marketOrder(self._instrument, -1*shares)	
			
			
class SMACrossOver(BaseStrategy):  # params = [smaPeriod]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(SMACrossOver, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__sma = ma.SMA(self.__prices, params[0])

	def getIndicators(self):
		return [['sma', self.__sma]]
	
	def availableData(self):
		return self.__sma[-1] is not None
		
	def strategies(self, bars):
		shares = self.getBroker().getShares(self._instrument)
		if shares == 0 and cross.cross_above(self.__prices, self.__sma) > 0:
			self.marketOrder(self._instrument, self.getNbSharesToTake(bars))
		elif shares > 0 and cross.cross_below(self.__prices, self.__sma) > 0:
			self.marketOrder(self._instrument, -1*shares)

			
class RSI2(BaseStrategy): # params = [entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(RSI2, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.__priceDS = feed[instrument].getPriceDataSeries()
		self.__entrySMA = ma.SMA(self.__priceDS, params[0])
		self.__exitSMA = ma.SMA(self.__priceDS, params[1])
		self.__rsi = rsi.RSI(self.__priceDS, params[2])
		self.__overBoughtThreshold = params[3]
		self.__overSoldThreshold = params[4]

	def getIndicators(self):
		return [['entry_sma', self.__entrySMA], ['exit_sma', self.__exitSMA], ['rsi', self.__rsi]]
		
	def availableData(self):
		return self.__exitSMA[-1] is not None and self.__entrySMA[-1] is not None and self.__rsi[-1] is not None
				
	def strategies(self, bars):
		bar = bars[self._instrument]
	
		shares = self.getBroker().getShares(self._instrument)
		if shares > 0 and self.exitLongSignal():
			self.marketOrder(self._instrument, -shares)
		elif shares < 0 and self.exitShortSignal():
			self.marketOrder(self._instrument, -shares) # need pos value but its minus by minus equal plus
		elif shares == 0:
			if self.enterLongSignal(bar):
				self.marketOrder(self._instrument, self.getNbSharesToTake(bars))
			elif self.enterShortSignal(bar):
				self.marketOrder(self._instrument, -self.getNbSharesToTake(bars))
		
	def enterLongSignal(self, bar):
		return bar.getPrice() > self.__entrySMA[-1] and self.__rsi[-1] <= self.__overSoldThreshold

	def exitLongSignal(self):
		return cross.cross_above(self.__priceDS, self.__exitSMA)#and not self._longPos.exitActive()# and self.getBroker().getShares(self._instrument) > 0 # not self._longPos.exitActive()

	def enterShortSignal(self, bar):
		return bar.getPrice() < self.__entrySMA[-1] and self.__rsi[-1] >= self.__overBoughtThreshold

	def exitShortSignal(self):
		return cross.cross_below(self.__priceDS, self.__exitSMA)# and not self._shortPos.exitActive()# and self.getBroker().getShares(self._instrument) > 0 #not self._shortPos.exitActive()