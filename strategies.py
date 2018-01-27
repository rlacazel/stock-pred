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
from pyalgotrade.technical import vwap
import datetime
import enum

# Strategies to study:
# - p&f targets
# - Fibonacco retracement levels

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

def getPercentDiff(init_v, new_v):
	return 0 if init_v == 0 else (new_v - init_v) / init_v

class BaseStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BaseStrategy, self).__init__(feed)
		# if feed.barsHaveAdjClose():
			# self.setUseAdjustedValues(True)
		self._init_price = None
		self._last_price = None
		self._instrument = instrument
		self._count = 0
		self._prices = feed[instrument].getPriceDataSeries()
		self.entryTypeWhenAskPrediction = entryTypeWhenAskPrediction
		if entryTypeWhenAskPrediction == PredictionFromType.Short:
			self.marketOrder(self._instrument, -1)
		elif entryTypeWhenAskPrediction == PredictionFromType.Long:
			self.marketOrder(self._instrument, 1)
		d =  pd.DataFrame(columns=['date'], dtype=datetime.date)
		v =  pd.DataFrame(columns=[self.__class__.__name__], dtype=int)
		# self._actions = pd.concat([d, v], axis=1)
		self._signals = dict()
		self._onGoingAcceptedOrder = dict()
		# Exit and stop loss orders.
		self._takeProfitOrder = None
		self._stopLossOrder = None
		self._stopLossThreshold = 0.25
		
	def setPredictionMode(self, mode):
		self.entryTypeWhenAskPrediction = mode
		if self.entryTypeWhenAskPrediction == PredictionFromType.Short:
			self.marketOrder(self._instrument, -1)
		elif self.entryTypeWhenAskPrediction == PredictionFromType.Long:
			self.marketOrder(self._instrument, 1)
			
	def getName(self):
		return self.__class__.__name__
	
	def onOrderUpdated(self, order):
		curr_date = self.getCurrentDateTime().date()
		shares = self.getBroker().getShares(self._instrument)
		datePositionSubmitted = order.getSubmitDateTime().date() if order.getSubmitDateTime() is not None else None
			
		if order.isAccepted():			
			self._onGoingAcceptedOrder[order.getId()] = str(datePositionSubmitted) + ' -> ' + getStrOrder(order.getAction(), shares)
			# self.debug('Accepted [' + str(order.getId()) + ']: ' + self._onGoingAcceptedOrder[order.getId()] + ' [shares=' + str(shares) + ']' )
		if order.isFilled():
			# Was the take profit order filled ?
			if self._takeProfitOrder is not None and order.getId() == self._takeProfitOrder.getId():
				entryPrice = order.getExecutionInfo().getPrice()
				self.debug('Take profit [' + str(order.getId()) + ']: ' + str(self.getFeed().getCurrentBars().getDateTime().date()) + " at " + str(entryPrice))
				self._takeProfitOrder = None
				if self._stopLossOrder is not None: 
					self.getBroker().cancelOrder(self._stopLossOrder)
			# Was the stop loss order filled ?
			elif self._stopLossOrder is not None and order.getId() == self._stopLossOrder.getId():
				entryPrice = order.getExecutionInfo().getPrice()
				self.debug('Stop loss   [' + str(order.getId()) + ']: ' + str(self.getFeed().getCurrentBars().getDateTime().date()) + " at " + str(entryPrice))
				self._stopLossOrder = None
				if self._takeProfitOrder is not None: 
					self.getBroker().cancelOrder(self._takeProfitOrder)
			else:
				self.debug('Activated   [' + str(order.getId()) + ']: ' + self._onGoingAcceptedOrder[order.getId()] + ' [shares=' + str(shares) + ']' )
				# self._signals[datePositionSubmitted] = 1 if shares > 0 else -1 if shares < 0 else 0
				if shares == 0: # meaning we have sold out everything
					self.cleanStopAndLimitOrder()
				else:
					self._signals[datePositionSubmitted] = 1 if shares > 0 else -1 if shares < 0 else 0

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
	
	def placeStopOrder(self, instr, price, share, goodTillCancelled):
		self.cleanStopAndLimitOrder()
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			if share > 0: # stop order or short position
				priceLimit = price * (1 + self._stopLossThreshold)
			else:
				priceLimit = price * (1 - self._stopLossThreshold)
			self._stopLossOrder = self.stopOrder(instr, priceLimit, share, goodTillCancelled)
	
	def placeLimitOrder(self, instr, price, share, goodTillCancelled):
		self.cleanStopAndLimitOrder()
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			if share > 0: # take profit from short position
				priceLimit = price * (1 - self._stopLossThreshold)
			else:
				priceLimit = price * (1 + self._stopLossThreshold)
			self._takeProfitOrder = self.limitOrder(instr, priceLimit, share, goodTillCancelled)
	
	def cleanStopAndLimitOrder(self):
		if self._takeProfitOrder is not None and self._takeProfitOrder.isActive():
			self.getBroker().cancelOrder(self._takeProfitOrder)
		if self._stopLossOrder is not None and self._stopLossOrder.isActive():
			self.getBroker().cancelOrder(self._stopLossOrder)
	
	def placeStopAndLimitOrder(self, instr, price, share, goodTillCancelled):
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			self.placeLimitOrder(instr, price, share, goodTillCancelled)
			self.placeStopOrder(instr, price, share, goodTillCancelled)
	
	def getChangePrice(self):
		return (self._last_price - self._init_price) / self._init_price * 100	

	def strategies(self, bars):
		raise NotImplementedError("Must override strategies")
	
	def getNbSharesToTake(self, bars):
		return int(self.getBroker().getCash() * 0.9 / bars[self._instrument].getPrice())
	
	def onBars(self, bars):
		shares = self.getBroker().getShares(self._instrument)
		# self._actions[bars.getDateTime().date()] = 1 if shares > 0 else -1 if shares < 0 else 0
		
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

class BaseBB(BaseStrategy): # params = [bands]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BaseBB, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self._bbands = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), params[0], 2)
		self._lows = feed[instrument].getLowDataSeries()
		self._volumes = feed[instrument].getVolumeDataSeries()
		self._stopLossThreshold = 0.1 #override
		self._lower = self._bbands.getLowerBand()
		self._upper = self._bbands.getUpperBand()
		self._middle = self._bbands.getMiddleBand()

	def getIndicators(self):
		return [['upper', self._bbands.getUpperBand()], ['middle', self._bbands.getMiddleBand()], ['lower', self._bbands.getLowerBand()]]
		
	def availableData(self):
		return len(self._upper) > 1 and self._upper[-2] is not None

class BBands(BaseBB): # params = [bBandsPeriod]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BBands, self).__init__(feed, instrument, params, entryTypeWhenAskPrediction)
		
	def strategies(self, bars):
		price = bars[self._instrument].getPrice()
		
		shares = self.getBroker().getShares(self._instrument)
		if shares == 0 and price < self._lower[-1]:
			shareToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, shareToBuy)
			self.placeStopOrder(self._instrument, price, -shareToBuy, True)
		elif shares > 0 and price > self._upper[-1]:
			self.marketOrder(self._instrument, -1*shares)	
			
class DoubleBottomBB(BaseBB): # params = [bands]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(DoubleBottomBB, self).__init__(feed, instrument, params, entryTypeWhenAskPrediction)
		
	def strategies(self, bars):
		shares = self.getBroker().getShares(self._instrument)
		price = bars[self._instrument].getPrice()
		diffVolum = getPercentDiff(self._volumes[-2], self._volumes[-1])
			
		if self._lows[-2] < self._lower[-2] and price < self._lower[-1] and diffVolum < -0.30 and shares == 0:
			sharesToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, sharesToBuy)
			self.placeStopOrder(self._instrument, price, -sharesToBuy, True)
		elif shares > 0 and price > self._upper[-1]:
			self.marketOrder(self._instrument, -shares)	

class ReversalBB(BaseBB): # params = [bands]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(ReversalBB, self).__init__(feed, instrument, params, entryTypeWhenAskPrediction)
		
	def strategies(self, bars):
		shares = self.getBroker().getShares(self._instrument)
		price = bars[self._instrument].getPrice()

		if self._lows[-2] > self._upper[-2] and getPercentDiff(self._lows[-2], self._prices[-2]) < 0.05 and shares == 0:
			sharesToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, -sharesToBuy)
			self.placeStopOrder(self._instrument, price, sharesToBuy, True)
		elif shares < 0 and self._prices[-1] < self._middle[-1]:
			self.marketOrder(self._instrument, -shares)	

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
			sharesToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, sharesToBuy)
			self.placeStopAndLimitOrder(self._instrument, self.__prices[-1], -sharesToBuy, True)
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