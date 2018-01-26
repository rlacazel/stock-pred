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
		self._signals = dict()
		self._onGoingAcceptedOrder = dict()
		# Exit and stop loss orders.
		self._takeProfitOrder = None
		self._stopLossOrder = None
		
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
		# print(datePositionSubmitted)
		# clear the order book of previous date if new are accepted or all orders if one is activated
		# if position.isAccepted() and datePositionSubmitted is not None:
			# for order in self.getBroker().getActiveOrders(self._instrument):
				# if order.getSubmitDateTime().date() < datePositionSubmitted:
					# self._onGoingAcceptedOrder[order.getId()] = str(datePositionSubmitted)
					# self.getBroker().cancelOrder(order)
		# elif position.isFilled():
			# for order in self.getBroker().getActiveOrders(self._instrument):
				# if order.getSubmitDateTime().date() < curr_date:
					# self._onGoingAcceptedOrder[order.getId()] = str(datePositionSubmitted)
					# self.getBroker().cancelOrder(order)
				
		# if position.isAccepted():			
			# self._onGoingAcceptedOrder[position.getId()] = str(datePositionSubmitted) + ' -> ' + getStrOrder(position.getAction(), shares)
			# self.debug('Accepted  [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
		# elif position.isFilled() and position.getId() in self._onGoingAcceptedOrder:
			# self.debug('Activated [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
			# self._actions[datePositionSubmitted] = 1 if shares > 0 else -1 if shares < 0 else 0
		# elif position.isCanceled() and position.getId() in self._onGoingAcceptedOrder:
			# self.debug('Cancelled [' + str(position.getId()) + ']: ' + self._onGoingAcceptedOrder[position.getId()])
			
		if order.isAccepted():			
			self._onGoingAcceptedOrder[order.getId()] = str(datePositionSubmitted) + ' -> ' + getStrOrder(order.getAction(), shares)
			# self.debug('Accepted [' + str(order.getId()) + ']: ' + self._onGoingAcceptedOrder[order.getId()] + ' [shares=' + str(shares) + ']' )
		if order.isFilled():
			# Was the take profit order filled ?
			if self._takeProfitOrder is not None and order.getId() == self._takeProfitOrder.getId():
				entryPrice = order.getExecutionInfo().getPrice()
				self.debug(str(self.getFeed().getCurrentBars().getDateTime()) + "Take profit order filled at" + str(entryPrice))
				self._takeProfitOrder = None
				# self._signals[curr_date] = 0
				# Cancel the other exit order to avoid entering a short position.
				self.getBroker().cancelOrder(self._stopLossOrder)
			# Was the stop loss order filled ?
			elif self._stopLossOrder is not None and order.getId() == self._stopLossOrder.getId():
				entryPrice = order.getExecutionInfo().getPrice()
				self.debug(str(self.getFeed().getCurrentBars().getDateTime()) + "Stop loss order filled at" + str(entryPrice))
				self._stopLossOrder = None
				# self._signals[curr_date] = 0
				# Cancel the other exit order to avoid entering a short position.
				self.getBroker().cancelOrder(self._takeProfitOrder)
			else:
				self.debug('Activated [' + str(order.getId()) + ']: ' + self._onGoingAcceptedOrder[order.getId()] + ' [shares=' + str(shares) + ']' )
				self._signals[datePositionSubmitted] = 1 if shares > 0 else -1 if shares < 0 else 0
			# print(datePositionSubmitted)
			# print(self._actions[datePositionSubmitted])
			# else: # It is the buy order that got filled.
				# entryPrice = order.getExecutionInfo().getPrice()
				# shares = order.getExecutionInfo().getQuantity()
				# print self.getFeed().getCurrentBars().getDateTime(), "Buy order filled at", entryPrice
				# Submit take-profit and stop-loss orders.
				# In the next version I'll provide a shortcut for this similar to self.order(...) for market orders.
				# takeProfitPrice = entryPrice * 1.01
				# self.__takeProfitOrder = self.getBroker().createLimitOrder(broker.Order.Action.SELL, self.__instrument, takeProfitPrice, shares)
				# self.__takeProfitOrder.setGoodTillCanceled(True)
				# self.getBroker().placeOrder(self.__takeProfitOrder)
				# stopLossPrice = entryPrice * 0.95
				# 7 = self.getBroker().createStopOrder(broker.Order.Action.SELL, self.__instrument, stopLossPrice, shares)
				# self.__stopLossOrder.setGoodTillCanceled(True)
				# self.getBroker().placeOrder(self.__stopLossOrder)
				# print "Take-profit set at", takeProfitPrice
				# print "Stop-loss set at", stopLossPrice

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
	
	def placeStopOrder(self, instr, stopPrice, share, goodTillCancelled):
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			return self.stopOrder(instr, stopPrice, share, goodTillCancelled)
	
	def placeLimitOrder(self, instr, limitPrice, share, goodTillCancelled):
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			return self.limitOrder(instr, limitPrice, share, goodTillCancelled)
	
	def placeStopLimitOrder(self, instr, stopPrice, limitPrice, share, goodTillCancelled):
		if self.entryTypeWhenAskPrediction == PredictionFromType.NoPred:
			self.stopLimitOrder(instr, stopPrice, limitPrice, share, goodTillCancelled)
	
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

class BBands(BaseStrategy): # params = [bBandsPeriod]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(BBands, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.__bbands = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), params[0], 2)

	def getIndicators(self):
		return [['upper', self.__bbands.getUpperBand()], ['middle', self.__bbands.getMiddleBand()], ['lower', self.__bbands.getLowerBand()]]
		
	def availableData(self):
		return self.__bbands.getUpperBand()[-1] is not None
		
	def strategies(self, bars):
		lower = self.__bbands.getLowerBand()[-1]
		upper = self.__bbands.getUpperBand()[-1]
		price = bars[self._instrument].getPrice()
		
		shares = self.getBroker().getShares(self._instrument)
		if shares == 0 and price < lower:
			shareToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, shareToBuy)
			self._stopLossOrder = self.placeStopOrder(self._instrument, price-2, -shareToBuy, True)
		elif shares > 0 and price > upper:
			self.marketOrder(self._instrument, -1*shares)	
			
class DoubleBottomBB(BaseStrategy): # params = [vwapWindowSize, threshold]
	def __init__(self, feed, instrument, params, entryTypeWhenAskPrediction = PredictionFromType.NoPred):
		super(DoubleBottomBB, self).__init__(feed, instrument, entryTypeWhenAskPrediction)
		self.__bbands = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), params[0], 2)
		self.__lows = feed[instrument].getLowDataSeries()
		self.__volumes = feed[instrument].getVolumeDataSeries()
		self._previousLowBelowBand = False

	def getIndicators(self):
		return [['upper', self.__bbands.getUpperBand()], ['lower', self.__bbands.getLowerBand()], ['low', self.__lows]]
		
	def availableData(self):
		return self.__bbands.getUpperBand()[-1] is not None
		
	def strategies(self, bars):
		# print(str(bars.getDateTime().date()))
		previous_bar = self.getFeed().getDataSeries(self._instrument)[-2]
		shares = self.getBroker().getShares(self._instrument)
		price = bars[self._instrument].getPrice()
		lower = self.__bbands.getLowerBand()[-1]
		upper = self.__bbands.getUpperBand()[-1]
		
		diffVolum = (self.__volumes[-1] - self.__volumes[-2]) / self.__volumes[-2]
		# if (diffVolum < -0.25):
			# print(self.__volumes[-2] )
			# print(self.__volumes[-1] )
			# print(str(bars.getDateTime().date()) + '->' + str(diffVolum))
			
		if self._previousLowBelowBand and price < lower and diffVolum < -0.25 and shares == 0:
			sharesToBuy = self.getNbSharesToTake(bars)
			self.marketOrder(self._instrument, sharesToBuy)
			self._takeProfitOrder = self.placeLimitOrder(self._instrument, price+5, -sharesToBuy, True)
			self._stopLossOrder = self.placeStopOrder(self._instrument, price-5, -sharesToBuy, True)
			# self.placeStopLimitOrder(self._instrument, price-1, price+1, -sharesToBuy, True)
		# elif shares > 0 and price > upper+5:
			# self.marketOrder(self._instrument, -shares)	
			
		self._previousLowBelowBand = self.__lows[-2] < lower
		
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