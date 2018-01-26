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
import datetime
from itertools import chain, combinations
from collections import defaultdict
import collections
import traceback

default_from_date  = '2015-01-01' 
default_to_date  = '2019-01-01'
scores = {'GNW': ['SMACrossOver_RSI2', 1.13], 'XEC': ['RSI2', 43.99], 'NAVI': ['SMACrossOver_BBands', 62.45], 'QRVO': ['SMACrossOver', 36.38], 'TWX': ['SMACrossOver', 50.4], 'DUK': ['SMACrossOver', 21.2], 'TRIP': ['RSI2', 29.1], 'HES': ['BBands_RSI2', 0.87], 'RLC': ['RSI2', 0.46], 'BBBY': ['SMACrossOver_RSI2', 0.88], 'FE': ['BBands', 28.64], 'NBL': ['RSI2', 31.29], 'GILD': ['RSI2', 1.81], 'WHR': ['BBands', 2.47], 'TGNA': ['BBands', 29.6], 'FLS': ['SMACrossOver_BBands_RSI2', 2.69], 'UNP': ['BBands', 45.73], 'APA': ['RSI2', 45.76], 'SLG': ['SMACrossOver_BBands', 38.25], 'WU': ['BBands', 38.96], 'SEE': ['RSI2', 14.7], 'STX': ['RSI2', 79.44], 'CTL': ['SMACrossOver_BBands', 3.42], 'CA': ['BBands', 40.73], 'MAC': ['BBands', 15.98], 'DISCK': ['SMACrossOver_BBands', 35.55], 'MRO': ['SMACrossOver_BBands', 51.88], 'ALTR': ['BBands', 49.28], 'RL': ['BBands', 21.58], 'CMG': ['BBands', 13.33], 'SCG': ['BBands', 15.66], 'JOY': ['SMACrossOver', 42.83], 'NE': ['SMACrossOver_BBands', 89.28], 'JNPR': ['BBands', 69.94], 'DO': ['SMACrossOver_BBands', 36.68], 'FTR': ['RSI2', 56.13], 'GGP': ['BBands', 45.52], 'CNX': ['BBands', 23.46], 'PXD': ['RSI2', 85.5], 'FOSL': ['RSI2', 18.68], 'R': ['RSI2', 40.73], 'AN': ['BBands', 23.13], 'HCP': ['SMACrossOver_BBands', 16.16], 'ORLY': ['RSI2', 36.3], 'MUR': ['SMACrossOver_BBands_RSI2', 20.69], 'BXP': ['SMACrossOver_BBands', 20.68], 'NWL': ['RSI2', 25.42], 'RRC': ['RSI2', 75.16], 'EFX': ['BBands', 73.05], 'PCG': ['RSI2', 11.18], 'HRB': ['BBands', 32.99], 'APC': ['RSI2', 41.54], 'IPG': ['BBands', 33.4], 'JWN': ['SMACrossOver_RSI2', 2.55], 'MOS': ['RSI2', 32.03], 'PRGO': ['RSI2', 38.7], 'URBN': ['SMACrossOver', 56.64], 'AKAM': ['RSI2', 9.8], 'HBI': ['BBands', 35.21], 'SWN': ['RSI2', 100.16], 'BHI': ['BBands', 90.42], 'SPLS': ['SMACrossOver_BBands', 20.41], 'NOV': ['SMACrossOver_RSI2', 15.03], 'AVB': ['BBands', 40.73], 'GE': ['RSI2', 27.16, strategies.PredictionFromType.Long], 'WMB': ['BBands', 34.71], 'KMI': ['RSI2', 50.07], 'ADS': ['BBands', 47.68], 'VNO': ['SMACrossOver_BBands', 25.84], 'THC': ['BBands_RSI2', 12.95], 'DVN': ['RSI2', 41.24], 'WDC': ['BBands_RSI2', 12.08], 'HOG': ['RSI2', 29.71], 'EQR': ['BBands', 40.03], 'FCX': ['SMACrossOver', 161.3], 'LVLT': ['BBands', 21.54], 'AGN': ['SMACrossOver', 7.5], 'PDCO': ['BBands', 13.9], 'WFM': ['SMACrossOver_BBands', 19.41], 'PBI': ['BBands', 10.82], 'MAT': ['BBands_RSI2', 12.48], 'AAP': ['SMACrossOver_RSI2', 7.2], 'GPS': ['SMACrossOver', 56.55], 'NLSN': ['BBands_RSI2', 11.39], 'SE': ['SMACrossOver', 10.6], 'DISCA': ['SMACrossOver_BBands', 16.28], 'BEN': ['BBands_RSI2', 6.45], 'WBA': ['BBands', 31.97], 'VIAB': ['SMACrossOver_BBands', 8.69], 'MNK': ['RSI2', 57.93], 'HCN': ['SMACrossOver_BBands', 33.05], 'CF': ['RSI2', 28.15], 'CHK': ['SMACrossOver_BBands', 135.03], 'MCK': ['BBands', 13.49], 'KORS': ['BBands', 42.25], 'SLB': ['RSI2', 28.9], 'KIM': ['BBands_RSI2', 7.45], 'ESV': ['RSI2', 31.8], 'EQT': ['SMACrossOver_BBands', 19.78], 'SRCL': ['SMACrossOver_BBands', 15.79], 'OI': ['BBands_RSI2', 34.41], 'CELG': ['BBands', 46.16], 'ENDP': ['RSI2', 105.25], 'MYL': ['RSI2', 15.04], 'RIG': ['SMACrossOver', 50.39], 'NRG': ['BBands_RSI2', 25.81], 'REGN': ['SMACrossOver_BBands', 13.91], 'NTAP': ['SMACrossOver', 113.21]}
failed = ['ADT', 'GAS', 'ARG', 'AA', 'BXLT', 'BSK', 'BRCM', 'CVC', 'CAM', 'DAL', 'DTV', 'EMC', 'FSIV', 'HAR', 'HSP', 'HCBK', 'GMCR', 'KRFT', 'LLTC', 'MHFI', 'PLL', 'POM', 'PSX', 'PCL', 'PCP', 'RLC', 'SNDK', 'SIAL', 'HOT', 'TE', 'TSO', 'TWC', 'TJK', 'TYC', 'UA']

# TUTO : http://gbeced.github.io/pyalgotrade/docs/v0.18/html/tutorial.html
list_symbols = ['ABT', 'ABBV', 'ACN', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']
good_perf = dict()

def fetch_data(stock_sym, override_file = False, from_date = default_from_date, to_date = default_to_date):

	filename = 'data/'+stock_sym+'.csv'
	
	# if os.path.isfile(filename):
		# df = pd.read_csv(filename)
	# else:
	if not os.path.isfile(filename) or override_file:
		ts = TimeSeries(key='XNL4', output_format='pandas')
		df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')
		# df, meta_data = ts.get_batch_stock_quotes(symbols=[stock_sym,'MSFT'])
			
		col_list = df.columns.tolist()
		col_list.remove('7. dividend amount') 
		col_list.remove('8. split coefficient') 
		# print(col_list)
		df = df[col_list]
		df = df.reset_index()
		df = df.rename(columns={'date': "Date", '1. open': "Open", '2. high': "High", '3. low': "Low", '4. close': "Close", '6. volume': "Volume", '5. adjusted close': "Adj Close"})
		df = df.set_index('Date')
		df = df[from_date:to_date]
		
		if len(df.index) < 500:
			raise ValueError('Not enough ' + stock_sym + ' data : ' + str(len(df.index)) + '\n')
			
		df.to_csv(filename)
	return filename

def run(strategy, symbol, plot = False):

	strategy.setDebugMode(plot)
	initial_capital = strategy.getResult()
	
	if plot: 
		# Attach a returns analyzers to the strategy.
		returnsAnalyzer = returns.Returns()
		strategy.attachAnalyzer(returnsAnalyzer)

		# Attach the plotter to the strategy.
		# plt = plotter.StrategyPlotter(strategy)
		plt = plotter.StrategyPlotter(strategy, True, True, True)
		for ind in strategy.getIndicators():
			plt.getInstrumentSubplot(symbol).addDataSeries(*ind)
			
		# Plot the simple returns on each bar.
		# plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
		

	# Run the strategy.
	strategy.run()
	percent_return = (strategy.getResult() - initial_capital) / initial_capital * 100
	perf = percent_return-strategy.getChangePrice()
	if perf > 20 and percent_return > 0:
		if symbol not in good_perf or (symbol in good_perf and good_perf[symbol][1] < percent_return):
			good_perf[symbol] = [strategy.getName(), round(percent_return,2)]
		
	strategy.info("%s %4s value: $%.2f (return: %.2f%% / p : %.2f%%) [perf: %.2f%%]" % (strategy.getName(), symbol, strategy.getResult(), percent_return, strategy.getChangePrice(), perf)) # daily return : percent_return / strategy._count)

	if plot: 
		plt.plot()


# ordered = collections.OrderedDict(sorted(bb._signals.items(), key=lambda t: t[0]))
# for k,v in ordered.items():
	# print(str(k) + ': ' + str(v))
			
# sma.run()
		
# feed = yahoofeed.Feed()
# file = fetch_data('MO')
# feed.addBarsFromCSV("data", file)
	
# sma = strategies.SMACrossOver(feed, "data", [20]) # , True, True
# run(sma, 'data', True)
# sma.run()
	
# feed = yahoofeed.Feed()
# feed.addBarsFromCSV("data", file)

# rsi2 = strategies.RSI2(feed, "data", 154, 5, 2, 91, 18)
# run(rsi2, 'data', True)
# rsi2.run()

# d1 = sma._signals.set_index('date')
# d2 = rsi2._signals.set_index('date')
# d = pd.concat([d1, d2], axis=1)
# print(d2.to_string())

# feed = yahoofeed.Feed()
# feed.addBarsFromCSV("data", file)
# dual = strategies.SMAandRSI2(feed, "data", d)
# run(dual, 'data', False)
# dual.run()

# def execStrtategy():
	# feed = yahoofeed.Feed()
	# feed.addBarsFromCSV(s, file)
	# strat = strategies.SMACrossOver(feed, s, 20) # , True, True
	# return strat._signals.set_index('date')

def predict_all():
	for score in scores.keys():
		if len(scores[score]) > 2:
			predict(scores[score][0], score, scores[score][2])
		else:
			predict(scores[score][0], score)

def predict(strats_name, stock, currentAction = strategies.PredictionFromType.Nothing):
	print('Prediction %s [%s]' % (stock, strats_name))
	names = strats_name.split('_')
	all_strats = []
	weights = dict()
	for strat_name in names:
		strat = getStrats(strat_name, stock, True)
		if strats_name == strat_name:
			strat.setPredictionMode(currentAction)
			strat.run()
			return
		else:
			all_strats.append(strat)
			weights[strat_name] = 1
	
	weights = dict()
	for name in names:
		weights[name] = 1
	runCombinedStrat(all_strats, weights, stock, True)

#SMACrossOver_RSI2
def run_symbol(strats_name, stock):
	names = strats_name.split('_')
	all_strats = []
	weights = dict()
	for strat_name in names:
		strat = getStrats(strat_name, stock)
		# strat.setPredictionMode(strategies.PredictionFromType.Nothing)
		# strat.run()
		run(strat, stock)
		all_strats.append(strat)
		weights[strat_name] = 1
	
	for i in range(2,len(names)+1):
		combi = list(combinations(names, i))
		for c in combi:
			weights = dict()
			for strat_n in c:
				weights[strat_n] = 1
			runCombinedStrat(all_strats, weights, stock)
		
def getStrats(strat_name, stock, override_file = False, from_date = default_from_date, to_date = default_to_date):
	feed = yahoofeed.Feed()
	file = fetch_data(stock, override_file, from_date, to_date)
	feed.addBarsFromCSV(stock, file)
	if strat_name == 'SMACrossOver':
		return strategies.SMACrossOver(feed, stock, [20])
	elif strat_name == 'RSI2':
		return strategies.RSI2(feed, stock, [154, 5, 2, 91, 18])
	elif strat_name == 'BBands':
		return strategies.BBands(feed, stock, [20])
	elif strat_name == 'DoubleBottomBB':
		return strategies.DoubleBottomBB(feed, stock, [20])
		
def runCombinedStrat(strats, weights, stock, predict_mode = False):
	orders = dict()
	for key in list(chain(*[s._signals.keys() for s in strats])):
		orders[key] = dict()
		for strat in strats:
			if key in strat._signals:
				orders[key][strat.__class__.__name__] = strat._signals[key]
					
	# ordered = collections.OrderedDict(sorted(orders.items(), key=lambda t: t[0]))
	# for k,v in ordered.items():
		# print(str(k) + ': ' + str(v))
			
	feed = yahoofeed.Feed()
	file = fetch_data(stock)
	feed.addBarsFromCSV(stock, file)
	all = strategies.AllStrat(feed, stock, orders, weights)
	if predict_mode:
		all.setPredictionMode(strategies.PredictionFromType.Nothing)
		all.run()
	else:
		run(all, stock)

# for s in ['AKAM']: # ['AAPL']
for s in list_symbols: 
	if s in failed: continue
	strat = getStrats('DoubleBottomBB', s)
	run(strat, s, True)
	# ordered = collections.OrderedDict(sorted(strat._signals.items(), key=lambda t: t[0]))
	# for k,v in ordered.items():
		# print(str(k) + ': ' + str(v))
	
# predict_all()
exit()

count = 0
failed_symbols = []
for s in list_symbols: #list_symbols
	if s in failed: continue
	count += 1
	if count % 10 == 0:
		print(good_perf)
		print('Failed symbols : ' + str(failed_symbols) + '\n')
	try:
		file = fetch_data(s)
	except ValueError as err:
		print(err)
		failed_symbols.append(s)
		continue
	except:
		print("Failed to fetch : %s\n" % s)
		failed_symbols.append(s)
		continue
	
	try:
		# predict('SMACrossOver', s)
		run_symbol('SMACrossOver_RSI2_BBands', s)
		print()
	except Exception as ex:
		template = "An exception of type {0} occurred. Arguments: {1!r} \n"
		message = template.format(type(ex).__name__, ex.args)
		print(message)
		failed_symbols.append(s)
		traceback.print_exc()