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
from itertools import chain
from collections import defaultdict

# TUTO : http://gbeced.github.io/pyalgotrade/docs/v0.18/html/tutorial.html
list_symbols = ['ABT', 'ABBV', 'ACN', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']
good_perf = dict()

def fetch_data(stock_sym):

	filename = 'data/'+stock_sym+'.csv'
	
	# if os.path.isfile(filename):
		# df = pd.read_csv(filename)
	# else:
	if not os.path.isfile(filename):
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
		df = df['2015-01-01':'2018-10-01']
		
		if len(df.index) < 500:
			raise ValueError('Not enough ' + stock_sym + ' data : ' + str(len(df.index)) + '\n')
			
		df.to_csv(filename)
	return filename

def run(strategy, symbol, plot = False):

	strategy.setDebugMode(False)
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
		plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
		

	# Run the strategy.
	strategy.run()
	percent_return = (strategy.getResult() - initial_capital) / initial_capital * 100
	perf = percent_return-strategy.getChangePrice()
	if perf > 20 and percent_return > 0:
		if symbol not in good_perf or (symbol in good_perf and good_perf[symbol][1] < perf):
			good_perf[symbol] = [strategy.__class__.__name__, perf]
		
	strategy.info("Final %4s value: $%.2f (return: %.2f%% / p : %.2f%%) [perf: %.2f%%]" % (symbol, strategy.getResult(), percent_return, strategy.getChangePrice(), perf)) # daily return : percent_return / strategy._count)

	if plot: 
		plt.plot()

# feed = yahoofeed.Feed()
# file = fetch_data('ABT')
# feed.addBarsFromCSV("data", file)
	
# bb = strategies.BBands(feed, "data", [40]) # , True, True
# run(bb, 'data', True)
# sma.run()
		
# feed = yahoofeed.Feed()
# file = fetch_data('AAPL')
# feed.addBarsFromCSV("data", file)
	
# sma = strategies.SMACrossOver(feed, "data", 20, False, strategies.PredictionFromType.Nothing) # , True, True
# run(sma, 'data', True)
# sma.run()
	
# feed = yahoofeed.Feed()
# feed.addBarsFromCSV("data", file)

# rsi2 = strategies.RSI2(feed, "data", 154, 5, 2, 91, 18)
# run(rsi2, 'data', True)
# rsi2.run()

# d1 = sma._actions.set_index('date')
# d2 = rsi2._actions.set_index('date')
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
	# return strat._actions.set_index('date')

# exit()
	
count = 0
for s in list_symbols:
	count += 1
	if count % 10 == 0:
		print(str(good_perf) + '\n')
	try:
		file = fetch_data(s)
	except ValueError as err:
		print(err)
		continue
	except:
		print("Failed to fetch : %s\n" % s)
		continue
	
	try:
		feed = yahoofeed.Feed()
		feed.addBarsFromCSV(s, file)
		sma = strategies.SMACrossOver(feed, s, [20]) # , True, True
		feed = yahoofeed.Feed()
		feed.addBarsFromCSV(s, file)
		rsi2 = strategies.RSI2(feed, s, [154, 5, 2, 91, 18]) # , True, True		
		feed = yahoofeed.Feed()
		feed.addBarsFromCSV(s, file)
		bb = strategies.BBands(feed, s, [40]) # , True, True
		
		run(sma, s)
		run(rsi2, s)
		run(bb, s)
		
		actions = dict()
		all_actions = [sma._actions, rsi2._actions, bb._actions]
		for key in list(chain(*[a.keys() for a in all_actions])):
			actions[key] = []
			for act in all_actions:
				if key in act:
					actions[key].append(act[key])
		
		feed = yahoofeed.Feed()
		feed.addBarsFromCSV(s, file)
		all = strategies.AllStrat(feed, s, actions)
		run(all, s)
		print()
	except Exception as ex:
		template = "An exception of type {0} occurred. Arguments: {1!r} \n"
		message = template.format(type(ex).__name__, ex.args)
		print(message)