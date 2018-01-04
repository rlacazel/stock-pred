import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt2
import tensorflow as tf
import os 
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

flags = tf.app.flags
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
FLAGS = flags.FLAGS

def fetch_data(stock_sym):
	filename = 'data/'+stock_sym+'_dailyadj.csv'
	if os.path.isfile(filename):
		df = pd.read_csv(filename)
	else:
		ts = TimeSeries(key='XNL4', output_format='pandas')
		df, meta_data = ts.get_daily_adjusted(symbol=stock_sym, outputsize='full')
		df.to_csv(filename)

	col_list = df.columns.tolist()
	col_list.remove('close')
	col_list.remove('dividend amount')
	col_list.remove('split coefficient') 
	col_list.remove('adjusted close') 
	col_list.append('close')
	prices = df['close']
	dates = df[col_list[1]]
	percent_change = [[dates[0], 0]] + [[dates[i], float(format((prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]*100, '.2f'))] for i in range(1,len(prices))]
	# print(percent_change)
	df = df[col_list[2:]]
	df['percent'] = [x[1] for x in percent_change]
	print(df.head())
	return df, prices, percent_change

def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape
    
    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
    return X_train, X_test
	
def preprocess_data(stock, seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix()

	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index : index + sequence_length])
		
	result = np.array(result)
	row = round(0.9 * result.shape[0])
	train = result[: int(row), :]

	train, result = standard_scaler(train, result)

	X_train = train[:, : -1]
	# print(X_train)
	y_train = train[:, -1][: ,-1]
	X_test = result[int(row) :, : -1]
	#print(X_test[:3])
	y_test = result[int(row) :, -1][ : ,-1]
	#print(y_test[:3])

	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

	return [X_train, y_train, X_test, y_test]
	
def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_shape=(None, layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model
	
df, prices, percent_changes = fetch_data(FLAGS.stock_symbol)
window = 20	
X_train, y_train, X_test, y_test = preprocess_data(df[::1], window)
# print(X_train[0:2])
# print(y_train[0:2])
# print("X_train", X_train.shape)
# print("y_train", y_train.shape)
# print("X_test", X_test.shape)
# print("y_test", y_test.shape)

#for i in range(len(y_train) + window + 1, len(y_train) + len(y_test) + window + 1):
# print(prices.iloc[i])
	
model = build_model([X_train.shape[2], window, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    verbose=1)
	
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

diff = []
ratio = []
pred = model.predict(X_test)
investment = 100
good_predications=0
for u in range(1,len(y_test)):
	# pr = pred[u][0]
	# ratio.append((y_test[u] / pr) - 1)
	# diff.append(abs(y_test[u] - pr))
	# predicted_percent = pred[u][0]*100/pred[u-1][0]-100
	predicted_percent = (pred[u][0]-pred[u-1][0])/pred[u-1][0]*100
	global_idx = len(y_train) + window + 1 + u
	if predicted_percent > 2: #buy
		investment += investment*(percent_changes[global_idx][1]/100)
	elif predicted_percent < 2: #sell
		investment -= investment*(percent_changes[global_idx][1]/100)
	good_predications += 1 if (predicted_percent * percent_changes[global_idx][1]) > 0 else 0

print(investment)
print('Predication ratio : ' + str(good_predications/float(len(y_test))))
	
plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_test, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()	