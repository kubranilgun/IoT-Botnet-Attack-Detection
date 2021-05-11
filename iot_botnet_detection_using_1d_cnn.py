# import libraries
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn import model_selection
import time

# define the parameter values
epochs, batch_size = 10, 64
print('Number of epochs: ', epochs)
print('Batch size: ', batch_size)

# load the dataset, returns train and test X and y elements
def load_dataset(datasetname):
	# load input data and output data
	print('\nLoading ' + datasetname + ' dataset...')
	dataX = loadtxt('Provision_PT_838/x_' + datasetname + '.csv', delimiter=',')
	datay = loadtxt('Provision_PT_838/y_' + datasetname + '.csv', delimiter=',')
	# split data into train data and test data
	trainX, testX = model_selection.train_test_split(dataX, test_size=0.10, random_state=42, shuffle=True)
	trainy, testy = model_selection.train_test_split(datay, test_size=0.10, random_state=42, shuffle=True)
	# get 3d training data
	#print('Preprocessing data...')
	#print('trainX.shape: ', trainX.shape)
	#print('testX.shape: ', testX.shape)
	listtrain = list()
	listtest = list()
	listtrain.append(trainX)
	listtest.append(testX)
	#print('len(listtrain): ', len(listtrain))
	#print('len(listtest): ', len(listtest))
	trainX = dstack(listtrain)
	testX = dstack(listtest)
	print('\nAfter stacking...')
	print('trainX.shape: ', trainX.shape)
	print('testX.shape: ', testX.shape)
	#trainX = dstack(trainX)
	#testX = dstack(testX)
	# convert output data to categorical form
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print('\nAfter categorizing...')
	print('trainy.shape: ', trainy.shape)
	print('testy.shape: ', testy.shape, '\n')
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	n_features, n_added_dimension, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_features, n_added_dimension)))
	#print('First Conv1D layer is done')
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
	#print('Second Conv1D layer is done')
	model.add(MaxPooling1D(pool_size=2))
	#print('MaxPooling1D layer is done')
	model.add(Flatten())
	#print('Flatten layer is done')
	model.add(Dense(50, activation='relu'))
	#print('First Dense layer is done')
	model.add(Dense(n_outputs, activation='softmax'))
	#print('Second dense layer is done')
	# model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print('Compilation is done')
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=0)
	#print('Fitting is done')
	# evaluate model
	redundant, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	#print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.2f%%' %m)
	print('Standard Deviation: %.2f' %s)

# run an experiment
def run_experiment(repeats, datasetname):
	experiment_start_time = time.time()
	# load data
	trainX, trainy, testX, testy = load_dataset(datasetname)
	print('Evauating for ' + datasetname + '...\n')
	# repeat experiment
	evaluation_start_time = time.time()
	scores = list()
	for r in range(repeats):
		print('Repeat: ', r+1)
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('Score: %.2f' %score, '\n')
		scores.append(score)
	# summarize results
	print('Results for ' + datasetname + ':')
	summarize_results(scores)
	print("Duration of loading data: %d seconds" %(evaluation_start_time - experiment_start_time))
	print("Duration of evaluating model: %d seconds" %(time.time() - evaluation_start_time))
	print("Duration of whole experiment: %d seconds" %(time.time() - experiment_start_time))

# run the experiment
run_experiment(5, 'bashlite_combo')
run_experiment(5, 'bashlite_junk')
run_experiment(5, 'bashlite_scan')
run_experiment(5, 'bashlite_tcp')
run_experiment(5, 'bashlite_udp')
run_experiment(5, 'mirai_ack')
run_experiment(5, 'mirai_scan')
run_experiment(5, 'mirai_syn')
run_experiment(5, 'mirai_udp')
run_experiment(5, 'mirai_udpplain')