import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

from tflearn.layers.merge_ops import merge
from tflearn.data_utils import to_categorical

from tflearn.data_preprocessing import ImagePreprocessing

def ANN(WIDTH, HEIGHT, CHANNELS, LABELS):
	dropout_value = 0.4
	filters = 10

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Building the network
	network = input_data(shape=[None, 28, 28, 1],
		data_preprocessing=img_prep,
		name='input')
	# Branch 1

	layer_7x7 = conv_2d(network, filters, [7, 7], activation = 'relu', name = 'Conv2d_7x7')
	#layer_7x7 = dropout(layer_7x7, dropout_value)

	layer_5x5 = conv_2d(layer_7x7, filters, [5, 5], activation = 'relu', name = 'Conv2d_5x5')
	#layer_5x5 = dropout(layer_5x5, dropout_value)

	sum_5x5 = merge((layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_5x5')

	layer_3x3 = conv_2d(sum_5x5, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3')
	#layer_3x3 = dropout(layer_3x3, dropout_value)

	sum_3x3 = merge((layer_3x3, layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_3x3')

	layer_2x2 = conv_2d(sum_3x3, filters, [2, 2], activation = 'relu', name = 'Conv2d_2x2')
	#layer_2x2 = dropout(layer_2x2, dropout_value)

	sum_2x2 = merge((layer_2x2, layer_3x3,layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_2x2')

	# Batch normalization
	#sum_2x2  = tflearn.batch_normalization(sum_2x2)
	
	# Fully connected 1
	fc1 = fully_connected(sum_2x2, 1000, activation='relu')
	fc1 = dropout(fc1, dropout_value)

	# Fully connected 2
	fc2 = fully_connected(fc1, 1000, activation='relu')
	fc2 = dropout(fc2, dropout_value)

	# Output layer
	final = fully_connected(fc2, 10, activation = 'softmax')
	
	network = regression(final, optimizer = 'adam', learning_rate = 0.0003,
	                     loss = 'categorical_crossentropy', name ='target')
	
	model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
	return model