import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


# Building the network
def ANN():
	network = input_data(shape=[None, 28, 28, 1], name='input')

	network = conv_2d(network, 64, 3, activation = 'relu')
	network = conv_2d(network, 64, 3, activation = 'relu')
	network = max_pool_2d(network, 2, strides = 2)

	network = conv_2d(network, 128, 3, activation = 'relu')
	network = conv_2d(network, 128, 3, activation = 'relu')
	network = max_pool_2d(network, 2, strides = 2)

	network = conv_2d(network, 256, 3, activation = 'relu')
	network = conv_2d(network, 256, 3, activation = 'relu')
	network = conv_2d(network, 256, 3, activation = 'relu')
	network = max_pool_2d(network, 2, strides = 2)

	network = conv_2d(network, 512, 3, activation = 'relu')
	network = conv_2d(network, 512, 3, activation = 'relu')
	network = conv_2d(network, 512, 3, activation = 'relu')
	network = max_pool_2d(network, 2, strides = 2)

	network = conv_2d(network, 512, 3, activation = 'relu')
	network = conv_2d(network, 512, 3, activation = 'relu')
	network = conv_2d(network, 512, 3, activation = 'relu')
	network = max_pool_2d(network, 2, strides = 2)

	network = fully_connected(network, 4096, activation = 'relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation = 'relu')
	network = dropout(network, 0.5)

	# Output layer
	merged_layers = fully_connected(network, 10, activation = 'softmax')
	network = regression(merged_layers, optimizer = 'adam', learning_rate = 0.0005,
	                     loss = 'categorical_crossentropy', name ='target')

	model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
	return model