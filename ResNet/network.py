import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn import residual_bottleneck, batch_normalization, global_avg_pool

# Building the network
def ANN():
	network = input_data(shape=[None, 28, 28, 1], name='input')

	network = conv_2d(network, 64, 3, activation='relu', bias=False)
	# Residual blocks
	network = tflearn.residual_bottleneck(network, 3, 16, 64)
	network = tflearn.residual_bottleneck(network, 1, 32, 128, downsample=True)
	network = tflearn.residual_bottleneck(network, 2, 32, 128)
	network = tflearn.residual_bottleneck(network, 1, 64, 256, downsample=True)
	network = tflearn.residual_bottleneck(network, 2, 64, 256)
	network = tflearn.batch_normalization(network)
	network = tflearn.activation(network, 'relu')
	network = tflearn.global_avg_pool(network)
	
	# Output layer
	network = fully_connected(network, 10, activation = 'softmax')

	'''
	network = regression(network, optimizer = 'adam', learning_rate = 0.01,
	                     loss = 'categorical_crossentropy', name ='target')
	'''
	# Regression
	network = tflearn.regression(network, optimizer = 'momentum',
	                         loss  = 'categorical_crossentropy',
	                         learning_rate = 0.1)

	model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
	return model