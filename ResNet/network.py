import tflearn
from tflearn.layers.core import input_data, activation, fully_connected
from tflearn.layers.conv import conv_2d, residual_bottleneck, global_avg_pool
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing

# Building the network
def ANN(WIDTH, HEIGHT, CHANNELS, LABELS):
	dropout_value = 0.35

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Building the network
	network = input_data(shape=[None, WIDTH, HEIGHT, CHANNELS],
		data_preprocessing=img_prep,
		name='input')

	network = conv_2d(network, 64, 3, activation='relu', bias=False)
	# Residual blocks
	network = residual_bottleneck(network, 3, 16, 64)
	network = residual_bottleneck(network, 1, 32, 128, downsample=True)
	network = residual_bottleneck(network, 2, 32, 128)
	network = residual_bottleneck(network, 1, 64, 256, downsample=True)
	network = residual_bottleneck(network, 2, 64, 256)
	network = batch_normalization(network)
	network = activation(network, 'relu')
	network = global_avg_pool(network)
	
	# Output layer
	network = fully_connected(network, LABELS, activation = 'softmax')

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