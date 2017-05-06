import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import ImagePreprocessing

# Building the network
def ANN(WIDTH, HEIGHT, CHANNELS, LABELS):
	dropout_value = 0.40

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	network = input_data(shape=[None, WIDTH, HEIGHT, CHANNELS],
		data_preprocessing=img_prep,
		name='input')

	# Branch 1
	branch1 = conv_2d(network, 32, [2, 2], activation = 'relu', name = 'B1Conv2d_2x2')
	#branch1 = dropout(branch1, dropout_value)

	# Branch 2
	branch2 = conv_2d(network, 16, [3, 3], activation = 'relu', name = 'B2Conv2d_3x3')
	#branch2 = dropout(branch2, dropout_value)
	branch2 = conv_2d(branch2, 32, [2, 2], activation = 'relu', name = 'B2Conv2d_2x2')
	#branch2 = dropout(branch2, dropout_value)

	# Branch 3
	branch3 = conv_2d(network, 8, [5, 5], activation = 'relu', name = 'B3Conv2d_5x5')
	#branch3 = dropout(branch3, dropout_value)
	branch3 = conv_2d(branch3, 16, [3, 3], activation = 'relu', name = 'B3Conv2d_3x3')
	#branch3 = dropout(branch3, dropout_value)
	branch3 = conv_2d(branch3, 32, [2, 2], activation = 'relu', name = 'B3Conv2d_2x2')
	#branch3 = dropout(branch3, dropout_value)

	# Branch 4
	branch4 = conv_2d(network, 4, [7, 7], activation = 'relu', name = 'B4Conv2d_7x7')
	#branch4 = dropout(branch4, dropout_value)
	branch4 = conv_2d(branch4, 8, [5, 5], activation = 'relu', name = 'B4Conv2d_5x5')
	#branch4 = dropout(branch4, dropout_value)
	branch4 = conv_2d(branch4, 16, [3, 3], activation = 'relu', name = 'B4Conv2d_3x3')
	#branch4 = dropout(branch4, dropout_value)
	branch4 = conv_2d(branch4, 32, [2, 2], activation = 'relu', name = 'B4Conv2d_2x2')
	#branch4 = dropout(branch4, dropout_value)

	# Merging the branches
	merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'elemwise_sum', name = 'Merge')

	# Batch normalization
	merged_layers = tflearn.batch_normalization(merged_layers)
	
	# Fully connected 1
	merged_layers = fully_connected(merged_layers, 1000, activation='relu')
	merged_layers = dropout(merged_layers, dropout_value)

	# Fully connected 2
	merged_layers = fully_connected(merged_layers, 1000, activation='relu')
	merged_layers = dropout(merged_layers, dropout_value)

	# Output layer
	merged_layers = fully_connected(merged_layers, LABELS, activation = 'softmax')

	network = regression(merged_layers, optimizer = 'adam', learning_rate = 0.0005,
	                     loss = 'categorical_crossentropy', name ='target')

	model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
	return model