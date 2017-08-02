import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected, activation, flatten
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import ImagePreprocessing

# Building the network
def ANN(NAME, WIDTH, HEIGHT, CHANNELS, LABELS):
	

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	#img_prep.add_featurewise_zero_center()
	#img_prep.add_featurewise_stdnorm()

	network = input_data(shape=[None, WIDTH, HEIGHT, CHANNELS],
		data_preprocessing=img_prep,
		name='Input')
# ------------------------------------------------------------------------------------------
	if NAME == 'MLP-3':
		dropout_value = 0.5

		network = flatten(network) 
		network = fully_connected(network, 1000, activation='relu')

		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		# Output layer
		network = fully_connected(network, LABELS, activation = 'softmax')

		network = regression(network, optimizer = 'adam', learning_rate = 0.0005,
		                     loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	if NAME == 'MLP-4':
		dropout_value = 0.5

		network = flatten(network) 
		network = fully_connected(network, 1000, activation='relu')
		network = fully_connected(network, 1000, activation='relu')
		
		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		# Output layer
		network = fully_connected(network, LABELS, activation = 'softmax')

		network = regression(network, optimizer = 'adam', learning_rate = 0.0005,
		                     loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	if NAME == 'MLP-5':
		dropout_value = 0.5

		network = flatten(network) 
		network = fully_connected(network, 1000, activation='relu')
		network = fully_connected(network, 1000, activation='relu')
		network = fully_connected(network, 1000, activation='relu')

		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		
		network = fully_connected(network, 1000, activation='relu')
		network = dropout(network, dropout_value)

		# Output layer
		network = fully_connected(network, LABELS, activation = 'softmax')

		network = regression(network, optimizer = 'adam', learning_rate = 0.0005,
		                     loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'SimpleCNN':
		dropout_value = 0.5

		network = conv_2d(network, 10, [2, 2], activation = 'relu', name = 'B1Conv2d_2x2')

		# Branch 2
		network = conv_2d(network, 10, [2, 2], activation = 'relu', name = 'B2Conv2d_2x2')

		# Fully connected 1
		network = fully_connected(network, 100, activation='relu')
		network = dropout(network, dropout_value)

		# Fully connected 2
		network = fully_connected(network, 100, activation='relu')
		network = dropout(network, dropout_value)

		# Output layer
		network = fully_connected(network, LABELS, activation = 'softmax')

		network = regression(network, optimizer = 'adam', learning_rate = 0.0005,
		                     loss = 'categorical_crossentropy', name ='target')	
# ------------------------------------------------------------------------------------------
	elif NAME == 'PrimeInception-ES':

		dropout_value = 0.5
		
		# Branch 1
		branch1 = conv_2d(network, 32, [2, 2], activation = 'relu', name = 'B1Conv2d_2x2')
		branch1 = batch_normalization(branch1)

		# Branch 2
		branch2 = conv_2d(network, 16, [3, 3], activation = 'relu', name = 'B2Conv2d_3x3')
		branch2 = conv_2d(branch2, 32, [2, 2], activation = 'relu', name = 'B2Conv2d_2x2')
		branch2 = batch_normalization(branch2)

		# Branch 3
		branch3 = conv_2d(network, 8, [5, 5], activation = 'relu', name = 'B3Conv2d_5x5')
		branch3 = conv_2d(branch3, 16, [3, 3], activation = 'relu', name = 'B3Conv2d_3x3')
		branch3 = conv_2d(branch3, 32, [2, 2], activation = 'relu', name = 'B3Conv2d_2x2')
		branch3 = batch_normalization(branch3)

		# Branch 4
		branch4 = conv_2d(network, 4, [7, 7], activation = 'relu', name = 'B4Conv2d_7x7')
		branch4 = conv_2d(branch4, 8, [5, 5], activation = 'relu', name = 'B4Conv2d_5x5')
		branch4 = conv_2d(branch4, 16, [3, 3], activation = 'relu', name = 'B4Conv2d_3x3')
		branch4 = conv_2d(branch4, 32, [2, 2], activation = 'relu', name = 'B4Conv2d_2x2')
		branch4 = batch_normalization(branch4)

		# Merging the branches
		merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'elemwise_sum', name = 'ElemwiseSum')
		
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
# ------------------------------------------------------------------------------------------
	elif NAME == 'PrimeInception-C':

		dropout_value = 0.5
		
		# Branch 1
		branch1 = conv_2d(network, 32, [2, 2], activation = 'relu', name = 'B1Conv2d_2x2')
		branch1 = batch_normalization(branch1)

		# Branch 2
		branch2 = conv_2d(network, 16, [3, 3], activation = 'relu', name = 'B2Conv2d_3x3')
		branch2 = conv_2d(branch2, 32, [2, 2], activation = 'relu', name = 'B2Conv2d_2x2')
		branch2 = batch_normalization(branch2)

		# Branch 3
		branch3 = conv_2d(network, 8, [5, 5], activation = 'relu', name = 'B3Conv2d_5x5')
		branch3 = conv_2d(branch3, 16, [3, 3], activation = 'relu', name = 'B3Conv2d_3x3')
		branch3 = conv_2d(branch3, 32, [2, 2], activation = 'relu', name = 'B3Conv2d_2x2')
		branch3 = batch_normalization(branch3)

		# Branch 4
		branch4 = conv_2d(network, 4, [7, 7], activation = 'relu', name = 'B4Conv2d_7x7')
		branch4 = conv_2d(branch4, 8, [5, 5], activation = 'relu', name = 'B4Conv2d_5x5')
		branch4 = conv_2d(branch4, 16, [3, 3], activation = 'relu', name = 'B4Conv2d_3x3')
		branch4 = conv_2d(branch4, 32, [2, 2], activation = 'relu', name = 'B4Conv2d_2x2')
		branch4 = batch_normalization(branch4)

		# Merging the branches
		merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'concat', name = 'Concatenation')
	
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
# ------------------------------------------------------------------------------------------
	elif NAME == 'SaraNet-P-ES':
		dropout_value = 0.5
		filters = 15

		layer_7x7 = conv_2d(network, filters, [7, 7], activation = 'relu', name = 'Conv2d_7x7')

		layer_5x5 = conv_2d(layer_7x7, filters, [5, 5], activation = 'relu', name = 'Conv2d_5x5')
	
		sum_5x5 = merge((layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_5x5')

		layer_3x3 = conv_2d(sum_5x5, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3')
	
		sum_3x3 = merge((layer_3x3, layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_3x3')

		layer_2x2 = conv_2d(sum_3x3, filters, [2, 2], activation = 'relu', name = 'Conv2d_2x2')

		sum_2x2 = merge((layer_2x2, layer_3x3,layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_2x2')

		# Fully connected 1
		fc1 = fully_connected(sum_2x2, 1000, activation='relu')
		fc1 = dropout(fc1, dropout_value)

		# Fully connected 2
		fc2 = fully_connected(fc1, 1000, activation='relu')
		fc2 = dropout(fc2, dropout_value)

		# Output layer
		final = fully_connected(fc2, 10, activation = 'softmax')

		network = regression(final, optimizer = 'adam', learning_rate = 0.0005,
				             loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'SaraNet-P-C':
		dropout_value = 0.5
		filters = 15

		layer_7x7 = conv_2d(network, filters, [7, 7], activation = 'relu', name = 'Conv2d_7x7')

		layer_5x5 = conv_2d(layer_7x7, filters, [5, 5], activation = 'relu', name = 'Conv2d_5x5')
	
		sum_5x5 = merge((layer_5x5, layer_7x7), mode = 'concat', name = 'Sum_5x5')

		layer_3x3 = conv_2d(sum_5x5, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3')
	
		sum_3x3 = merge((layer_3x3, layer_5x5, layer_7x7), mode = 'concat', name = 'Sum_3x3')

		layer_2x2 = conv_2d(sum_3x3, filters, [2, 2], activation = 'relu', name = 'Conv2d_2x2')

		sum_2x2 = merge((layer_2x2, layer_3x3,layer_5x5, layer_7x7), mode = 'concat', name = 'Sum_2x2')

		# Fully connected 1
		fc1 = fully_connected(sum_2x2, 1000, activation='relu')
		fc1 = dropout(fc1, dropout_value)

		# Fully connected 2
		fc2 = fully_connected(fc1, 1000, activation='relu')
		fc2 = dropout(fc2, dropout_value)

		# Output layer
		final = fully_connected(fc2, 10, activation = 'softmax')

		network = regression(final, optimizer = 'adam', learning_rate = 0.0005,
				             loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'SaraNet-3x3-ES':
		dropout_value = 0.5
		filters = 15

		layer_7x7 = conv_2d(network, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_1')

		layer_5x5 = conv_2d(layer_7x7, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_2')
	
		sum_5x5 = merge((layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_1')

		layer_3x3 = conv_2d(sum_5x5, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_3')
	
		sum_3x3 = merge((layer_3x3, layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_2')

		layer_2x2 = conv_2d(sum_3x3, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_4')

		sum_2x2 = merge((layer_2x2, layer_3x3,layer_5x5, layer_7x7), mode = 'elemwise_sum', name = 'Sum_3')

		# Fully connected 1
		fc1 = fully_connected(sum_2x2, 1000, activation='relu')
		fc1 = dropout(fc1, dropout_value)

		# Fully connected 2
		fc2 = fully_connected(fc1, 1000, activation='relu')
		fc2 = dropout(fc2, dropout_value)

		# Output layer
		final = fully_connected(fc2, 10, activation = 'softmax')

		network = regression(final, optimizer = 'adam', learning_rate = 0.0005,
				             loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'SaraNet-3x3-C':
		dropout_value = 0.5
		filters = 15

		layer_7x7 = conv_2d(network, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_1')

		layer_5x5 = conv_2d(layer_7x7, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_2')
	
		sum_5x5 = merge((layer_5x5, layer_7x7), mode = 'concat', name = 'Concat_1')

		layer_3x3 = conv_2d(sum_5x5, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_3')
	
		sum_3x3 = merge((layer_3x3, layer_5x5, layer_7x7), mode = 'concat', name = 'Concat_2')

		layer_2x2 = conv_2d(sum_3x3, filters, [3, 3], activation = 'relu', name = 'Conv2d_3x3_4')

		sum_2x2 = merge((layer_2x2, layer_3x3,layer_5x5, layer_7x7), mode = 'concat', name = 'Concat_3')

		# Fully connected 1
		fc1 = fully_connected(sum_2x2, 1000, activation='relu')
		fc1 = dropout(fc1, dropout_value)

		# Fully connected 2
		fc2 = fully_connected(fc1, 1000, activation='relu')
		fc2 = dropout(fc2, dropout_value)

		# Output layer
		final = fully_connected(fc2, 10, activation = 'softmax')

		network = regression(final, optimizer = 'adam', learning_rate = 0.0005,
				             loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'VGG-16':

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
		merged_layers = fully_connected(network, LABELS, activation = 'softmax')
		network = regression(merged_layers, optimizer = 'adam', learning_rate = 0.0005,
		                     loss = 'categorical_crossentropy', name ='target')
# ------------------------------------------------------------------------------------------
	elif NAME == 'ResNet-18':
		# Residual blocks
		network = conv_2d(network, 16, 7, regularizer='L2', weight_decay=0.0001)
		network = max_pool_2d(network, 3, strides = 2)

		network = residual_block(network, 1, 64)
		network = residual_block(network, 1, 64, downsample = True)

		network = residual_block(network, 1, 128)
		network = residual_block(network, 1, 128, downsample = True)

		network = residual_block(network, 1, 256)
		network = residual_block(network, 1, 256, downsample = True)

		network = residual_block(network, 1, 512)
		network = residual_block(network, 1, 512, downsample = True)

		network = batch_normalization(network)
		network = activation(network, 'relu')
		network = global_avg_pool(network)

		# Output layer
		network = fully_connected(network, LABELS, activation = 'softmax')

		
		network = regression(network, optimizer = 'adam', learning_rate = 0.01, loss = 'categorical_crossentropy', name ='target')
		'''
		# Regression
		network = regression(network, optimizer = 'momentum',
		                         loss  = 'categorical_crossentropy',
		                         learning_rate = 0.1)

		'''
	# -----------------------------------------------
	# Returning the network and tensorboard settings.
	# -----------------------------------------------	
	model = tflearn.DNN(network, 
						tensorboard_verbose = 0, 
						tensorboard_dir = './logs/', 
						best_checkpoint_path = './checkpoints/best/'+ NAME + '/best_' + NAME + '-', 
						max_checkpoints = 1)
	return model
