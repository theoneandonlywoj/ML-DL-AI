import tflearn
import numpy as np
from tqdm import tqdm

from tflearn.layers.core import input_data, activation, fully_connected
from tflearn.layers.conv import conv_2d, residual_bottleneck, global_avg_pool, residual_block
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing


# Building the network ResNet-18
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
	# Residual blocks'
	
	network = residual_bottleneck(network, 3, 16, 64)
	network = residual_bottleneck(network, 1, 32, 128, downsample=True)
	network = residual_bottleneck(network, 2, 32, 128)
	network = residual_bottleneck(network, 1, 64, 256, downsample=True)
	network = residual_bottleneck(network, 2, 64, 256)
	network = residual_bottleneck(network, 1, 128, 512, downsample=True)
	network = residual_bottleneck(network, 2, 128, 512)
	
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
	model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', 
		best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
	return model

def big_dataset_prediction(model, DATA = []):
    # Predicting
    test_data_predicted = np.empty((0, 10))
    test_data_predicted_label = np.empty((0, 10))
    print('Prediction in progress...')
    for i in tqdm(range(0, DATA.shape[0])):
        current_example = DATA[i].reshape([-1,28,28,1])
        test_data_predicted = np.append(test_data_predicted, model.predict(current_example), axis = 0)
        test_data_predicted_label = np.append(test_data_predicted_label, model.predict_label(current_example), axis = 0)

    print('The test data has been successfully labeled.')
    print('*' * 70)
    return test_data_predicted_label
