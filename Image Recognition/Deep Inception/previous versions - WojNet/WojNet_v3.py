from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

from tflearn.helpers.trainer import Trainer
from import_data import *

# Acquiring the data

folder = 'Digit Recognizer'
file_name = 'train.csv'
specific_dataset_source = folder + '/' + file_name
output_columns = ['label']

data = import_csv(specific_dataset_source, shuffle = True)
x_data, y_data = get_xy_mutual(data, output_columns, type = 'numpy')

x_data = standalization_divide(x_data, 255)
get_info(x_data, 'input')

num_samples = x_data.shape[0]
input_features = x_data.shape[1]

number_of_labels = labels_info(y_data)
y_data_as_numbers = labels_as_numbers(y_data)

split_percentage = 80
x_train, x_val = cross_validation(x_data, split_percentage)

y_train = np.array(y_data_as_numbers[0:(int(x_data.shape[0]/(100/split_percentage)))])
y_val = np.array(y_data_as_numbers[(int(x_data.shape[0]/(100/split_percentage))):x_data.shape[0]])

# Shaping data to the correct shape.
x_train = x_train.reshape([-1, 28, 28, 1])
x_val = x_val.reshape([-1, 28, 28, 1])
y_train = to_categorical(y_train, nb_classes = 10)
y_val = to_categorical(y_val, nb_classes = 10)


print('Size of the intput '+ str(x_data.shape))
print('Size of the output '+ str(y_data.shape))
print('First five examples of one-hot encoded output:')
print(y_train[:5, :])

# --------------------------------------------------------------------------------------------------------

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')

#
branch1 = conv_2d(network, 32, [2, 2], activation = 'relu')
branch1 = dropout(branch1, 0.5)

branch2 = conv_2d(network, 16, [3, 3], activation = 'relu')
branch2 = dropout(branch2, 0.5)
branch2 = conv_2d(branch2, 32, [2, 2], activation = 'relu')
branch2 = dropout(branch2, 0.5)

branch3 = conv_2d(network, 8, [5, 5], activation = 'relu')
branch3 = dropout(branch3, 0.5)
branch3 = conv_2d(branch3, 16, [3, 3], activation = 'relu')
branch3 = dropout(branch3, 0.5)
branch3 = conv_2d(branch3, 32, [2, 2], activation = 'relu')
branch3 = dropout(branch3, 0.5)

branch4 = conv_2d(network, 4, [7, 7], activation = 'relu')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 8, [5, 5], activation = 'relu')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 16, [3, 3], activation = 'relu')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 32, [2, 2], activation = 'relu')
branch4 = dropout(branch4, 0.5)

# Hidden layer 3
merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'elemwise_sum', name = 'Merge')

# Hidden layer 4
merged_layersk = fully_connected(merged_layers, 256, activation='relu')
merged_layers = dropout(merged_layers, 0.5)

merged_layers = fully_connected(merged_layers, 10, activation='softmax')
network = regression(merged_layers, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# ---------------------------------------------------------------------------------------


# Training
# model = tflearn.DNN(network, tensorboard_verbose = 3, tensorboard_dir='./logs')

model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir='./logs') # checkpoint_path = '/logs/model_vgg',

model.fit(x_train, y_train, n_epoch = 2, validation_set=(x_val, y_val),
          show_metric = True, batch_size = 200, shuffle = True, #snapshot_step = 100,
          snapshot_epoch = True, run_id = 'vgg_digits_woj')

model.save("WojNet.tfl")
#tensorboard --logdir=logs/

for index in range(0,2):
	predicted_as_prob = np.array(model.predict(x_data[index].reshape([1,28,28,1])))
	print('*' * 70)
	print(predicted_as_prob)
	print(predicted_as_prob.max())
	predicted_as_idx = np.where(predicted_as_prob == predicted_as_prob.max())[1]
	print(predicted_as_idx)


'''
for index in range(11,15):
    predicted_as_prob = np.array(model.predict(x_data[index].reshape([1,28,28,1])))
    print('*' * 70)
    print(type(predicted_as_prob))
    print(predicted_as_prob.shape)
    #print(predicted_as_prob)
    #print(predicted_as_prob.max())
    print(np.amax(predicted_as_prob))

'''
	
