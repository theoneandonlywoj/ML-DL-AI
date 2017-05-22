
# coding: utf-8

# In[18]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

from tflearn.helpers.trainer import Trainer
from import_data import *

from tqdm import tqdm

# Acquiring the data

folder = 'Digit Recognizer'
file_name = 'train.csv'
specific_dataset_source = folder + '/' + file_name
output_columns = ['label']

data = import_csv(specific_dataset_source, shuffle = True)


# In[2]:

x_data = data
y_data = np.array(data.pop('label'))


# In[3]:

print('Shape of the input data:', x_data.shape)
print('Shape of the output data:', y_data.shape)


# In[4]:

x_data = x_data / 255

num_samples = x_data.shape[0]
input_features = x_data.shape[1]

print('Number of samples:', num_samples)
print('Number of the input features:', input_features)


# In[5]:

#number_of_labels = labels_info(y_data)
y_data_as_numbers = labels_as_numbers(y_data)


# In[6]:

split_percentage = 80
split_index = int(x_data.shape[0]/(100/split_percentage))


# In[7]:

x_train = np.array(x_data[:split_index])
x_val = np.array(x_data[split_index:])

y_train = np.array(y_data_as_numbers[:split_index])
y_val = np.array(y_data_as_numbers[split_index:])


# In[8]:

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[9]:

# Shaping data to the correct shape.
x_train = x_train.reshape([-1, 28, 28, 1])
x_val = x_val.reshape([-1, 28, 28, 1])
y_train = to_categorical(y_train, nb_classes = 10)
y_val = to_categorical(y_val, nb_classes = 10)


# In[10]:

# Building the network
network = input_data(shape=[None, 28, 28, 1], name='input')

# Branch 1
branch1 = conv_2d(network, 32, [2, 2], activation = 'relu', name = 'B1Conv2d_2x2')
branch1 = dropout(branch1, 0.5)

# Branch 2
branch2 = conv_2d(network, 16, [3, 3], activation = 'relu', name = 'B2Conv2d_3x3')
branch2 = dropout(branch2, 0.5)
branch2 = conv_2d(branch2, 32, [2, 2], activation = 'relu', name = 'B2Conv2d_2x2')
branch2 = dropout(branch2, 0.5)

# Branch 3
branch3 = conv_2d(network, 8, [5, 5], activation = 'relu', name = 'B3Conv2d_5x5')
branch3 = dropout(branch3, 0.5)
branch3 = conv_2d(branch3, 16, [3, 3], activation = 'relu', name = 'B3Conv2d_3x3')
branch3 = dropout(branch3, 0.5)
branch3 = conv_2d(branch3, 32, [2, 2], activation = 'relu', name = 'B3Conv2d_2x2')
branch3 = dropout(branch3, 0.5)

# Branch 4
branch4 = conv_2d(network, 4, [7, 7], activation = 'relu', name = 'B4Conv2d_7x7')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 8, [5, 5], activation = 'relu', name = 'B4Conv2d_5x5')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 16, [3, 3], activation = 'relu', name = 'B4Conv2d_3x3')
branch4 = dropout(branch4, 0.5)
branch4 = conv_2d(branch4, 32, [2, 2], activation = 'relu', name = 'B4Conv2d_2x2')
branch4 = dropout(branch4, 0.5)

# Merging the branches
merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'elemwise_sum', name = 'Merge')

# Fully connected 1
merged_layers = fully_connected(merged_layers, 10, activation='relu')
merged_layers = dropout(merged_layers, 0.5)
# Output layer
merged_layers = fully_connected(merged_layers, 10, activation = 'softmax')
network = regression(merged_layers, optimizer = 'adam', learning_rate = 0.003,
                     loss = 'categorical_crossentropy', name ='target')

# ---------------------------------------------------------------------------------------


# In[11]:

# Training
model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
# checkpoint_path ='./checkpoints/checkpoint',

preTrained = False

if preTrained == False:
	model.fit(x_train, y_train, n_epoch = 1, validation_set = (x_val, y_val), 
	show_metric = True, batch_size = 200, shuffle = True, #snapshot_step = 100,
          snapshot_epoch = True, run_id = 'WojNet')
else:
	# Loading the best accuracy checkpoint (accuracy over the validation data)
	model.load('./checkpoints/best/best_val9723')

	print('*' * 70)
	print('Model is successfully loaded for the best performance!')


# In[12]:

print('Evaluation in progress...')
print('Training data:', model.evaluate(x_train, y_train))
print('Validation data:',model.evaluate(x_val, y_val))


# In[13]:

file_name_test = 'test.csv'
folder = 'Digit Recognizer'

source = folder + '/' + file_name_test
data = pd.read_csv(source)


# In[14]:

test_input = data.loc[:, :]


# In[15]:

test_input_numpy = test_input.as_matrix()


# In[16]:

# Standalization
test_input_standarized = test_input_numpy / 255


# In[20]:

test_data_predicted = np.empty((0, 10))
test_data_predicted_label = np.empty((0, 10))
print('Prediction in progress...')
for i in range(0, test_input_numpy.shape[0]):
    current_example = test_input_standarized[i].reshape([-1,28,28,1])
    test_data_predicted = np.append(test_data_predicted, model.predict(current_example), axis = 0)
    test_data_predicted_label = np.append(test_data_predicted_label, model.predict_label(current_example), axis = 0)

print('The test data has been successfully labeled.')


# In[ ]:

index = np.arange(1, test_data_predicted.shape[0] + 1, 1)


# In[ ]:

test_data_predicted = test_data_predicted_label[:, -1]


# In[ ]:

col = ['ImageId', 'Label']
output_data = np.stack((index, test_data_predicted))
output_data = output_data.T
output_data = output_data.astype(int)
test_data_prediction = pd.DataFrame(output_data, columns=col)

predict_output = 'labels.csv'
predicted_output_path= folder + '/' + predict_output

test_data_prediction.to_csv(predicted_output_path, sep = ',', index = False)

print('The test data CSV file has been successfully uploaded!')

#tensorboard --logdir=logs/	


# In[ ]:



