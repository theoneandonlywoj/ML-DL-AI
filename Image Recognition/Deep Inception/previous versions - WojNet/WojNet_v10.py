
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

from tflearn.helpers.trainer import Trainer
from import_data import *

import matplotlib.pyplot as plt

# Acquiring the data

folder = 'Digit Recognizer'
file_name = 'train.csv'
specific_dataset_source = folder + '/' + file_name
output_columns = ['label']

data = import_csv(specific_dataset_source, shuffle = True)


# In[2]:

#x_data, y_data = get_xy_mutual(data, output_columns, type = 'numpy')
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

split_index


# In[8]:

x_train = np.array(x_data[:split_index])
x_val = np.array(x_data[split_index:])

y_train = np.array(y_data_as_numbers[:split_index])
y_val = np.array(y_data_as_numbers[split_index:])
# --------------------------------------------------------------------------------------------------------


# In[9]:

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[10]:

#%matplotlib inline
#plt.hist(y_train)


# In[11]:

#%matplotlib inline
#plt.hist(y_val)


# In[12]:

print(y_train[:5])


# In[13]:

# Shaping data to the correct shape.
x_train = x_train.reshape([-1, 28, 28, 1])
x_val = x_val.reshape([-1, 28, 28, 1])
y_train = to_categorical(y_train, nb_classes = 10)
y_val = to_categorical(y_val, nb_classes = 10)

print('First five examples of one-hot encoded output:')
print(y_train[:5, :])


# In[14]:

# Building convolutional network
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

# Hidden layer 3
merged_layers = merge((branch1, branch2, branch3, branch4), mode = 'elemwise_sum', name = 'Merge')

# Hidden layer 4
merged_layers = fully_connected(merged_layers, 10, activation='relu')
merged_layers = dropout(merged_layers, 0.5)

merged_layers = fully_connected(merged_layers, 10, activation = 'softmax')
network = regression(merged_layers, optimizer = 'adam', learning_rate = 0.001,
                     loss = 'categorical_crossentropy', name ='target')

# ---------------------------------------------------------------------------------------


# In[15]:

# Training
# model = tflearn.DNN(network, tensorboard_verbose = 3, tensorboard_dir='./logs')

model = tflearn.DNN(network, tensorboard_verbose = 1, tensorboard_dir = './logs', best_checkpoint_path = './checkpoints/best/best_val', max_checkpoints = 1)
# checkpoint_path ='./checkpoints/checkpoint',

preTrained = True

if preTrained == False:
	model.fit(x_train, y_train, n_epoch = 2000, validation_set = (x_val, y_val), 
	show_metric = True, batch_size = 200, shuffle = True, #snapshot_step = 100,
          snapshot_epoch = True, run_id = 'WojNet')
else:
	# Loading the best accuracy checkpoint (accuracy over the validation data)
	model.load('./checkpoints/best/best_val9798')

	print('*' * 70)
	print('Model is successfully loaded for the best performance!')


# In[16]:

print('Training data:', model.evaluate(x_train, y_train))
print('Validation data:',model.evaluate(x_val, y_val))


# In[17]:

file_name_test = 'test.csv'
folder = 'Digit Recognizer'

source = folder + '/' + file_name_test
data = pd.read_csv(source)


# In[18]:

test_input = data.loc[:, :]


# In[19]:

#test_input.shape


# In[20]:

#test_input.shape[0]


# In[21]:

test_input_numpy = test_input.as_matrix()
# test_input_numpy = test_input_numpy.reshape([-1,28,28,1])
#test_input_numpy = test_input_numpy.reshape([test_input.shape[0],28,28,1])


# In[22]:

test_input_numpy.shape


# In[23]:

#test_input_numpy.shape[0]


# In[24]:

# Standalization
test_input_standarized = test_input_numpy / 255
#test_input_standarized.shape


# In[25]:

current_example = test_input_standarized[0]


# In[26]:

current_example.shape


# In[27]:

test_data_predicted = np.empty((0, 10))
test_data_predicted_label = np.empty((0, 10))
for i in range (0, test_input_numpy.shape[0]):
    current_example = test_input_standarized[i].reshape([-1,28,28,1])
    test_data_predicted = np.append(test_data_predicted, model.predict(current_example), axis = 0)
    test_data_predicted_label = np.append(test_data_predicted_label, model.predict_label(current_example), axis = 0)

    if i%2000 == 0:
        print(i)

print('Shape', test_data_predicted.shape)
   
# Choosing the most probable label
#test_data_predicted = test_data_predicted[:, -1]
# Indexing from 1 to number_of_examples


# In[28]:

index = np.arange(1, test_data_predicted.shape[0] + 1, 1)


# In[29]:

print(index)


# In[30]:

testing_index = 10


# In[31]:

current_example_woj = test_input_standarized[testing_index].reshape([-1,28,28,1])
example_by_woj = model.predict_label(current_example_woj)


# In[32]:

print(example_by_woj)


# In[33]:

print(test_data_predicted_label[testing_index])


# In[34]:

test_data_predicted_label.shape


# In[35]:

test_data_predicted_label.shape


# In[36]:

test_data_predicted = test_data_predicted_label[:, -1]


# In[37]:

test_data_predicted.shape


# In[38]:

test_data_predicted[testing_index]


# In[39]:




# In[40]:




# In[41]:

print(index)


# In[ ]:




# In[42]:

print(test_data_predicted[:10])


# In[43]:

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

