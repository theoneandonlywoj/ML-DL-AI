import pandas as pd
import numpy as np
from tflearn.data_utils import to_categorical

def import_csv(file_path, shuffle = False):
    data = pd.read_csv(file_path)
    print('*' * 70)
    print('Import CSV file has been successful!')
    if shuffle == True:
        data.reindex(np.random.permutation(data.index))
        print('The data has been shuffled!')
    else:
        print('The data has not been shuffled!')
    return data

def labels_info(output_data):
    labels_names = np.unique(output_data)
    number_of_labels = labels_names.shape[0]
    print('*' * 70)
    print("Number of uniques categories:", number_of_labels)
    labels_as_numbers = np.arange(number_of_labels)
    print("Categories as numbers", labels_as_numbers)

    for _ in labels_as_numbers:
        print('Category ' + str(_) + ' is ' + str(labels_names[_]))
    return number_of_labels

def labels_as_numbers(output_data):
    _, output_data_as_numbers = np.unique(output_data, return_inverse=True)
    return output_data_as_numbers

# -------------------------------------------------------------------------------
# Acquiring the data
def get_data_MNIST():
	folder = 'Digit Recognizer'
	file_name = 'train.csv'
	specific_dataset_source = folder + '/' + file_name
	output_columns = ['label']

	data = import_csv(specific_dataset_source, shuffle = True)

	# Data split into the input and output
	x_data = data
	y_data = np.array(data.pop('label'))

	print('Shape of the input data:', x_data.shape)
	print('Shape of the output data:', y_data.shape)


	# Standalization
	x_data = x_data / 255

	num_samples = x_data.shape[0]
	input_features = x_data.shape[1]

	print('Number of samples:', num_samples)
	print('Number of the input features:', input_features)

	y_data_as_numbers = labels_as_numbers(y_data)


	# Cross validation data preparation
	split_percentage = 80
	split_index = int(x_data.shape[0]/(100/split_percentage))

	x_train = np.array(x_data[:split_index])
	x_val = np.array(x_data[split_index:])

	y_train = np.array(y_data_as_numbers[:split_index])
	y_val = np.array(y_data_as_numbers[split_index:])


	# Information about the data
	print(x_train.shape)
	print(x_val.shape)
	print(y_train.shape)
	print(y_val.shape)

	# Shaping data into the correct shape.
	x_train = x_train.reshape([-1, 28, 28, 1])
	x_val = x_val.reshape([-1, 28, 28, 1])
	y_train = to_categorical(y_train, nb_classes = 10)
	y_val = to_categorical(y_val, nb_classes = 10)

	return x_train, x_val, y_train, y_val   

# Oxford Flowers Dataset
def get_data_oxford_flowers():
	import tflearn.datasets.oxflower17 as oxflower17
	X, Y = oxflower17.load_data(one_hot = True, resize_pics = (227, 227))

	split_percentage = 80
	split_index = int(X.shape[0]/(100/split_percentage))

	x_train = np.array(X[:split_index])
	x_val = np.array(X[split_index:])

	y_train = np.array(Y[:split_index])
	y_val = np.array(Y[split_index:])
	return x_train, x_val, y_train, y_val 

def get_data_CIFAR10(dataset = 'Train + Val'):
	from tflearn.datasets import cifar10
	(X, Y), (X_test, Y_test) = cifar10.load_data()
	
	# Size is 32, 32, 3
	
	split_percentage = 80
	split_index = int(X.shape[0]/(100/split_percentage))

	x_train = np.array(X[:split_index])
	x_val = np.array(X[split_index:])

	y_train = np.array(Y[:split_index])
	y_val = np.array(Y[split_index:])
	
	Y = to_categorical(Y, 10)
	Y_test = to_categorical(Y_test, 10)

	if dataset == 'Test':
		return x_train, x_val, y_train, y_val 
	else:
		return X_test, Y_test 

	


