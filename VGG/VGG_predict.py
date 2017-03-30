import tflearn

import numpy as np
import pandas as pd
from tqdm import tqdm
from network import *

model = ANN()

# Loading the best accuracy checkpoint (accuracy over the validation data)
model_name = input('Input name of the best model: ')
model_source = './checkpoints/best/' + str(model_name)
model.load(model_source)

print('*' * 70)
print('Model is successfully loaded for the best performance!')
print('*' * 70)

# Loading the test data
file_name_test = 'test.csv'
folder = 'Digit Recognizer'

source = folder + '/' + file_name_test
data = pd.read_csv(source)

test_input = data.loc[:, :]

test_input_numpy = test_input.as_matrix()

# Standalization
test_input_standarized = test_input_numpy / 255

# Predicting
test_data_predicted = np.empty((0, 10))
test_data_predicted_label = np.empty((0, 10))
print('Prediction in progress...')
for i in tqdm(range(0, test_input_numpy.shape[0])):
    current_example = test_input_standarized[i].reshape([-1,28,28,1])
    test_data_predicted = np.append(test_data_predicted, model.predict(current_example), axis = 0)
    test_data_predicted_label = np.append(test_data_predicted_label, model.predict_label(current_example), axis = 0)

print('The test data has been successfully labeled.')
print('*' * 70)
# Indexing
index = np.arange(1, test_data_predicted.shape[0] + 1, 1)

# Picking the last label = label with the highest probability
test_data_predicted = test_data_predicted_label[:, -1]

col = ['ImageId', 'Label']
output_data = np.stack((index, test_data_predicted))
output_data = output_data.T
output_data = output_data.astype(int)
test_data_prediction = pd.DataFrame(output_data, columns=col)

predict_output = 'labels.csv'
predicted_output_path= folder + '/' + predict_output

test_data_prediction.to_csv(predicted_output_path, sep = ',', index = False)

print('The test data CSV file has been successfully uploaded!')
print('*' * 70)
#tensorboard --logdir=logs/	