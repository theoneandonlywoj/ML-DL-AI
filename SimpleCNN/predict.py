import tflearn

import numpy as np
import pandas as pd
from network import *
from import_data import *
from export_prediction import *

model = ANN(WIDTH = 28, HEIGHT = 28, CHANNELS = 1, LABELS = 10)

# Loading the best accuracy checkpoint (accuracy over the validation data)
model_name = input('Input name of the best model: ')
model_source = './checkpoints/best/' + str(model_name)
model.load(model_source)

print('*' * 70)
print('Model is successfully loaded for the best performance!')
print('*' * 70)

# Data
test_input_numpy = get_data_MNIST_test()

# Standalization
test_input_numpy = test_input_numpy / 255

# Prediction
test_data_predicted_label = big_dataset_prediction(model, DATA = test_input_numpy)
# Exporting the prediction
prediction_export(test_data_predicted_label)

#tensorboard --logdir=logs/	
