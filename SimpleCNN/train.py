from __future__ import division, print_function, absolute_import

import tflearn
from import_data import *
from network import *

# Getting the data
x_train, x_val, y_train, y_val = get_data()
#x_train, x_val, y_train, y_val = get_data_oxford_flowers()

# Recalling the network defined in network.py
model = ANN()
# Training
model.fit(x_train, y_train, n_epoch = 1000, validation_set = (x_val, y_val), 
show_metric = True, batch_size = 100, shuffle = True, #snapshot_step = 100,
      snapshot_epoch = True, run_id = 'SimpleCNN')

# Loading the best network
model_name = input('Input name of the best model: ')
model_source = './checkpoints/best/' + str(model_name)
model.load(model_source)

print('*' * 70)
print('Model is successfully loaded for the best performance!')

# Evaluation

print('Evaluation in progress...')
print('Training data:', model.evaluate(x_train, y_train))
print('Validation data:',model.evaluate(x_val, y_val))

#tensorboard --logdir=logs/	




