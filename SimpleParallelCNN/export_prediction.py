import numpy as np
import pandas as pd

def prediction_export(predicted):
	# Indexing
	index = np.arange(1, predicted.shape[0] + 1, 1)

	# Picking the last label = label with the highest probability
	test_data_predicted = predicted[:, -1]

	col = ['ImageId', 'Label']
	output_data = np.stack((index, test_data_predicted))
	output_data = output_data.T
	output_data = output_data.astype(int)
	test_data_prediction = pd.DataFrame(output_data, columns=col)

	folder = 'Digit Recognizer'
	predict_output = 'labels.csv'
	predicted_output_path= folder + '/' + predict_output

	test_data_prediction.to_csv(predicted_output_path, sep = ',', index = False)

	print('The test data CSV file has been successfully uploaded!')
	print('*' * 70)