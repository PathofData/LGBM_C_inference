import ctypes
import os
import unittest
from itertools import islice
from main import run_booster


# Path of the Script
PATH = os.path.abspath(os.path.dirname(__file__))
# Path of the LGBM Library
LIB_PATH = os.path.join(PATH, 'lib_lightgbm.so')
LIB = ctypes.cdll.LoadLibrary(LIB_PATH)
# Path of the trained LGB model
MODEL_PATH = 'saved_model.txt'
NUM_CLASSES = 3

dtype_float32 = 0
dtype_float64 = 1
dtype_int32 = 2
dtype_int64 = 3


class TestListElements(unittest.TestCase):
    """Class to test Python library predictions with
    C++ library predictions. The inputs are:
    python_predictions: CSV file with each line being a python
    prediction for a single sample separated by comma.
    test_data: CSV file with each line being features from a
    single sample separated by comma.
    """
    def setUp(self):
        """Read the predictions into an nested list and cast it into float
        """
        with open(os.path.join(PATH, 'test_predictions.csv')) as label_f:
            python_predictions = [line.rstrip().split(',') for line in label_f]
        python_predictions = [[float(i) for i in sample] for sample in python_predictions]

        """Read the samples into a nested list and cast it into float
        """
        with open(os.path.join(PATH, 'test_data.csv')) as data_f:
            data_input = [line.rstrip().split(',') for line in data_f]

        data_input = [[float(i) for i in sample] for sample in data_input]
        """Perform the prediction. Keep in mind that the predictions are
        a flattened list of shape n_samples * n_classes
        """
        c_predictions = run_booster(data=data_input, num_classes=NUM_CLASSES)
        prediction_size = len(c_predictions)
        c_predictions = iter(c_predictions)
        """Reshape the flattened list into a nested list
        """
        c_predictions = [list(islice(c_predictions, NUM_CLASSES))
                         for _ in range(prediction_size // NUM_CLASSES)]

        self.expected = python_predictions
        self.result = c_predictions

    def test_count_eq(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq(self):
        self.assertListEqual(self.result, self.expected)


if __name__ == '__main__':
    unittest.main()
