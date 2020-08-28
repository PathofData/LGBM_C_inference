import ctypes
import os
import unittest
from itertools import islice


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


def c_array(ctype, values):
    """Util function that converts a python list variable
    into a C++ array variable whose elements have the selected Ctype Dtype
    :param ctype: The selected C++ Dtype of the C++ array elements
    :param values: Python list to be converted
    :return: A C++ Array variable whose elements have the selected Ctype Dtype
    """
    return (ctype * len(values))(*values)


def c_str(string):
    """Util function that converts a Python string variable
    into a C++ string variable
    :param string: A Python string to be converted
    :return: A C++ converted string
    """
    return ctypes.c_char_p(string.encode('utf-8'))


def run_booster(data, num_classes):
    """Function that uses the C++ LGBM API to perform predictions.
    Expects a nested list of samples which converts to a C++ Double Array
    in order to compute the predictions.
    :param data: Python List of Lists that contain a list of samples where each sample
    is a list of features.
    :param num_classes: The number of classes to predict in case of multi class prediction.
    For regression set num_classes=1.
    :return: A Python list of floats with length equal to the length of samples * num_classes.
    """
    booster = ctypes.c_void_p()
    num_total_model = ctypes.c_long()
    matrix_shape_0 = len(data)
    matrix_shape_1 = len(data[0])
    predictions = [0.0]*matrix_shape_0*num_classes
    predictions = c_array(ctypes.c_double, predictions)
    num_predictions = ctypes.c_long()
    data = [item for sublist in data for item in sublist]
    data = c_array(ctypes.c_double, data)

    LIB.LGBM_BoosterCreateFromModelfile(
        c_str(os.path.join(PATH, MODEL_PATH)),
        ctypes.byref(num_total_model),
        ctypes.byref(booster))

    LIB.LGBM_BoosterPredictForMat(
        booster,
        data,
        dtype_float64,
        matrix_shape_0,
        matrix_shape_1,
        1,
        0,
        100,
        c_str(''),
        ctypes.byref(num_predictions),
        predictions)

    LIB.LGBM_BoosterFree(booster)
    return [ctypes.c_double(i).value for i in predictions]


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
