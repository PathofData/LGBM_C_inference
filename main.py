import ctypes
import os
import csv
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


def main():
    with open(os.path.join(PATH, 'test_data.csv')) as data_f:
        data_input = [line.rstrip().split(',') for line in data_f]

    data_input = [[float(i) for i in sample] for sample in data_input]
    c_predictions = run_booster(data=data_input, num_classes=NUM_CLASSES)
    prediction_size = len(c_predictions)
    c_predictions = iter(c_predictions)
    c_predictions = [list(islice(c_predictions, NUM_CLASSES))
                     for _ in range(prediction_size // NUM_CLASSES)]

    with open(os.path.join(PATH, 'output.csv'), 'w', newline='') as out_f:
        wr = csv.writer(out_f)
        for L in iter(c_predictions):
            wr.writerow(L)


if __name__ == '__main__':
    main()
