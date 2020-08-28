import os
import csv
from itertools import islice
from c_interface import run_booster


# Path of the Script
PATH = os.path.abspath(os.path.dirname(__file__))
NUM_CLASSES = 3


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
