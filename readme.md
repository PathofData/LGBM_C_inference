##Lightweight LGBM inference using C++ API

This repository provides a sample python script which demonstrates the use of the C++ API
provided by the LGBM package in order to perform inference without using the Python package
or its 3rd party dependencies (numpy, scipy, etc...).

This approach has minimal dependencies and is ideal for environments with limited resources
such as mobile and IoT devices.

### Explanation of the code
The code in this project is inspired from
[the official repository](https://github.com/microsoft/LightGBM/blob/master/tests/c_api_test/test_.py)

This example uses a model trained on the Iris dataset on a normal python environment. A
sample script to train the model is provided in `train.py`. After training is complete you
can save the model by calling:

```
# Assuming the model instance is called model
model.save_model('saved_model.txt')
```

We can now perform inference by pointing to the saved model through the C++ API. In order
to use the library we have to convert the python variables into the corresponding
C++ variables. The functions `c_array` and `c_str` take care of that. The `run_booster`
wrapper function integrates the logic so that we can feed python data directly without
having to deal with the conversions running under the hood.

### Script execution

Executing the `main.py` will read the data from a test_data.csv file, and using the
saved model will perform predictions which will be saved in the output.csv file.

### Testing

In order to test the predictions between the python package and the C++ API execute the
`test.py` which will load python predictions from a test_predictions.csv file and then
compare them with predictions inferred from the test_data.csv

