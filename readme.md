# LGBM inference using C++ API (Python/C++ Interface)

This repository provides sample scripts which demonstrate the use of the C++ API
provided by the LGBM package in order to perform inference without using the Python package
or its 3rd party dependencies (numpy, scipy, etc...).

This approach has minimal dependencies and is ideal for environments with limited resources
such as mobile and IoT devices.

For reference both `Python` and `C++` interface options are provided. One can access
the C++ API both from Python and C++ scripts, depending on the environment of choice.

## Python Code Explanation
The code in this project is inspired from
[the official repository](https://github.com/microsoft/LightGBM/blob/master/tests/c_api_test/test_.py)

The linking between C++ and Python is presented in the `c_interface.py` file.
This file contains the needed functions to use the C++ library through Python.

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

### Predicting

Executing the `predict.py` will read the data from a test_data.csv file, and using the
saved model will perform predictions which will be saved in the output.csv file.

### Testing

In order to test the predictions between the python package and the C++ API execute the
`test.py` which will load python predictions from a test_predictions.csv file and then
compare them with predictions inferred from the test_data.csv

## C++ Code Explanation

This tutorial provides the steps to:

* build the LGBM Package from source
* Build an executable that performs inference using a pre-trained model

### Instructions for building the LGBM Package from source

Follow the steps as provided in the [Official Instalation Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

```
git clone --recursive https://github.com/microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
```
The above code will build the LGBM Package on a local folder called LightGBM.

### Instructions for building executable for inference (prediction)

In order to build the executable we have to link the Include folder
as well as the main library. These are provided with the -I,-L and -l arguments:

* `-I{path_to_LightGBM}/include`
* `-L{path_to_LightGBM}`
* `-l_lightgbm`

For example if the user is on the `cpp` directory of this repo you can use
the following command to build an executable with the name `lgbm_predict`

`g++ main_c.cpp -I${PWD}/LightGBM/include -L${PWD}/LightGBM -l_lightgbm -o lgbm_predict`

Then in order to run the executable first export the library path to your env:

`export LD_LIBRARY_PATH=${PWD}/LightGBM/`

Then run the executable:

`./lgbm_predict`

### Exploration of the code

The `main_c.cpp` file is used to load the model (the model in this repo is `saved_model.txt`)
and perform predictions.
The main function for predictions called `predict` is responsible for:

* Reading the data from a csv file (here `test_data.csv`)
* Loading the pre-trained model
* Performing the prediction and saving the result to a csv file (here `c_test_predictions.csv`)
