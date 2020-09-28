#include <LightGBM/c_api.h>
#include <iostream>

void predict()
{
    int temp;
    int p = 100;
    BoosterHandle handle;
    // load model
    temp = LGBM_BoosterCreateFromModelfile(
      "saved_model.txt",
      &p,
      &handle
    );

    // Perform prediction
    int64_t res;
    res = LGBM_BoosterPredictForFile(
      handle,
      "test_data.csv",
      0,
      C_API_PREDICT_NORMAL,
      0,
      -1,
      "",
      "c_test_predictions.csv"
    );

}

int main() {

  predict();
  return 0;

}