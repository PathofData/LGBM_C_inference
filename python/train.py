import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report


def main():
    data = load_iris(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    df = df.sample(frac=1)
    df.reset_index(inplace=True, drop=True)

    train_set = df.loc[:100, :]
    test_set = df.loc[100:, :]

    X_train = train_set.drop('target', axis=1)
    X_test = test_set.drop('target', axis=1)
    y_train = train_set['target']
    y_test = test_set['target']

    train_params = {
        'objective': 'multiclass',
        'metric': "multi_logloss",
        'num_class': 3,
        'verbose': -1
    }

    train_ds = lgb.Dataset(data=X_train, label=y_train)
    valid_ds = lgb.Dataset(data=X_test, label=y_test)
    model = lgb.train(train_params, train_set=train_ds, valid_sets=valid_ds)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    print(classification_report(y_test, predicted_classes))

    X_test.to_csv('test_data.csv', index=False, header=False)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv('test_predictions.csv', index=False, header=False)
    model.save_model('saved_model.txt')


if __name__ == '__main__':
    main()
