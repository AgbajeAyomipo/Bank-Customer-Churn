import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import json
import yaml
import joblib
import os

def evaluate() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Desktop/Bank-Customer-Churn')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    model = joblib.load(config__['model']['out_path'])

    X_test = pd.read_csv(config__['data']['x_test_path'])
    y_test = pd.read_csv(config__['data']['y_test_path'])

    X_test = X_test.values
    y_test = y_test['Exited'].values

    preds_ = model.predict(X_test)

    roc = roc_auc_score(y_true = y_test, y_score = preds_)
    accuracy_score_ = accuracy_score(y_true = y_test, y_pred = preds_)
    precision_score_ = precision_score(y_true = y_test, y_pred = preds_)

    _metric = {
        'roc': roc,
        'accuracy_score': accuracy_score_,
    }

    json.dump(
        obj = _metric,
        fp = open(config__['metric']['path'] + '/' 'metrics.json', 'w'),
        indent = 4,
        sort_keys=True
    )
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(y_true = y_test, y_pred = preds_, cmap = 'gist_gray', display_labels = ['NO', 'YES'], ax = ax)
    plt.savefig(config__['metric']['path'] + '/confusion_matrix.png')
    print("Successful")

if __name__ == '__main__':
    evaluate()