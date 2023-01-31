import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import joblib
import os

def train() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Desktop/Bank-Customer-Churn')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    xgbc = XGBClassifier(
    n_estimators = config__['train']['params']['_n_estimators'],
    learning_rate = config__['train']['params']['_learning_rate'],
    max_depth = config__['train']['params']['max_depth']
    )

    X_train = pd.read_csv(config__['data']['x_train_path'])
    y_train = pd.read_csv(config__['data']['y_train_path'])

    X_train = X_train.values
    y_train = y_train['Exited'].values

    xgbc.fit(X = X_train,
            y = y_train)

    joblib.dump(xgbc, config__['model']['path_'] + '/' + 'model.joblib')
    print('model trained and saved successfully')

if __name__ == '__main__':
    train()