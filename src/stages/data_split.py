import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os

def data_split() -> None:
    os.chdir('../')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    df_ = pd.read_csv(config__['data']['_train_path'] + '/features.csv')

    X = df_.drop(df_.columns[-1], axis = 1)
    y = df_[[df_.columns[-1]]]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = config__['base']['test_size'],
                                                    random_state = config__['base']['random_state'])
    cols_ = X_train.columns

    scale_ = MinMaxScaler(feature_range=(0,1))
    scale_.fit(X_train)
    X_train = scale_.transform(X_train)
    X_test = scale_.transform(X_test)

    x_train_df = pd.DataFrame(data = X_train, columns = cols_)
    x_test_df = pd.DataFrame(data = X_test, columns = cols_)
    y_train_df = pd.DataFrame(data = y_train)
    y_test_df = pd.DataFrame(data = y_test)

    x_train_df.to_csv(config__['data']['_train_path'] + '/' + 'xtrain.csv')
    x_test_df.to_csv(config__['data']['_train_path'] + '/' + 'xest.csv')
    y_train_df.to_csv(config__['data']['_train_path'] + '/' + 'ytrain.csv')
    y_test_df.to_csv(config__['data']['_train_path'] + '/' + 'ytest.csv')

    print("Data Splitted and saved successfully")

if __name__ == '__main__':
    data_split()