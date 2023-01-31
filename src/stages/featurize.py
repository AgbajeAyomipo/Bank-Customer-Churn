import pandas as pd
import numpy as np
import os
import json
import yaml
import joblib

def featurize() -> None:
    os.chdir('../')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    df_ = pd.read_csv(config__['data']['load_data_path'])
    df_ = df_.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)

    geo_map = {
    'France': 0,
    'Germany': 1,
    'Spain': 2
    }
    gender_map = {
        'Female': 0,
        'Male': 1
    }

    df_['Geography'] = df_['Geography'].map(geo_map)
    df_['Gender'] = df_['Gender'].map(gender_map)

    print('Features Successfully updated')

if __name__ == '__main__':
    featurize()