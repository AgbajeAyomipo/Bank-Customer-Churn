import pandas as pd
import numpy as np
import os
import json
import yaml
import joblib

def data_load() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Desktop/Bank-Customer-Churn')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    df_ = pd.read_csv(config__['data']['load_data_path'])
    df_.head()
    print('Data Loaded Successfully')

if __name__ == '__main__':
    data_load()