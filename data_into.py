# -*-coding:utf-8-*-
# @Time   : 2022/10/16 10:12
# @Author : 王梓涵

import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from collections import Counter
from sklearn import preprocessing
from collections import Counter

def data_into(filepath):
    data = pd.read_csv(filepath)
    data_encoded = pd.get_dummies(data)
    X_dataset = data_encoded.iloc[:, 1:]  #得到所有的特征
    y_dataset = data_encoded.iloc[:, 0]  #得到标签，0无链接关系，1有链接关系
    #归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_dataset = min_max_scaler.fit_transform(X_dataset)
    X_dataset = pd.DataFrame(X_dataset)
    #打乱数据
    # X_dataset, y_dataset = shuffle(X_dataset, y_dataset, random_state=17)
    print("Raw X data statistics:" )
    print(Counter(X_dataset))
    return X_dataset, y_dataset,data

