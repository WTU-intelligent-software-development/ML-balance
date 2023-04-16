# -*-coding:utf-8-*-
# @Time   : 2022/10/16 10:15
# @Author : 王梓涵

# 使用imlbearn库中上采样方法中的SMOTE接口
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from typing import Counter
#过采样包
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE 
from imblearn.over_sampling import SMOTE
#欠采样包
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn import preprocessing
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.under_sampling import NearMiss 

#综合采样包
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek

##过采样

def Randomos(X_dataset, y_dataset):
    ros=RandomOverSampler(random_state=42)
    X_ros, y_ros=ros.fit_resample(X_dataset, y_dataset)
    print("After RandomOverSampler X data statistics:")
    print(Counter(y_ros))
    return X_ros, y_ros

def smote(X_dataset, y_dataset):
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(sampling_strategy="not majority",random_state=42,
                k_neighbors=3,n_jobs=-1)
    X_dataset_smote, y_dataset_smote = smo.fit_resample(X_dataset, y_dataset)
    print("After smote X data statistics:")
    print(Counter(y_dataset_smote))
    #打乱一下数据
    X_dataset_smote, y_dataset_smote = shuffle(X_dataset_smote, y_dataset_smote, random_state=28) 
    return X_dataset_smote, y_dataset_smote

def svm_smote(X_dataset, y_dataset):
    sm = SVMSMOTE(random_state=42)
    X_svm, y_svm = sm.fit_resample(X_dataset, y_dataset)
    print("After SVMSMOTE X data statistics:")
    print(Counter(y_svm))
    return X_svm, y_svm




##欠采样

def undersmapling(X_dataset, y_dataset):
    rus = RandomUnderSampler(random_state=42)
    X_res,y_res = rus.fit_resample(X_dataset, y_dataset)
    print("After RandomUnderSampler X data statistics:")
    print(Counter(y_res))
    return X_res, y_res

def CondenNearN(X_dataset, y_dataset):
    cnn=CondensedNearestNeighbour(random_state=42)
    X_cnn, y_cnn = cnn.fit_resample(X_dataset, y_dataset)
    print("After CondensedNearestNeighbour X data statistics:")
    print(Counter(y_cnn))
    return X_cnn, y_cnn

def TomekLink(X_dataset, y_dataset):
    tl=TomekLinks()
    X_tl, y_tl = tl.fit_resample(X_dataset, y_dataset)
    print("After TomekLinks X data statistics:")
    print(Counter(y_tl))
    return X_tl, y_tl

def nearmiss(X_dataset, y_dataset):
    nm=nm = NearMiss()
    X_nm,y_nm=nm.fit_resample(X_dataset, y_dataset)
    print("After NearMiss X data statistics:")
    print(Counter(y_nm))
    return X_nm,y_nm

##综合采样

def Smote_Tomek(X_dataset, y_dataset):
    kos = SMOTETomek(random_state=42)  # 综合采样
    X_stk, y_stk= kos.fit_resample(X_dataset, y_dataset)
    print("After Smote_Tomek X data statistics:")
    print(Counter(y_stk))
    return X_stk, y_stk

def smotenn(X_dataset, y_dataset):
    smen=SMOTEENN(random_state=42)
    X_smen, y_smen=smen.fit_resample(X_dataset, y_dataset)
    print("After SMOTEENN X data statistics:")
    print(Counter(y_smen))
    return X_smen, y_smen
    







