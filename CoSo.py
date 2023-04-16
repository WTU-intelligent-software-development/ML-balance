# -*-coding:utf-8-*-
# @Time   : 2022/11/19 12:20
# @Author : 王梓涵

###本文件计算轮廓系数

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import data_into as dt
import data_blance as db
import clustingstab as cb
import time


def nobanlanceclustingcs(name,X,dataset):
    model_kmeans=KMeans(n_clusters=2,random_state=32)
    model_kmeans.fit(X)
    #聚类轮廓系数
    cs=metrics.silhouette_score(X,model_kmeans.labels_,metric='euclidean')
    #保存到文件
    with open(name, 'a',encoding='utf-8') as f:
        f.write("数据集："+str(dataset)+'\n')
        f.write("轮廓系数："+str(cs)+'\n')
        f.write("---------------------------------------"+'\n')

def balanceclustingcs(name,X,y,dbm,dataset):
    model_kmeans=KMeans(n_clusters=2,random_state=32)
    #数据平衡
    X1,y1=dbm(X,y)
    #模型训练
    model_kmeans.fit(X1)
    #聚类轮廓系数
    cs=metrics.silhouette_score(X1,model_kmeans.labels_,metric='euclidean')
    #保存到文件
    with open(name, 'a',encoding='utf-8') as f:
        f.write("数据集："+str(dataset)+'\n')
        f.write("平衡方法"+str(dbm)+'\n')
        f.write("轮廓系数："+str(cs)+'\n')
        f.write("---------------------------------------"+'\n')

def runcs():
    #读取数据
    IDTC= "IDTC"
    IDUC= "IDUC"
    TCCC= "TCCC"
    UCCC= "UCCC"
    UCTC= "UCTC"
    iTrust= "iTrust"
    SMOS= "SMOS"
    #----------------读取数据----------------
    filepath1='./data/Data_ID_TC.csv'
    filepath2='./data/Data_ID_UC.csv'
    filepath3='./data/Data_TC_CC.csv'
    filepath4='./data/Data_UC_CC.csv'
    filepath5='./data/Data_UC_TC.csv'
    filepath6='./data/Data_iTrust.csv'
    filepath7='./data/Data_SMOS.csv'
    X_dataset_ID_TC, y_dataset_ID_TC,data_ID_TC = dt.data_into(filepath1)

    X_dataset_ID_UC, y_dataset_ID_UC,data_ID_UC = dt.data_into(filepath2)

    X_dataset_TC_CC, y_dataset_TC_CC,data_TC_CC = dt.data_into(filepath3)

    X_dataset_UC_CC, y_dataset_UC_CC,data_UC_CC = dt.data_into(filepath4)

    X_dataset_UC_TC, y_dataset_UC_TC,data_UC_TC = dt.data_into(filepath5)

    #代码制品

    X_dataset_iTrust, y_dataset_iTrust,data_iTrust = dt.data_into(filepath6)

    X_dataset_SMOS, y_dataset_SMOS,data_SMOS = dt.data_into(filepath7)
    
    #不平衡数据集
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_ID_TC,IDTC)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_ID_UC,IDUC)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_TC_CC,TCCC)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_UC_CC,UCCC)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_UC_TC,UCTC)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_SMOS,SMOS)
    nobanlanceclustingcs("./CVres/nobalanceCS.txt",X_dataset_iTrust,iTrust)
    #平衡数据集
    #RUS
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.undersmapling,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.undersmapling,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.undersmapling,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.undersmapling,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.undersmapling,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.undersmapling,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.undersmapling,iTrust)

    #Tomeklink

    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.TomekLink,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.TomekLink,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.TomekLink,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.TomekLink,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.TomekLink,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.TomekLink,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.TomekLink,iTrust)


    #NearMiss
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.nearmiss,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.nearmiss,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.nearmiss,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.nearmiss,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.nearmiss,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.nearmiss,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.nearmiss,iTrust)

    #ROS
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.Randomos,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.Randomos,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.Randomos,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.Randomos,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.Randomos,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.Randomos,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.Randomos,iTrust)


    #SMOTE
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.smote,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.smote,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.smote,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.smote,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.smote,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.smote,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.smote,iTrust)

    #Somteenn
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.smotenn,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.smotenn,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.smotenn,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.smotenn,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.smotenn,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.smotenn,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.smotenn,iTrust)



    #smotetomeklink
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_TC,y_dataset_ID_TC,db.Smote_Tomek,IDTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_ID_UC,y_dataset_ID_UC,db.Smote_Tomek,IDUC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_TC_CC,y_dataset_TC_CC,db.Smote_Tomek,TCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_CC,y_dataset_UC_CC,db.Smote_Tomek,UCCC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_UC_TC,y_dataset_UC_TC,db.Smote_Tomek,UCTC)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_SMOS,y_dataset_SMOS,db.Smote_Tomek,SMOS)
    balanceclustingcs("./CVres/balanceCS.txt",X_dataset_iTrust,y_dataset_iTrust,db.Smote_Tomek,iTrust)
    


    