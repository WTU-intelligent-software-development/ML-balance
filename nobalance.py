# -*-coding:utf-8-*-
# @Time   : 2022/11/19 12:20
# @Author : 王梓涵
from sklearn.metrics import  classification_report

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold
import math
import pandas as pd
import sklearn.model_selection as ms
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#读取数据
import data_into as dt
import data_blance as db
import clustingstab as cb

import time




#十折交叉验证结果
def resultnobalancerun(model,X,y):

    #10折交叉验证得分
    p=[]
    sp = cross_val_score(model,X,y, cv=10,scoring='precision',verbose=0)
    p.append(sp)
    p1=np.mean(p)
    r=[]
    sr = cross_val_score(model,X,y, cv=10,scoring='recall',verbose=0)
    r.append(sr)
    r1=np.mean(r)
    f=[]
    sf = cross_val_score(model,X,y, cv=10,scoring='f1',verbose=0)
    f.append(sf)
    f1=np.mean(f)
    a=0

    #对于10次交叉验证的RMSE

    return a,p1,r1,f1#,Rmseavg

#分类模型的最优参数选取与10折交叉验证平均结果
def ModeGnobalancerun(params, X, y,mlmodel,name):
    model = ms.GridSearchCV(estimator=mlmodel, param_grid=params, cv=10,verbose=0)
    # 网格搜索训练后的副产品
    model.fit(X, y)
    #保存到文件
    with open('./CVres/Bestparameter.txt', 'a',encoding='utf-8') as f:
        f.write("数据集：" + str(name) + '\n')
        f.write("机器学习模型："+str(mlmodel)+'\n')
        f.write("模型的最优参数："+str(model.best_params_)+'\n')
        f.write("最优模型分数："+str(model.best_score_)+'\n')
        f.write("最优模型对象："+str(model.best_estimator_)+'\n')
        f.write("--------------------------------------------\n")
    # 用最优参数训练模型
    modelbest = model.best_estimator_
    #数据平衡
    #10折交叉验证得分
    a,p1,r1,f1=resultnobalancerun(modelbest,X,y)
    #对于10次交叉验证的RMSE
    return a,p1,r1,f1

def nobalancerun(name,X,y,dataset):
    #----------------KNN----------------
    model_knn=KNeighborsClassifier(n_neighbors=5)
    
    #----------------结果输出----------------
    aknn,pknn,rknn,fknn=resultnobalancerun(model_knn,X,y)
    
    #----------------DT----------------
    model_dt=DecisionTreeClassifier(random_state=32)#随机种子变化
    #----------------结果输出----------------
    adt,pdt,rdt,fdt=resultnobalancerun(model_dt,X,y)

    #----------------Kmeans----------------
    model_kmeans=KMeans(n_clusters=2,random_state=32)
    #----------------结果输出----------------
    akmeans,pkmeans,rkmeans,fkmeans=resultnobalancerun(model_kmeans,X,y)


    #----------------NB----------------
    model_nb=GaussianNB()
    #----------------结果输出----------------
    anb,pnb,rnb,fnb=resultnobalancerun(model_nb,X,y)

    #----------------RF----------------
    model_rf=RandomForestClassifier(random_state=32)
    #----------------结果输出----------------
    arf,prf,rrf,frf=resultnobalancerun(model_rf,X,y)
   

    #----------------LR----------------
    model_lr=LogisticRegression(random_state=32)
    #----------------结果输出----------------
    alr,plr,rlr,flr=resultnobalancerun(model_lr,X,y)

    #----------------SVM----------------
    model_svm = svm.SVC(probability=True,random_state=32)
    #----------------结果输出----------------
    asvm,psvm,rsvm,fsvm=resultnobalancerun(model_svm,X,y)

    #保存结果
    with open(name, 'a',encoding='utf-8') as f:
        f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
        f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
        f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
        f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
        f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
        f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
        f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
        f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
        f.close()




def runexp():
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
    
    #----------------IDTC----------------
    nobalancerun("./CVres/None/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,IDTC)

    #---------------IDUC----------------
    #欠采样
    nobalancerun("./CVres/None/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,IDUC)

    #---------------TCCC----------------
    #欠采样
    nobalancerun("./CVres/None/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,TCCC)
    
    
    #---------------UCCC----------------
    #欠采样
    nobalancerun("./CVres/None/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,UCCC)


    #---------------UCTC----------------
    #欠采样
    nobalancerun("./CVres/None/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,UCTC)
    
    #---------------SMOS---------------- 

    #欠采样
    nobalancerun("./CVres/None/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,SMOS)

    #---------------iTrust---------------- 
    #欠采样
    nobalancerun("./CVres/None/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,iTrust)

if __name__ == '__main__':
     #开始计时
    time_start=time.time()
    runexp()
    #结束计时
    time_end=time.time()
    #计算运行时间
    print('time cost',time_end-time_start,'s')
