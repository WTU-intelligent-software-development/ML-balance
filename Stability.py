# -*-coding:utf-8-*-
# @Time   : 2022/11/13 16:20
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
def resultcv(model,X,y,dbm):
    #数据平衡
    X11,y11=dbm(X,y)
    #10折交叉验证得分
    p=[]
    sp = cross_val_score(model,X11,y11, cv=10,scoring='precision',verbose=0)
    p.append(sp)
    p1=np.mean(p)
    r=[]
    sr = cross_val_score(model,X11,y11, cv=10,scoring='recall',verbose=0)
    r.append(sr)
    r1=np.mean(r)
    f=[]
    sf = cross_val_score(model,X11,y11, cv=10,scoring='f1',verbose=0)
    f.append(sf)
    f1=np.mean(f)

    pstd=np.std(p, ddof = 1)
    rstd=np.std(r,ddof=1)
    fstd=np.std(f,ddof=1)

    # a=[]
    # sa = cross_val_score(model,X11,y11, cv=10,scoring='accuracy',verbose=0)
    # a.append(1-sa)


    #对于10次交叉验证的RMSE

    return pstd,rstd,fstd,p1,r1,f1


def result(model,X,y,dbm):
    X1,Xt,y1,yt=train_test_split(X,y,test_size=0.1)
        #数据平衡
    X11,y11=dbm(X1,y1)
    model.fit(X11,y11)
    yp=model.predict(Xt)
    report=classification_report(yt,yp,output_dict=True)
    """
        {'1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1}, 
        'accuracy': 1.0, 'macro avg': {'precision': 1.0, 
        'recall': 1.0, 'f1-score': 1.0, 'support': 1}, 
        'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1}}
    """
    p=report[u'1']['precision']
    r=report[u'1']['recall']
    f=report[u'1']['f1-score']
    return p,r,f

#分类模型的最优参数选取与10折交叉验证平均结果
def ModeGcv(params, X, y, dbm,mlmodel,name):
    model = ms.GridSearchCV(estimator=mlmodel, param_grid=params, cv=10,verbose=0)
    # 网格搜索训练后的副产品
    X1, y1 = dbm(X, y)
    model.fit(X1, y1)
    #保存到文件
    with open('./CVres/Bestparameter.txt', 'a',encoding='utf-8') as f:
        f.write("数据集：" + str(name) + '\n')
        f.write("机器学习模型："+str(mlmodel)+'\n')
        f.write("平衡方法："+str(dbm.__name__)+'\n')
        f.write("模型的最优参数："+str(model.best_params_)+'\n')
        f.write("最优模型分数："+str(model.best_score_)+'\n')
        f.write("最优模型对象："+str(model.best_estimator_)+'\n')
        f.write("--------------------------------------------\n")
    # 用最优参数训练模型
    #10折交叉验证得分
    # a,p1,r1,f1=resultcv(model.best_estimator_,X,y,dbm)
    p1,r1,f1=result(model.best_estimator_,X,y,dbm)
    a=0
    #对于10次交叉验证的RMSE
    return a,p1,r1,f1

#保存错误率结果
def saveerror(a,dbm,mlmodel):
    with open('./CVres/errorrate.txt', 'a',encoding='utf-8') as f:
        f.write("机器学习模型：" + str(mlmodel) + '\n')
        f.write("平衡方法：" + str(dbm.__name__) + '\n')
        f.write("错误率：" + str(a) + '\n')
        f.write("--------------------------------------------")

#分类模型错误率统计检验


##聚类前霍普金斯统计量


#聚类交叉验证


#聚类稳定性    


# def itrusttmoek(name,X,y,dbm):
#     model_knn=KNeighborsClassifier(n_neighbors=6)
#     model_dt=DecisionTreeClassifier(class_weight='balanced',criterion='entropy',max_depth=2,random_state=32)
#     model_nb=GaussianNB(var_smoothing=1e-07)
#     model_lr=LogisticRegression(random_state=32,C=0.1,penalty='l2')
#     model_svm=svm.SVC(C=1000,kernel='rbf',gamma=0.0001,random_state=32)

#     pknn,rknn,fknn=result(model_knn,X,y,dbm)
#     pdt,rdt,fdt=result(model_dt,X,y,dbm)
#     pnb,rnb,fnb=result(model_nb,X,y,dbm)
#     plr,rlr,flr=result(model_lr,X,y,dbm)
#     psvm,rsvm,fsvm=result(model_svm,X,y,dbm)

#     with open(name, 'a',encoding='utf-8') as f:
#         f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
#         f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
#         f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
#         f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
#         # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
#         f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
#         f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
#         # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
#         f.close()

# def itrustros(name,X,y,dbm):
#     model_knn=KNeighborsClassifier(n_neighbors=1)
#     model_dt=DecisionTreeClassifier(class_weight='balanced',criterion='gini',max_depth=9,random_state=32)
#     model_nb=GaussianNB(var_smoothing=1e-07)
#     model_lr=LogisticRegression(random_state=32,C=1.5,penalty='l2')
#     model_svm=svm.SVC(C=0.001,kernel='rbf',gamma=0.001,random_state=32)

#     pknn,rknn,fknn=result(model_knn,X,y,dbm)
#     pdt,rdt,fdt=result(model_dt,X,y,dbm)
#     pnb,rnb,fnb=result(model_nb,X,y,dbm)
#     plr,rlr,flr=result(model_lr,X,y,dbm)
#     psvm,rsvm,fsvm=result(model_svm,X,y,dbm)

#     with open(name, 'a',encoding='utf-8') as f:
#         f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
#         f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
#         f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
#         f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
#         # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
#         f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
#         f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
#         # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
#         f.close()

# def itrustsmote(name,X,y,dbm):
#     model_knn=KNeighborsClassifier(n_neighbors=1)
#     model_dt=DecisionTreeClassifier(class_weight='balanced',criterion='entropy',max_depth=2,random_state=32)
#     model_nb=GaussianNB(var_smoothing=1e-07)
#     model_lr=LogisticRegression(random_state=32,C=0.1,penalty='l2')
#     model_svm=svm.SVC(C=1000,kernel='rbf',gamma=0.01,random_state=32)
#     psvm,rsvm,fsvm=result(model_svm,X,y,dbm)

#     pknn,rknn,fknn=result(model_knn,X,y,dbm)
#     pdt,rdt,fdt=result(model_dt,X,y,dbm)
#     pnb,rnb,fnb=result(model_nb,X,y,dbm)
#     plr,rlr,flr=result(model_lr,X,y,dbm)

#     with open(name, 'a',encoding='utf-8') as f:
#         f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
#         f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
#         f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
#         f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
#         # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
#         f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
#         f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
#         # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
#         f.close()

# def itrustsmotetomek(name,X,y,dbm):
#     model_knn=KNeighborsClassifier(n_neighbors=6)
#     model_dt=DecisionTreeClassifier(class_weight='balanced',criterion='entropy',max_depth=9,random_state=32)
#     model_nb=GaussianNB(var_smoothing=1e-07)
#     model_lr=LogisticRegression(random_state=32,C=1.7,penalty='l2')
#     model_svm=svm.SVC(C=0.001,kernel='rbf',gamma=0.01,random_state=32)

#     pknn,rknn,fknn=result(model_knn,X,y,dbm)
#     pdt,rdt,fdt=result(model_dt,X,y,dbm)
#     pnb,rnb,fnb=result(model_nb,X,y,dbm)
#     plr,rlr,flr=result(model_lr,X,y,dbm)
#     psvm,rsvm,fsvm=result(model_svm,X,y,dbm)

#     with open(name, 'a',encoding='utf-8') as f:
#         f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
#         f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
#         f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
#         f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
#         # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
#         f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
#         f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
#         # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
#         f.close()



# def itrustsmoteenn(name,X,y,dbm):
#     model_knn=KNeighborsClassifier(n_neighbors=2)
#     model_dt=DecisionTreeClassifier(class_weight='balanced',criterion='entropy',max_depth=2,random_state=32)
#     model_nb=GaussianNB(var_smoothing=1e-07)
#     model_lr=LogisticRegression(random_state=32,C=1.7,penalty='l2')
#     model_svm=svm.SVC(C=0.1,random_state=32,kernel='rbf',gamma=0.01)

#     pknn,rknn,fknn=result(model_knn,X,y,dbm)
#     pdt,rdt,fdt=result(model_dt,X,y,dbm)
#     pnb,rnb,fnb=result(model_nb,X,y,dbm)
#     plr,rlr,flr=result(model_lr,X,y,dbm)
#     psvm,rsvm,fsvm=result(model_svm,X,y,dbm)

#     with open(name, 'a',encoding='utf-8') as f:
#         f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
#         f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
#         f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
#         f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
#         # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
#         f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
#         f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
#         # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
#         f.close()


def cv(name, X,y,dbm,dataset):
    """
    函数说明：
    输入参数：
    name：文件名
    X：特征
    y：标签
    dbm：数据平衡方法
    """

    #----------------KNN----------------
    model_knn=KNeighborsClassifier(n_neighbors=5)
    # params_svm={'n_neighbors':range(1,10)}
    #----------------结果输出----------------
    # aknn,pknn,rknn,fknn=ModeGcv(params_svm,X,y,dbm,model_knn,dataset)
    pknnstd,rknnstd,fknnstd,pknn,rknn,fknn=resultcv(model_knn,X,y,dbm)
    
    #----------------DT----------------
    # model_dt=DecisionTreeClassifier(class_weight='balanced',random_state=32)#随机种子变化
    # params_dt={'criterion':['gini','entropy'],'max_depth':range(1,10)}
    # adt,pdt,rdt,fdt=ModeGcv(params_dt,X,y,dbm,model_dt,dataset)
    model_dt=DecisionTreeClassifier(random_state=32)
    pdtstd,rdtstd,fdtstd,pdt,rdt,fdt=resultcv(model_dt,X,y,dbm)

    #----------------Kmeans----------------
    # model_kmeans=KMeans(n_clusters=2,random_state=32)
    # akmeans,pkmeans,rkmeans,fkmeans=resultcv(model_kmeans,X,y,dbm)


    #----------------NB----------------
    model_nb=GaussianNB()
    # params_nb={'var_smoothing':[1e-7,1e-8,1e-9,1e-10,1e-11]}
    # anb,pnb,rnb,fnb=ModeGcv(params_nb,X,y,dbm,model_nb,dataset)
    pnbstd,rnbstd,fnbstd,pnb,rnb,fnb=resultcv(model_nb,X,y,dbm)

    #----------------RF----------------
    # model_rf=RandomForestClassifier(random_state=32)
    # params_rf= {'n_estimators':range(1,20),'criterion':['gini','entropy'],'max_depth':range(1,15)}
    # arf,prf,rrf,frf=ModeGcv(params_rf,X,y,dbm,model_rf,dataset)
   

    #----------------LR----------------
    model_lr=LogisticRegression(random_state=32)
    # params_lr={'C':[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9],'penalty':['l1','l2']}
    # alr,plr,rlr,flr=ModeGcv(params_lr,X,y,dbm,model_lr,dataset)
    plrstd,rlrstd,flrstd,plr,rlr,flr=resultcv(model_lr,X,y,dbm)

    #----------------SVM----------------
    # model_svm = svm.SVC(probability=True,class_weight='balanced',kernel='rbf',random_state=32)
    # params_svm = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.01,0.001, 0.0001]} 
    # asvm,psvm,rsvm,fsvm=ModeGcv(params_svm,X,y,dbm,model_svm,dataset)
    model_svm = svm.SVC(random_state=32)
    psvmstd,rsvmstd,fsvmstd,psvm,rsvm,fsvm=resultcv(model_svm,X,y,dbm)

    #保存errorrate
    # saveerror(aknn,dbm,model_knn)
    # saveerror(adt,dbm,model_dt)
    # saveerror(anb,dbm,model_nb)
    # saveerror(arf,dbm,model_rf)
    # saveerror(alr,dbm,model_lr)
    # saveerror(asvm,dbm,model_svm)
    # saveerror(akmeans,dbm,model_kmeans)

    #保存结果
    with open(name, 'a',encoding='utf-8') as f:
        f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
        f.write('KNN,'+str(pknn)+','+str(rknn)+','+str(fknn)+'\n')
        f.write('DT,'+str(pdt)+','+str(rdt)+','+str(fdt)+'\n')
        f.write('NB,'+str(pnb)+','+str(rnb)+','+str(fnb)+'\n')
        # f.write('RF,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
        f.write('LR,'+str(plr)+','+str(rlr)+','+str(flr)+'\n')
        f.write('SVM,'+str(psvm)+','+str(rsvm)+','+str(fsvm)+'\n')
        # f.write('Kmeans,'+str(pkmeans)+','+str(rkmeans)+','+str(fkmeans)+'\n')
        f.write('-----------------------------------------'+'\n')
        f.write('Modelstd,'+'P-std'+','+'R-std'+','+'F-std'+'\n')
        f.write('KNN,'+str(pknnstd)+','+str(rknnstd)+','+str(fknnstd)+'\n')
        f.write('DT,'+str(pdtstd)+','+str(rdtstd)+','+str(fdtstd)+'\n')
        f.write('NB,'+str(pnbstd)+','+str(rnbstd)+','+str(fnbstd)+'\n')
        # f.write('RF,'+str(prfstd)+','+str(rrfstd)+','+str(frfstd)+'\n')
        f.write('LR,'+str(plrstd)+','+str(rlrstd)+','+str(flrstd)+'\n')
        f.write('SVM,'+str(psvmstd)+','+str(rsvmstd)+','+str(fsvmstd)+'\n')
        # f.write('Kmeans,'+str(pkmeansstd)+','+str(rkmeansstd)+','+str(fkmeansstd)+'\n')
        f.close()

def runexp():
    IDTC= "IDTC"
    IDUC= "IDUC"
    TCCC= "TCCC"
    UCCC= "UCCC"
    UCTC= "UCTC"
    iTrust= "iTrust"
    SMOS= "SMOS"
    #霍普金斯统计量
    cb.mainrun()
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
    
    #----------------数据平衡与模型运行----------------
    #----------------IDTC----------------
    #欠采样
    cv("./CVres/RUS/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.undersmapling,IDTC)
    cv("./CVres/Tomeklink/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.TomekLink,IDTC)
    cv("./CVres/NearMiss/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.nearmiss,IDTC)
    #过采样
    cv("./CVres/ROS/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.Randomos,IDTC)
    cv("./CVres/SMOTE/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.smote,IDTC)
    #综合采样
    cv("./CVres/SMOTEENN/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.smotenn,IDTC)
    cv("./CVres/SMOTETomeklink/IDTC.csv",X_dataset_ID_TC, y_dataset_ID_TC,db.Smote_Tomek,IDTC)
    #---------------IDUC----------------
    #欠采样
    cv("./CVres/RUS/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.undersmapling,IDUC)
    cv("./CVres/Tomeklink/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.TomekLink,IDUC)
    cv("./CVres/NearMiss/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.nearmiss,IDUC)
    #过采样
    cv("./CVres/ROS/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.Randomos,IDUC)
    cv("./CVres/SMOTE/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.smote,IDUC)
    #综合采样
    cv("./CVres/SMOTEENN/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.smotenn,IDUC)
    cv("./CVres/SMOTETomeklink/IDUC.csv",X_dataset_ID_UC, y_dataset_ID_UC,db.Smote_Tomek,IDUC)
    #---------------TCCC----------------

    #欠采样
    cv("./CVres/RUS/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.undersmapling,TCCC)
    cv("./CVres/Tomeklink/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.TomekLink,TCCC)
    cv("./CVres/NearMiss/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.nearmiss,TCCC)
    #过采样
    cv("./CVres/ROS/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.Randomos,TCCC)
    cv("./CVres/SMOTE/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.smote,TCCC)
    #综合采样
    cv("./CVres/SMOTEENN/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.smotenn,TCCC)
    cv("./CVres/SMOTETomeklink/TCCC.csv",X_dataset_TC_CC, y_dataset_TC_CC,db.Smote_Tomek,TCCC)
    
    #---------------UCCC----------------
    #欠采样
    cv("./CVres/RUS/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.undersmapling,UCCC)
    cv("./CVres/Tomeklink/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.TomekLink,UCCC)
    cv("./CVres/NearMiss/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.nearmiss,UCCC)
    #过采样
    cv("./CVres/ROS/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.Randomos,UCCC)
    cv("./CVres/SMOTE/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.smote,UCCC)
    #综合采样
    cv("./CVres/SMOTEENN/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.smotenn,UCCC)
    cv("./CVres/SMOTETomeklink/UCCC.csv",X_dataset_UC_CC, y_dataset_UC_CC,db.Smote_Tomek,UCCC)

    #---------------UCTC----------------
    #欠采样
    cv("./CVres/RUS/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.undersmapling,UCTC)
    cv("./CVres/Tomeklink/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.TomekLink,UCTC)
    cv("./CVres/NearMiss/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.nearmiss,UCTC)
    #过采样
    cv("./CVres/ROS/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.Randomos,UCTC)
    cv("./CVres/SMOTE/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.smote,UCTC)
    #综合采样
    cv("./CVres/SMOTEENN/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.smotenn,UCTC)
    cv("./CVres/SMOTETomeklink/UCTC.csv",X_dataset_UC_TC, y_dataset_UC_TC,db.Smote_Tomek,UCTC)
    
    #---------------SMOS---------------- 

    #欠采样
    cv("./CVres/RUS/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.undersmapling,SMOS)
    cv("./CVres/Tomeklink/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.TomekLink,SMOS)
    cv("./CVres/NearMiss/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.nearmiss,SMOS)
    #过采样
    cv("./CVres/ROS/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.Randomos,SMOS)
    cv("./CVres/SMOTE/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.smote,SMOS)
    #综合采样
    cv("./CVres/SMOTEENN/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.smotenn,SMOS)
    cv("./CVres/SMOTETomeklink/SMOS.csv",X_dataset_SMOS, y_dataset_SMOS,db.Smote_Tomek,SMOS)


    #---------------iTrust---------------- 
    #欠采样
    cv("./CVres/RUS/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.undersmapling,iTrust)
    cv("./CVres/Tomeklink/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.TomekLink,iTrust) ####
    cv("./CVres/NearMiss/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.nearmiss,iTrust)
    #过采样
    cv("./CVres/ROS/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.Randomos,iTrust)
    cv("./CVres/SMOTE/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.smote,iTrust)
    #综合采样
    cv("./CVres/SMOTEENN/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.smotenn,iTrust)
    cv("./CVres/SMOTETomeklink/iTrust.csv",X_dataset_iTrust, y_dataset_iTrust,db.Smote_Tomek,iTrust)

    
    #错误率统计检验

if __name__ == '__main__':
    #开始计时
    time_start=time.time()
    runexp()
    #结束计时
    time_end=time.time()
    #计算运行时间
    print('time cost',time_end-time_start,'s')
    


    