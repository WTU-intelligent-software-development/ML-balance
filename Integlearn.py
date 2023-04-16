# -*-coding:utf-8-*-
# @Time   : 2022/11/20 16:20
# @Author : 王梓涵
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
import data_into
import data_blance
import Classfier
import ROC_PR
import xgboost as xgb
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.ensemble import BalancedBaggingClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import data_into as dt
import time


# def resultcv(model,X,y):
    #10折交叉验证得分
    # p=[]
    # sp = cross_val_score(model,X,y, cv=10,scoring='precision',verbose=0)
    # p.append(sp)
    # p1=np.mean(p)
    # r=[]
    # sr = cross_val_score(model,X,y, cv=10,scoring='recall',verbose=0)
    # r.append(sr)
    # r1=np.mean(r)
    # f=[]
    # sf = cross_val_score(model,X,y, cv=10,scoring='f1',verbose=0)
    # f.append(sf)
    # f1=np.mean(f)
    #对于10次交叉验证的RMSE
    # return p1,r1,f1#,Rmseavg
def resultcv(model,Xtrain,ytrain,Xtest,ytest):
    ##最终结果
    modelf=model.fit(Xtrain,ytrain)
    yp=modelf.predict(Xtest)
    #得到p，r，f
    pfinal= precision_score(ytest,yp,average='binary')
    rfinal= recall_score(ytest, yp, average='binary')
    ffinal= f1_score(ytest, yp, average='binary')
    return pfinal,rfinal,ffinal



def runensemble(name,X1,y1,dataset):
    #数据集划分
    X11,Xt,y11,yt=train_test_split(X1,y1,test_size=0.2,random_state=32)
    #随机森林
    rf=RandomForestClassifier(random_state=32)
    rf=rf.fit(X11,y11)
    prf,rrf,frf=resultcv(rf,X11,y11,Xt,yt)

    # #平衡随机森林
    # clf = BalancedRandomForestClassifier(random_state=32)
    # clf.fit(X11,y11)  
    # pbrf,rbrf,fbrf=resultcv(clf,X11,y11,Xt,yt)

    # # #RUSboost
    # rusboost = RUSBoostClassifier(random_state=32)
    # rusboost.fit(X11,y11)
    # prus,rrus,frus=resultcv(rusboost,X11,y11,Xt,yt)

    # #平衡bagging
    # bbc = BalancedBaggingClassifier(random_state=32)
    # bbc.fit(X11,y11)
    # pbb,rbb,fbb=resultcv(bbc,X11,y11,Xt,yt)

    #GBDT
    gbdt = GradientBoostingClassifier(verbose=0,random_state=32)
    gbdt.fit(X11,y11)
    pgbdt,rgbdt,fgbdt=resultcv(gbdt,X11,y11,Xt,yt)

    #XGBoost
    # xgb = xgb.XGBClassifier(random_state=32)
    # xgb.fit(X,y)
    # pxgb,rxgb,fxgb=resultcv(xgb,X,y)


    with open(name, 'a',encoding='utf-8') as f:
        f.write('数据集：'+dataset+'\n')
        f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
        f.write('随机森林,'+str(prf)+','+str(rrf)+','+str(frf)+'\n')
        f.write('GBDT,'+str(pgbdt)+','+str(rgbdt)+','+str(fgbdt)+'\n')
        # f.write('平衡随机森林：'+str(pbrf)+','+str(rbrf)+','+str(fbrf)+'\n')
        # f.write('RUSboost：'+str(prus)+','+str(rrus)+','+str(frus)+'\n')
        # f.write('平衡bagging：'+str(pbb)+','+str(rbb)+','+str(fbb)+'\n')
        # f.write('GBDT：'+str(pgbdt)+','+str(rgbdt)+','+str(fgbdt)+'\n')

def runexp():
    #IDTC= "IDTC"
    IDUC= "IDUC"
    #TCCC= "TCCC"
    UCCC= "UCCC"
    UCTC= "UCTC"
    iTrust= "iTrust"
    SMOS= "SMOS"
    #----------------读取数据----------------
    #filepath1='./data/Data_ID_TC.csv'
    filepath2='./data/Data_ID_UC.csv'
    #filepath3='./data/Data_TC_CC.csv'
    filepath4='./data/Data_UC_CC.csv'
    filepath5='./data/Data_UC_TC.csv'
    filepath6='./data/Data_iTrust.csv'
    filepath7='./data/Data_SMOS.csv'
    #X_dataset_ID_TC, y_dataset_ID_TC,data_ID_TC = dt.data_into(filepath1)

    X_dataset_ID_UC, y_dataset_ID_UC,data_ID_UC = dt.data_into(filepath2)

    #X_dataset_TC_CC, y_dataset_TC_CC,data_TC_CC = dt.data_into(filepath3)

    X_dataset_UC_CC, y_dataset_UC_CC,data_UC_CC = dt.data_into(filepath4)

    X_dataset_UC_TC, y_dataset_UC_TC,data_UC_TC = dt.data_into(filepath5)

    #代码制品

    X_dataset_iTrust, y_dataset_iTrust,data_iTrust = dt.data_into(filepath6)

    X_dataset_SMOS, y_dataset_SMOS,data_SMOS = dt.data_into(filepath7)

    #IDTC
    #runensemble('./CVres/ensemble/IDTCensmble.txt',X_dataset_ID_TC,y_dataset_ID_TC,IDTC)

    #IDUC

    runensemble('./CVres/ensemble/IDUCensmble.txt',X_dataset_ID_UC,y_dataset_ID_UC,IDUC)


    #TCCC

    #runensemble('./CVres/ensemble/TCCCensmble.txt',X_dataset_TC_CC,y_dataset_TC_CC,TCCC)

    #UCCC

    runensemble('./CVres/ensemble/UCCCensmble.txt',X_dataset_UC_CC,y_dataset_UC_CC,UCCC)

    #UCTC

    runensemble('./CVres/ensemble/UCTCensmble.txt',X_dataset_UC_TC,y_dataset_UC_TC,UCTC)

    #iTrust

    runensemble('./CVres/ensemble/iTrustensmble.txt',X_dataset_iTrust,y_dataset_iTrust,iTrust)

    #SMOS

    runensemble('./CVres/ensemble/SMOSensmble.txt',X_dataset_SMOS,y_dataset_SMOS,SMOS)


if __name__ == '__main__':
    #开始计时
    time_start=time.time()
    runexp()
    #结束计时
    time_end=time.time()
    #计算运行时间
    print('time cost',time_end-time_start,'s')

   

    

