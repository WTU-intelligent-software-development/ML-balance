from sklearn.cluster import KMeans
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split

import data_into as dt
import data_blance as db

import numpy as np

def cluban(name,X,y,dbm,dataset):
    #数据集划分
    X1,Xt,y1,yt=train_test_split(X,y,test_size=0.2,random_state=32)
    #数据平衡
    X11,y11=dbm(X1,y1)
    #数据平衡
    model_kmeans=KMeans(n_clusters=2,random_state=32)
    # p=[]
    # sp = cross_val_score(model_kmeans,X1,y1, cv=10,scoring='precision',verbose=0)
    # p.append(sp)
    # p1=np.mean(p)
    # r=[]
    # sr = cross_val_score(model_kmeans,X1,y1, cv=10,scoring='recall',verbose=0)
    # r.append(sr)
    # r1=np.mean(r)
    # f=[]
    # sf = cross_val_score(model_kmeans,X1,y1, cv=10,scoring='f1',verbose=0)
    # f.append(sf)
    # f1=np.mean(f)
    modelf=model_kmeans.fit(X11)
    yp=modelf.predict(Xt)
    p1= precision_score(yt,yp,average='binary')
    r1= recall_score(yt, yp, average='binary')
    f1= f1_score(yt, yp, average='binary')
    yp=modelf.predict(Xt)

    #最终结果
    with open(name, 'a',encoding='utf-8') as f:
        f.write('数据集：'+dataset+'\n')
        f.write('Model,'+'P-cvscore'+','+'R-cvscore'+','+'F-cvscore'+'\n')
        f.write('Kmeans：'+str(p1)+','+str(r1)+','+str(f1)+'\n')
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

    #IDTC
    cluban('./CVres/kmeans/RUS/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.undersmapling,IDTC)
    cluban('./CVres/kmeans/Tomeklink/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.TomekLink,IDTC)
    cluban('./CVres/kmeans/NearMiss/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.nearmiss,IDTC)
    
    cluban('./CVres/kmeans/ROS/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.Randomos,IDTC)
    cluban('./CVres/kmeans/SMOTE/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.smote,IDTC)

    cluban('./CVres/kmeans/SMOTENN/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.smotenn,IDTC)
    cluban('./CVres/kmeans/SMOTETomeklink/IDTCkmeans.txt',X_dataset_ID_TC,y_dataset_ID_TC,db.Smote_Tomek,IDTC)

    #IDUC

    cluban('./CVres/kmeans/RUS/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.undersmapling,IDUC)
    cluban('./CVres/kmeans/Tomeklink/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.TomekLink,IDUC)
    cluban('./CVres/kmeans/NearMiss/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.nearmiss,IDUC)
    
    cluban('./CVres/kmeans/ROS/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.Randomos,IDUC)
    cluban('./CVres/kmeans/SMOTE/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.smote,IDUC)

    cluban('./CVres/kmeans/SMOTENN/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.smotenn,IDUC)
    cluban('./CVres/kmeans/SMOTETomeklink/IDUCkmeans.txt',X_dataset_ID_UC,y_dataset_ID_UC,db.Smote_Tomek,IDUC)

    


    # #TCCC

    cluban('./CVres/kmeans/RUS/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.undersmapling,TCCC)
    cluban('./CVres/kmeans/Tomeklink/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.TomekLink,TCCC)
    cluban('./CVres/kmeans/NearMiss/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.nearmiss,TCCC)

    cluban('./CVres/kmeans/ROS/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.Randomos,TCCC)

    cluban('./CVres/kmeans/SMOTE/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.smote,TCCC)

    cluban('./CVres/kmeans/SMOTENN/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.smotenn,TCCC)

    cluban('./CVres/kmeans/SMOTETomeklink/TCCCkmeans.txt',X_dataset_TC_CC,y_dataset_TC_CC,db.Smote_Tomek,TCCC)

    #UCCC

    cluban('./CVres/kmeans/RUS/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.undersmapling,UCCC)
    cluban('./CVres/kmeans/Tomeklink/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.TomekLink,UCCC)
    cluban('./CVres/kmeans/NearMiss/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.nearmiss,UCCC)

    cluban('./CVres/kmeans/ROS/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.Randomos,UCCC)
    cluban('./CVres/kmeans/SMOTE/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.smote,UCCC)

    cluban('./CVres/kmeans/SMOTENN/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.smotenn,UCCC)
    cluban('./CVres/kmeans/SMOTETomeklink/UCCCkmeans.txt',X_dataset_UC_CC,y_dataset_UC_CC,db.Smote_Tomek,UCCC)

    #UCTC

    cluban('./CVres/kmeans/RUS/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.undersmapling,UCTC)
    cluban('./CVres/kmeans/Tomeklink/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.TomekLink,UCTC)
    cluban('./CVres/kmeans/NearMiss/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.nearmiss,UCTC)

    cluban('./CVres/kmeans/ROS/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.Randomos,UCTC)
    cluban('./CVres/kmeans/SMOTE/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.smote,UCTC)

    cluban('./CVres/kmeans/SMOTENN/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.smotenn,UCTC)
    cluban('./CVres/kmeans/SMOTETomeklink/UCTCkmeans.txt',X_dataset_UC_TC,y_dataset_UC_TC,db.Smote_Tomek,UCTC)



    #iTrust

    cluban('./CVres/kmeans/RUS/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.undersmapling,iTrust)
    cluban('./CVres/kmeans/Tomeklink/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.TomekLink,iTrust)
    cluban('./CVres/kmeans/NearMiss/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.nearmiss,iTrust)
    
    cluban('./CVres/kmeans/ROS/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.Randomos,iTrust)
    cluban('./CVres/kmeans/SMOTE/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.smote,iTrust)

    cluban('./CVres/kmeans/SMOTENN/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.smotenn,iTrust)
    cluban('./CVres/kmeans/SMOTETomeklink/iTrustkmeans.txt',X_dataset_iTrust,y_dataset_iTrust,db.Smote_Tomek,iTrust)

    #SMOS

    cluban('./CVres/kmeans/RUS/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.undersmapling,SMOS)
    cluban('./CVres/kmeans/Tomeklink/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.TomekLink,SMOS)
    cluban('./CVres/kmeans/NearMiss/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.nearmiss,SMOS)

    cluban('./CVres/kmeans/ROS/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.Randomos,SMOS)
    cluban('./CVres/kmeans/SMOTE/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.smote,SMOS)

    cluban('./CVres/kmeans/SMOTENN/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.smotenn,SMOS)
    cluban('./CVres/kmeans/SMOTETomeklink/SMOSkmeans.txt',X_dataset_SMOS,y_dataset_SMOS,db.Smote_Tomek,SMOS)


if __name__ == '__main__':
    #开始计时
    time_start=time.time()
    runexp()
    #结束计时
    time_end=time.time()
    #计算运行时间
    print('time cost',time_end-time_start,'s.txt')

   