# -*-coding:utf-8-*-
# @Time   : 2022/11/13 16:20
# @Author : 王梓涵
from sklearn.metrics import  classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import Classfier as cls

def modeload(model,X,y,dbm,p=[],r=[],f=[]):
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
    print(yp.shape)
    p.append(report[u'1']['precision'])
    r.append(report[u'1']['recall'])
    f.append(report[u'1']['f1-score'])
    return p,r,f

def stdavg(s):
    return (np.std(s)-np.mean(s)*0.1)

def ReportStability(name, X,y,dbm,n):
    """
    函数说明：
    输入参数：
    name：文件名
    X：特征
    y：标签
    n：重复次数
    输出：
    包含KNN、DT、GBAYES、SVM、LR、RF、Kmeans的P、R、F的标准差与0.1倍均值的结果。
    输出在CSV文件中。
    """
    #定义列表
    row={}
    allprecision_knn=[]
    allrecall_knn=[]
    allf_knn=[]
    allprecision_dt=[]
    allrecall_dt=[]
    allf_dt=[]
    allprecision_kmeans=[]
    allrecall_kmeans=[]
    allf_kmeans=[]
    allprecision_nb=[]
    allrecall_nb=[]
    allf_nb=[]
    allprecision_rf=[]
    allrecall_rf=[]
    allf_rf=[]
    allprecision_lr=[]
    allrecall_lr=[]
    allf_lr=[]
    allprecision_svm=[]
    allrecall_svm=[]
    allf_svm=[]
    for i in range(n):
        #----------------KNN----------------
        model_knn=KNeighborsClassifier(n_neighbors=5)
        pknn,rknn,fknn=modeload(model_knn,X,y,dbm,allprecision_knn,allrecall_knn,allf_knn)
        #----------------DT----------------
        model_dt=DecisionTreeClassifier()#随机种子变化
        pdt,rdt,fdt=modeload(model_dt,X,y,dbm,allprecision_dt,allrecall_dt,allf_dt)
        #----------------Kmeans----------------
        model_kmeans=KMeans(n_clusters=2)
        pkmeans,rkmeans,fkmeans=modeload(model_kmeans,X,y,dbm,allprecision_kmeans,allrecall_kmeans,allf_kmeans)
        #----------------NB----------------
        model_nb=GaussianNB()
        pnb,rnb,fnb=modeload(model_nb,X,y,dbm,allprecision_nb,allrecall_nb,allf_nb)
        #----------------RF----------------
        model_rf=RandomForestClassifier()
        prf,rrf,frf=modeload(model_rf,X,y,dbm,allprecision_rf,allrecall_rf,allf_rf)
        #----------------LR----------------
        model_lr=LogisticRegression()
        plr,rlr,flr=modeload(model_lr,X,y,dbm,allprecision_lr,allrecall_lr,allf_lr)
        #----------------SVM----------------
        model_svm=svm.SVC()
        psvm,rsvm,fsvm=modeload(model_svm,X,y,dbm,allprecision_svm,allrecall_svm,allf_svm)
    # #KNN
    row['Model1']='KNN'
    #p标准差-0.1p均值
    row['Precision']=stdavg(pknn)
    #r标准差-0.1r均值
    row['Recall']=stdavg(rknn)
    #f标准差-0.1f均值
    row['F1']=stdavg(fknn)
    #标准差
    row['avgknn1']=np.mean(pknn)
    row['avgknn2']=np.mean(rknn)
    row['avgknn3']=np.mean(fknn)
    row['en1']='\n'

    #DT
    row["Model2"]='DT'
    row['Dtpstd'] = stdavg(pdt)
    row['Dtrstd'] = stdavg(rdt)
    row['Dtfstd'] = stdavg(fdt)
    row['avgdt1']=np.mean(pdt)
    row['avgdt2']=np.mean(rdt)
    row['avgdt3']=np.mean(fdt)
    row['en2']='\n'

    #NB
    row['Model3']='NB'
    row['Nbstdp'] = stdavg(pnb)
    row['Nbstdr'] = stdavg(rnb)
    row['Nbstdf'] = stdavg(fnb)
    row['avgnb1']=np.mean(pnb)
    row['avgnb2']=np.mean(rnb)
    row['avgnb3']=np.mean(fnb)

    row['en3']='\n'


    #SVM
    row["Model4"]='SVM'
    row['Svmstdp'] = stdavg(psvm)
    row['Svmstdr'] = stdavg(rsvm)
    row['Svmstdf'] = stdavg(fsvm)
    row['avgsvm1']=np.mean(psvm)
    row['avgsvm2']=np.mean(rsvm)
    row['avgsvm3']=np.mean(fsvm)

    row['en4']='\n'

    #LR
    row["Model5"]='LR'
    row['Lrstdp'] = stdavg(plr)
    row['Lrstdr'] = stdavg(rlr)
    row['Lrstdf'] = stdavg(flr)
    row['avglr1']=np.mean(plr)
    row['avglr2']=np.mean(rlr)
    row['avglr3']=np.mean(flr)
    row['en5']= '\n'

    #RF
    row["Model6"]='RF'
    row['Rfstdp'] = stdavg(prf)
    row['Rfstdr'] = stdavg(rrf)
    row['Rfstdf'] = stdavg(frf)
    row['avgrf1']=np.mean(prf)
    row['avgrf2']=np.mean(rrf)
    row['avgrf3']=np.mean(frf)

    row['en6']='\n'
    #Kmeans
    row["Model7"]='Kmeans'
    row['Kmeansstdp'] = stdavg(pkmeans)
    row['Kmeansstdr'] = stdavg(rkmeans)
    row['Kmeansstdf'] = stdavg(fkmeans)
    row['avgkmeans1']=np.mean(pkmeans)
    row['avgkmeans2']=np.mean(rkmeans)
    row['avgkmeans3']=np.mean(fkmeans)
    row['en7']='\n'

    #dict转换为dataframe
    df = pd.DataFrame(row,index=[0]
                #index=['Model','Standard deviation Precision','Standard deviation Recall ',
                #'Standard deviation Fmeasure','0.1 * Average Precision','0.1 * Average Recall','0.1 * Average Fmeasure']
                )
    #dataframe转换为csv
    df.to_csv(name, mode='a',header=0,encoding='utf-8')
    