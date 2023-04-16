import pandas as pd
from numpy.random import uniform,normal
from scipy.spatial.distance import cdist
import Stability as st
import data_into as dt
import data_blance as db
import clustingstab as cb
 
 
#n维霍普金斯统计量计算，input:DataFrame类型的二维数据，output:float类型的霍普金斯统计量
#默认从数据集中抽样的比例为0.3
def hopkins_statistic(data:pd.DataFrame,sampling_ratio:float = 0.3,):
    #抽样比例超过0.1到0.5区间任意一端则用端点值代替
    sampling_ratio = min(max(sampling_ratio,0.1),0.5)
    #抽样数量
    n_samples = int(data.shape[0] * sampling_ratio)
    #原始数据中抽取的样本数据
    sample_data = data.sample(n_samples)
    #原始数据抽样后剩余的数据
    data = data.drop(index = sample_data.index) #,inplace = True)
    #原始数据中抽取的样本与最近邻的距离之和
    data_dist = cdist(data,sample_data).min(axis = 0).sum()
    #人工生成的样本点，从平均分布中抽样(artificial generate samples)
    ags_data = pd.DataFrame({col:uniform(data[col].min(),data[col].max(),n_samples)\
                             for col in data})
    #人工样本与最近邻的距离之和
    ags_dist = cdist(data,ags_data).min(axis = 0).sum()
    #计算霍普金斯统计量H
    H_value = ags_dist / (data_dist + ags_dist)
    return H_value

def balancehopkins(dbm,X,y,name):
    i=1
    X1,y1=dbm(X,y)
    with open('./CVres/banlancehopkins.txt','a+',encoding='utf-8') as f:
        f.write(str(dbm.__name__)+"+"+name+"霍普金斯统计量"+str(hopkins_statistic(X1))+'\n')
        f.close()
        

def nonebalancehopkins(X,name):
    with open('./CVres/nonebanlancehopkins.txt','a+',encoding='utf-8') as f:
        f.write(name+"霍普金斯统计量"+str(hopkins_statistic(X))+'\n')
        f.close()
        
# 主函数
def mainrun():
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

    IDTC= "IDTC"
    IDUC= "IDUC"
    TCCC= "TCCC"
    UCCC= "UCCC"
    UCTC= "UCTC"
    iTrust= "iTrust"
    SMOS= "SMOS"

    nonebalancehopkins(X_dataset_ID_TC,IDTC)
    nonebalancehopkins(X_dataset_ID_UC,IDUC)
    nonebalancehopkins(X_dataset_TC_CC,TCCC)
    nonebalancehopkins(X_dataset_UC_CC,UCCC)
    nonebalancehopkins(X_dataset_UC_TC,UCTC)
    nonebalancehopkins(X_dataset_iTrust,iTrust)
    nonebalancehopkins(X_dataset_SMOS,SMOS)

    #balanced

    balancehopkins(db.undersmapling,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.undersmapling,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.undersmapling,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.undersmapling,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.undersmapling,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.undersmapling,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.undersmapling,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    balancehopkins(db.TomekLink,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.TomekLink,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)  
    balancehopkins(db.TomekLink,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.TomekLink,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.TomekLink,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.TomekLink,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.TomekLink,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    balancehopkins(db.nearmiss,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.nearmiss,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.nearmiss,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.nearmiss,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.nearmiss,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.nearmiss,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.nearmiss,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    
    balancehopkins(db.Randomos,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.Randomos,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.Randomos,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.Randomos,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.Randomos,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.Randomos,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.Randomos,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    balancehopkins(db.smote,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.smote,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.smote,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.smote,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.smote,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.smote,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.smote,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    balancehopkins(db.Smote_Tomek,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.Smote_Tomek,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.Smote_Tomek,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.Smote_Tomek,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.Smote_Tomek,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.Smote_Tomek,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.Smote_Tomek,X_dataset_SMOS,y_dataset_SMOS,SMOS)

    balancehopkins(db.smotenn,X_dataset_ID_TC,y_dataset_ID_TC,IDTC)
    balancehopkins(db.smotenn,X_dataset_ID_UC,y_dataset_ID_UC,IDUC)
    balancehopkins(db.smotenn,X_dataset_TC_CC,y_dataset_TC_CC,TCCC)
    balancehopkins(db.smotenn,X_dataset_UC_CC,y_dataset_UC_CC,UCCC)
    balancehopkins(db.smotenn,X_dataset_UC_TC,y_dataset_UC_TC,UCTC)
    balancehopkins(db.smotenn,X_dataset_iTrust,y_dataset_iTrust,iTrust)
    balancehopkins(db.smotenn,X_dataset_SMOS,y_dataset_SMOS,SMOS)
    
if __name__ == '__main__':
    mainrun()