# -*-coding:utf-8-*-
# @Time   : 2022/10/16 10:10
# @Author : 王梓涵

#导入sklearn库
from sklearn.ensemble import RandomForestClassifier
# Path: main.py
from sklearn.neighbors import KNeighborsClassifier
#
from sklearn.linear_model import LogisticRegression
#
from sklearn.cluster import KMeans
#
from sklearn.ensemble import GradientBoostingClassifier
#
from sklearn.ensemble import AdaBoostClassifier  
#
from sklearn.tree import DecisionTreeClassifier
#
from sklearn import svm
#
from sklearn.naive_bayes import MultinomialNB
#
import xgboost as xgb
#
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
#
from sklearn.neural_network import MLPClassifier 
#
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM  
from sklearn.pipeline import Pipeline 
#

import matplotlib.pyplot as plt

#
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import calinski_harabasz_score
from xgboost import plot_importance
from sklearn.metrics import accuracy_score

##分割数据集
def trainsplit(X_dataset, y_dataset):
    X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=32,test_size=0.1)
    return X_train,y_train,X_test,y_test

def trainsplit_stability(X_dataset, y_dataset,i):
    X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=i,test_size=0.1)
    return X_train,y_train,X_test,y_test


#随机森林 参数请参考技术文档

def RF(X_train, y_train, X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=32,test_size=0.1)
    model_RF=RandomForestClassifier(n_estimators=100,random_state=32)
    model_RF.fit(X_train,y_train)
    y_pred_RF = model_RF.predict(X_test)
    score=model_RF.score(X_test,y_test)
    print("Score：", score)
    y_score_RF = model_RF.predict_proba(X_test)[:,1]
    y_test_RF=y_test
    print(classification_report(y_test, y_pred_RF))
    return y_pred_RF,y_score_RF,y_test_RF

#KNN,输入数据集与k

def KNN(X_train, y_train, X_test,y_test,k):
    #X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=32, test_size=0.1)
    model_Knn = KNeighborsClassifier(n_neighbors=k)
    model_Knn .fit(X_train, y_train)
    y_pred_Knn = model_Knn .predict(X_test)
    score=model_Knn.score(X_test,y_test)
    print("Score：", score)
    y_test_Knn=y_test
    y_score_Knn = model_Knn.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_Knn))
    return y_pred_Knn,y_score_Knn,y_test_Knn

#LR,输入数据集

def LR(X_train, y_train,X_test,y_test ):
    ##X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=50,test_size=0.1)
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)
    score=model_LR.score(X_test,y_test)
    print("Score：", score)
    y_test_LR=y_test
    y_score_LR = model_LR.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_LR))
    return y_pred_LR,y_score_LR,y_test_LR

#K-means

def Kmeans(X_dataset, y_dataset):
    #print(type(X_dataset.values))
    model_Kmeans = KMeans(n_clusters=2)
    model_Kmeans.fit(X_dataset)
    y_pred_Kmeans = model_Kmeans.predict(X_dataset)
    score = calinski_harabasz_score(X_dataset,y_pred_Kmeans)
    plt.scatter(X_dataset.values[:,0],X_dataset.values[:,1],c=y_pred_Kmeans)
    plt.text(0,1,'k=%d, ch-score: %.2f' % (2, score))
    #plt.savefig('kmeans.png')
    plt.show()
    #y_score_Kmeans = model_Kmeans.predict_proba(X_dataset)
    print(classification_report(y_dataset, y_pred_Kmeans))
    return y_pred_Kmeans,#y_score_Kmeans

#GBDT

def GBDT(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=32,test_size=0.1)
    model_GBDT = GradientBoostingClassifier(n_estimators=120,verbose=1,random_state=32)
    model_GBDT.fit(X_train, y_train)
    y_pred_GBDT = model_GBDT.predict(X_test)
    score=model_GBDT.score(X_test,y_test)
    print("Score：", score)
    y_score_GBDT = model_GBDT.predict_proba(X_test)[:,1]
    y_test_GBDT=y_test
    print(classification_report(y_test, y_pred_GBDT))
    return y_pred_GBDT,y_score_GBDT,y_test_GBDT

#AdaBoost

def AdaBoost(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=46,test_size=0.2)
    #迭代120次 ,学习率为0.01
    model_Adaboost = AdaBoostClassifier(n_estimators=240,learning_rate=0.1)
    model_Adaboost.fit(X_train,y_train)
    y_pred_Adaboost = model_Adaboost.predict( X_test)
    score=model_Adaboost.score(X_test,y_test)
    print("Score：", score)
    y_test_Adaboost=y_test
    y_score_Adaboost = model_Adaboost.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_Adaboost))
    return y_pred_Adaboost,y_score_Adaboost,y_test_Adaboost

#DT

def DT(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, random_state=32,test_size=0.1)
    model_DT = DecisionTreeClassifier(random_state=32)
    model_DT.fit(X_train, y_train)
    y_pred_DT = model_DT.predict(X_test)
    score=model_DT.score(X_test,y_test)
    print("Score：", score)
    y_test_DT=y_test
    y_score_DT = model_DT.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_DT))
    return y_pred_DT,y_score_DT,y_test_DT

#SVM,输入数据集
def SVM(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    # 分类器
    clf = svm.SVC(kernel="linear",probability=True)   # 参数kernel为线性核函数
    clf.fit(X_train, y_train)  # 训练分类器
    print("Support Vector：\n", clf.n_support_)  # 每一类中属于支持向量的点数目
    y_pred_SVM = clf.predict(X_test)
    score = clf.score(X_test, y_test)  # 模型得分
    print("Score：", score)
    y_score_SVM = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_SVM))
    y_test_SVM=y_test
    return y_pred_SVM,y_score_SVM,y_test_SVM

#XGboost

def XGBoost(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    plst = list(params.items())


    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 500
    model = xgb.train(plst, dtrain, num_rounds)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred_XGboost = model.predict(dtest)
    y_test_XGboost=y_test
    accuracy = accuracy_score(y_test,y_pred_XGboost)
    print("accuarcy: %.2f%%" % (accuracy*100.0))
    # 显示重要特征
    plot_importance(model)
    plt.show()
    #print(y_pred_XGboost)
    y_score_XGboost = model.predict(dtest)
    print(classification_report(y_test, y_pred_XGboost))
    return y_pred_XGboost,y_score_XGboost,y_test_XGboost

#bayes-MultinomialNB
#如果样本特征的大部分是多元离散值，使用MultinomialNB比较合适。
def mbayes(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    model_mbayes = MultinomialNB(alpha=0.01)
    model_mbayes.fit(X_train, y_train)
    y_pred_mbayes =model_mbayes.predict(X_test)
    score=model_mbayes.score(X_test,y_test)
    print("Score：", score)
    y_score_mbayes = model_mbayes.predict_proba(X_test)[:,1]
    y_test_mbayes=y_test
    print(classification_report(y_test, y_pred_mbayes))
    return y_pred_mbayes,y_score_mbayes,y_test_mbayes

#bayes-BernoulliNB
#如果样本特征的大部分是二元离散值，使用BernoulliNB比较合适。

def bbayes(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    model_bbayes = BernoulliNB(alpha=0.01)
    model_bbayes.fit(X_train, y_train)
    y_pred_bbayes =model_bbayes.predict(X_test)
    score=model_bbayes.score(X_test,y_test)
    print("Score：", score)
    y_test_bbayes=y_test
    y_score_bbayes = model_bbayes.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_bbayes))
    return y_pred_bbayes,y_score_bbayes,y_test_bbayes

#bayes-GaussianNB
#如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。

def gbayes(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    model_gbayes = GaussianNB()
    model_gbayes.fit(X_train, y_train)
    y_pred_gbayes =model_gbayes.predict(X_test)
    score=model_gbayes.score(X_test,y_test)
    print("Score：", score)
    y_test_gbayes=y_test
    y_score_gbayes = model_gbayes.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_gbayes))
    return y_pred_gbayes,y_score_gbayes,y_test_gbayes


#多层感知机MLP

def MLP(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    model_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model_MLP.fit(X_train, y_train)
    y_pred_MLP =model_MLP.predict(X_test)
    score=model_MLP.score(X_test,y_test)
    print("Score：", score)
    y_score_MLP = model_MLP.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_MLP))
    y_test_MLP=y_test
    return y_pred_MLP,y_score_MLP,y_test_MLP

#受限玻尔兹曼机RBM与LR

def RBM(X_train, y_train,X_test,y_test):
   # X_train,X_test,y_train,y_test = train_test_split(X_dataset, y_dataset, test_size=0.1, random_state=33)
    logistic = linear_model.LogisticRegression()  
    rbm = BernoulliRBM(random_state=0, verbose=True)  
    model_RBM = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])  
    #设置一些参数
    rbm.learning_rate = 0.1
    rbm.n_iter = 20   
    rbm.n_components = 100  
    #正则化强度参数
    logistic.C = 1000   
    model_RBM.fit(X_train, y_train)  
    y_pred_RBM = model_RBM.predict(X_test)
    print(classification_report(y_test, y_pred_RBM))
    score=model_RBM.score(X_test,y_test)
    print("Score：", score)
    y_score_RBM = model_RBM.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred_RBM))
    y_test_RBM=y_test
    return y_pred_RBM,y_score_RBM,y_test_RBM

#
