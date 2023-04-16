## IR特征说明

0,1 -VSM
2,3 -LSI
4,5 -LDA
6,7 -BM25

label：0 -无效链接

label：1 -有效链接

## 模型参数说明

>随机森林

RandomForestClassifier(

bootstrap=True, 

ccp_alpha=0.0, 

class_weight=None,#这个参数主要是用于样本不平衡数据集，当设置为None时，所有类别样本权重都为1。也可以利用列表或者字典手动设置各个类别样本的权重，将样本较少的类别赋予更大的权重。

criterion='gini', #字符串类型，默认值为 ‘gini’。这个参数指定划分子树的评估标准：1.‘entropy’，使用基于信息熵的方法，即计算信息增益；2.‘gini’，使用基尼系数（Gini Impurity）

max_depth=None, #最大深度

max_features='auto',#此参数用于限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。

max_leaf_nodes=None, #这个参数通过限制树的最大叶子数量来防止过拟合

max_samples=None,

min_impurity_decrease=0.0, 

min_impurity_split=None,

min_samples_leaf=1,#数值型，默认值1，指定每个叶子结点包含的最少的样本数 

min_samples_split=2,#数值型，默认值2，指定每个内部节点

min_weight_fraction_leaf=0.0, 

n_estimators=100,#默认100，指定树弱分类器个数，值越大越精确,推荐的参数值为：[120, 300, 500, 800, 1200]

>KNN

n_neighbors：KNN中的k值，默认为5（对于k值的选择，前面已经给出解释）；

weights：用于标识每个样本的近邻样本的权重，可选择"uniform",“distance” 或自定义权重。默认"uniform"，所有最近邻样本权重都一样。如果是"distance"，则权重和距离成反比例；如果样本的分布是比较成簇的，即各类样本都在相对分开的簇中时，我们用默认的"uniform"就可以了，如果样本的分布比较乱，规律不好寻找，选择"distance"是一个比较好的选择；

algorithm：限定半径最近邻法使用的算法，可选‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’。

>LR

参考链接：
[LR参数](https://blog.csdn.net/weixin_50304531/article/details/109717609)

>K-means

n_clusters：整形，缺省值=8 【生成的聚类数，即产生的质心（centroids）数。】

max_iter：整形，缺省值=300
执行一次k-means算法所进行的最大迭代数。

n_init：整形，缺省值=10
用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果。
init：有三个可选值：’k-means++’， ‘random’，或者传递一个ndarray向量。

>SVM
可以通过调节class_weight参数来调节不同类别的权重，平衡数据增强泛化能力

```python
sklearn.svm.SVC(C=1.0, 
kernel='rbf', degree=3, 
gamma='auto', coef0=0.0, 
shrinking=True, probability=False,
tol=0.001, cache_size=200, 
class_weight=None, 
verbose=False, max_iter=-1, 
decision_function_shape=None,
random_state=None)
```
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
```

  　　0 – 线性：u'v

 　　 1 – 多项式：(gamma*u'*v + coef0)^degree

  　　2 – RBF函数：exp(-gamma|u-v|^2)

  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
```
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。

gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features

coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

probability ：是否采用概率估计？.默认为False

shrinking ：是否采用shrinking heuristic方法，默认为true

tol ：停止训练的误差值大小，默认为1e-3

cache_size ：核函数cache缓存大小，默认为200

class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)

verbose ：允许冗余输出？

max_iter ：最大迭代次数。-1为无限制。

decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3

random_state ：数据洗牌时的种子值，int值

>GBDT

```python
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
```

参数参考链接：
[参数参考](https://blog.csdn.net/VariableX/article/details/107200334)

>其他部分请自行查询sklearn官方文档

[Sklearn官方文档超链接](https://scikit-learn.org/stable/user_guide.html)

