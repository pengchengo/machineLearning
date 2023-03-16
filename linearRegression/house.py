#!/usr/local/bin/python3

# 导入数据处理库
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 导入数据
path = './house_data.csv'
data = pd.read_csv(path)

# 特征缩放 （x-平均值）/标准差
data = (data - data.mean())/data.std()
# 查看特征缩放后的数据
data.head(10)
# 绘制数据散点图
#data.plot(kind = 'scatter', x = 'CRIM', y = 'MEDV')
#data.plot(kind = 'scatter', x = 'ZN', y = 'MEDV')
#plt.show()

# 变量初始化
# 最后一列为y，其余为x
cols = data.shape[1] #列数 shape[0]行数 [1]列数
X = data.iloc[:,0:cols-1]       #取前cols-1列，即输入向量
y = data.iloc[:,cols-1:cols]    #取最后一列，即目标变量
#print(X.head(10))

# 划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# 将数据转换成numpy矩阵
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)
# 初始化theta矩阵
theta = np.matrix([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#添加偏置列，值为1，axis = 1 添加列 
X_train = np.insert(X_train, 0, 1, axis=1) 
X_test = np.insert(X_test,0,1,axis=1)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# 代价函数
def CostFunction(X,y,theta):
    inner = np.power(X*theta.T-y, 2)
    return np.sum(inner)/(2*len(X))

# 正则化代价函数
def regularizedcost(X,y,theta,l):
    reg = (l/(2*len(X)))*(np.power(theta, 2).sum())    
    return CostFunction(X,y,theta) + reg

# 梯度下降
def GradientDescent(X,y,theta,l,alpha,epoch):
    temp = np.matrix(np.zeros(np.shape(theta)))   # 定义临时矩阵存储theta
    parameters = int(theta.flatten().shape[1])    # 参数 θ的数量
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m
    for i in range(epoch):
        # 利用向量化一步求解
        temp = theta - (alpha / m) * (X * theta.T - y).T * X - (alpha*l/m)*theta     # 添加了正则项
        theta = temp
        cost[i] = regularizedcost(X, y, theta, l)      # 记录每次迭代后的代价函数值
    return theta,cost   

alpha = 0.01  #学习速率
epoch = 1000  #迭代步数
l = 50      #正则化参数

#运行梯度下降算法 并得出最终拟合的theta值 代价函数J(theta)
final_theta, cost = GradientDescent(X_train, y_train, theta, l, alpha, epoch)
print(final_theta)