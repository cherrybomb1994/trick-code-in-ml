import pandas as pd
import numpy as np

####################################
table = pd.read_csv("300w_chi2_1hot.csv")
#使用fastFM，label要求为1与-1
table['y']=table['y'].apply(lambda x: -1 if x==0 else 1)
####################################

####################################
#开始切割样本——jiaru做正负样本真实的分布
#切分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.3, random_state=5)  #train_test_split方法分割数据集

#del table
train_total = pd.concat([X_train,y_train],axis=1)
table_sample_n = train_total[train_total['y']==-1].sample(n=y_train[y_train==1].sum(),axis=0,random_state=20)

table_sample_o = train_total[train_total['y']==1]
table_sample = pd.concat([table_sample_o,table_sample_n])
import numpy as np
table_sample = table_sample.reindex(np.random.permutation(table_sample.index))   #concat之后index需要打乱重排序一下，再切割？
table_sample.sort_index(inplace=True) 
table_sample.index = range(len(table_sample))
y_train =  table_sample['y']
X_train = table_sample.drop(['y'],axis=1)
####################################

####################################
#fastFM的预处理，in转成矩阵，out转为array
import scipy.sparse as sp

X_train = sp.csc_matrix(np.array(X_train))
y_train = np.array(y_train)
#X_train_sam = sp.csc_matrix(np.array(X_train))
#y_train_sam = np.array(y_train)
X_test = sp.csc_matrix(np.array(X_test))
y_test = np.array(y_test)
####################################


####################################
#train，选择SGD下降方式，iter貌似要很大，l2_reg若被赋值，说明一阶二阶均为此系数，l2_reg_V，l2_reg_w单独设定，是一阶二阶独立的系数
#rank是embedding的大小，即为k

from fastFM import sgd

fm = sgd.FMClassification(n_iter=1500000,
                          init_stdev=0,
                          l2_reg=0.001,
                          #l2_reg_V=0.01, l2_reg_w=0.01,
                          rank=20,
                          step_size=0.001)
fm.fit(X_train, y_train)
y_pred = fm.predict(X_test)
y_pred_proba = fm.predict_proba(X_test)
####################################



####################################
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))

from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test, y_pred_proba)
print "auc: "+str(test_auc)
####################################



####################################
#label转回0，1，看混淆矩阵

for i in range(len(y_pred)):
    if y_pred[i]==-1:
        y_pred[i]=0

for i in range(len(y_test)):
    if y_test[i]==-1:
        y_test[i]=0

def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    #labels = [1,0]
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i],"\t",
    print 
    for i in range(len(conf_mat)):
        print i,"\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j],'\t',
        print 
    print 
    
my_confusion_matrix(y_test, y_pred)
####################################

####################################
#看一下auc的排序能力

pro_dit = {}
#index_list = y_test.index
for i in range(len(y_test)):
    pro_dit[i] = y_pred_proba[i]
sort_pro_dit = sorted(pro_dit.items(), key=lambda x: x[1], reverse=True)


count = 0
count_1 = 0
count_2 = 0
for i in range(900000):
    index = sort_pro_dit[i][0]
    if sort_pro_dit[i][1] > 0.75 and y_test[index] == 1:
        count += 1
    if sort_pro_dit[i][1] > 0.75:
        count_2 += 1
    if y_test[index] == 1:
        count_1 += 1

count_total = 0
for i in range(len(y_test)):
    index = sort_pro_dit[i][0]
    if y_test[index] == 1:
        count_total += 1

print(u'排序后前n预测正确：' + str(count))
print(u'排序后前n预测为1：' + str(count_2))
print(u'排序后前n真实为正：' + str(count_1))
print(u'测试集中全部为正： ' + str(count_total))
print('全部样本： '+str(len(y_test)))
####################################
