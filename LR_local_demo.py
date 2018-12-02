import pandas as pd
import numpy as np

####################################
table = pd.read_csv("hot.csv")

#开始切割样本——train训练集1：1均衡，test原始分布
#切分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.3, random_state=5)  #train_test_split方法分割数据集

#del table
train_total = pd.concat([X_train,y_train],axis=1)
table_sample_n = train_total[train_total['y']==0].sample(n=y_train[y_train==1].sum(),axis=0,random_state=20)

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
#LR模型
#特征数值归一化
from sklearn.preprocessing import Normalizer
sc = Normalizer()
sc.fit(X_train_sam)
X_train_std = sc.transform(X_train_sam)

sc.fit(X_test)
X_test_std = sc.transform(X_test)
####################################

####################################
#打印LR模型输出的概率值
#LR模型使用，打印概率
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1,penalty='l1', random_state=0)
lr.fit(X_train_std, y_train_sam)
pro = lr.predict_proba(X_test)   #预测概率值
y_pred = lr.predict(X_test_std)    #预测值

from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))

from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,y_pred)
print "auc: "+str(test_auc)
####################################

####################################
#打印混淆矩阵
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
#绘制ROC曲线
import matplotlib.pyplot as plt

def get_auc(y_true, y_pred_pos_prob, plot_ROC=False):
    """计算 AUC 值。
    Args:
        y_true: 真实标签，如 [0, 1, 1, 1, 0]
        y_pred_pos_prob: 预测每个样本为 positive 的概率。
        plot_ROC: 是否绘制  ROC 曲线。
    Returns:
       roc_auc: AUC 值.
       fpr, tpr, thresholds: see roc_curve.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_pos_prob)
    roc_auc = metrics.auc(fpr, tpr)  # auc 值
    if plot_ROC:
        plt.plot(fpr, tpr, '-*', lw=1, label='auc=%g' % roc_auc)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('FM_ROC.jpg')
    return roc_auc, fpr, tpr, thresholds

roc_auc,fpr,tpr,thresholds = get_auc(y_test,y_pred_proba,plot_ROC=True)
####################################


####################################
#观察LR系数，代表特征权重
weight_list = lr.coef_
feature_list = []
for j in X_not_y.columns:
    feature_list.append(j)
print(len(weight_list[0]))
print(len(feature_list))
weight_dit={}

for i in range(len(feature_list)):
    if weight_list[0][i]!=0:
        weight_dit[feature_list[i]] = weight_list[0][i]

sort_dit = sorted(weight_dit.items(),key = lambda x:x[1],reverse = True)

##LR模型排序结果，写出文件
outfile = open('LR_1+2_feature_300w_chi2.txt', 'w')
for f in range(len(sort_dit)):
    outfile.write("%2d) %-*s %f" % (f + 1, 70, sort_dit[f][0], sort_dit[f][1]))
    outfile.write("\n")
outfile.close()
####################################
