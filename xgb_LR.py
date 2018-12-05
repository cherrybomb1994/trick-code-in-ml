import pandas as pd
import numpy as np

table = pd.read_csv("./chi2_xgb_.csv")

from sklearn.model_selection import train_test_split
y = table["y"]
X_data = table.drop(['y'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=5)

#对于后续的LR，相当于又训练了一个模型，在这里对于train继续切分，一些训练xgb，一些训练LR，这里test_size比较小，因为运行的资源比较少
#后续拼接的时候内存回爆掉
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.3)

#XGB训练，此处使用的试sklearn封装版本，直接是XGBClassifier，也可以使用xgboost自带的booster方法，区别一个是xgb.fit，一个是xgb.train
import xgboost as xgb
import os 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.externals import joblib

import datetime
starTime = datetime.datetime.now()

save_model_path =  '360_xgb.ml'

params={'n_estimators':50, 
        'max_depth':5,
        'subsample':0.75,
        'min_child_weight':50,
        'colsample_bytree':0.75,
        'eta': 0.04,
        'gamma':0.1,
        'lambda':30,
        'seed':0,
        'nthread':8,
        'silent':0,
        'scale_pos_weight':(np.sum(y_train==0)/np.sum(y_train==1))     
       }
xgboost = xgb.XGBClassifier(**params)
print "xgb param done!"
xgboost.fit(X_train, y_train)


y_xgboost_test = xgboost.predict_proba(X_test)[:, 1]
fpr_xgboost, tpr_xgboost, _ = roc_curve(y_test, y_xgboost_test)
auc = roc_auc_score(y_test, y_xgboost_test)
print "xgb AUC:"+str(auc)
print "xgb finish!"
#save model
joblib.dump(xgboost,save_model_path)

endTime = datetime.datetime.now()
print("Spend %.2f S(%.2f Min)" % ((endTime-starTime).seconds,(endTime-starTime).seconds / 60))


#拼接叶子节点，LR部分
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.externals import joblib
xgboost = joblib.load("360_xgb.ml")

xgb_enc = OneHotEncoder()
xgb_lr = LogisticRegression(C=1, penalty='l1')
xgb_enc.fit(xgboost.apply(X_train)[:, :])
#xgb_lr.fit(xgb_enc.transform(xgboost.apply(X_train_lr)[:, :]), y_train_lr)
xgb_lr.fit(np.hstack([xgb_enc.transform(xgboost.apply(X_train_lr)[:, :]).toarray(),X_train_lr.values]), y_train_lr)
print "LR fit!"
joblib.dump(xgb_lr, "360_LR.m")

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_test, y_test, test_size=0.5, random_state=5) 
del X_train_t
del y_train_t
y_xgb_lr_test = xgb_lr.predict_proba(np.hstack([xgb_enc.transform(xgboost.apply(X_test_t)[:, :]).toarray(),X_test_t.values]))[:, 1]

print "LR pred finish!"
fpr_xgb_lr, tpr_xgb_lr, _ = roc_curve(y_test_t, y_xgb_lr_test)
auc = roc_auc_score(y_test_t, y_xgb_lr_test)
print("Xgboost + LR:", auc)
