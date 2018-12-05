#前面有个xgb+LR，其中用的是sklearn封装的版本，此处加一个xgb自己的booster实现
#二者区别之一是，一个用xgb.fit，一个是xgb.trian


    import datetime
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.externals import joblib

    save_model_path = path + '/push_only360sample_xgb.ml'
    tree_path = path + '/push_only360sample_xgb.txt'
    starTime = datetime.datetime.now()
    print("===============String Trainning...================")
#训练集测试集切割
    X_data = result360.drop(['open_id','union_id','user_id','materiel_name','y','rovince1','age_group',
                           'mobile_type', 'city_rank','clicknum'],axis=1)
    y_data = result360["y"]
    #print(X_data)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3,stratify=y_data,random_state=2)
    train_matrix = xgb.DMatrix(X_train,label=y_train)
    eva_matrix = xgb.DMatrix(X_val,label=y_val)

    params={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':6,
        'subsample':0.75,#0.65->
        'min_child_weight':20,#20-->
        'colsample_bytree':0.75,
        'eta': 0.04,
        'gamma':0.1,
        'lambda':30,
        'seed':0,
        'nthread':8,
        'silent':0,
        'scale_pos_weight':(np.sum(y_train==0)/np.sum(y_train==1))
        }
    watchlist  = [(train_matrix,'train'),(eva_matrix,'eval')]
    #model=xgb.train(params,train_matrix,num_boost_round=179,evals=watchlist,early_stopping_rounds=100)
    dtrain = xgb.DMatrix(X_data,label=y_data)
    cvresult=xgb.cv(params,dtrain,num_boost_round=1000,nfold=5,metrics='auc',
    stratified=True,verbose_eval=True,early_stopping_rounds=100)
    print(cvresult)
     print("="*100)
     #特征选择
    # selection=SelectFromModel(xgb,threshold=2.0,prefit=False)
    # selection_x_predictors=selection.fit_transform(x,y)
    # xgb.fit(selection_x_predictors,y,eval_metric="auc")
    
    print ("best best_ntree_limit",model.best_ntree_limit )
    
    #save model
    joblib.dump(model,save_model_path)
    model.dump_model(tree_path)
    #print(model.boost)
    endTime = datetime.datetime.now()
    print("Spend %.2f S(%.2f Min)" % ((endTime-starTime).seconds,(endTime-starTime).seconds / 60))
    
    
    #创建特征map
def ceate_feature_map(features):
    outfile = open(dd+'合作率xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
ceate_feature_map(X_hot.columns)

#
from base import parseModel
bst.dump_model('xgb_model',fmap='xgb_model.fmap')
#parseModel('xgb_model','xgb_model.csv')

#查看模型内部特征重要性
for importance_type in('weight','gain','cover','total_gain','total_cover'):
    df=bst.get_score(fmap='xgb_model.fmap', importance_type=importance_type)
    df=pd.DataFrame.from_dict(df,orient='index').reset_index()
    df.columns=['fea',str(importance_type)]
    df=df.sort_index(axis = 0,ascending = False,by = str(importance_type))
    df.to_csv('fea_importance'.csv',index=False)
df
