# -*- coding: utf-8 -*-

import os,sys

reload(sys)
sys.setdefaultencoding("utf-8")
import json,re
import numpy as np
import math

from pyspark import SparkContext
#from pyspark.sql import HiveContext
from pyspark.sql.functions import concat
from pyspark.sql import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def LR_train(df):
    train, test = df.randomSplit([0.7, 0.3], seed=12345)
    lr = LogisticRegression(maxIter=3, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, threshold=0.5,
                            probabilityCol="probability")
    lrModel = lr.fit(train)
    result = lrModel.transform(test)
    #print('----------->>>>>>>>>>', lrModel.coefficients)
    #print('----------->>>>>>>>>>', lrModel.intercept)
    #print('----------->>>>>>>>>>', lrModel.summary.areaUnderROC)
    #print('[[[[[[[[[[[[[[[[',
    #      result.select("features", "rawPrediction", "probability", "prediction", "features").rdd.collect())
    #return lrModel.coefficients,lrModel.intercept,lrModel.summary.areaUnderROC
    return result


def formatstring(line):
    res=[]
    for i in line:
        if i==None:
            res.append(0.0)
        else:
            res.append(float(i))
    return res


# --------------------------------------------------
# 求预测结果的混淆矩阵，要求predictions中有两列：prediction,label
def getHMatrix(predictions,sqlContext):

    rocEv = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
    auc = rocEv.evaluate(predictions)

    predictions.createGlobalTempView("mytest")

    #posCnt = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.label = 1.0").collect()
    #negCnt = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.label = 0.0").collect()

    #posPredCnt = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction = 1.0").collect()
    #negPredCnt = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction = 0.0").collect()

    tp = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction = t1.label and t1.prediction = 1.0").collect()
    fp = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction <> t1.label and t1.prediction = 1.0").collect()

    tn = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction = t1.label and t1.prediction = 0.0").collect()
    fn = sqlContext.sql("select count(*) from global_temp.mytest t1 where t1.prediction <> t1.label and t1.prediction = 0.0").collect()
    fn_tmp = re.findall(r'\=(\d+)',str(fn))
    fp_tmp = re.findall(r'\=(\d+)', str(fp))
    tn_tmp = re.findall(r'\=(\d+)', str(tn))
    tp_tmp = re.findall(r'\=(\d+)', str(tp))

    print ('================================')
    print "labels\tpred_1\tpred_0"
    print "true_1\t"+str(int(tp_tmp[0]))+'\t'+str(int(fn_tmp[0]))
    print "true_0\t"+str(int(fp_tmp[0]))+'\t'+str(int(tn_tmp[0]))
    print ('================================')

    percent = float(tp_tmp[0]) / (float(tp_tmp[0]) + float(fp_tmp[0]))
    recall = float(tp_tmp[0])/(float(tp_tmp[0])+float(fn_tmp[0]))
    print ("per: "+str(percent))
    print ("recall: "+str(recall))
    print('auc: ' + str(auc))
    print ('================================')

    return fn_tmp,fp_tmp,tn_tmp,tp_tmp,percent,recall,auc

def main():

    sc = SparkContext()

    path_in = "/xgb_merge.csv"
    path_out = "LR_train_result.txt"

    sqlContext = SQLContext(sc)
    table = sqlContext.read.format('csv').options(header='true').load(path_in)
    table.na.fill(0.0)

    trainingData = table.rdd.map(formatstring).map(lambda x: (Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])
    trainingData.show()
    '''
    result = LR_train(trainingData)
    fn_tmp, fp_tmp, tn_tmp, tp_tmp,percent,recall,auc = getHMatrix(result,sqlContext)
    with open(path_out,'w')as f:
        f.write('================================'+'\n')
        f.write("labels\tpred_1\tpred_0"+'\n')
        f.write("true_1\t"+str(int(tp_tmp[0]))+'\t'+str(int(fn_tmp[0]))+'\n')
        f.write("true_0\t"+str(int(fp_tmp[0]))+'\t'+str(int(tn_tmp[0]))+'\n')
        f.write('================================' + '\n')
        f.write("per: "+str(percent)+'\n')
        f.write("recall: "+str(recall)+'\n')
        f.write('auc: ' + str(auc)+'\n')
        f.write('================================' + '\n')


    '''




    sc.stop()

if __name__ == "__main__":
    main()
