# -*- coding: utf-8 -*-

import os,sys

reload(sys)
sys.setdefaultencoding("utf-8")
import json,re
import math

from pyspark import SparkContext
#from pyspark.sql import HiveContext
from pyspark.sql import *
#from pyspark.ml.classification import LogisticRegression

from pyspark.mllib.util import MLUtils

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



def formatstring(line):
    res=[]
    for i in line:
        if i==None:
            res.append(0.0)
        else:
            res.append(float(i))
    return res

def countloss(line):
    if line[1]==1.0:
        return -math.log(max(min(line[0], 1.0 - 1.0e-15), 10e-15))
    elif line[1]==0.0:
        return -math.log(1.0 - max(min(line[0], 1.0 - 1.0e-15), 10e-15))
    else:
        return 0.0


def sample(line):
    if line.label==1:
        return (1,line)
    else:
        return (0,line)

def recover(line):
    return line[1]

def main():

    sc = SparkContext()
    path_in = "xgb_merge_mapfea.libsvm"
    path_out = "LR_train_result"

    '''
    ##如果输入时csv格式
    sqlContext = SQLContext(sc)
    table = sqlContext.read.format('csv').options(header='true').load(path_in)

    trainingData = table.rdd.map(formatstring).map(lambda x: (Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])

    result = LR_train(trainingData)
    fn_tmp, fp_tmp, tn_tmp, tp_tmp,percent,recall,auc = getHMatrix(result,sqlContext)


    rdd = sc.parallelize([' tp: '+str(int(tp_tmp[0]))+
                          ' fn: '+str(int(fn_tmp[0]))+
                          ' fp: '+str(int(fp_tmp[0]))+
                          ' tn: '+str(int(tn_tmp[0]))+
                          ' per: '+str(percent)+
                          ' recall '+str(recall)+
                          ' auc： '+str(auc)])
    rdd.repartition(1).saveAsTextFile(path_out)
    '''

    parsedData = MLUtils.loadLibSVMFile(sc, path_in)

    training, test = parsedData.randomSplit([0.8, 0.2], seed=223)
    trainPositiveSampleNum = training.filter(lambda point: point.label == 1).count()
    trainNum = training.count()
    pos_neg_ratio = trainPositiveSampleNum*5*1.0/trainNum
    fractions = {1: 1-pos_neg_ratio, 0: pos_neg_ratio}
    train = training.map(sample).sampleByKey(True,fractions,23).map(recover)
    one = train.filter(lambda point: point.label == 1).count()
    zero = train.filter(lambda point: point.label == 0).count()
    one_0 = test.filter(lambda point: point.label == 1).count()
    zero_0 = test.filter(lambda point: point.label == 0).count()
    print "==================================="
    print 'OriTrainPositiveNum: '+ str(trainPositiveSampleNum)
    print 'totalTrainNum: '+str(trainNum)
    print 'clickRatio: '+str(pos_neg_ratio)
    print 'train_sample_label:1_num: '+str(one)
    print 'train_sample_label:0_num: '+str(zero)
    print 'test_1: '+str(one_0)
    print 'test_0: '+str(zero_0)
    print "==================================="
    print ""

    model = LogisticRegressionWithLBFGS.train(train,regType="l2",regParam=0.05)

    predictionAndLabels_train = training.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    trainloss =  predictionAndLabels_train.map(countloss).mean()
    testloss = predictionAndLabels.map(countloss).mean()

    metrics = BinaryClassificationMetrics(predictionAndLabels)

    predictionAndLabels.map(lambda x: (x[1], x[0])).toDF(["prediction", "label"]).createGlobalTempView("mytest")
    sqlContext = SQLContext(sc)
    tp = sqlContext.sql(
        "select count(*) from global_temp.mytest t1 where t1.prediction = t1.label and t1.prediction = 1.0").collect()
    fp = sqlContext.sql(
        "select count(*) from global_temp.mytest t1 where t1.prediction <> t1.label and t1.prediction = 1.0").collect()
    tn = sqlContext.sql(
        "select count(*) from global_temp.mytest t1 where t1.prediction = t1.label and t1.prediction = 0.0").collect()
    fn = sqlContext.sql(
        "select count(*) from global_temp.mytest t1 where t1.prediction <> t1.label and t1.prediction = 0.0").collect()
    fn_tmp = re.findall(r'\=(\d+)', str(fn))
    fp_tmp = re.findall(r'\=(\d+)', str(fp))
    tn_tmp = re.findall(r'\=(\d+)', str(tn))
    tp_tmp = re.findall(r'\=(\d+)', str(tp))

    percent = float(tp_tmp[0]) / (float(tp_tmp[0]) + float(fp_tmp[0]) + 1)
    recall = float(tp_tmp[0]) / (float(tp_tmp[0]) + float(fn_tmp[0]) + 1)

    weight_list = model.weights
    weight_dit = {}
    for i in range(len(weight_list)):
        weight_dit[i] = weight_list[i]

    '''
    sort_dit = sorted(weight_dit.items(), key=lambda x: x[1], reverse=True)

    print "==================================="
    print "             sort weight            "
    for f in range(len(sort_dit)):
        print sort_dit[f][0], sort_dit[f][1]
    print "===================================" 
    '''
    sort_dit = sorted(weight_dit.items(), key=lambda x: x[1], reverse=True)

    rdd = sc.parallelize(['Feature Num: '+ str(model.numFeatures)+
                          'test_1: ' + str(one_0)+
                          'test_0: ' + str(zero_0)+
                          '; tp: '+str(int(tp_tmp[0]))+        #!!!!!此处fn与fp好像反了
                          '; fn: '+str(int(fn_tmp[0]))+
                          '; fp: '+str(int(fp_tmp[0]))+
                          '; tn: '+str(int(tn_tmp[0]))+
                          '; per: ' + str(percent) +
                          '; recall: ' + str(recall) +
                          '; auc：' + str(metrics.areaUnderROC)+
                          '; trainloss: '+str(trainloss)+
                          '; testloss: '+str(testloss)
                          #'; fea_weight: '+str(sort_dit)
                          ])

    print "==================================="
    print "              Eval"
    print  ' tp: '+str(int(tp_tmp[0]))
    print  ' fn: '+str(int(fn_tmp[0]) )
    print  ' fp: '+str(int(fp_tmp[0]))
    print  ' tn: '+str(int(tn_tmp[0]))
    print  ' per: ' + str(percent)
    print  ' recall: ' + str(recall)
    print  ' auc：' + str(metrics.areaUnderROC)
    print  ' trainloss: '+str(trainloss)
    print  ' testloss: '+str(testloss)
    print "==================================="
    print ""
    print "==================================="
    print "             sort weight            "
    for f in range(len(sort_dit)):
        print sort_dit[f][0], sort_dit[f][1]
    print "==================================="

    rdd.repartition(1).saveAsTextFile(path_out)


    sc.stop()

if __name__ == "__main__":
    main()
