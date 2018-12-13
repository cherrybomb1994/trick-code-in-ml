# encoding: utf-8
# -*- coding: utf-8 -*-
#基于https://github.com/blebreton/spark-FM-parallelSGD，注意label需要变成1与-1

import os,sys

reload(sys)
sys.setdefaultencoding("utf-8")

import re
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.storagelevel import *
import pyspark.mllib.linalg
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score, log_loss, mean_squared_error
import time
import pickle

from matplotlib.colors import LinearSegmentedColormap
from pyspark.mllib.regression import LabeledPoint

# -------------------------------------------------------------------------------
# Factorization machines


def fm_get_p(x, w):
    """
    Computes the probability of an instance given a model
    """
    # use the compress trick if x is a sparse vector
    # The compress trick allows to upload the weight matrix for the rows corresponding to the indices of the non-zeros X values
    if type(x) == pyspark.mllib.linalg.SparseVector:
        W = w[x.indices]
        X = x.values
    elif type(x) == pyspark.mllib.linalg.DenseVector:
        W = w
        X = x
    else:
        return 'data type error'

    xa = np.array([X])
    VX = xa.dot(W)
    VX_square = (xa * xa).dot(W * W)
    phi = 0.5 * (VX * VX - VX_square).sum()

    return 1.0 / (1.0 + np.exp(-phi))


def fm_get_p_old(X, W):
    """
    Computes the probability of an instance given a model
    """
    w_triu = np.triu(np.dot(W, W.T), 1)
    xa = np.array([X])
    x_triu = np.triu(np.dot(xa.T, xa), 1)
    phi = np.sum(x_triu * w_triu)
    return 1.0 / (1.0 + np.exp(-phi))


def fm_gradient_sgd_trick(X, y, W, regParam):
    """
    Computes the gradient for one instance using Rendle FM paper (2010) trick (linear time computation)
    """
    xa = np.array([X])
    x_matrix = xa.T.dot(xa)

    VX = xa.dot(W)
    VX_square = (xa * xa).dot(W * W)
    phi = 0.5 * (VX * VX - VX_square).sum()

    expnyt = np.exp(-y * phi)
    np.fill_diagonal(x_matrix, 0)
    result = (-y * expnyt) / (1 + expnyt) * (np.dot(x_matrix, W))

    return regParam * W + result


def fm_gradient_sgd(X, y, dim, W, regParam):
    """
    Computes the gradient for one instance
    """
    w_matrix = np.dot(W, W.T)
    w_triu = np.triu(w_matrix, 1)
    xa = np.array([X])
    x_matrix = np.dot(xa.T, xa)
    x_triu = np.triu(x_matrix, 1)
    phi = np.sum(x_triu * w_triu)
    expnyt = np.exp(-y * phi)
    x_matrix_negeye = (1 - np.eye(dim)) * x_matrix
    return regParam * W + (-y * expnyt) / (1 + expnyt) * (np.dot(x_matrix_negeye, W))


def predictFM(data, w):
    """
    Computes the probabilities given a model for the complete data set
    """
    return data.map(lambda row: fm_get_p(row.features, w))


def logloss(X, w, y):
    """
    Computes the logloss of the model for one instance
    """
    # p = max(min(phi, 1.0 - 10e-12), 10e-12)
    phi = get_phi(X, w)
    # y01 = 1 if y==1 else 0

    return np.log(1 + np.exp(-y * phi))


def logloss2(y_pred, y_true):
    """
    Computes the logloss given the true label and the predictions
    """
    # avoid NaN value
    y_pred[y_pred == 0] = 1e-12
    y_pred[y_pred == 1] = 1 - 1e-12

    losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(losses)


def get_phi(X, W):
    """
    Computes the phi-value for an instance given a model
    """
    xa = np.array([X])
    VX = xa.dot(W)
    VX_square = (xa * xa).dot(W * W)
    phi = 0.5 * (VX * VX - VX_square).sum()
    return phi


# -----------------------------------------------------------------------
# Train with parallel sgd

def trainFM_parallel_sgd(sc, data, iterations=50, iter_sgd=5, alpha=0.005, regParam=0.0001, factorLength=4, \
                         verbose=False, savingFilename=None, evalTraining=None):
    """
    Train a Factorization Machine model using parallel stochastic gradient descent.

    Parameters:
    data : RDD of LabeledPoints
        Training data. Labels should be -1 and 1
        Features should be either SparseVector or DenseVector from mllib.linalg library
    iterations : numeric
        Nr of iterations of parallel SGD. default=50
    iter_sgd : numeric
    	Nr of iteration of sgd in each partition. default = 5
    alpha : numeric
        Learning rate of SGD. default=0.01
    regParam : numeric
        Regularization parameter. default=0.01
    factorLength : numeric
        Length of the weight vectors of the FMs. default=4
    verbose: boolean
        Whether to ouptut iteration numbers, time, logloss for train and validation sets
    savingFilename: String
    	Whether to save the model after each iteration
    evalTraining : instance of the class evaluation
    	Plot the evaluation during the training (on a train and a validation set)
        The instance should be created before using trainFM_parallel_sgd

    returns: w
        numpy matrix holding the model weights
    """

    # split the data in train and validation sets if evalTraining or verbose
    if evalTraining:
        verbose2 = True
    else:
        verbose2 = False
    if verbose or verbose2:
        train, val = data.randomSplit([0.8, 0.2])
        train.persist(StorageLevel.MEMORY_ONLY_SER)
        val.persist(StorageLevel.MEMORY_ONLY_SER)
        # train.cache()
        # val.cache()
    else:
        train = data.persist(StorageLevel.MEMORY_ONLY_SER)
    # train= data.cahe()

    # glom() allows to treat a partition as an array rather as a single row at time
    train_Y = train.map(lambda row: row.label).glom()
    train_X = train.map(lambda row: row.features).glom()
    train_XY = train_X.zip(train_Y).persist(StorageLevel.MEMORY_ONLY_SER)
    # train_XY = train_X.zip(train_Y).cache()

    # Initialize weight vectors
    nrFeat = len(train_XY.first()[0][0])
    np.random.seed(int(time.time()))
    w = np.random.ranf((nrFeat, factorLength))
    w = w / np.sqrt((w * w).sum())

    if evalTraining:
        evalValidation = evaluation(val)
        evalValidation.modulo = evalTraining.modulo
        evalValidation.evaluate(w)
        evalTraining = evaluation(train)
        evalTraining.evaluate(w)
        evalTraining.modulo = evalValidation.modulo
    if verbose:
        print 'iter \ttime \ttrain_logl \tval_logl'
        # compute original logloss (0 iteration)
        if evalTraining:
            print '%d \t%d \t%5f \t%5f' % (0, 0, evalTraining.logl[-1], evalValidation.logl[-1])
        else:
            print '%d \t%d \t%5f \t%5f' % (0, 0, evaluate(train, w)[2], evaluate(val, w)[2])
        start = time.time()

    for i in xrange(iterations):
        wb = sc.broadcast(w)
        wsub = train_XY.map(lambda (X, y): sgd_subset(X, y, wb.value, iter_sgd, alpha, regParam))
        w = wsub.mean()

        # evaluate and store the evaluation figures each 'evalTraining.modulo' iteration
        if evalTraining and i % evalTraining.modulo == 0:
            evalTraining.evaluate(w)
            evalValidation.evaluate(w)
        if verbose:
            if evalTraining:
                if i % evalTraining.modulo == 0:
                    print '%d \t%d \t%5f \t%5f' % (
                    i + 1, time.time() - start, evalTraining.logl[-1], evalValidation.logl[-1])
            else:
                print '%d \t%d \t%5f \t%5f' % (i + 1, time.time() - start, evaluate(train, w)[2], evaluate(val, w)[2])
        if savingFilename:
            saveModel(w, savingFilename + '_iteration_' + str(i + 1))


    if verbose:
        print 'Train set: ';
        print '(rtv_pr_auc, rtv_auc, logl, mse, accuracy)';
        print evaluate(train, w)
        print 'Validation set:';
        print evaluate(val, w)

    train_XY.unpersist()

    return w


def sgd_subset(train_X, train_Y, w, iter_sgd, alpha, regParam):
    """
    Computes stochastic gradient descent for a partition (in memory)
    Automatically detects which vector representation is used (dense or sparse)
    Parameter:
	train_X : list of pyspark.mllib.linalg dense or sparse vectors
	train_Y : list of labels
	w : numpy matrix holding the model weights
	iter_sgd : numeric
		Nr of iteration of sgd in each partition.
	alpha : numeric
		Learning rate of SGD.
	regParam : numeric
		Regularization parameter.

    return:
	numpy matrix holding the model weights for this partition
    """
    if type(train_X[0]) == pyspark.mllib.linalg.DenseVector:
        return sgd_subset_dense(train_X, train_Y, w, iter_sgd, alpha, regParam)
    elif type(train_X[0]) == pyspark.mllib.linalg.SparseVector:
        return sgd_subset_sparse(train_X, train_Y, w, iter_sgd, alpha, regParam)
    else:
        return 'data type error'


def sgd_subset_dense(train_X, train_Y, w, iter_sgd, alpha, regParam):
    """
    Computes stochastic gradient descent for a partition (in memory)
    Parameter:
	train_X : list of pyspark.mllib.linalg dense or sparse vectors
	train_Y : list of labels
	w : numpy matrix holding the model weights
	iter_sgd : numeric
		Nr of iteration of sgd in each partition.
	alpha : numeric
		Learning rate of SGD.
	regParam : numeric
		Regularization parameter.

    return:
	wsub: numpy matrix holding the model weights for this partition
    """
    N = len(train_X)
    wsub = w
    G = np.ones(w.shape)
    for i in xrange(iter_sgd):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in xrange(N):
            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            grads = fm_gradient_sgd_trick(X, y, wsub, regParam)
            G += grads * grads
            wsub -= alpha * grads / np.sqrt(G)

    return wsub


def sgd_subset_sparse(train_X, train_Y, w, iter_sgd, alpha, regParam):
    """
    Computes stochastic gradient descent for a partition (in memory)
    The compress trick allows to upload the weight matrix for the rows corresponding to the indices of the non-zeros X values
    Parameter:
	train_X : list of pyspark.mllib.linalg dense or sparse vectors
	train_Y : list of labels
	w : numpy matrix holding the model weights
	iter_sgd : numeric
		Nr of iteration of sgd in each partition.
	alpha : numeric
		Learning rate of SGD.
	regParam : numeric
		Regularization parameter.

    return:
	wsub: numpy matrix holding the model weights for this partition
    """
    N = len(train_X)
    wsub = w
    G = np.ones(w.shape)
    for i in xrange(iter_sgd):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in xrange(N):
            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            grads_compress = fm_gradient_sgd_trick(X.values, y, wsub[X.indices], regParam)
            G[X.indices] += grads_compress * grads_compress
            wsub[X.indices] -= alpha * grads_compress / np.sqrt(G[X.indices])

    return wsub


# -----------------------------------------------------------------------
# Train with non-parallel sgd
def trainFM_sgd(data, iterations=300, alpha=0.01, regParam=0.01, factorLength=4):
    """
    Train a Factorization Machine model using stochastic gradient descent, non-parallel.

    Parameters:
    data : RDD of LabeledPoints
            Training data. Labels should be -1 and 1
    iterations : numeric
            Nr of iterations of SGD. default=300
    alpha : numeric
            Learning rate of SGD. default=0.01
    regParam : numeric
            Regularization parameter. default=0.01
    factorLength : numeric
            Length of the weight vectors of the FMs. default=4

    returns: w
            numpy matrix holding the model weights
    """
    # data is labeledPoint RDD
    train_Y = np.array(data.map(lambda row: row.label).collect())
    train_X = np.array(data.map(lambda row: row.features).collect())
    (N, dim) = train_X.shape
    w = np.random.ranf((dim, factorLength))
    w = w / np.sqrt((w * w).sum())
    G = np.ones(w.shape)
    for i in xrange(iterations):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in xrange(N):
            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            grads = fm_gradient_sgd_trick(X, y, wsub, regParam)
            G += grads * grads
            w -= alpha * grads / np.sqrt(G)

    return w


# -----------------------------------------------------------------------
def evaluate(data, w):
    """
    Evaluate a Factorization Machine model on a data set.

    Parameters:
    data : RDD of LabeledPoints
            Evaluation data. Labels should be -1 and 1
    w : numpy matrix
            FM model, result from trainFM_sgd or trainFM_parallel_sgd

    returns : (rtv_pr_auc, rtv_auc, logl, mse, accuracy)
            rtv_pr_auc : Area under the curve of the Recall/Precision graph (average precision score)
            rtv_auc : Area under the curve of the ROC-curve
            logl : average logloss
            MSE : mean square error
            accuracy
    """
    # data.cache()
    data.persist(StorageLevel.MEMORY_ONLY_SER)
    y_true_rdd = data.map(lambda lp: 1 if lp.label == 1 else 0)
    y_true = y_true_rdd.collect()
    y_pred_rdd = predictFM(data, w)
    y_pred = y_pred_rdd.collect()

    logl = logloss2(np.array(y_pred), np.array(y_true))

    # rtv_pr_auc and rtv_auc
    y_pair = np.column_stack((y_pred, y_true))
    sort_y_pair = y_pair[y_pair[:, 0].argsort()[::-1]]

    fpr, tpr, _ = roc_curve(sort_y_pair[:, 1], sort_y_pair[:, 0])

    if np.isnan(tpr[0]):
        rtv_pr_auc = 0
        rtv_auc = 0
        print 'cant compute AUC'
    else:
        rtv_auc = auc(fpr, tpr)
        rtv_pr_auc = average_precision_score(sort_y_pair[:, 1], sort_y_pair[:, 0])

    # mse
    mse = mean_squared_error(sort_y_pair[:, 1], sort_y_pair[:, 0])

    # accuracy
    y_pred_label = np.zeros(len(y_pred))
    y_pred_label[np.array(y_pred) > 0.5] = 1
    truePred = ((y_pred_label - y_true) == 0).sum()
    accuracy = float(truePred) / len(y_true)

    return rtv_pr_auc, rtv_auc, logl, mse, accuracy


def saveModel(w, fileName):
    """
    Saves the model in a pickle file
    """
    # with open('model/'+fileName, 'wb') as handle :
    with open(fileName, 'wb') as handle:
        pickle.dump(w, handle)


def loadModel(fileName):
    """
    Load the model from a pickle file
    """
    # with open('model/'+fileName, 'rb') as handle :
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def transform_data(data_01_label):
    """
    Transforms LabeledPoint RDDs that have 0/1 labels to -1/1 labels (as is needed for the FM models)
    """
    data = data_01_label.map(lambda row: LabeledPoint(-1 if row.label == 0 else 1, row.features))


# -----------------------------------------------------------------------
# Plot the error

class evaluation(object):
    """ Store the evaluation figures (rtv_pr_auc, rtv_auc, logl, mse, accuracy) in lists
        Print the final error
        Plot the evolution of the error function of the number of iterations
    """

    def __init__(self, data):
        self.data = data
        self.rtv_pr_auc = []
        self.rtv_auc = []
        self.logl = []
        self.mse = []
        self.accuracy = []
        # choose the modulo of the iterations to compute the evaluation
        self.modulo = 1

    def evaluate(self, w):
        eval = evaluate(self.data, w)
        self.rtv_pr_auc.append(eval[0])
        self.rtv_auc.append(eval[1])
        self.logl.append(eval[2])
        self.mse.append(eval[3])
        self.accuracy.append(eval[4])

    def display(self):
        """ print the evaluation figures (mse, logl, rtv_pr_auc, rtv_auc, accuracy) (last element of the corresponding evaluation list)
        """
        print 'MSE: {0:3f} \nlogl: {1:3f} \nrtv_pr_auc: {2:3f} \nrtv_auc: {3:3f} \nAccuracy: {3:3f}\n' \
            .format(self.mse[-1], self.logl[-1], self.rtv_pr_auc[-1], self.rtv_auc[-1], self.accuracy[-1])


def main():

    sc = SparkContext()

    path_in = "xgb_merge_mapfea_50.libsvm"

    parsedData = MLUtils.loadLibSVMFile(sc, path_in)
    training, testing = parsedData.randomSplit([0.9, 0.1], seed=13L)
    trainPosSampleNum = training.filter(lambda point: point.label == 1).count()
    trainNegSampleNum = training.filter(lambda point: point.label == 0).count()
    testPosSampleNum = testing.filter(lambda point: point.label == 1).count()
    testNegSampleNum = testing.filter(lambda point: point.label == 0).count()

    train = training.map(lambda row: LabeledPoint(-1 if row.label == 0 else 1, row.features))
    test = testing.map(lambda row: LabeledPoint(-1 if row.label == 0 else 1, row.features))
    iterationnum = 3
    w = trainFM_parallel_sgd(sc, train, iterations=iterationnum)

    rtv_pr_auc, rtv_auc, logl, mse, accuracy = evaluate(train,w)

    predictionAndLabels = test.map(lambda lp: (float(fm_get_p(lp.features,w)), lp.label))
    #predictionAndLabels.saveAsTextFile('pySpark_Tmp/')

    labelnum = 0.05
    tp = predictionAndLabels.filter(lambda x:x[0]>=labelnum and x[1]==1.0).count()
    tn = predictionAndLabels.filter(lambda x:x[0]<labelnum and x[1]==-1.0).count()
    fn = predictionAndLabels.filter(lambda x:x[0]<labelnum and x[1]==1.0).count()
    fp = predictionAndLabels.filter(lambda x:x[0]>=labelnum and x[1]==-1.0).count()

    metrics = BinaryClassificationMetrics(predictionAndLabels)

    print "==================================="
    print "              Eval"
    print  ' trainPosSampleNum：' + str(trainPosSampleNum)
    print  ' trainNegSampleNum：' + str(trainNegSampleNum)
    print  ' testPosSampleNum：' + str(testPosSampleNum)
    print  ' testNegSampleNum：' + str(testNegSampleNum)
    print  ' trainLogLoss: '+str(logl)
    print  ' tp: ' + str(tp)
    print  ' fn: ' + str(fn)
    print  ' fp: ' + str(fp)
    print  ' tn: ' + str(tn)
    print  ' per: '+str(tp*1.0/(tp+fp)+1)
    print  ' cal:'+str(tp*1.0/(tp+fn)+1)
    print  ' auc：' + str(metrics.areaUnderROC)
    print "==================================="

    '''
    path_out ="label_"+str(labelnum)+"_iter"+str(iterationnum)
    rdd = sc.parallelize([' trainPosSampleNum：' + str(trainPosSampleNum) +
                          ' trainNegSampleNum：' + str(trainNegSampleNum) +
                          ' testPosSampleNum：' + str(testPosSampleNum) +
                          ' testNegSampleNum：' + str(testNegSampleNum) +
                          ' trainLogLoss: '+str(logl)+
                          ' tp: ' + str(tp) +
                          ' fn: ' + str(fn) +
                          ' fp: ' + str(fp) +
                          ' tn: ' + str(tn) +
                          ' per: '+str(tp*1.0/(tp+fp)+1) +
                          ' cal:'+str(tp*1.0/(tp+fn)+1)+
                          ' auc：' + str(metrics.areaUnderROC) 
                          ])
    '''




    sc.stop()

if __name__ == "__main__":
    main()


