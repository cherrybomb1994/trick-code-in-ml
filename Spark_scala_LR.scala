##!!!!scala与java相似，需要有一个包地址，就是下面这句！，运行的时候文件在这个地址下？？？
package anruiqi.acquirer.data.demo

import org.apache.log4j.Level
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.log4j.Level
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

/**
 * Created with IDEA.
 * 效果要比spark_python_LR要好，这个待考证，实验中Scala版本不用分层auc也正常，更变一下判别label阈值即可。
 * 但是spark_python_LR中表现差，train必须分层抽样auc才正常，而且效果不如Scala好。待考证
 */
object LRTrainDriver {

  def main(args: Array[String]){

    val dataPath = "xgb_merge_mapfea_50_clickadd_onebase.libsvm"
    val modelPath = "lr/"

    // Set logger level
    org.apache.log4j.Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val sc = new SparkContext(new SparkConf().setAppName("LR Model Training Job"))

    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, dataPath)
    //val data = loadTrainDataSet(sc, dataPath, featureCountPath, minFeatureCount)

    // Split data into training (90%) and evaluation (10%).
    val splits = data.randomSplit(Array(0.9, 0.1), seed = 13L)
    val trainData = splits(0).cache()
    val testData = splits(1)

    val trainPositiveSampleNum = trainData.filter(point => point.label == 1).count()
    val trainNegativeSampleNum = trainData.filter(point => point.label == 0).count()
    val evaluatePositiveSampleNum = testData.filter(point => point.label == 1).count()
    val evaluateNegativeSampleNum = testData.filter(point => point.label == 0).count()

    // Train model
    println("******************** Start Training ********************")
    val classNum = 2
    val updater = new SquaredL2Updater()
    val alpha = 0.05
    val lr = new LogisticRegressionWithLBFGS()
    lr.setIntercept(true).setNumClasses(classNum).optimizer.setUpdater(updater).setRegParam(alpha)

    val model = lr.run(trainData)
    println("******************** End   Training ********************")

    // Save model
    println("******************** Save     Model ********************")
    //model.save(sc, modelPath)

    model.clearThreshold()

    // Evaluate Model
    val trainPredictionAndLabels = trainData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)

      (prediction, label)
    }

    val trainLogloss = trainPredictionAndLabels.map {
      case (v1, 1.0) =>
        -math.log(math.max(math.min(v1, 1.0 - 10e-15), 10e-15))

      case (v1, 0.0) =>
        -math.log(1.0 - math.max(math.min(v1, 1.0 - 10e-15), 10e-15))

      case _ => 0.0

    }.mean()

    val testPredictionAndLabels = testData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)

      (prediction, label)
    }

    val testLogloss = testPredictionAndLabels.map {
      case (v1, 1.0) =>
        -math.log(math.max(math.min(v1, 1.0 - 10e-15), 10e-15))

      case (v1, 0.0) =>
        -math.log(1.0 - math.max(math.min(v1, 1.0 - 10e-15), 10e-15))

      case _ => 0.0

    }.mean()
    println("******************************")

    println("******************************")

    val metrics = new BinaryClassificationMetrics(testPredictionAndLabels)
    val auc = metrics.areaUnderROC()
    val pr  = metrics.areaUnderPR()

    val tp = testPredictionAndLabels.filter(point => (point._1 >= 0.05 && point._2==1.0)).count()
    val tf = testPredictionAndLabels.filter(point => (point._1 < 0.05 && point._2==0.0)).count()
    val fn = testPredictionAndLabels.filter(point => (point._1 < 0.05 && point._2==1.0)).count()
    val fp = testPredictionAndLabels.filter(point => (point._1 >=0.05 && point._2==0.0)).count()

    // Output metrics
    println()
    println("******************** Evaluate Model ********************")
    println("                  Classes Num: " + model.numClasses)
    println("                  Feature Num: " + model.numFeatures)
    println("                  Weight Size: " + model.weights.size)
    println("    Train Positive Sample Num: " + trainPositiveSampleNum)
    println("    Train Negative Sample Num: " + trainNegativeSampleNum)
    println(" Evaluate Positive Sample Num: " + evaluatePositiveSampleNum)
    println(" Evaluate Negative Sample Num: " + evaluateNegativeSampleNum)
    println("                          AUC: " + auc)
    println("                           PR: " + pr)
    println("                           tp: " + tp)
    println("                           tf: " + tf)
    println("                           fn: " + fn)
    println("                           fp: " + fp)
    println("               Train Log Loss: " + trainLogloss)
    println("                Test Log Loss: " + testLogloss)
    println("********************************************************")
    println()

    //val resultPath = "hdfs://hadoop-meituan/user/hadoop-gct/anruiqi/pySpark_Tmp/"
    //testPredictionAndLabels.saveAsTextFile(resultPath)
    // Shutdown
    sc.stop()
  }
}
