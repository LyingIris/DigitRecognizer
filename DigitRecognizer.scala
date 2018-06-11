// Databricks notebook source
val trainingRDD = sqlContext.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/FileStore/tables/train.csv")
val testRDD = sqlContext.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/FileStore/tables/test.csv")

val training = trainingRDD.toDF()
val test = testRDD.toDF()

// COMMAND ----------

training.cache()
test.cache()

println(s"Observation counts - ${training.count} training / ${test.count} test")

// COMMAND ----------

/// representation of a training observation. 
//  index 0 - the label 
//  remaining indexes: pixel position. 
//  Value at remaining indexes: grayscale value - (255) is black 
training.take(1)

// COMMAND ----------

val pixelCol = training.columns.filter( _.contains("pixel")) 

// COMMAND ----------

import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// COMMAND ----------


// Chain indexer + dtc together into a single ML Pipeline.
val assembler = new VectorAssembler()
  .setInputCols(pixelCol)
  .setOutputCol("features")

val dtc = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

val pipeline = new Pipeline().setStages(Array(assembler, dtc))

// COMMAND ----------

val model = pipeline.fit(training)

// COMMAND ----------

val tree = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]

// COMMAND ----------

display(tree)

// COMMAND ----------

val variedMaxDepthModels = (0 until 8).map { maxDepth =>
  // For this setting of maxDepth, learn a decision tree.
  
  dtc.setMaxDepth(maxDepth)
  // Create a Pipeline with our feature processing stage (indexer) plus the tree algorithm
  val pipeline = new Pipeline().setStages(Array(assembler, dtc))
  
  // Run the ML Pipeline to learn a tree.
  pipeline.fit(training)
}

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")

// COMMAND ----------

val predictions = model.transform(test)

// COMMAND ----------

predictions.take(1)

// COMMAND ----------

predictions.select("prediction").rdd.map(r => r(0)).collect()

// COMMAND ----------

import org.apache.spark.sql.functions._
val solution = predictions.select("prediction")
val solutionFinal = solution.withColumn("ImageId",monotonically_increasing_id()+1)

// COMMAND ----------

// zip(solution)
solutionFinal.take(10)

// COMMAND ----------

// solutionFinal.write.format("com.databricks.spark.csv").option("header", "true")
//   .save("/Users/sages/OneDrive/Documents/School/CS 4301")

// COMMAND ----------

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


// COMMAND ----------

/// Turning our RDD rows into "LabeledPoint" objects
// this wasn't easy whatsoever. I spent quite a bit of time trying to get this to work, 
// and then I found an example online using "MLUtils.loadLibSVMFile" which streamlined this coersion 

        // training.map( row => LabeledPoint(0.0, Vectors.dense( row ) ) ) 
        // val dataPoints = training.map(row => 
        //     new LabeledPoint(
        //           row(0).toDouble, 
        //           Vectors.dense(row.take(row.length - 1).map(str => str.toDouble))
        //     )
        //   ).cache()

val observations: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
val splits = observations.randomSplit(Array(0.7, 0.3))

val (trainingData, testData) = (splits(0), splits(1))

// COMMAND ----------

/// Generating a random forest model 
// using the common pattern for tweaking model parameters as outlined in the spark docs 
// https://spark.apache.org/docs/latest/mllib-ensembles.html#random-forests
val numClasses = 10        // 0-9
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10          // arbitrarily chosen, tweakable  
val featureSubsetStrategy = "auto"
val impurity = "entropy"
val maxDepth = 8           // arbitrarily chosen, tweakable 
val maxBins = 32           // arbitrarily chosen, tweakable 


val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)





// COMMAND ----------

// Evaluate model on test instances and compute test error
val pairs = testData.map { obs =>
  val prediction = model.predict(obs.features)
  (obs.label, prediction)
}

// COMMAND ----------

val classErr = pairs.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Classification Error of $classErr \n\n ${model.toDebugString}")


// COMMAND ----------


