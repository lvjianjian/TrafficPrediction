package com.ada.roadstatistics

import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.rdd.RDD

/**
  * Created by zhongjian on 2017/5/24.
  */
class GraphLoader(val sc:SparkContext) extends Logging{


  def loadEdgeFromDataSource(edgeFile: String): RDD[(Long, (Long, Long, Float))] = {
    logInfo("Loading Edge data from %s".format(edgeFile))
    sc.textFile(edgeFile).map {
      x => val temp = x.split("\t")
        (temp(0).toLong, (temp(1).toLong, temp(2).toLong, temp(3).toFloat))
    }
  }



  def loadVertexFromDataSource(vertexFile: String): RDD[(Long, (Double, Double))] = {
    logInfo("Loading Vertex data from %s".format(vertexFile))
    sc.textFile(vertexFile).map {
      x => val temp = x.split("\t")
        (temp(0).toLong, (temp(1).toDouble, temp(2).toDouble))
    }
  }


}
