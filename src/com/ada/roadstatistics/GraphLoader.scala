package com.ada.roadstatistics

import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.rdd.RDD

/**
  * Created by zhongjian on 2017/5/24.
  */
class GraphLoader(val sc: SparkContext) extends Logging with Serializable{


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

  /**
    * 载入新的地图版本
    *
    * @param edgeFile (edgeid,((起点经纬度),(终点经纬度),长度)
    * @return
    */
  def loadNewEdgeFromDataSource(edgeFile: String): RDD[(Long, ((Double, Double), (Double, Double), Int, Int))] = {
    logInfo("Loading Edge data from %s".format(edgeFile))
    sc.textFile(edgeFile).map {
      x => val temp = x.split(":")
//        if (temp.length != 10) {
//          logError("Loading Edge Failure: " + temp(0))
//          System.exit(1)
//        }
        val id: Long = temp(0).toLong
        val length: Int = temp(7).toInt
        val road_class = temp(4)
        val ids = temp(9).split("\\(")(1).split("\\)")(0).split(",")
        val start = ids(0) //经度 纬度格式
        val startLonLat = start.trim.split(" ")
        val end = ids(ids.length - 1)
        val endLonLat = end.trim.split(" ")
        (id, ((startLonLat(0).toDouble, startLonLat(1).toDouble), (endLonLat(0).toDouble, endLonLat(1).toDouble), length, Tool.road_class2int(road_class)))
    }
  }


}
