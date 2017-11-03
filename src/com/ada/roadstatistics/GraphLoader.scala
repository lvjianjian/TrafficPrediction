package com.ada.roadstatistics

import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.rdd.RDD

/**
  * Created by zhongjian on 2017/5/24.
  */
class GraphLoader(val sc: SparkContext) extends Logging with Serializable {


  def loadEdgeFromDataSource(edgeFile: String): RDD[(Long, (Long, Long, Float))] = {
    logInfo("Loading Edge data from %s".format(edgeFile))
    sc.textFile(edgeFile).map {
      x =>
        val temp = x.split("\t")
        (temp(0).toLong, (temp(1).toLong, temp(2).toLong, temp(3).toFloat))
    }
  }


  def loadVertexFromDataSource(vertexFile: String): RDD[(Long, (Double, Double))] = {
    logInfo("Loading Vertex data from %s".format(vertexFile))
    sc.textFile(vertexFile).map {
      x =>
        val temp = x.split("\t")
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
      x =>
        val temp = x.split(":")
        //        if (temp.length != 10) {
        //          logError("Loading Edge Failure: " + temp(0))
        //          System.exit(1)
        //        }
        val id: Long = temp(0).toLong
        val length: Int = temp(7).toInt
        val road_class = temp(4)
        val ids = temp(9).split("\\(")(1).split("\\)")(0).split(",")
        val start = ids(0)
        //经度 纬度格式
        val startLonLat = start.trim.split(" ")
        val end = ids(ids.length - 1)
        val endLonLat = end.trim.split(" ")
        (id, ((startLonLat(0).toDouble, startLonLat(1).toDouble), (endLonLat(0).toDouble, endLonLat(1).toDouble), length, Tool.road_class2int(road_class)))
    }
  }


  /**
    * 载入新的地图版本, 自己抽取出的子图
    *
    * @param edgeFile (edgeid,((起点经纬度),(终点经纬度),长度)
    * @return
    */
  def loadNewPartEdgeFromDataSource(edgeFile: String): RDD[(Long, (Long, Long, String, Int, Int, Int))] = {
    logInfo("Loading Part Edge data from %s".format(edgeFile))
    sc.textFile(edgeFile).filter(x => !x.contains("edge_id")).map {
      x =>
        val temp = x.split(",")
        //        if (temp.length != 10) {
        //          logError("Loading Edge Failure: " + temp(0))
        //          System.exit(1)
        //        }
        val edgeid: Long = temp(1).toLong
        val s_id: Long = temp(2).toLong
        val e_id: Long = temp(3).toLong

        val length: Int = temp(8).toInt
        val road_class = temp(5)
        val width: Int = temp(7).toInt
        val speed_limit: Int = temp(9).toInt
        (edgeid, (s_id, e_id, road_class, width, length, speed_limit))
    }
  }

}

object test {
  def main(args: Array[String]): Unit = {
    //    val s = """57325,294053,288907,288908,2,0x03,0x00,11,75,40,"LINESTRING (116.39998 39.83186, 116.40005 39.83177, 116.40007 39.83174, 116.40011 39.8317, 116.40017 39.83166, 116.40023 39.83161, 116.40026 39.83159, 116.40036 39.83154, 116.40043 39.83151, 116.40065 39.83149)"""
    //    val ss = s.split(",")
    //    print(ss(1))
    val s2 = Parameter.HDFS_NODE_FRONT_PART + "/user/lvzhongjian/data/gaotong2016/R_G_0123class_45huan.csv"
    val ss = s2.split("/")
    print(ss(ss.length - 1).replace(".csv",""))
  }
}
