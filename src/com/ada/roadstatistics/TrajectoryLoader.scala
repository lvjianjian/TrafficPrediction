package com.ada.roadstatistics


import java.text.SimpleDateFormat

import org.apache.log4j.Logger
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}


/**
  * Created by zhongjian on 2017/5/23.
  */
class TrajectoryLoader(val sc: SparkContext) extends Logging with Serializable {
//  val logger:Logger = Logger.getLogger(getClass.getName)


  /**
    * 从文件中读取轨迹 返回RDD[(Long//轨迹id,Array[(Long)]路段id 点id,Array[Long]时间)//]
    *
    * @param trajectoryFile
    * @param minEdges 轨迹所拥有的最小路段数
    * @param maxEdges 轨迹所拥有的最大路段数
    * @return
    */
  def loadTrajectoryFromDataSource(trajectoryFile: String, minEdges: Int, maxEdges: Int): RDD[(Long, Array[Long], Array[String])] = {
    logInfo("Loading Trajectory data from %s".format(trajectoryFile))
    sc.textFile(trajectoryFile).map(x => x.split(",")).filter(x => x.length >= minEdges).filter(x => x.length <= maxEdges).map {
      temp =>
        var arrayList: List[Long] = Nil
        var timeList: List[String] = Nil
        var preEdgeId: Long = 0
        for (i <- 1 until temp.size) {
          val split = temp(i).split("\\|")
          arrayList = split(1).toLong :: arrayList
          timeList = split(0) :: timeList
        }
        (temp(0).toLong, arrayList.reverse.toArray, timeList.reverse.toArray)
    }
  }

  /**
    * 从新的轨迹文件中读取轨迹（16年轨迹） 返回RDD[(Long//轨迹id,Array[(Long)]路段id 点id,Array[Long]时间)//]
    *
    * @param trajectoryFile
    * @param minEdges 轨迹所拥有的最小路段数
    * @param maxEdges 轨迹所拥有的最大路段数
    * @return RDD, 每个元素是 （edgeid数组，time数组，（开始点经纬度），（结束点经纬度）,(第一个路段长度，最后一个路段长度)）. time数组比edgeid数组多一个
    */
  def loadNewTrajectoryFromDataSource(trajectoryFile: String, minEdges: Int, maxEdges: Int): RDD[(Array[Long], Array[Long], (Double, Double), (Double, Double), (Int, Int))] = {
    logInfo("Loading Trajectory data from %s".format(trajectoryFile))
    sc.textFile(trajectoryFile).map {
      temp =>
        try {
          val strings = temp.split(",")
          val edgeids = strings(4)
          val times = strings(8)
          val edgeNum = strings(11).toInt
          val startLon = strings(22).toDouble
          val startLat = strings(23).toDouble
          val endLon = strings(24).toDouble
          val endLat = strings(25).toDouble
          val endTime = strings(27).toLong
          val lengths = strings(10).split("\\|")
          (edgeids, times, edgeNum, (startLon, startLat), (endLon, endLat), endTime, lengths(0).toInt, lengths(lengths.length - 1).toInt)
        }catch{
          case e:Exception=>
            ("","",-100,(0D,0D),(0D,0D),0L,0,0) //第三项设置负数 后面可以成功过滤
//          None
        }
    }.filter({x => (x._3 >= minEdges && x._3 <= maxEdges)}).map {
      temp =>
        val edgeids: List[Long] = temp._1.split("\\|").toList.map(edge => edge.toLong)
        val times: List[Long] = temp._2.split("\\|").toList.map(edge => edge.toLong)
        val startLonLat: (Double, Double) = temp._4
        val endLonLat = temp._5
        (edgeids.toArray, (times :+ temp._6).toArray, startLonLat, endLonLat, (temp._7, temp._8))
    }

  }

}
