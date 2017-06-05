package com.ada.roadstatistics


import java.text.SimpleDateFormat

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}



/**
  * Created by zhongjian on 2017/5/23.
  */
class TrajectoryLoader(val sc:SparkContext) extends Logging{


  /**
    * 从文件中读取轨迹 返回RDD[(Long//轨迹id,Array[(Long)]路段id 点id,Array[Long]时间)//]
    * @param trajectoryFile
    * @param minEdges 轨迹所拥有的最小路段数
    * @param maxEdges 轨迹所拥有的最大路段数
    * @return
    */
  def loadTrajectoryFromDataSource(trajectoryFile: String, minEdges:Int, maxEdges:Int): RDD[(Long, Array[Long],Array[String])] = {
    logInfo("Loading Trajectory data from %s".format(trajectoryFile))
    sc.textFile(trajectoryFile).map(x => x.split(",")).filter(x => x.length > minEdges).filter(x=>x.length<=maxEdges).map {
      temp =>
        var arrayList: List[Long] = Nil
        var timeList:List[String] = Nil
        var preEdgeId: Long = 0
        for (i<-1 until temp.size){
          val split = temp(i).split("\\|")
          arrayList = split(1).toLong :: arrayList
          timeList = split(0)  :: timeList
        }
        (temp(0).toLong, arrayList.reverse.toArray,timeList.reverse.toArray)
    }
  }



}
