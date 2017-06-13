package com.ada.roadstatistics

import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}

/**
  * Created by zhongjian on 2017/5/23.
  */
class Statisticer(val sc: SparkContext) extends Logging with Serializable {

  val region1: Array[Double] = Array(116.258726, 39.820333, 116.505918, 40.028333);
  val region2: Array[Double] = Array(116.275935, 39.840503, 116.495319, 40.000593);
  val region3: Array[Double] = Array(116.26954, 39.828598, 116.49167, 39.997132);

  /**
    * 统计轨迹为以edge为单位的起始时间，总运行时间和轨迹数
    *
    * @param time_window        统计的时间窗大小
    * @param min_edges          过滤的轨迹的最小路段数
    * @param max_edges          过滤的轨迹的最大路段数
    * @param min_section_length 过滤的路段的最小长度
    */
  def count(time_window: Int, min_edges: Int, max_edges: Int, min_section_length: Int): Unit = {
    val edges_RDD = new GraphLoader(sc).loadEdgeFromDataSource(Parameter.edge_data_path)
    val edges_broadcast = sc.broadcast(edges_RDD.collect().toMap)
    val source: RDD[(Long, Array[Long], Array[String])] = new TrajectoryLoader(sc).loadTrajectoryFromDataSource(Parameter.traj_data_path, min_edges, max_edges) //Parameter.traj_data_path
    val map: RDD[((Long, String), (Int, Int))] = source.flatMap({
      temp =>
        var list: List[((Long, String), (Int, Int))] = Nil
        val edgeids = temp._2
        val time = temp._3
        for (i <- 0 until edgeids.length - 1) {
          //最后一个路段放弃
          val edgeid = edgeids(i)
          val length = edges_broadcast.value.get(edgeid).get._3
          if (length >= min_section_length) {
            val start_time = time(i)
            val end_time = time(i + 1)
            list = ((edgeid, Tool.timeFormatByMinute(start_time, time_window)), (Tool.timeDifference(start_time, end_time), 1)) :: list
          }
        }
        list
    })

    val savePath = Parameter.HDFS_BASE_RESULT_DIR + "TrafficConditionStatistic/" + ("%dTimeWindow_%dMinEdges_%dMaxEdges_%dMinSectionLength".format(time_window, min_edges, max_edges, min_section_length))
    map.reduceByKey((e1, e2) => (e1._1 + e2._1, e1._2 + e2._2)).map({
      temp => (temp._1._1, (temp._1._2, temp._2._1, temp._2._2))
    }).groupByKey().map({
      temp =>
        (temp._1, temp._2.toList.sortWith((a, b) => a._1.toLong < b._1.toLong))
    }).map({
      temp =>
        temp._1 + "|" + temp._2.mkString("|")
    }).saveAsTextFile(savePath)

    println("save in " + savePath)
  }

  /**
    * 划分网格,并统计网格中的起点为单位的路段的轨迹运行总时间和总轨迹数(统计的比较粗)
    *
    * @param path      count 统计完后的保存路径
    * @param region    区域
    * @param lon_split 经度划分数
    * @param lat_split 纬度划分数
    */
  def regionCount(path: String, region: Array[Double], lon_split: Int, lat_split: Int): Unit = {
    val loader = new GraphLoader(sc)
    val source = loader.loadVertexFromDataSource(Parameter.vertex_data_path)
    val edge_RDD = loader.loadEdgeFromDataSource(Parameter.edge_data_path)
    val broadcast = sc.broadcast(source.collect().toMap)
    val edge_map = sc.broadcast(edge_RDD.collect().toMap)
    //(time,((x,y),(traveltime,num)))
    val count_RDD: RDD[(String, ((Int, Int), (Int, Int)))] = sc.textFile(path + "/part*").flatMap({
      s =>
        var list: List[(String, ((Int, Int), (Int, Int)))] = Nil
        val split = s.split("\\|")
        val roadid = split(0).toLong
        val xy = Tool.getGridXY(region, lon_split, lat_split, broadcast.value.get(edge_map.value.get(roadid).get._1).get)
        if (xy._1 != -1 && xy._2 != -1) {
          // 不超过区域界限
          for (i <- 1 until split.length) {
            val time_alltraveltime_num = split(i).substring(1, split(i).length - 1).split(",")
            list = (time_alltraveltime_num(0), (xy, (time_alltraveltime_num(1).toInt, time_alltraveltime_num(2).toInt))) :: list
          }
        }
        list
    })

    val savePath = path + "/lon%d_lat%d_region(%f,%f,%f,%f)".format(lon_split, lat_split, region(0), region(1), region(2), region(3))
    count_RDD.map({ temp => ((temp._1, temp._2._1._1, temp._2._1._2), (temp._2._2._1, temp._2._2._2)) })
      .reduceByKey((e1, e2) => (e1._1 + e2._1, e1._2 + e2._2))
      .map({
        temp => (temp._1._1, (temp._1._2, temp._1._3, temp._2._1, temp._2._2))
      }).groupByKey().map({
      temp =>
        temp._1 + "|" + temp._2.mkString("|")
    }).coalesce(1).saveAsTextFile(savePath)
  }


  /**
    * 划分网格,并统计网格中的起点为单位的路段和各个路段的总运行时间和轨迹数量
    *
    * @param path count 统计完后的保存路径
    * @param region
    * @param lon_split
    * @param lat_split
    */
  def regionCount2(path: String, region: Array[Double], lon_split: Int, lat_split: Int): String = {
    val loader = new GraphLoader(sc)
    val source = loader.loadVertexFromDataSource(Parameter.vertex_data_path)
    val edge_RDD = loader.loadEdgeFromDataSource(Parameter.edge_data_path)
    val broadcast = sc.broadcast(source.collect().toMap)
    val edge_map = sc.broadcast(edge_RDD.collect().toMap)
    //(time,((x,y),(traveltime,num)))
    val count_RDD: RDD[(String, ((Int, Int, Long), (Int, Int)))] = sc.textFile(path + "/part*").flatMap({
      s =>
        var list: List[(String, ((Int, Int, Long), (Int, Int)))] = Nil
        val split = s.split("\\|")
        val roadid = split(0).toLong
        val xy = Tool.getGridXY(region, lon_split, lat_split, broadcast.value.get(edge_map.value.get(roadid).get._1).get)
        if (xy._1 != -1 && xy._2 != -1) {
          // 不超过区域界限
          for (i <- 1 until split.length) {
            val time_alltraveltime_num = split(i).substring(1, split(i).length - 1).split(",")
            list = (time_alltraveltime_num(0), ((xy._1, xy._2, roadid), (time_alltraveltime_num(1).toInt, time_alltraveltime_num(2).toInt))) :: list
          }
        }
        list
    })

    val savePath = path + "/detail/lon_split=%d/lat_split=%d/region=%f_%f_%f_%f".format(lon_split, lat_split, region(0), region(1), region(2), region(3))

    count_RDD.map({ temp => ((temp._1, temp._2._1._1, temp._2._1._2, temp._2._1._3), (temp._2._2._1, temp._2._2._2)) })
      .reduceByKey((e1, e2) => (e1._1 + e2._1, e1._2 + e2._2))
      .map({
        temp => ((temp._1._1, temp._1._2, temp._1._3), (temp._1._4, temp._2._1, temp._2._2))
      }).groupByKey()
      .map({
        temp => (temp._1._1, (temp._1._2, temp._1._3, temp._2.mkString(",")))
      }).groupByKey()
      .map({
        temp =>
          temp._1 + "|" + temp._2.mkString("|")
      })
      .saveAsTextFile(savePath)

    val savePath2 = path + "/detail/lon_split=%d/lat_split=%d/region=%f_%f_%f_%f/EdgeSize".format(lon_split, lat_split, region(0), region(1), region(2), region(3))

    count_RDD.map({
      temp => ((temp._2._1._1, temp._2._1._2), temp._2._1._3)
    }).groupByKey().map({
      temp => (temp._1, temp._2.toSet.size)
    }).coalesce(1).saveAsTextFile(savePath2)

    savePath
  }


  /**
    * 在regionCount2统计完后，基于统计的数据统计各个grid在各个时间区间的平均速度（这里是以路段为单位统计的，意义不强）
    *
    * @param path        : regionCount2统计完后的路径
    * @param time_window : 时间窗大小，这里需要是5的倍数,同时是24*60的因子，建议5，10，15，20，30
    */
  def regionAvgSpeedFromRegionCount2(path: String, time_window: Int): Unit = {
    val loader = new GraphLoader(sc)
    val edge_RDD = loader.loadEdgeFromDataSource(Parameter.edge_data_path)
    val edge_map = sc.broadcast(edge_RDD.collect().toMap)
    val savePath = path + "/%dtime_window_averageSpeed".format(time_window)
    sc.textFile(path + "/part*").flatMap({
      temp =>
        val strings = temp.split("\\|")
        val time = strings(0)
        val newTime = Tool.timeFormatByMinute(time + "00", time_window)
        var list: List[((String, Int, Int), (Float, Int))] = Nil
        for (i <- 1 until strings.length) {
          val split = strings(i).substring(1, strings(i).length - 1).replace("(", "").replace(")", "").split(",")
          val x = split(0).toInt
          val y = split(1).toInt
          var allAverageSpeed: Float = 0
          for (j <- 2 until(split.length, 3)) {
            val edgeid = split(j).toLong
            val travelTime = split(j + 1).toInt
            val num = split(j + 2).toInt
            if (travelTime != 0) {
              val averageSpeed = (edge_map.value.get(edgeid).get._3) / (travelTime / num.toFloat)
              allAverageSpeed += averageSpeed
            }
          }

          list = ((newTime, x, y), (allAverageSpeed, (split.length - 2) / 3)) :: list
        }
        list
    }

    ).reduceByKey((e1, e2) => (e1._1 + e2._1, e1._2 + e2._2)).map({
      temp => (temp._1._1, (temp._1._2, temp._1._3, temp._2._1.toFloat / temp._2._2.toFloat))
    }).groupByKey().map({
      temp => temp._1 + "|" + temp._2.mkString("|")
    }).coalesce(1).saveAsTextFile(savePath)
  }

  /**
    * 从原始 map maching 后的轨迹出发统计各个grid在各个时间区间的平均速度（这里是以轨迹为单位统计的，物理意义比上面的方式强）
    *
    * @param time_window 统计的时间窗大小
    * @param min_edges   过滤轨迹的最小边数
    * @param max_edges   过滤轨迹的最大边数
    * @param region      划分区域（左下角经纬度+右上角经纬度）
    * @param lon_split   经度划分数
    * @param lat_split   纬度划分数
    */
  def regionAvgSpeedFromRowTraj(time_window: Int, min_edges: Int, max_edges: Int, region: Array[Double], lon_split: Int, lat_split: Int): Unit = {
    val loader = new GraphLoader(sc)
    val edges_RDD = loader.loadEdgeFromDataSource(Parameter.edge_data_path)
    val edges_broadcast = sc.broadcast(edges_RDD.collect().toMap)
    val vertex_RDD = loader.loadVertexFromDataSource(Parameter.vertex_data_path)
    val vertex_broadcast = sc.broadcast(vertex_RDD.collect().toMap)
    val traj_RDD: RDD[(Long, Array[Long], Array[String])] = new TrajectoryLoader(sc).loadTrajectoryFromDataSource(Parameter.traj_data_path, min_edges, max_edges) //Parameter.traj_data_path
    val xytspeed_RDD: RDD[((Int, Int, String), Float)] = traj_RDD.flatMap({
      temp =>
        val edges = temp._2
        val times = temp._3
        var list: List[((Int, Int, String), Float)] = Nil
        var index: Int = 0
        while (index < edges.length - 1) {
          var xy: (Int, Int) = null
          xy = Tool.getGridXY(region, lon_split, lat_split, vertex_broadcast.value.get(edges_broadcast.value.get(edges(index)).get._1).get)
          if(xy._1 != -1 && xy._2!= -1) {
            var startTime: String = null;
            startTime = Tool.timeFormatByMinute(times(index), time_window)
            var distance: Float = 0;
            var timeDifference: Int = 0;
            while (index < edges.length - 1
              && xy == Tool.getGridXY(region, lon_split, lat_split, vertex_broadcast.value.get(edges_broadcast.value.get(edges(index)).get._1).get)
              && startTime == Tool.timeFormatByMinute(times(index), time_window)) {
              distance = distance + edges_broadcast.value.get(edges(index)).get._3
              timeDifference = timeDifference + Tool.timeDifference(times(index), times(index + 1))
              index = index + 1
            }
            if (timeDifference != 0) {
              list = ((xy._1, xy._2, startTime), distance / timeDifference.toFloat) :: list
            }
          }else{
            index = index + 1
          }
        }
        list
    })

    val savePath = Parameter.HDFS_BASE_RESULT_DIR + "TrafficConditionStatistic/regionAvgSpeedFromRowTraj" +
      "/min_edges=%d/max_edges=%d".format(min_edges,max_edges) +
      "/region=%f_%f_%f_%f/lon_split=%d/lat_split=%d".format(region(0), region(1), region(2), region(3), lon_split, lat_split) +
      "/time_window=%d/withNum".format(time_window)

    xytspeed_RDD.map({
      temp =>
        (temp._1, (temp._2, 1))
    }).reduceByKey((e1, e2) => (e1._1 + e2._1, e1._2 + e2._2)).map({
      temp => (temp._1._3, (temp._1._1, temp._1._2, temp._2._1 / temp._2._2.toFloat,temp._2._2))
    }).groupByKey().map({
      temp => temp._1 + "|" + temp._2.mkString("|")
    }).coalesce(1).saveAsTextFile(savePath)

  }

  /**
    * 挑选 包含从开始时间到结束时间（不包括）的轨迹段 的完整轨迹
    *
    * @param startTime
    * @param endTime
    */
  def chooseTrajs(startTime: String, endTime: String): Unit = {
    val source: RDD[(Long, Array[Long], Array[String])] = new TrajectoryLoader(sc).loadTrajectoryFromDataSource(Parameter.traj_data_path, 10, Int.MaxValue) //Parameter.traj_data_path
    val map: RDD[String] = source.flatMap({
      temp =>
        var list: List[String] = Nil
        val edgeids = temp._2
        val time = temp._3
        var choose: Boolean = false
        var s: String = "" + temp._1 + ","
        for (i <- 0 until edgeids.length) {
          val edgeid = edgeids(i)
          val start_time = time(i)
          if (start_time.toLong >= startTime.toLong && start_time.toLong < endTime.toLong) {
            choose = true
          }
          s += (start_time + "|" + edgeid + ",")
        }

        if (choose) {
          list = s :: list
        }
        list
    })
    val savePath = Parameter.HDFS_BASE_RESULT_DIR + "TrafficConditionStatistic/" + "chooseTrajs/%s/%s/".format(startTime, endTime)
    map.coalesce(1).saveAsTextFile(savePath)
  }

}

object Statisticer {
  def main(args: Array[String]): Unit = {
    //    System.setProperty("hadoop.home.dir", "c:\\winutils\\")
    val conf = new SparkConf()
      //      .setMaster("local[2]")
      .setAppName("TrafficConditionStatistics")
    val sc = new SparkContext(conf)
    val statisticer = new Statisticer(sc)
    //        statisticer.count(5,10,Int.MaxValue,0)
    //    val savePath = statisticer.regionCount2(Parameter.HDFS_BASE_RESULT_DIR + "TrafficConditionStatistic/5TimeWindow_10MinEdges_2147483647MaxEdges_0MinSectionLength",
    //      statisticer.region3, 48, 48)
    //    val savePath = "E:\\"
    //            val savePath = "/user/lvzhongjian/result/TrafficConditionStatistic/5TimeWindow_10MinEdges_2147483647MaxEdges_0MinSectionLength/detail/lon_split=80/lat_split=80/region=116.269540_39.828598_116.491670_39.997132"
    //    statisticer.regionAvgSpeedFromRegionCount2(savePath, 20)
    //    statisticer.chooseTrajs("20160229233000","20160301000000")

    statisticer.regionAvgSpeedFromRowTraj(20, 10, Int.MaxValue, statisticer.region3, 48, 48)

  }
}
