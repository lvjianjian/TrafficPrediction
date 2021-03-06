package com.ada.roadstatistics

import java.time.{Duration, LocalDateTime, ZoneOffset}
import java.time.format.DateTimeFormatter

import org.apache.log4j.Logger

/**
  * Created by zhongjian on 2017/5/23.
  */

object Tool {

  val logger: Logger = Logger.getLogger(getClass.getName)


  /**
    * 求时间差
    *
    * @param start_time yyyyMMddHHmmss形式
    * @param end_time
    * @return
    */
  def timeDifference(start_time: String, end_time: String): Int = {
    if (start_time.length != 14 || end_time.length != 14) {
      throw new Exception("time should be yyyyMMddHHmmss")
    }

    val parse = LocalDateTime.parse(start_time, DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
    return Duration.between(parse, LocalDateTime.parse(end_time, DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))).getSeconds.toInt


  }


  /**
    * 将time转化为某个时间起点，这个time包含在时间起点+time_window下
    *
    * @param time        YYYYMMDDHHmmss形式
    * @param time_window 单位：分钟
    * @return
    */
  def timeFormatByMinute(time: String, time_window: Int): String = {
    if (time_window > 60 || time_window < 1)
      throw new Exception("time_window should be in 1 to 60 minutes")

    var substring:String = time.substring(0, 8)
    val substring1 = time.substring(10, 12)
    val minute = substring1.toInt
    val hour = time.substring(8,10).toInt
    var z = ((hour * 60) + minute) / time_window
    z = z * time_window
    val temp = z / 60
    val m = (z - temp * 60)
    if (temp < 10)
      substring = substring + "0" + temp
    else
      substring = substring + temp
    if (m < 10)
      substring = substring + "0" + m
    else
      substring = substring + m
    substring
  }


  /**
    *
    * @param timeLong 长整形时间
    * @return 字符串时间，格式: YYYYMMDDHHmmss形式
    */
  def longToStringTime(timeLong: Long): String = {
    return LocalDateTime.ofEpochSecond(timeLong, 0, ZoneOffset.of("+0800")).format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))
  }

  /**
    * 获取某点在grid中的位置，X对应经度划分后所在第几列（从0开始），Y对应纬度划分后点所在第几行（0开始）
    *
    * 如下划分成 2x2
    * -     -     -
    * |0，1 |1，1 |
    * -     -     -
    * |0，0 |1，0 |
    * -     -     -
    *
    * 若点不在任何格子返回（-1，-1）
    *
    * @param region       左下角经纬度和右上角经纬度
    * @param lon_split    经度划分数
    * @param lat_split    纬度划分数
    * @param vertexLonLat 点的经纬度
    * @return
    */
  def getGridXY(region: Array[Double], lon_split: Int, lat_split: Int, vertexLonLat: (Double, Double)): (Int, Int) = {
    val vertex_lon = vertexLonLat._1
    val vertex_lat = vertexLonLat._2
    val left_bottom_lon = region(0)
    val left_bottom_lat = region(1)
    val right_up_lon = region(2)
    val right_up_lat = region(3)


    if (vertex_lon < left_bottom_lon || vertex_lon > right_up_lon
      || vertex_lat < left_bottom_lat || vertex_lat > right_up_lat)
      return (-1, -1)

    val x_step = (right_up_lon - left_bottom_lon) / lon_split
    //    println(x_step)
    val y_step = (right_up_lat - left_bottom_lat) / lat_split
    //    println(y_step)
    var x, y = 0
    x = ((vertex_lon - left_bottom_lon) / x_step).toInt
    y = ((vertex_lat - left_bottom_lat) / y_step).toInt
    if (x >= lon_split)
      x = x - 1
    if (y >= lat_split)
      y = y - 1
    (x, y)
  }

  /**
    * 获取轨迹第index段的起点
    *
    * @param edges
    * @param index
    * @param start
    * @param end
    * @param traj
    * @return
    */
  def getLonLat(edges: Map[Long, ((Double, Double), (Double, Double), Int, Int)], index: Int, start: (Double, Double), end: (Double, Double), traj: Array[Long]): (Double, Double) = {
    if (index == 0)
      start
    else if (index == traj.length - 1)
      end
    else {
      val edge = edges.get(traj(index))
      if (edge != None)
        edge.get._1
      else {
        logger.error("warning！！！ miss " + traj(index))
        (-1D, -1D)
      }
    }
  }


  def inRoadClass(edges: Map[Long, ((Double, Double), (Double, Double), Int, Int)], edge_id: Long, road_classes: Array[Int]): Boolean = {
    val edge = edges.get(edge_id)
    if (edge != None) {
      val rc: Int = edge.get._4
      val r:Boolean = road_classes.contains(rc) // 加了这句话就不行？？ = .= 在调用地方引入了死循环，需要注意
      r
    }else{
      false
    }
  }

  def road_class2int(road_class:String): Int ={
    var r = -1
    r = road_class match {
      case "0x00" => 0
      case "0x01" => 1
      case "0x02" => 2
      case "0x03" => 3
      case "0x04" => 4
      case "0x06" => 6
      case "0x08" => 8
      case "0x09" => 9
      case "0x0b" => 11
    }

    r
  }

  def main(args: Array[String]): Unit = {

    val s = "1|(1,1,1)"
    val split = s.split("\\|")
    println(split(0))
    println(split(1))

    println(split(1).split("\\(")(1).split("\\)")(0).split(",")(0))
    //    println(timeFormatByMinute("20160301000501",5))
    //    println(timeDifference("20160229235957", "20160301000000"))
    //    println(timeDifference("20160229235957", "20160229235959"))
    //    println(timeDifference("20160229235957", "20160302000001"))
    //    println(timeFormatByMinute("20160301153959", 5))
    //    println("%dTimeWindow_%dMinEdges_%dMaxEdges_%dMinSectionLength".format(1, 1, 1, 1))
    //    println(getGridXY(Array(116.26954, 39.828598, 116.49167, 39.997132),80,80,(116.386463,39.845849)))
    //    val str = "LINESTRING (116.41563 39.7214, 116.41946 39.72185)"
    //    println(str.split("\\(")(1).split("\\)")(0).split(",")(0).split(" ")(0))

    //    println(longToStringTime(1459999642))
    //    println(longToStringTime(1460002237))
    println(timeFormatByMinute("201607290740",20))
    println(timeFormatByMinute("20160301124501",50))
    println(timeFormatByMinute("20160301123501",50))
    println(timeFormatByMinute("20160301125501",50))
    println(timeFormatByMinute("20160301122201",50))
    println(timeFormatByMinute("20160301120501",50))
    println(timeFormatByMinute("20160301235501",50))
  }
}

