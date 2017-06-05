package com.ada.roadstatistics

import java.text.SimpleDateFormat
import java.util.Date

/**
  * Created by zhongjian on 2017/5/23.
  */

object Tool {


  /**
    * 求时间差
    *
    * @param start_time YYYYMMDDHHmmss形式
    * @param end_time
    * @return
    */
  def timeDifference(start_time: String, end_time: String): Int = {
    if (start_time.length != 14 || end_time.length != 14) {
      throw new Exception("time should be YYYYMMDDHHmmss")
    }

    var df: SimpleDateFormat = new SimpleDateFormat("YYYYMMDDHHmmss")
    var begin: Date = df.parse(start_time)
    var end: Date = df.parse(end_time)
    var between: Long = (end.getTime() - begin.getTime()) / 1000 //转化成秒
    between.toInt
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

    val substring = time.substring(0, 10)
    val substring1 = time.substring(10, 12)
    val minute = substring1.toInt
    var temp = time_window * (minute / time_window)
    if (temp < 10)
      substring + "0" + temp
    else
      substring + temp
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
    println(x_step)
    val y_step = (right_up_lat - left_bottom_lat) / lat_split
    println(y_step)
    var x, y = 0
    x = ((vertex_lon - left_bottom_lon) / x_step).toInt
    y = ((vertex_lat - left_bottom_lat) / y_step).toInt
    if (x >= lon_split)
      x = x - 1
    if (y >= lat_split)
      y = y - 1
    (x, y)
  }




  def main(args: Array[String]): Unit = {
    println(timeFormatByMinute("20160301000501",5))
//    println(timeDifference("20160301153959", "20160301155111"))
//    println(timeFormatByMinute("20160301153959", 5))
//    println("%dTimeWindow_%dMinEdges_%dMaxEdges_%dMinSectionLength".format(1, 1, 1, 1))
  }
}

