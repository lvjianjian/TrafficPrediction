package com.ada.roadstatistics;
/**
  * Created by zhongjian on 2017/5/23.
  */

object Parameter{
  /**
    * hdfs node front part of url
    */
  val HDFS_NODE_FRONT_PART = "hdfs://node1:9000"
  /**
    * traj data path in hdfs
    */
  val traj_data_path: String = HDFS_NODE_FRONT_PART+"/user/caojiaqing/JqCao/data/trajectory_beijing_new.txt"

  /**
    * edge data path in hdfs
    */
  val edge_data_path: String = HDFS_NODE_FRONT_PART+"/user/lvzhongjian/data/edges_new.txt"

  /**
    * vertex data path in hdfs
    */
  val vertex_data_path:String = HDFS_NODE_FRONT_PART + "/user/caojiaqing/JqCao/data/vertices_new.txt"

  /**
    * base result dir to save in hdfs
    */
  val HDFS_BASE_RESULT_DIR = "/user/lvzhongjian/result/"


  /**
    * 2016 edge data path in hdfs
    */
  val new_edge_data_path: String = HDFS_NODE_FRONT_PART+"/data/GaoTong/2016/R-G.csv"


  /**
    * 2016 traj data path in hdfs
    */
  val new_traj_data_path: String = HDFS_NODE_FRONT_PART+"/data/GaoTong/2016/"
}
