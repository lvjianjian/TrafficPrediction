package com.ada.global;

import java.io.File;
import java.time.format.DateTimeFormatter;

/**
 * Created by zhongjian on 2017/5/24.
 */
public class Parameter {

    public static final String PROJECTPATH = "E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\MapShow\\";
    public static final String VERTEXPATH = Parameter.PROJECTPATH + "data" + File.separator + "vertices_new.txt";
    public static final String EDGEPATH = Parameter.PROJECTPATH + "data" + File.separator + "edges_new.txt";

    public static final  DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm");

    public static int MAXSPEEDFIXEDTHRESHOLD = 5;

}
