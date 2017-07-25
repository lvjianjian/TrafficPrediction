package com.ada.preprocess;

import com.ada.mapshow.AllAvgSpeedCondition;
import com.ada.mapshow.AvgSpeedCondition;

import java.util.List;

/**
 * Created by zhongjian on 2017/7/14.
 */
public class MaxSpeedFillingFixed extends SpeedFixed {


    @Override
    public void fixed(String olddatapath) {
        System.out.println("start " + getMethodName());
        AllAvgSpeedCondition allAvgSpeedCondition = AllAvgSpeedCondition.reLoad(olddatapath);
        int WEIGHT_THRESHOLD = allAvgSpeedCondition.getTime_window() * 1;
        String savePath = olddatapath + "_" + getMethodName() + "_" + WEIGHT_THRESHOLD;
        List<Integer> noSpeedRegion = findNoSpeedRegion(olddatapath);
        List<String> times = allAvgSpeedCondition.getTimes();
        int x_num = allAvgSpeedCondition.getX_num();
        int y_num = allAvgSpeedCondition.getY_num();
        for (int i = 0; i < times.size(); i++)
        {
            String timeIndex = times.get(i);
            AvgSpeedCondition avgSpeedCondition = allAvgSpeedCondition.get(timeIndex);
            for (int k = 0; k < x_num; k++)
            {
                for (int j = 0; j < y_num; j++)
                {
                    if (noSpeedRegion.contains(xyToInt(k, j, y_num)))
                    {
                        avgSpeedCondition.getSpeeds()[k][j] = 0;
                    }
                    else if (avgSpeedCondition.getWeight()[k][j] < WEIGHT_THRESHOLD)
                    {
                        //                        System.out.println(timeIndex);
                        //                        System.out.println(k+","+j);
                        //                        System.out.println("old:"+avgSpeedCondition.getSpeeds()[k][j]);
                        avgSpeedCondition.getSpeeds()[k][j] = 15;
                        //                        System.out.println("fixed:"+avgSpeedCondition.getSpeeds()[k][j]);
                        //                        System.out.println(allAvgSpeedCondition.get("201602291800").getSpeeds()[0][0]);
                        //                        System.exit(1);
                    }
                }
            }
        }
        //        System.out.println(noSpeedRegion.size());
        //        for (int i = 0; i < noSpeedRegion.size(); i++) {
        //            System.out.println(noSpeedRegion.get(i)[0]+","+noSpeedRegion.get(i)[1]);
        //        }
//        modifyNoCondition(allAvgSpeedCondition,noSpeedRegion);
        allAvgSpeedCondition.save(savePath);
        System.out.println("finish " + getMethodName());
    }

    @Override
    public String getMethodName() {
        return "MaxSpeedFillingFixed";
    }


    public static void main(String[] args) {
        new MaxSpeedFillingFixed().fixed("E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\MapShow\\data\\avgspeedfromrow\\2016\\03\\48_48_20");
    }
}
