package com.ada.preprocess;

import com.ada.mapshow.AllAvgSpeedCondition;
import com.ada.mapshow.AvgSpeedCondition;

import java.util.List;

/**
 * Created by zhongjian on 2017/7/17.
 */
public class LinearInterpolationFixed extends SpeedFixed {
    @Override
    public void fixed(String oldDataPath) {
        System.out.println("start " + getMethodName());
        AllAvgSpeedCondition allAvgSpeedCondition = AllAvgSpeedCondition.reLoad(oldDataPath);

        String savePath = oldDataPath + "_" + getMethodName();

        List<String> times = allAvgSpeedCondition.getTimes();
        List<Integer> noSpeedRegion = findNoSpeedRegion(oldDataPath);
        int y_num = allAvgSpeedCondition.getY_num();
        int x_num = allAvgSpeedCondition.getX_num();
        double min_speed = 0.25;
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < y_num; j++)
            {
                //                System.out.println("process "+i+","+j);
                if (noSpeedRegion.contains(xyToInt(i, j, y_num)))//无路况区域，直接置为0
                {
                    for (int k = 0; k < times.size(); k++)
                    {
                        AvgSpeedCondition avgSpeedCondition = allAvgSpeedCondition.get(times.get(k));
                        avgSpeedCondition.getSpeeds()[i][j] = 0;
                    }
                }
                else//其他情况作线性插值
                {
                    int k = 0;
                    float preSpeed = 0;
                    int startIndex = 0;
                    while (k < times.size())
                    {

                        String timeIndex = times.get(k);
                        AvgSpeedCondition avgSpeedCondition = allAvgSpeedCondition.get(timeIndex);
                        float speed = avgSpeedCondition.getSpeeds()[i][j];

                        if (speed >= min_speed)
                        {
                            preSpeed = speed;
                            ++startIndex;
                        }
                        else
                        {
                            while (k < times.size())//找到后一个速度不为0的值
                            {
                                if (allAvgSpeedCondition.get(times.get(k)).getSpeeds()[i][j] >= min_speed)
                                    break;
                                ++k;
                            }
                            if (k == times.size())
                                break;
                            float postSpeed = allAvgSpeedCondition.get(times.get(k)).getSpeeds()[i][j];
                            //如果前速度没有，按后速度填
                            if (preSpeed < min_speed)
                            {
                                for (int l = startIndex; l < k; l++)
                                {
                                    allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] = postSpeed;
                                    if (allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] == 0)
                                    {
                                        System.out.println(i + "," + j + "," + l);
                                    }
                                }
                            }
                            else//前后速度都有，线性插值
                            {
                                int interpolationSize = k - startIndex;
                                double addValue = (postSpeed - preSpeed) / (interpolationSize + 1);
                                for (int l = startIndex; l < k; l++)
                                {
                                    allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] = (float) (preSpeed + addValue * (l - startIndex + 1));
                                    if (allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] == 0)
                                    {
                                        System.out.println(i + "," + j + "," + l);
                                    }
                                }
                            }

                            startIndex = k;
                            preSpeed = postSpeed;
                        }
                        ++k;
                    }
                    if (startIndex < k)//后面没有的按preSpeed填充
                    {
                        for (int l = startIndex; l < k; l++)
                        {
                            allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] = preSpeed;
                            if (allAvgSpeedCondition.get(times.get(l)).getSpeeds()[i][j] == 0)
                            {
                                System.out.println(i + "," + j + "," + l);
                            }
                        }
                    }
                }
            }

        }
        //        modifyNoCondition(allAvgSpeedCondition,noSpeedRegion);

        allAvgSpeedCondition.save(savePath);
        allAvgSpeedCondition.info(noSpeedRegion);
        System.out.println("finish " + getMethodName());


    }


    @Override
    public String getMethodName() {
        return "LinearInterpolationFixed";
    }

    public static void main(String[] args) {
        new LinearInterpolationFixed().fixed("E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\MapShow\\data\\avgspeedfromrow\\2016\\03\\48_48_20");
    }
}
