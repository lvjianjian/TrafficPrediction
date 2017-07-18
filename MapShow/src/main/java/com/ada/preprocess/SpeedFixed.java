package com.ada.preprocess;

import com.ada.mapshow.AllAvgSpeedCondition;
import com.ada.mapshow.AvgSpeedCondition;

import java.io.*;
import java.util.*;
import java.util.function.Consumer;

/**
 * Created by zhongjian on 2017/7/14.
 */
public abstract class SpeedFixed {


    public abstract void fixed(String oldDataPath);


    public abstract String getMethodName();

    public List<Integer> findNoSpeedRegion(String oldDataPath) {
        AllAvgSpeedCondition allAvgSpeedCondition = AllAvgSpeedCondition.load(oldDataPath);
        double threshold = 0.05;
        String savePath = oldDataPath + "_" + "noSpeedRegion" + "_" + threshold;
        int y_num = allAvgSpeedCondition.getY_num();
        int x_num = allAvgSpeedCondition.getX_num();
        List<Integer> result = new LinkedList<>();
        try
        {
            File file = new File(savePath);
            if (file.exists())
            {
                System.out.println("read " + savePath + " ing...");
                BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
                int size = new Integer(bufferedReader.readLine());
                String[] ints = bufferedReader.readLine().split(",");
                for (int i = 0; i < ints.length; i++)
                {
                    result.add(new Integer(ints[i]));
                }
                if (size != result.size())
                {
                    System.err.println("Error: " + savePath + " is wrong, please delete it.");
                    System.exit(1);
                }

            }
            else//有缓存文件
            {
                int[][] count = new int[x_num][y_num];

                List<String> times = allAvgSpeedCondition.getTimes();
                for (int k = 0; k < times.size(); k++)
                {
                    String timeIndex = times.get(k);
                    AvgSpeedCondition avgSpeedCondition = allAvgSpeedCondition.get(timeIndex);
                    for (int i = 0; i < allAvgSpeedCondition.getX_num(); i++)
                    {
                        for (int j = 0; j < allAvgSpeedCondition.getY_num(); j++)
                        {
                            if (avgSpeedCondition.getSpeeds()[i][j] != 0 && avgSpeedCondition.getWeight()[i][j] != 0)
                            {
                                ++count[i][j];
                            }
                        }
                    }
                }
                int size = times.size();
                for (int i = 0; i < allAvgSpeedCondition.getX_num(); i++)
                {
                    for (int j = 0; j < allAvgSpeedCondition.getY_num(); j++)
                    {

                        if ((float) count[i][j] / size < threshold)
                        {
                            result.add(xyToInt(i, j, y_num));
                        }
                    }
                }
            }
            try
            {
                BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(savePath));
                bufferedWriter.write(result.size() + "");
                bufferedWriter.newLine();
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < result.size(); i++)
                {
                    Integer integer = result.get(i);
                    stringBuilder.append(integer + ",");
                }
                bufferedWriter.write(stringBuilder.toString());

                bufferedWriter.flush();
                bufferedWriter.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }

        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        System.out.println("no speed region size is " + result.size());


        return result;
    }

    public static int xyToInt(int x, int y, int y_num) {
        return x * y_num + y;
    }

    public static int[] intToXY(int i, int y_num) {
        int y = i % y_num;
        int x = i / y_num;
        return new int[]{x, y};
    }

    public static void main(String[] args) {
        LinkedList<int[]> l = new LinkedList<>();
        l.add(new int[]{1, 2, 3});
        l.remove(new int[]{1, 2, 3});
        System.out.println(l.size());

        LinkedList<String> l2 = new LinkedList<>();
        l2.add("1,2");
        l2.remove("1,2");
        System.out.println(l2.size());
    }

}
