package com.ada.mapshow;

import com.ada.global.Parameter;

import java.io.*;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalField;
import java.time.temporal.WeekFields;
import java.util.*;
import java.util.function.BiConsumer;

/**
 * Created by zhongjian on 2017/6/1.
 */
public class AllAvgSpeedCondition {

    private Map<String, AvgSpeedCondition> map = new HashMap<>();

    private int time_window;

    private static AllAvgSpeedCondition allAvgSpeedCondition;

    private int x_num;

    private int y_num;

    public int getX_num() {
        return x_num;
    }

    public void setX_num(int x_num) {
        this.x_num = x_num;
    }

    public int getY_num() {
        return y_num;
    }

    public void setY_num(int y_num) {
        this.y_num = y_num;
    }

    public List<String> getTimes() {
        Set<String> strings = map.keySet();
        LinkedList<String> list = new LinkedList<>(strings);
        Collections.sort(list);
        return list;
    }

    public int getTime_window() {
        return time_window;
    }

    private void addAvgSpeedCondition(AvgSpeedCondition avgSpeedCondition) {
        map.put(avgSpeedCondition.getTime(), avgSpeedCondition);
    }

    /**
     * @param line 格式： time|(x,y,speed)|...
     */
    private void addAvgSpeedCondition(String line, int x_num, int y_num) {
        String[] split = line.split("\\|");
        String time = split[0];
        AvgSpeedCondition avgSpeedCondition = new AvgSpeedCondition(time, x_num, y_num);
        for (int i = 1; i < split.length; i++)
        {
            String[] split1 = split[i].substring(1, split[i].length() - 1).split(",");
            avgSpeedCondition.setSpeed(Integer.valueOf(split1[0]), Integer.valueOf(split1[1]), Float.valueOf(split1[2]));
            if (split1.length > 3)
            {
                avgSpeedCondition.setWeight(Integer.valueOf(split1[0]), Integer.valueOf(split1[1]), Integer.valueOf(split1[3]));
            }
        }
        addAvgSpeedCondition(avgSpeedCondition);
    }

    public AvgSpeedCondition get(String time) {
        return map.get(time);
    }


    private AllAvgSpeedCondition(int time_window) {
        this.time_window = time_window;
    }

    public static AllAvgSpeedCondition reLoad(String path) {
        File file = new File(path);
        String path1 = file.getName();
        String[] split = path1.split("_");
        int x_num = Integer.valueOf(split[0]);
        int y_num = Integer.valueOf(split[1]);
        int time_window = Integer.valueOf(split[2]);
        allAvgSpeedCondition = new AllAvgSpeedCondition(time_window);
        allAvgSpeedCondition.setX_num(x_num);
        allAvgSpeedCondition.setY_num(y_num);
        try
        {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));

            String s = null;
            while ((s = bufferedReader.readLine()) != null)
            {
                if (!s.trim().equals(""))
                    allAvgSpeedCondition.addAvgSpeedCondition(s, x_num, y_num);
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
        return allAvgSpeedCondition;
    }


    public static AllAvgSpeedCondition load(String path) {
        synchronized (AllAvgSpeedCondition.class)
        {
            if (allAvgSpeedCondition == null)
            {
                synchronized (AllAvgSpeedCondition.class)
                {
                    reLoad(path);
                }
            }
        }
        return allAvgSpeedCondition;
    }


    /**
     * 将平均速度转换成 拥堵（1），较为拥堵（2），较为通畅（3），通畅（4） 4个类别, 有些区域无路况信息，label用0表示
     *
     * @param dataPath
     */
    public static void toCondition(String dataPath) {
        AllAvgSpeedCondition load = AllAvgSpeedCondition.reLoad(dataPath);
        List<String> times = load.getTimes();
        int time_window = allAvgSpeedCondition.getTime_window();
        System.out.println("check data integrity");
        String startTime = times.get(0);
        String endTime = times.get(times.size() - 1);
        LocalDateTime startDateTime = LocalDateTime.parse(startTime, DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
        LocalDateTime endDateTime = LocalDateTime.parse(endTime, DateTimeFormatter.ofPattern("yyyyMMddHHmm"));
        long size = Duration.between(startDateTime, endDateTime).get(ChronoUnit.SECONDS) / (time_window * 60);
        //        assert size == times.size() : "data is not integrity";
        System.out.println(String.join(",", "start time is " + startTime, "end time is " + endTime
                , "size should be " + size, "actual is " + times.size(), "data integrity "+ (size == times.size())));


        String savePath = dataPath + "_condition";
        try
        {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(savePath));
            for (int i = 0; i < times.size(); i++)
            {
                AvgSpeedCondition avgSpeedCondition = load.get(times.get(i));
                float[][] speeds = avgSpeedCondition.getSpeeds();
                int[][] weight = avgSpeedCondition.getWeight();
                int x_num = avgSpeedCondition.getX_num();
                int y_num = avgSpeedCondition.getY_num();
                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.append(times.get(i));
                for (int j = 0; j < x_num; j++)
                {
                    for (int k = 0; k < y_num; k++)
                    {
                        int label = -1;
                        float v = (float) (speeds[j][k] * 3.6);
                        if (v >= 1 && v <= 15)
                        {//拥堵
                            label = 1;
                        }
                        else if (v > 15 && v <= 30)
                        {//较为拥堵
                            label = 2;
                        }
                        else if (v > 30 && v <= 45)
                        {//比较通畅
                            label = 3;
                        }
                        else if (v > 45)
                        {//通畅
                            label = 4;
                        }
                        else if (v == 0)//无路况区域
                        {
                            label = 0;
                        }
                        if (label != -1)
                        {
                            stringBuilder.append(String.format("|%d,%d,%d,%d", j, k, label,weight[j][k]));
                        }
                    }
                }
                bufferedWriter.write(stringBuilder.toString());
                bufferedWriter.write("\n");
            }
        }
        catch (IOException e)

        {
            e.printStackTrace();
        }

    }


    public void save(String path) {
        try
        {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(path));
            this.map.forEach(new BiConsumer<String, AvgSpeedCondition>() {
                @Override
                public void accept(String s, AvgSpeedCondition avgSpeedCondition) {
                    try
                    {
                        bufferedWriter.write(avgSpeedCondition.getLineString());
                        bufferedWriter.newLine();
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                    }
                }
            });
            bufferedWriter.flush();
            bufferedWriter.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        AllAvgSpeedCondition.toCondition(Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow\\2016\\03" + File.separator + "48_48_20_LinearInterpolationFixed");
        AllAvgSpeedCondition.toCondition(Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow\\2016\\03" + File.separator + "48_48_20_MaxSpeedFillingFixed_20");
        //        System.out.println(load.getTimes().size());
    }


}
