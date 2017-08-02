package com.ada.mapshow;

import com.ada.global.Parameter;
import com.ada.preprocess.LinearInterpolationFixed;
import com.ada.preprocess.MaxSpeedFillingFixed;
import com.ada.preprocess.SpeedFixed;

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

    public static double inCompleteDaysThreshold = 0.3;

    private Map<String, AvgSpeedCondition> map = new HashMap<>();

    private List<String> removeDays = new ArrayList<>();

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
        allAvgSpeedCondition.removeIncompleteDays(inCompleteDaysThreshold);
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
        System.out.println(String.join(",", "start time is " + startTime, "end time is " + endTime, "size should be " + size, "actual is " + times.size(), "data integrity " + (size == times.size())));

        int count = 0;
        //        System.out.println("size:"+times.size());
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
                        if (v > 0 && v <= 15)
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
                            stringBuilder.append(String.format("|(%d,%d,%d,%d)", j, k, label, weight[j][k]));
                        }

                        if (label > 0)
                            count++;
                    }
                }
                bufferedWriter.write(stringBuilder.toString());
                bufferedWriter.write("\n");
                bufferedWriter.flush();
            }
        }
        catch (IOException e)

        {
            e.printStackTrace();
        }

        System.out.println(">0 count:" + count);


    }


    public void save(String path) {
        try
        {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(path));
            List<String> times = this.getTimes();
            for (int i = 0; i < times.size(); i++)
            {
                String s1 = times.get(i);
                String s = s1.substring(0, 8);
                if (removeDays.contains(s))
                {
                    System.out.println("ignore " + s1);
                    continue;
                }
                AvgSpeedCondition avgSpeedCondition = this.map.get(s1);
                bufferedWriter.write(avgSpeedCondition.getLineString());
                bufferedWriter.newLine();
            }

            bufferedWriter.flush();
            bufferedWriter.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public void info(List<Integer> noSpeedRegions) {
        if (noSpeedRegions == null)
            noSpeedRegions = new LinkedList<>();
        List<String> times = allAvgSpeedCondition.getTimes();
        int x_num = allAvgSpeedCondition.getX_num();
        int y_num = allAvgSpeedCondition.getY_num();
        int count = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        String timeMin = null;
        String timeMax = null;
        for (int i = 0; i < times.size(); i++)
        {
            String timeIndex = times.get(i);
            AvgSpeedCondition avgSpeedCondition = allAvgSpeedCondition.get(timeIndex);
            float[][] speeds = avgSpeedCondition.getSpeeds();
            for (int x = 0; x < x_num; x++)
            {
                for (int y = 0; y < y_num; y++)
                {
                    if (noSpeedRegions.contains(SpeedFixed.xyToInt(x, y, y_num)))
                        continue;
                    if (speeds[x][y] > 0)
                    {
                        count++;
                    }
                    if (speeds[x][y] > max)
                    {
                        max = speeds[x][y];
                        timeMax = timeIndex;
                    }
                    if (speeds[x][y] < min)
                    {
                        min = speeds[x][y];
                        timeMin = timeIndex;
                    }

                }
            }

        }


        System.out.println("speed count :" + count + ", should be: " + (times.size() * (x_num * y_num - noSpeedRegions.size())));
        System.out.println("max speed " + max + ", in " + timeMax);
        System.out.println("min speed " + min + ", in " + timeMin);
    }

    public void removeIncompleteDays(double threshold) {
        int time_window = getTime_window();
        int size = 24 * 60 / time_window;
        size *= threshold;
        List<String> times = getTimes();
        int count = 1;
        String preTime = null;
        for (int i = 0; i < times.size(); i++)
        {
            String substring = times.get(i).substring(0, 8);
            if (!substring.equals(preTime))
            {

                if (preTime != null)
                {
                    if (count < size)
                    {
                        removeDays.add(preTime);
                    }

                    LocalDate pre = LocalDate.parse(preTime, DateTimeFormatter.ofPattern("yyyyMMdd"));
                    while (pre.plusDays(1).compareTo(LocalDate.parse(substring, DateTimeFormatter.ofPattern("yyyyMMdd"))) != 0)
                    {
                        pre = pre.plusDays(1);
                        removeDays.add(pre.format(DateTimeFormatter.ofPattern("yyyyMMdd")));
                    }

                }
                preTime = substring;
                count = 1;
            }
            else
            {
                count++;
            }

        }

        if (preTime != null)
        {
            if (count < size)
            {
                removeDays.add(preTime);
            }
        }

        for (int i = 0; i < removeDays.size(); i++)
        {
            System.out.println("remove" + removeDays.get(i));
        }
        fill();
    }

    private void fill() {
        List<String> times = getTimes();
        String startTime = times.get(0);
        for (int i = 1; i < times.size(); i++)
        {
            String timeIndex = times.get(i);
            LocalDateTime pre = LocalDateTime.parse(startTime, Parameter.dateTimeFormatter);
            LocalDateTime now = LocalDateTime.parse(timeIndex, Parameter.dateTimeFormatter);
            while (pre.plusMinutes(getTime_window()).compareTo(now) != 0)
            {
                pre = pre.plusMinutes(getTime_window());
                String format = pre.format(Parameter.dateTimeFormatter);
                if (removeDays.contains(format.substring(0, 8)))
                {
                    continue;
                }
                this.map.put(format, new AvgSpeedCondition(format, getX_num(), getY_num()));
            }
            startTime = timeIndex;
        }

    }


    public static void main(String[] args) {
        String path = "E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\MapShow\\data\\avgspeedfromrow\\2016\\all\\48_48_20";
        AllAvgSpeedCondition load = AllAvgSpeedCondition.reLoad(path);
        new MaxSpeedFillingFixed().fixed(path);
        new LinearInterpolationFixed().fixed(path);
        //        load.save("E:\\ZhongjianLv\\project\\jamprediction\\RoadStatistics\\MapShow\\data\\avgspeedfromrow\\48_48_20");
        //        AllAvgSpeedCondition.toCondition(Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow\\2016\\03" + File.separator + "48_48_20_LinearInterpolationFixed");
        //        AllAvgSpeedCondition.toCondition(Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow\\2016\\03" + File.separator + "48_48_20_MaxSpeedFillingFixed_20");


        //        System.out.println(load.getTimes().size());
    }


}
