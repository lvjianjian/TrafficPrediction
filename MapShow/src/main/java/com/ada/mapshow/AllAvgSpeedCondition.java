package com.ada.mapshow;

import com.ada.global.Parameter;

import java.io.*;
import java.util.*;

/**
 * Created by zhongjian on 2017/6/1.
 */
public class AllAvgSpeedCondition {

    private Map<String,AvgSpeedCondition> map = new HashMap<>();

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

    public List<String> getTimes(){
        Set<String> strings = map.keySet();
        LinkedList<String> list = new LinkedList<>(strings);
        Collections.sort(list);
        return list;
    }

    public int getTime_window() {
        return time_window;
    }

    private void addAvgSpeedCondition(AvgSpeedCondition avgSpeedCondition){
            map.put(avgSpeedCondition.getTime(),avgSpeedCondition);
    }

    /**
     *
     * @param line 格式： time|(x,y,speed)|...
     */
    private void addAvgSpeedCondition(String line,int x_num,int y_num){
        String[] split = line.split("\\|");
        String time = split[0];
        AvgSpeedCondition avgSpeedCondition = new AvgSpeedCondition(time,x_num,y_num);
        for (int i = 1; i < split.length; i++) {
            String[] split1 = split[i].substring(1, split[i].length() - 1).split(",");
            avgSpeedCondition.setSpeed(Integer.valueOf(split1[0]),Integer.valueOf(split1[1]),Float.valueOf(split1[2]));
        }
        addAvgSpeedCondition(avgSpeedCondition);
    }

    public AvgSpeedCondition get(String time){
        return map.get(time);
    }


    private AllAvgSpeedCondition(int time_window){
        this.time_window = time_window;
    }

    public static AllAvgSpeedCondition reLoad(String path){
        File file = new File(path);
        String path1 = file.getName();
        String[] split = path1.split("_");
        int x_num = Integer.valueOf(split[0]);
        int y_num = Integer.valueOf(split[1]);
        int time_window = Integer.valueOf(split[2]);
        allAvgSpeedCondition = new AllAvgSpeedCondition(time_window);
        allAvgSpeedCondition.setX_num(x_num);
        allAvgSpeedCondition.setY_num(y_num);
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));

            String s = null;
            while ((s = bufferedReader.readLine())!=null){
                if(!s.trim().equals(""))
                    allAvgSpeedCondition.addAvgSpeedCondition(s,x_num,y_num);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return allAvgSpeedCondition;
    }


    public static void main(String[] args) {
        AllAvgSpeedCondition load = AllAvgSpeedCondition.load(Parameter.PROJECTPATH + "data" + File.separator + "80_80_30");
        System.out.println(load.getTimes().size());
    }

    public static AllAvgSpeedCondition load(String path) {
        synchronized (AllAvgSpeedCondition.class) {
            if (allAvgSpeedCondition == null) {
                synchronized (AllAvgSpeedCondition.class) {
                    reLoad(path);
                }
            }
        }
        return allAvgSpeedCondition;
    }




}
