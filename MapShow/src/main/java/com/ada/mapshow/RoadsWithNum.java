package com.ada.mapshow;

import com.ada.global.Parameter;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zhongjian on 2017/5/24.
 *
 * 存放路段和通过路段的总轨迹数
 */
public class RoadsWithNum {
    private int minNum = 0;
    private List<RoadWithNum> roadWithNumList;

    public RoadsWithNum(int minNum) {
        this.minNum = minNum;
        roadWithNumList = new ArrayList<RoadWithNum>();
    }

    public RoadsWithNum() {
        roadWithNumList = new ArrayList<RoadWithNum>();
    }

    public RoadsWithNum(String path, int minNum) throws Exception {
        roadWithNumList = new ArrayList<RoadWithNum>();
        setMinNum(minNum);
        read(path);
    }

    public void setMinNum(int minNum) {
        this.minNum = minNum;
    }

    public int getMinNum() {
        return minNum;
    }

    public List<RoadWithNum> getRoadsWithNumList() {
        return roadWithNumList;
    }

    public void addRoadWithNum(RoadWithNum roadWithNum){
        if(roadWithNum.getNum() >= minNum) {
            getRoadsWithNumList().add(roadWithNum);
        }
    }


    /**
     *
     * @param s 格式：roadid|(time1,alltraveltime1,num1)|(time2,alltraveltime2,num2)|...
     */
    public void addRoadWithNum(String s){
        String[] split = s.split("\\|");
        Long edgeid = Long.valueOf(split[0]);
        RoadWithNum roadWithNum = new RoadWithNum(edgeid);
        for (int i = 1; i < split.length; i++) {
            String[] temp = split[i].substring(1, split[i].length() - 1).split(",");
            roadWithNum.addNum(Integer.valueOf(temp[2]));
        }
        addRoadWithNum(roadWithNum);
    }

    private void read(String path) throws Exception {
        File file = new File(path);
        if(!file.exists() || !file.isDirectory()){
            throw new Exception("path should be hadoop result directory");
        }
        File[] files = file.listFiles();
        for (int i = 0; i < files.length; i++) {
            File file1 = files[i];
            String name = file1.getName();
            System.out.println("read "+ name);
            if(name.startsWith("part")){
                BufferedReader bufferedReader = new BufferedReader(new FileReader(file1));
                String s = null;
                while ((s=bufferedReader.readLine())!=null){
                    addRoadWithNum(s);
                }
            }
        }

    }

    public void save(String path){
        try {
            path += String.format("_min%d", getMinNum());
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(path));
            bufferedWriter.write(getMinNum()+"");
            bufferedWriter.newLine();
            for (int i = 0; i < getRoadsWithNumList().size(); i++) {
                RoadWithNum roadWithNum = getRoadsWithNumList().get(i);
                bufferedWriter.write(String.format("%d,%d", roadWithNum.getRoadid(),roadWithNum.getNum()));
                bufferedWriter.newLine();
            }

            bufferedWriter.flush();
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static RoadsWithNum load(String path){
        RoadsWithNum roadsWithNum = new RoadsWithNum();
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
            String s = bufferedReader.readLine();
            roadsWithNum.setMinNum(Integer.valueOf(s));
            while ((s = bufferedReader.readLine())!=null){
                if(s.contains(",")){
                    String[] split = s.split(",");
                    roadsWithNum.addRoadWithNum(new RoadWithNum(Long.valueOf(split[0]),Integer.valueOf(split[1])));
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return roadsWithNum;
    }

    public static void main(String[] args) {
//        try {
//            RoadsWithNum roadsWithNum = new RoadsWithNum(Parameter.PROJECTPATH + "5TimeWindow_10MinEdges_2147483647MaxEdges_20MinSectionLength",100);
//            roadsWithNum.save(Parameter.PROJECTPATH+"result" + File.separator + "r");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        RoadsWithNum roadsWithNumList = RoadsWithNum.load(Parameter.PROJECTPATH + "result\\r_min100");
        System.out.println(roadsWithNumList.getRoadsWithNumList().size());

    }

}
