package com.ada.mapshow;

/**
 * Created by zhongjian on 2017/6/1.
 */
public class AvgSpeedCondition {

    private String time;

    private int x_num;

    private int y_num;

    private float[][] speeds;
    private int[][] weight;//每个grid所含的轨迹数

    public AvgSpeedCondition(String time,int x_num,int y_num){
        this.time = time;
        this.x_num = x_num;
        this.y_num = y_num;
        speeds = new float[x_num][y_num];
        weight = new int[x_num][y_num];
    }

    public void setSpeed(int x,int y, float speed){
        speeds[x][y] = speed;
    }

    public void setWeight(int x,int y,int num){
        weight[x][y] = num;
    }

    public String getTime() {
        return time;
    }


    public int getX_num() {
        return x_num;
    }


    public int getY_num() {
        return y_num;
    }


    public float[][] getSpeeds() {
        return speeds;
    }


    public int[][] getWeight() {
        return weight;
    }
}
