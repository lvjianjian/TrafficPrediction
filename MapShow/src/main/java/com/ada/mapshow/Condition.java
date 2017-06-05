package com.ada.mapshow;

/**
 * Created by zhongjian on 2017/5/24.
 */
public class Condition {
    private String time;//YYYYMMDDHHmm格式
    private int allTravelTime;
    private int num;


    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public int getAllTravelTime() {
        return allTravelTime;
    }

    public void setAllTravelTime(int allTravelTime) {
        this.allTravelTime = allTravelTime;
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    public Condition(String time, int allTravelTime, int num) {
        this.time = time;
        this.allTravelTime = allTravelTime;
        this.num = num;
    }

    public Condition(){}
}
