package com.ada.mapshow;

/**
 * Created by zhongjian on 2017/5/24.
 */
public class RoadWithNum {

    private long roadid;

    private int num;

    public RoadWithNum() {
        setRoadid(-1);
        setNum(0);
    }


    public RoadWithNum(long roadid) {
        setRoadid(roadid);
        setNum(0);
    }

    public RoadWithNum(long roadid, int num) {
        setRoadid(roadid);
        setNum(num);
    }

    public long getRoadid() {
        return roadid;
    }

    public int getNum() {
        return num;
    }

    public void setRoadid(long roadid) {
        this.roadid = roadid;
        setNum(0);
    }

    public void setNum(int num) {
        this.num = num;
    }

    public void addNum(int num) {
        this.num += num;
    }

    @Override
    public String toString() {
        return "RoadWithNum{" +
                "roadid=" + roadid +
                ", num=" + num +
                '}';
    }
}
