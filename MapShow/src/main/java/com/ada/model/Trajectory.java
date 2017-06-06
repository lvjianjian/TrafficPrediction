package com.ada.model;

import com.ada.global.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhongjian on 2017/6/5.
 */
public class Trajectory {

    private long id;
    private List<String> times;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private List<Long> edgeIds;
    private List<double[]> lonlats;

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }


    public List<String> getTimes() {
        return times;
    }

    public void setTimes(List<String> times) {
        this.times = times;
    }

    public List<Long> getEdgeIds() {
        return edgeIds;
    }


    public List<double[]> getLonlats() {
        return lonlats;
    }

    public void setEdgeIds(List<Long> edgeIds) {
        this.edgeIds = edgeIds;
        //转换到经纬度时候抛掉最后一个点使得长度与时间对应
        lonlats = new ArrayList<double[]>();
        for (int i = 0; i < edgeIds.size(); i++) {
            Long edgeID = edgeIds.get(i);
            RoadMap load = RoadMap.load(Parameter.VERTEXPATH,Parameter.EDGEPATH);
            long startVertex = load.getEdgeMap().get(edgeID).getStartVertex();
            double longitude = load.getVertexMap().get(startVertex).getLongitude();
            double latitude = load.getVertexMap().get(startVertex).getLatitude();
            lonlats.add(new double[]{longitude,latitude});
        }
    }

    /**
     *
     * @param s 格式 id,time|edgeid,...,
     * @return
     */
    public static Trajectory readOne(String s){
        Trajectory trajectory = new Trajectory();
        String[] split = s.split(",");
        trajectory.setId(Long.valueOf(split[0]));
        List<String> times = new ArrayList<>();
        List<Long> edgeids = new ArrayList<>();
        for (int i = 1; i < split.length; i++) {
            String[] timeAndEdgeID = split[i].split("\\|");
            times.add(timeAndEdgeID[0]);
            edgeids.add(Long.valueOf(timeAndEdgeID[1]));
        }

        trajectory.setEdgeIds(edgeids);
        trajectory.setTimes(times);
        return trajectory;
    }



}
