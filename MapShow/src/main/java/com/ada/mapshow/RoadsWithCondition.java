package com.ada.mapshow;

import java.util.ArrayList;
import java.util.List;
/**
 * Created by zhongjian on 2017/5/24.
 */
public class RoadsWithCondition {

    private List<RoadWithCondition> roadWithConditions;

    public RoadsWithCondition(String path){
        roadWithConditions = new ArrayList<>();
        read(path);
    }

    private void read(String path){

    }

    public List<RoadWithCondition> getRoadWithConditions() {
        return roadWithConditions;
    }

    public static void main(String[] args) {
        new RoadsWithCondition("5TimeWindow_10MinEdges_2147483647MaxEdges_20MinSectionLength");
    }
}
