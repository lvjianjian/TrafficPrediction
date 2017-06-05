package com.ada.mapshow;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhongjian on 2017/5/24.
 */
public class RoadWithCondition {

    private long roadid;

    private List<Condition> conditionList;


    public RoadWithCondition(long roadid) {
        this.roadid = roadid;
        conditionList = new ArrayList<>();
    }

    public long getRoadid() {
        return roadid;
    }

    public List<Condition> getConditionList() {
        return conditionList;
    }

    public void insertCondition(Condition condition){
        this.getConditionList().add(condition);
    }

    public void insertCondition(String time,int allTravelTime,int num){
        insertCondition(new Condition(time,allTravelTime,num));
    }
}
