package com.ada.mapshow.servlet;

import com.ada.global.Parameter;
import com.ada.mapshow.AllAvgSpeedCondition;
import com.ada.mapshow.AvgSpeedCondition;
import com.ada.mapshow.RoadsWithNum;
import com.ada.mapshow.Tool;
import com.ada.model.Edge;
import com.ada.model.RoadMap;
import com.ada.model.Trajectory;
import com.ada.model.Vertex;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zhongjian on 2017/5/25.
 */
@Controller
@RequestMapping(value = "/show")
public class ShowController {

    private double[] region = new double[]{116.26954, 39.828598, 116.49167, 39.997132};
    //    private String speedPath2 = Parameter.PROJECTPATH + "data" + File.separator + "5TimeWindow_10MinEdges_maxMaxEdges_0MinSectionLength" + File.separator + "48_48_20";
    private String rawSpeedPath = Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow" + File.separator + "2016" + File.separator + "03" + File.separator + "48_48_20";
    private String maxFixedSpeedPath = Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow" + File.separator + "2016" + File.separator + "03" + File.separator + "48_48_20_MaxSpeedFillingFixed_20";
    private String inearInterpolationSpeedFixed = Parameter.PROJECTPATH + "data" + File.separator + "avgspeedfromrow" + File.separator + "2016" + File.separator + "03" + File.separator + "48_48_20_LinearInterpolationFixed";
    private String speedPath = inearInterpolationSpeedFixed;


    @RequestMapping(value = "/map.do")
    public String show() {
        return "map";
    }


    @RequestMapping(value = "/roadsWithNum.json", method = RequestMethod.POST)
    @ResponseBody
    public HashMap<String, Object> roadsWithNum() {
        RoadMap load = RoadMap.load(Parameter.VERTEXPATH, Parameter.EDGEPATH);
        System.out.println("request /show/roadsWithNum.json");
        int[] split = new int[]{2500, 5000, 10000};
        List<Long>[] lists = Tool.roadClassificationByNum(RoadsWithNum.load(Parameter.PROJECTPATH + "result\\r_min100"), split, null);
        List<double[]>[] doubles = new ArrayList[lists.length];
        Map<Long, Vertex> vertexMap = load.getVertexMap();
        for (int i = 0; i < lists.length; i++)
        {
            List<Long> list = lists[i];
            doubles[i] = new ArrayList<>();
            for (int j = 0; j < list.size(); j++)
            {
                Edge edge = load.getEdgeMap().get(list.get(j));
                doubles[i].add(new double[]{vertexMap.get(edge.getStartVertex()).getLongitude(), vertexMap.get(edge.getStartVertex()).getLatitude(), vertexMap.get(edge.getEndVertex()).getLongitude(), vertexMap.get(edge.getEndVertex()).getLatitude()});
            }
        }
        HashMap<String, Object> result = new HashMap<>();
        result.put("split", split);
        result.put("lists", doubles);
        return result;
    }

    @RequestMapping(value = "/times.json", method = RequestMethod.POST)
    @ResponseBody
    public Map<String, Object> getTimes() {
        AllAvgSpeedCondition load = AllAvgSpeedCondition.load(speedPath);
        HashMap<String, Object> objectObjectHashMap = new HashMap<>();
        objectObjectHashMap.put("xnum", load.getX_num());
        objectObjectHashMap.put("ynum", load.getY_num());
        objectObjectHashMap.put("region", region);
        objectObjectHashMap.put("times", load.getTimes());
        return objectObjectHashMap;
    }

    @RequestMapping(value = "/getByTime.json", method = RequestMethod.POST)
    @ResponseBody
    public HashMap<String, Object> getAvgSpeedConditionByTime(String time) {
        AllAvgSpeedCondition load = AllAvgSpeedCondition.load(speedPath);
        AvgSpeedCondition avgSpeedCondition = load.get(time);
        float[][] speeds = avgSpeedCondition.getSpeeds();
        int[][] weights = avgSpeedCondition.getWeight();
        HashMap<String, Object> objectObjectHashMap = new HashMap<>();
        objectObjectHashMap.put("speeds", speeds);
        objectObjectHashMap.put("weights", weights);
        return objectObjectHashMap;
    }


    @RequestMapping(value = "/someTrajs.json", method = RequestMethod.POST)
    @ResponseBody
    public List<Trajectory> getSomeTrajs() {
        List<Trajectory> trajectories = new ArrayList<>();
        try
        {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(Parameter.PROJECTPATH + "data\\20160229233000_20160301000000"));
            String s = null;
            while ((s = bufferedReader.readLine()) != null)
            {
                if (s.contains(","))
                {
                    Trajectory trajectory = Trajectory.readOne(s);
                    trajectories.add(trajectory);
                }
            }
        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        } catch (IOException e)
        {
            e.printStackTrace();
        }
        return trajectories;
    }

}
