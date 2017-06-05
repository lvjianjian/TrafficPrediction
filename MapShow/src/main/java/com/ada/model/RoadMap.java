package com.ada.model;

import com.ada.global.Parameter;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by zhongjian on 2017/5/25.
 */
public class RoadMap {

    private static RoadMap roadMap;


    //暂时只有边和点，没有拓扑结构
    private Map<Long,Vertex> vertexMap;
    private Map<Long,Edge> edgeMap;


    public Map<Long, Edge> getEdgeMap() {
        return edgeMap;
    }

    public Map<Long, Vertex> getVertexMap() {
        return vertexMap;
    }

    private RoadMap(){
        this.vertexMap = new HashMap<>();
        this.edgeMap = new HashMap<>();
    }

    public static RoadMap load(String vertex_path,String edge_path){
        if(roadMap == null){
            synchronized (RoadMap.class){
                if(roadMap == null){
                    read(vertex_path,edge_path);
                }
            }
        }
        return roadMap;
    }

    private static void read(String vertex_path,String edge_path){
        roadMap = new RoadMap();
        try {
            BufferedReader vertexReader = new BufferedReader(new FileReader(vertex_path));
            String s = null;
            while ((s = vertexReader.readLine())!=null){
                if(s.trim().equals(""))
                    continue;
                String[] split = s.split("\t");
                Vertex vertex = new Vertex(Long.valueOf(split[0]),Double.valueOf(split[2]),Double.valueOf(split[1]));
                roadMap.vertexMap.put(vertex.getId(),vertex);
            }
            vertexReader.close();
            BufferedReader edgeReader = new BufferedReader(new FileReader(edge_path));
            while ((s= edgeReader.readLine())!=null){
                if(s.trim().equals(""))
                    continue;
                String[] split = s.split("\t");
                Edge edge = new Edge(Long.parseLong(split[0]), Long.parseLong(split[1]), Long.parseLong(split[2]), Double.parseDouble(split[3]));
                roadMap.edgeMap.put(edge.getId(),edge);
            }
            edgeReader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
        RoadMap load = RoadMap.load(Parameter.PROJECTPATH + "data" + File.separator + "vertices_new.txt",
                Parameter.PROJECTPATH + "data" + File.separator + "edges_new.txt");
        System.out.println(load.getEdgeMap().size());
        System.out.println(load.getVertexMap().size());


    }
}
