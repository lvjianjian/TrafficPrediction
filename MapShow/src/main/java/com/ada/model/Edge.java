package com.ada.model;

/**
 * Created by JQ-Cao on 2016/6/6.
 */
public class Edge {
    private long id;
    private long startVertex;
    private long endVertex;
    private double dis;

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public long getStartVertex() {
        return startVertex;
    }

    public void setStartVertex(long startVertex) {
        this.startVertex = startVertex;
    }

    public long getEndVertex() {
        return endVertex;
    }

    public void setEndVertex(long endVertex) {
        this.endVertex = endVertex;
    }

    public double getDis() {
        return dis;
    }

    public void setDis(double dis) {
        this.dis = dis;
    }

    public Edge(long id, long startVertex, long endVertex, double dis) {
        this.id = id;
        this.startVertex = startVertex;
        this.endVertex = endVertex;
        this.dis = dis;
    }


    @Override
    public String toString() {
        return "Edge{" +
                "id=" + id +
                ", startVertex=" + startVertex +
                ", endVertex=" + endVertex +
                ", dis=" + dis +
                '}';
    }
}
