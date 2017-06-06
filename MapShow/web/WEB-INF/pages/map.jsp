<%--
  Created by IntelliJ IDEA.
  User: zhongjian
  Date: 2017/5/24
  Time: 21:56
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Map</title>
    <%--<script type="text/javascript" src="http://webapi.amap.com/maps?v=1.3&key=f3718f6d9327dc4fcce481a250bfd306"></script>--%>
    <script src="/js/jquery-3.2.1.min.js"></script>
    <link href="/css/bootstrap.css" rel="stylesheet">
    <script src="/js/bootstrap.js"></script>
    <script src="/leaflet/leaflet.js"></script>
    <script src="/leaflet/leaflet.ChineseTmsProviders.js"></script>
    <link rel="stylesheet" href="/leaflet/leaflet.css">


    <style type="text/css">
        .block {
            width: 20px;
            height: 20px;
            margin: 5px;
            opacity: 1;
            display: inline-block;
        }

        .label {
            float: right;
            margin: 5px;
            height: 20px;
            color: #0f0f0f;
        }

    </style>
</head>
<body>
<div class="row">
    <div id="container" style="width: 100% ;height: 100%;"></div>
    <div style="position:absolute; top: 58px; right: 8px; z-index: 1000;margin: 10px;padding: 10px;margin: 5px">
        <div class="btn-group" data-toggle="buttons" style="margin-left: 10px;width: 100%;">
            <label class="btn btn-primary" style="width: 44%" id="noshow">
                <input type="radio" name="show" autocomplete="off"> 不显示
            </label>
            <label class="btn btn-primary active" style="width: 44%;margin-left: 5px" id="speedshow">
                <input type="radio" name="show" autocomplete="off" checked="checked"> 平均速度
            </label>
        </div>
        <br/>
        <select name="selectTime" id="selectTime" style="margin: 10px;font-size: 25px;padding-right: 20px">
        </select>
        <br/>
        <div>
            <button id="previous" type="button" class="btn btn-success"
                    style="width: 44%;font-size: 15px;margin-left: 10px">Previous
            </button>
            <button id="next" type="button" class="btn btn-success" style="width: 44%;font-size: 15px">Next</button>
        </div>

        <div style="padding-top: 5px">
            <button id="showTrajs" type="button" class="btn btn-success"
                    style="width: 65%;font-size: 15px;margin-left: 10px">显示轨迹
            </button>
        </div>
        <div>
            <textarea id="trajsInfo" style = "font-size:20px;margin-top: 5px;margin-left: 10px" rows="10" hidden readonly></textarea>
        </div>
    </div>


    <div id="legendSwitch" class="legendSwitchClose"
         style="position:absolute; bottom: 58px; right: 8px; z-index: 1000;margin: 10px;padding: 10px; cursor: pointer; background: rgba(255,255,255,0.7);">
    </div>
</div>

<script type="text/javascript">

    var color = ["#00CCFF", "#0033FF", "9933FF", "FF0033"]


    //显示部分路段轨迹数量
    //    $(document).ready(function () {
    //        $.ajax({
    //            type: "POST",
    //            url: "/show/roadsWithNum.json",
    //            async: true,
    //            cache: false,
    //            dataType: "json",
    //            success: function (data) {
    //                var split = data.split;
    //                var lists = data.lists
    //                var opacity = 1
    //                for (i = 0; i < lists.length; ++i) {
    //                    var speedsPolygon = lists[i]
    //                    for (j = 0; j < speedsPolygon.length; j += 10) {
    //                        var doubles = speedsPolygon[j]
    //                        var line = getPolyline(color[i], 5, "solid", opacity)
    //                        lineArr = []
    //                        lineArr.push([doubles[0], doubles[1]])
    //                        lineArr.push([doubles[2], doubles[3]])
    //                        line.setPath(lineArr);
    //                        line.setMap(map);
    //
    //                    }
    //                }
    //            },
    //            error: function (XMLHttpRequest, textStatus, errorThrown) {
    //                alert(XMLHttpRequest.responseText);
    //            }
    //        });
    //    })


    var speedColors = ["#FF6600", "#FFCC00", "#99CC00", "#CCFF00", "#33FF00"]
    var speedSplit = 15;


    //leaflet 加载高德地图
    var normalm = L.tileLayer.chinaProvider('GaoDe.Normal.Map', {
//        maxZoom: 18,
        minZoom: 5
    });
    var normal = L.layerGroup([normalm])
    var map = L.map("container", {
        center: [39.90923, 116.397428],
        zoom: 13,
        layers: [normal],
        zoomContro: false,
//        maxZoom: 18,
        minZoom: 13
    });


    //高德地图
    //    var map = new AMap.Map('container', {
    //        mapStyle: "amap://styles/light",
    //        zoom: 13,
    //        center: [116.397428, 39.90923]
    //    });

    //设置经纬度分为多少段
    var partX = 80, partY = 80;
    //北京区域所在经度
    var rightUpX = 116.49167, leftBottomX = 116.26954;
    //北京区域所在纬度
    var rightUpY = 39.997132, leftBottomY = 39.828598;

    //计算经纬度相差数量
    var differX = rightUpX - leftBottomX;
    var differY = rightUpY - leftBottomY;
    //计算经纬度每段间隔多少
    var intervalX = differX / partX;
    var intervalY = differY / partY;
    var speedsPolygon = new Array()
    var initialPolygon = new Array()
    var times;
    var timeIndex = -1;
    var showIndex = 1;//显示按钮组索引

    var trajsPolyline = new Array()
    var pointsCircle = new Array()
    var trajs = null;


    function getBytime() {
        if (showIndex == 1) {
            $.ajax({
                type: "POST",
                url: "/show/getByTime.json",
                async: false,
                cache: false,
                dataType: "json",
                data: {
                    time: times[timeIndex]
                },
                success: function (data) {
                    clearSpeeds()
                    drawSpeeds(data)
                }
            })
        }
    }

    $(document).ready(function () {
        //去除建筑物，标注，背景，只保留道路  bg，road，building，point
//        var features = [];
//        features.push("road");
//        map.setFeatures(features)

        //显示时间下拉框
        for (i = 0; i < speedColors.length; ++i) {
            var value
            if (i != 0)
                if (i != speedColors.length - 1)
                    value = "" + (i) * speedSplit + "-" + (i + 1) * speedSplit
                else
                    value = "" + (i) * speedSplit + "+"
            else
                value = "1-" + (i + 1) * speedSplit
            $("#legendSwitch").append('<div> <div class="block" style="background:' + speedColors[i] + ';opacity:0.8"></div> <div class="label">' + value + '</div> </div>')
        }

        $.ajax({
            type: "POST",
            url: "/show/times.json",
            async: false,
            cache: false,
            dataType: "json",
            success: function (data) {
                partX = data.xnum;
                partY = data.ynum;
                leftBottomX = data.region[0];
                leftBottomY = data.region[1];
                rightUpX = data.region[2];
                rightUpY = data.region[3];

                //计算经纬度相差数量
                differX = rightUpX - leftBottomX;
                differY = rightUpY - leftBottomY;
                //计算经纬度每段间隔多少
                intervalX = differX / partX;
                intervalY = differY / partY;

                times = data.times;
                for (i = 0; i < times.length; ++i) {
                    $("#selectTime").append('<option value="' + i + '">' + timeFormat(times[i]) + '</option>')
                }


                initialize();

                timeIndex = 0;
                getBytime()

                $("#selectTime").change(function () {
                    timeIndex = $("#selectTime").val();
                    getBytime()

                })

                //显示前一时间段的网格图
                $("#previous").click(function () {
                    if (timeIndex <= 0) {
                        timeIndex = 0
                    } else {
                        timeIndex = parseInt(timeIndex) - 1;
                        getBytime()
                        $("#selectTime").val(timeIndex)
                    }

                })


                //显示后一时间段的网格图
                $("#next").click(function () {
                    if (timeIndex < times.length - 1) {
                        timeIndex = parseInt(timeIndex) + 1;
                        getBytime()
                        $("#selectTime").val(timeIndex)
                    }
                })


                //不显示网格图
                $("#noshow").click(function () {
                    if (showIndex != 0) {
                        showIndex = 0
                        clearSpeeds()
                        clearInitial()
                    }
                })


                //显示平均速度网格图
                $('#speedshow').on('click', function () {
                    if (showIndex != 1) {
                        showIndex = 1
                        initialize()
                        getBytime()
                    }
                })
                //显示轨迹
                $('#showTrajs').click(function () {
                    if ($('#showTrajs').text().trim() == "显示轨迹") {
                        $('#showTrajs').text("取消显示轨迹")
                        if (trajs == null) {
                            $.ajax({
                                type: "POST",
                                url: "/show/someTrajs.json",
                                async: false,
                                cache: false,
                                dataType: "json",
                                success: function (data) {
                                    trajs = data
                                }
                            })
                        }
                        $('#trajsInfo').attr("hidden",false)
                        trajsPolyline.push(drawOneTrajs(0))

                    } else {
                        $('#showTrajs').text("显示轨迹")
                        $('#trajsInfo').attr("hidden",true)
                        $('#trajsInfo').text("")
                        var length = trajsPolyline.length
                        for (i = 0; i < length; ++i)
                            trajsPolyline.pop().remove()
                        length = pointsCircle.length
                        for (i = 0; i < length; ++i)
                            pointsCircle.pop().remove()
                    }
                })


            }
        })
    })


    function timeFormat(time) {
        return time.substring(0, 4) + "/" + time.substring(4, 6) + "/" + time.substring(6, 8) + " " + time.substring(8, 10) + ":" + time.substring(10, 12)
    }

    //从m/s转换到km/h
    function speedFormat(speed) {
        return speed * 3.6
    }

    function clearSpeeds() {
        var length = speedsPolygon.length
        for (i = 0; i < length; ++i) {
            speedsPolygon.pop().remove()
        }
    }

    function clearInitial() {
        var length = initialPolygon.length
        for (i = 0; i < length; ++i) {
            initialPolygon.pop().remove()
        }
    }


    function drawSpeeds(speeds) {
        for (i = 0; i < speeds.length; ++i) {
            for (j = 0; j < speeds.length; ++j) {
                var speed = speedFormat(speeds[i][j])
                if (speed > 1) {
                    var number = speed / speedSplit;
                    if (number > speedColors.length) {
                        number = speedColors.length - 1;
                    }
                    lX = (leftBottomX + intervalX * i).toString();
                    lTopX = (leftBottomX + intervalX * i).toString();
                    rTopX = (leftBottomX + intervalX * (i + 1)).toString();
                    rX = (leftBottomX + intervalX * (i + 1)).toString();

                    lY = (leftBottomY + intervalY * j).toString();
                    lTopY = (leftBottomY  + intervalY * (j + 1)).toString();
                    rTopY = (leftBottomY  + intervalY * (j + 1)).toString();
                    rY = (leftBottomY  + intervalY * j).toString();
                    speedsPolygon.push(drawGrid([lX, lY, lTopX, lTopY, rTopX, rTopY, rX, rY], speedColors[parseInt(number)], 0.8))
                }

            }
        }
    }

    function drawOneTrajs(trajIndex) {


        var traj = trajs[trajIndex];
        var lonlats = traj.lonlats;
        var latlngs = new Array()
        for (i = 0; i < lonlats.length; ++i) {
            latlngs.push([lonlats[i][1], lonlats[i][0]])
        }
//        alert(traj.times.length)
        var polyline = L.polyline(latlngs, {color: 'red'}).addTo(map);
        for (i = 0; i < latlngs.length; ++i) {
//            alert(traj.times[i]+","+latlngs[i])
            $('#trajsInfo').text($('#trajsInfo').text()+traj.times[i]+","+latlngs[i][1]+","+latlngs[i][0] + "\n")
            pointsCircle.push(L.circle(latlngs[i], {radius: 1, color: "blue", attribution: "1"}).addTo(map).on('click', function (e) {
//                alert(traj.times[i])

//                alert(e.getAttribution())
            }));

        }
        // zoom the map to the polyline
        return polyline
    }


    //        var arr = new Array();//经纬度坐标数组
    //        var num = 0;//设置多边形id
    //        arr.push(new AMap.LngLat(lonlats[0], lonlats[1]));
    //        arr.push(new AMap.LngLat(lonlats[2], lonlats[3]));
    //        arr.push(new AMap.LngLat(lonlats[4], lonlats[5]));
    //        arr.push(new AMap.LngLat(lonlats[6], lonlats[7]));
    //        var polygon = new AMap.Polygon({
    //            id: "polygon" + num,//多边形ID
    //            path: arr,//多边形顶点经纬度数组
    //            strokeColor: "#272727",//线颜色
    //            strokeOpacity: 0.2,//线透明度
    //            strokeWeight: 3, //线宽
    //            fillColor: fillColor,//填充色
    //            fillOpacity: fillOpacity, //填充透明度
    //            map: map
    //        });
    //        num++;
    function drawGrid(lonlats, fillColor, fillOpacity) {
        var polygon = L.polygon([
            [lonlats[1], lonlats[0]],
            [lonlats[3], lonlats[2]],
            [lonlats[5], lonlats[4]],
            [lonlats[7], lonlats[6]]
        ], {color: "#272727", opacity: 0.2, weight: 1, fillColor: fillColor, fillOpacity: fillOpacity}).addTo(map);
        return polygon
    }

    function initialize() {
        //var point = new AMap.LngLat(116.404, 39.915); // 创建点坐标
        // mapObj.setCenter(point); // 设置地图中心点坐标

        //116.26954, 39.828598, 116.49167, 39.997132

        //北京：北纬39度54分20秒，东经116度25分29秒；全市南北跨纬度1度37分，东西跨经度2度05分。


        // alert(intervalX+":"+intervalY);
        //以上参数设置完毕后，这里是按照从左到右的顺序来确定各个区域的经纬度

        var lX, lTopX, rTopX, rX;
        var lY, lTopY, rTopY, rY;


        for (var i = 0; i < partX; i++) {
            for (var j = 0; j < partY; j++) {
                lX = (leftBottomX + intervalX * i).toString();
                lTopX = (leftBottomX + intervalX * i).toString();
                rTopX = (leftBottomX + intervalX * (i + 1)).toString();
                rX = (leftBottomX + intervalX * (i + 1)).toString();

                lY = (leftBottomY + intervalY * j).toString();
                lTopY = (leftBottomY  + intervalY * (j + 1)).toString();
                rTopY = (leftBottomY  + intervalY * (j + 1)).toString();
                rY = (leftBottomY  + intervalY * j).toString();
                initialPolygon.push(drawGrid([lX, lY, lTopX, lTopY, rTopX, rTopY, rX, rY], "#4f4f4f", 0.3))
            }
        }
    }


    //    var polyline1 = new AMap.Polyline({
    //        //设置线覆盖物路径
    //        strokeColor: "#3366FF", //线颜色
    //        strokeOpacity: 1,       //线透明度
    //        strokeWeight: 5,        //线宽
    //        strokeStyle: "solid",   //线样式
    //        strokeDasharray: [10, 5], //补充线样式
    //        map: map
    //    });
    //
    //    function getPolyline(color, width, style, opacity) {
    //        return polyline1 = new AMap.Polyline({
    //            //设置线覆盖物路径
    //            strokeColor: color, //线颜色
    //            strokeOpacity: opacity,       //线透明度
    //            strokeWeight: width,        //线宽
    //            strokeStyle: style,   //线样式
    //            strokeDasharray: [10, 5] //补充线样式
    //        });
    //    }


    //地图点击出现经纬度坐标事件
    //    var clickEventListener = map.on('click', function (e) {
    //        alert(e.lnglat.getLng() + ',' + e.lnglat.getLat())
    //    });


</script>
</body>
</html>
