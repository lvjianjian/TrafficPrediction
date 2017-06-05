package com.ada.mapshow;

import com.ada.global.Parameter;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhongjian on 2017/5/24.
 */
public class Tool {

    public static List<Long>[] roadClassificationByNum(RoadsWithNum roadsWithNum, int[] split, String savepath) {
        List<Long>[] splits = new List[split.length];
        for (int i = 0; i < split.length; i++) {
            splits[i] = new ArrayList<>();
        }
        for (int i = 0; i < roadsWithNum.getRoadsWithNumList().size(); i++) {
            RoadWithNum roadWithNum = roadsWithNum.getRoadsWithNumList().get(i);
            int num = roadWithNum.getNum();
            for (int j = 0; j < splits.length - 1; j++) {
                if (num >= split[j] && num < split[j + 1]) {
                    splits[j].add(roadWithNum.getRoadid());
                    break;
                }
            }
            if (num >= split[split.length - 1])
                splits[split.length - 1].add(roadWithNum.getRoadid());
        }

        if (savepath != null) {
            try {
                BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(savepath));
                for (int i = 0; i < splits.length; i++) {
                    List<Long> split1 = splits[i];
                    StringBuilder stringBuilder = new StringBuilder();
                    for (int j = 0; j < split1.size(); j++) {
                        stringBuilder.append(split1.get(j)+",");
                    }
                    int n = split[i];
                    bufferedWriter.write(n+","+stringBuilder.toString());
                    bufferedWriter.newLine();
                }

                bufferedWriter.flush();
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        } else {
            System.err.println("save path is null, can't save");
        }

        return splits;

    }


//    public static void countGridInfo(String path){
//        try {
//            BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
//            String line = null;
//            while ((line = bufferedReader.readLine())!=null){
//                String substring = line.substring(1, line.length() - 1);
//                String[] split = substring.split(",");
//                String[] split1 = split[0].substring(1, split[0].length() - 1).split(",");
//                int x = Integer.valueOf(split1[0]);
//                int y = Integer.valueOf(split1[1]);
//                int edgeSize = Integer.valueOf(split[1]);
//
//            }
//
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

    public static void main(String[] args) {
//        int[] split = new int[]{100, 500, 1000, 2500, 5000, 10000};
//        roadClassificationByNum(RoadsWithNum.load(Parameter.PROJECTPATH + "result\\r_min100"), split,Parameter.PROJECTPATH + "result\\r_min100_classfication");
    }
}
