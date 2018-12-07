import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;

//说明：
//正常是可以直接用sklearn中xgb结果直接xgb.apply之类的拼接，此处是如果给了一个TXT的dump文件，再用java运行，速度更快
//xgb dump tree 恢复的第二步，第一步先用parse_xgboost_model.py文件处理dump.txt，再用本java文件进行树结构的还原，并生成样本的叶子节点进行拼接


class XgboostNode{
   int treeIndex ;
   String nodeType ;
   int nodeIndex ;
   float weight ;

   String feature;
   double maxValue;
   int leftIndex;
   int rightIndex;
   int defaultIndex;
}

public class xgb_dump_recover {

    Map<Integer, HashMap<Integer, XgboostNode>> xg_model = new HashMap<Integer, HashMap<Integer, XgboostNode> >();


    boolean loadXgboostModel() {
        String path = "xgb_dump_recover.data";
        try {
            String line = null;
            BufferedReader br = new BufferedReader(new FileReader(path));
            HashMap<Integer, XgboostNode> xg_tree = null;
            while ((line = br.readLine()) != null) {
                XgboostNode node = new XgboostNode();
                String[] segs = line.trim().split("\t");
                int tree_idx = Integer.parseInt(segs[0]);
                //可以通过tree_idx进行截断，选取前多少棵树
                if(xg_model.containsKey(tree_idx)){
                    xg_tree = xg_model.get(tree_idx);
                }
                else{
                    xg_tree = new HashMap<Integer, XgboostNode> ();
                    xg_model.put(tree_idx, xg_tree);
                }
                node.treeIndex = tree_idx;
                node.nodeType = segs[1].trim();
                node.nodeIndex = Integer.parseInt(segs[2]);
                if(node.nodeType.equals("leaf")){
                    node.weight = Float.parseFloat(segs[3]);
                }
                else if(node.nodeType.equals("node")){

                    node.feature = segs[3];
                    node.maxValue = Double.parseDouble(segs[4]);
                    node.leftIndex = Integer.parseInt(segs[5]);
                    node.rightIndex = Integer.parseInt(segs[6]);
                    node.defaultIndex = Integer.parseInt(segs[7]);
                }
                xg_tree.put(node.nodeIndex, node);
            }
        }
        catch(IOException e) {

            System.out.println("load xgboost model failed.");
            return false;
        }

        System.out.println("load xgboost model successfully.");
        return true;
    }

    List<String> constructNewFeatureByXgboost(Map<String,Double> onelineMap){
        List<String> new_feature_list = new ArrayList<String>();
        for (Integer tree_idx : xg_model.keySet()) {
            HashMap<Integer, XgboostNode> tree = xg_model.get(tree_idx);
            int node_idx = 0;
            while(true){
                XgboostNode node = tree.get(node_idx);
                if(node.nodeType.equals("node")){
                    if(onelineMap.containsKey(node.feature)){
                        double fea_value = onelineMap.get(node.feature);
                        node_idx = fea_value < node.maxValue ? node.leftIndex : node.rightIndex;
                    }
                    else{
                        node_idx = node.defaultIndex;
                        continue;
                    }
                }
                else if(node.nodeType.equals("leaf")){
                    String temp = "xg_{" + node.treeIndex + ":" + node.nodeIndex + "}";
                    new_feature_list.add(temp);


                    break;
                }
            }
        }
        return new_feature_list;
    }

    Map<String,Double> createMap(String line,String[] feature_list){

        Map<String, Double> xg_feature_build_map = new HashMap<String, Double>();

        String[] item = line.split(",");
        for(int i=0; i<feature_list.length; i++){
            if(item[i].length()!=0){
                xg_feature_build_map.put(feature_list[i],Double.parseDouble(item[i]));
            }
        }
        return xg_feature_build_map;
    }

    String map2csv(Map<String,Double> oneline_Map,LinkedList<String> all_feature){

        StringBuilder res = new StringBuilder();
        for(int i=0;i<all_feature.size();i++){
            String tmp = all_feature.get(i);
            if(oneline_Map.containsKey(tmp)){
                Double value = oneline_Map.get(tmp);

                res.append(value.toString());
                res.append(",");
            }else {
                res.append(",");
            }
        }
        res.deleteCharAt(res.length()-1);
        return res.toString();
    }

    public static void main(String[] args) throws IOException{

        xgb_dump_recover source = new xgb_dump_recover();
        source.loadXgboostModel();



        try {

            BufferedReader reader = new BufferedReader(new FileReader("xgb_test.csv"));
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File("xgb_result.csv"),true));


            String featureline = reader.readLine();
            String[] feature_list = featureline.split(",");

            HashSet<String> new_feature_all = new HashSet<String>();
            String line = null ;
            while((line=reader.readLine())!=null){
                Map<String,Double> oneline_Map = source.createMap(line , feature_list);
                List<String> xgb_new_feature_list = source.constructNewFeatureByXgboost(oneline_Map);

                for(int i=0;i<xgb_new_feature_list.size();i++){
                    new_feature_all.add(xgb_new_feature_list.get(i));
                }

            }

            reader.close();

            LinkedList<String> all_feature = new LinkedList<>();
            for(int i=0;i<feature_list.length;i++){
                all_feature.add(feature_list[i]);
            }
            Iterator< String> iterator = new_feature_all.iterator();
            while(iterator.hasNext()){
                all_feature.add(iterator.next());
            }


            StringBuilder firstline = new StringBuilder();
            for(int i=0;i<all_feature.size();i++){
                firstline.append(all_feature.get(i));
                firstline.append(",");
            }
            firstline.deleteCharAt(firstline.length()-1);
            writer.write(firstline.toString());


            BufferedReader reader_true = new BufferedReader(new FileReader("xgb_test.csv"));
            String featureline_tmp = reader_true.readLine();
            String line_true = null ;
            while((line_true=reader_true.readLine())!=null){
                Map<String,Double> oneline_Map = source.createMap(line_true , feature_list);
                List<String> xgb_new_feature_list = source.constructNewFeatureByXgboost(oneline_Map);

                for(int i=0;i<xgb_new_feature_list.size();i++){
                    oneline_Map.put(xgb_new_feature_list.get(i),1.0);
                }

                String newline = source.map2csv(oneline_Map,all_feature);
                writer.newLine();
                writer.write(newline);
            }
            reader_true.close();
            writer.close();
            }
        catch (Exception e) {
            e.printStackTrace();
        }



    }
}
