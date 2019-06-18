/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package KNNClassificationRegression;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

/**
 *
 * @author dev
 */
public class KNN {

    private boolean isClassification = true,
            isNormalized = false,
            isTestInstanceNormalized = false;

    private int nearestNeighborCount = 1,
            instanceCount = 0,
            instanceAttributesCount = 0,
            maxAttributesCount = 0,
            indexOfMaxAttributesCountInstance = 0;

    private HashMap currentTrainingInstance,
            testInstance,
            testInstanceNormalized,
            attrDataMap,
            attrMean,
            attrStddev,
            nearestNeighbours;

    private ArrayList trainingInstanceList,
            trainingInstanceListNormalized,
            trainingInstanceLabels,
            instancesMSE,
            attrDataList;

    public KNN(int nearestNeighborCount, boolean isClassification) {

        //condition checks
        this.nearestNeighborCount = nearestNeighborCount;
        this.isClassification = isClassification;

        trainingInstanceList = new ArrayList<HashMap<String, Double>>();
        trainingInstanceLabels = new ArrayList<Object>();

    }

    public void createInstance() {
        //

        if (trainingInstanceLabels.size() != trainingInstanceList.size()) {
            System.err.println("Perhaps a label wasn't assigned to a previous instance.");
            return;
        }

        // Count the number of attributes of new instance
        this.instanceAttributesCount = 0;

        // Create a new instance and points to it
        HashMap<String, Double> newInstance = new HashMap<String, Double>();
        this.currentTrainingInstance = newInstance;
        trainingInstanceList.add(newInstance);
        instanceCount++;
        isNormalized = false;
    }

    public void setInstanceAttr(String attrName, Double val) {
        //this.currentInst.put(attrName, val);

        this.currentTrainingInstance.put(attrName, val);

        this.instanceAttributesCount++;
        if (this.instanceAttributesCount > this.maxAttributesCount) {
            this.maxAttributesCount = this.instanceAttributesCount;
            this.indexOfMaxAttributesCountInstance = (instanceCount - 1);
        }
    }

    public void setInstanceLabel(Object label) {
        trainingInstanceLabels.add((instanceCount - 1), label);
    }

    public int getCurrentInstanceCount() {
        return instanceCount;
    }

    public int getMaxAttributesInstanceCountIndex() {
        return indexOfMaxAttributesCountInstance;
    }

    public void createTestInstance() {
        this.testInstance = new HashMap<String, Double>();
        isTestInstanceNormalized = false;
    }

    public void setTestInstanceAttr(String attrName, Double val) {

        this.testInstance.put(attrName.toString(), val);

        this.instanceAttributesCount++;
        if (this.instanceAttributesCount > this.maxAttributesCount) {
            this.maxAttributesCount = this.instanceAttributesCount;
            this.indexOfMaxAttributesCountInstance = (instanceCount - 1);
        }
    }

    public void zScoreNormalization() {

        attrDataMap = new HashMap<String, LinkedList<Double>>();
        attrMean = new HashMap<String, Double>();
        attrStddev = new HashMap<String, Double>();
        HashMap<String, Double> maxAttrInstance = (HashMap<String, Double>) this.trainingInstanceList.get(this.indexOfMaxAttributesCountInstance);
        HashMap<String, Double> tempInstance;

        // Find mean of each attribute and aggregate attribute value to a linkedlist
        for (String key : maxAttrInstance.keySet()) {
            double sum = 0, mean = 0;
            attrDataList = new ArrayList();
            for (int instanceCounter = 0; instanceCounter < this.trainingInstanceList.size(); instanceCounter++) {
                tempInstance = (HashMap<String, Double>) this.trainingInstanceList.get(instanceCounter);
                if (tempInstance.containsKey(key.toString())) {
                    double attrVal = tempInstance.get(key.toString());
                    attrDataList.add(attrVal);
                    sum += attrVal;
                    //System.out.println(key.toString()+":"+sum);
                }
            }
            mean = sum / attrDataList.size();
            //System.out.println("sum:"+sum+",attrDataList.size()"+attrDataList.size());
            //System.out.println(key+"mean:"+mean);
            attrMean.put(key.toString(), mean);
            attrDataMap.put(key.toString(), attrDataList);

            //System.out.println("mean"+mean);
        }

        // For each attribute, calculate the corresponding stddev
        for (String key : maxAttrInstance.keySet()) {
            ArrayList tempAttrDataList = (ArrayList) attrDataMap.get(key.toString());

            double attrMeanVal = (double) attrMean.get(key.toString());
            double meanDiff = 0;
            double meanSquaredDiffSum = 0;
            double stdDev = 0;

            for (int tempAttrDataListCounter = 0; tempAttrDataListCounter < tempAttrDataList.size(); tempAttrDataListCounter++) {
                //System.out.println("size of "+key.toString()+" is "+tempAttrDataList.size());
                double x = (double) tempAttrDataList.get(tempAttrDataListCounter);
                meanDiff = x - attrMeanVal;
                /*
                 System.out.println("key:"+key.toString());
                 System.out.println("x:"+x+",attrMeanVal:"+attrMeanVal);
                 System.out.println("meanDiff"+meanDiff);
                 */
                meanSquaredDiffSum += (meanDiff * meanDiff);
                //System.out.println(x+","+meanDiff+","+meanSquaredDiffSum);
            }

            stdDev = Math.sqrt((meanSquaredDiffSum / tempAttrDataList.size()));
            attrStddev.put(key.toString(), stdDev);
        }

        trainingInstanceListNormalized = new ArrayList();
        HashMap<String, Double> normalizedInstance, tempTrainingInstance;

        for (int trainingInstanceCounter = 0; trainingInstanceCounter < trainingInstanceList.size(); trainingInstanceCounter++) {

            tempTrainingInstance = (HashMap<String, Double>) trainingInstanceList.get(trainingInstanceCounter);
            normalizedInstance = new HashMap();
            for (String key : maxAttrInstance.keySet()) {
                if (tempTrainingInstance.containsKey(key.toString())) {
                    double instanceAttrVal = tempTrainingInstance.get(key.toString());
                    double attributeMean = (double) attrMean.get(key.toString());
                    double attributeStddev = (double) attrStddev.get(key.toString());
                    double z = 0;
                    if (attributeStddev != 0) {
                        z = (instanceAttrVal - attributeMean) / attributeStddev;
                    }
                    normalizedInstance.put(key.toString(), z);
                    //System.out.println("normalized " + instanceAttrVal + " to " + z);
                    //System.out.println("\t mean=" + attributeMean + ",stddev=" + attributeStddev);
                } else {
                    normalizedInstance.put(key.toString(), 3.0);
                }
            }
            trainingInstanceListNormalized.add(normalizedInstance);
        }

        isNormalized = true;
    }

    public void testInstNormalization() {
        HashMap<String, Double> tempTestInstance = this.testInstance;
        testInstanceNormalized = new HashMap<String, Double>();

        for (String key : tempTestInstance.keySet()) {

            ArrayList tempAttrDataList = (ArrayList) attrDataMap.get(key);
            int attrDataSize = tempAttrDataList.size();

            Double mean = (double) attrMean.get(key.toString());
            Double stddev = (double) attrStddev.get(key.toString());
            Double x = (double) tempTestInstance.get(key.toString());

            if (mean == null || stddev == null) {
                System.err.println("Execution halted: cannot find corresponding attributes in training dataset.");
                new Exception().printStackTrace();
                return;
            }

            double z = 0;
            if (stddev != 0) {
                z = (x - mean) / stddev;
            }

            testInstanceNormalized.put(key.toString(), z);

        }

        isTestInstanceNormalized = true;
    }

    public Object predict() {

        // Check if every instance has a label assigned
        if (trainingInstanceLabels.size() != trainingInstanceList.size()) {
            System.err.println("Perhaps a label wasn't assigned to a previous instance.");
            return false;
        }

        // Check if value in every attributes are normalized
        if (!isNormalized) {
            zScoreNormalization();
        }

        // Normalize test isntance
        if (!isTestInstanceNormalized) {
            testInstNormalization();
        }

        instancesMSE = new ArrayList();

        // Get the normalized instance that contains maximum number of attributes
        HashMap<String, Double> maxAttrInstance = (HashMap<String, Double>) trainingInstanceListNormalized.get(indexOfMaxAttributesCountInstance);
        HashMap<String, Double> tempTrainingInst = null;
        for (int trainingInstanceListCounter = 0; trainingInstanceListCounter < trainingInstanceListNormalized.size(); trainingInstanceListCounter++) {
            double mse = 0;
            tempTrainingInst = (HashMap<String, Double>) trainingInstanceListNormalized.get(trainingInstanceListCounter);
            for (String key : maxAttrInstance.keySet()) {
                if (!testInstanceNormalized.containsKey(key.toString())) {
                    testInstanceNormalized.put(key.toString(), -3);
                }
                double trainingInstValue = (double) tempTrainingInst.get(key.toString());
                double testInstValue = (double) testInstanceNormalized.get(key.toString());
                //System.out.println(trainingInstValue + "," + testInstValue);

                mse += Math.abs(testInstValue - trainingInstValue);
            }
            instancesMSE.add(trainingInstanceListCounter, mse);
        }

        // Find the k nearest neighbour
        nearestNeighbours = new HashMap<Integer, Double>();
        Object[] tempMSEList = instancesMSE.toArray();

        for (int j = 0; j < tempMSEList.length; j++) {
            //System.out.println(tempMSEList[j]);
        }

        // Find the nearest neighbour
        for (int nn_counter = 0; nn_counter < this.nearestNeighborCount; nn_counter++) {
            double min_mse = Double.MAX_VALUE;
            int min_mse_index = -1;
            for (int k = 0; k < tempMSEList.length; k++) {
                double mse = (double) tempMSEList[k];
                if (mse < min_mse) {
                    min_mse = mse;
                    min_mse_index = k;
                }
            }
            tempMSEList[min_mse_index] = Double.MAX_VALUE;
            nearestNeighbours.put(min_mse_index, min_mse);
        }

        return (this.isClassification) ? classify(nearestNeighbours) : regress(nearestNeighbours);
    }

    public String classify(HashMap nearestNeighbours) {
        HashMap<String,Integer> labelFrequency=new HashMap();
        for(Object key:nearestNeighbours.keySet()){
            int instanceIndex=(int)key;
            String instanceLabel=(String) this.trainingInstanceLabels.get(instanceIndex);
            if(labelFrequency.containsKey(instanceLabel)){
                int frequency=labelFrequency.get(instanceLabel);
                labelFrequency.put(instanceLabel, frequency+1);
            }else{
                labelFrequency.put(instanceLabel, 1);
            }
        }
        
        String predictedLabel="null";
        int maxFreq=0;
        for(String key:labelFrequency.keySet()){
            if(labelFrequency.get(key) > maxFreq){
                maxFreq=labelFrequency.get(key);
                predictedLabel=key;
            }
            
        }
        
        
        return predictedLabel;
    }

    public double regress(HashMap nearestNeighbours) {

        return 0.0;
    }
}
