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
            instancesCount = 0,
            instanceAttributesCount = 0,
            maxAttributesCount = 0,
            indexOfMaxAttributesCountInstance = 0;

    private HashMap currentTrainingInstance,
            testInstance,
            testInstanceNormalized,
            mapOfAggregatedAttributeValue,
            attrMean,
            attrStddev,
            nearestNeighbours;

    private ArrayList listOfTrainingInstance,
            listOfTrainingInstancesNormalized,
            listOfTrainingInstanceLabel,
            listOfInstanceMeanSquaredError,
            listOfAttributeValue;

    public KNN(int nearestNeighborCount, boolean isClassification) {

        // Setting for count for nearest neighbours
        // Setting for KNN classification or regression
        this.nearestNeighborCount = nearestNeighborCount;
        this.isClassification = isClassification;

        // Training instance is represent using HashMap<String,Double>
        // listOfTrainingInstance links all the training instances
        listOfTrainingInstance = new ArrayList<HashMap<String, Double>>();

        // listOfTrainingInstanceLabel stores the label for each training instance
        // The position of each label corresponds to listOfTrainingInstance
        // E.g. listOfTrainingInstance.get(0) has label of 'listOfTrainingInstanceLabel.get(0)'
        listOfTrainingInstanceLabel = new ArrayList<Object>();

    }

    // This method creates training instance using an iterative approach
    public void createInstance() {

        // If label(s) are missing in training data, return
        if (listOfTrainingInstanceLabel.size() != listOfTrainingInstance.size()) {
            System.err.println("Perhaps a label wasn't assigned to a previous instance.");
            return;
        }

        // Records the attributes count for the training instance
        // 
        this.instanceAttributesCount = 0;

        // Create a new instance
        HashMap<String, Double> newInstance = new HashMap<String, Double>();

        // Classwide reference points to the new instance, the reference is used at setInstanceAttr subsequently
        this.currentTrainingInstance = newInstance;

        // Append the new training instance to the list
        listOfTrainingInstance.add(newInstance);
        instancesCount++;

        // Any new training instance add, mark the flag to false
        isNormalized = false;
    }

    public void setInstanceAttr(String attrName, Double val) {

        // Adding attribute and its value to the training instance
        this.currentTrainingInstance.put(attrName, val);

        // Increments the counter, to detects the maximum number of attributes
        // The purspose is used to find the training instance with maximum number of attributes
        // Because some training instance might have missing attributes
        this.instanceAttributesCount++;
        if (this.instanceAttributesCount > this.maxAttributesCount) {
            this.maxAttributesCount = this.instanceAttributesCount;
            this.indexOfMaxAttributesCountInstance = (instancesCount - 1);
        }
    }

    public void setInstanceLabel(Object label) {
        listOfTrainingInstanceLabel.add((instancesCount - 1), label);
    }

    public int getCurrentInstanceCount() {
        return instancesCount;
    }

    public int getMaxAttributesInstanceCountIndex() {
        return indexOfMaxAttributesCountInstance;
    }

    public void createTestInstance() {
        // Create a new test instance
        // Mark the flag to false since every test instance requires normalization
        this.testInstance = new HashMap<String, Double>();
        isTestInstanceNormalized = false;
    }

    public void setTestInstanceAttr(String attrName, Double val) {

        this.testInstance.put(attrName.toString(), val);

        this.instanceAttributesCount++;
        if (this.instanceAttributesCount > this.maxAttributesCount) {
            this.maxAttributesCount = this.instanceAttributesCount;
            this.indexOfMaxAttributesCountInstance = (instancesCount - 1);
        }
    }

    /**
     * Performs Z-score normalization on training data set.
     */
    public void zScoreNormalization() {

        /*  Z-Score Normalization
         **************************************
         * - Find mean for each attribute
         * - Find stddev for each attribute
         * - Compute z score
         **************************************/
        
        // Each key(attribute) contains a list of values of the attribute collected from training instanes
        mapOfAggregatedAttributeValue = new HashMap<String, LinkedList<Double>>();

        // Each key(attribute) contains mean value for the attribute
        attrMean = new HashMap<String, Double>();

        // Each key(attribute) contains stddev for the attribute
        attrStddev = new HashMap<String, Double>();

        // Get the instance with maximum attributes
        // The normalization will iterate using the keys from maximum attributes
        HashMap<String, Double> maxAttrInstance = (HashMap<String, Double>) this.listOfTrainingInstance.get(this.indexOfMaxAttributesCountInstance);
        HashMap<String, Double> tempInstance;

        //* - Find mean for each attributes
        for (String key : maxAttrInstance.keySet()) {
            double sum = 0, mean = 0;
            listOfAttributeValue = new ArrayList();

            // Iterate through each training instance and retrieve the value for correspoding attribute
            // Place the attribute value inside the corresponding list
            // Compute mean at the same time
            for (int instanceCounter = 0; instanceCounter < this.listOfTrainingInstance.size(); instanceCounter++) {
                tempInstance = (HashMap<String, Double>) this.listOfTrainingInstance.get(instanceCounter);
                if (tempInstance.containsKey(key.toString())) {
                    double attrVal = tempInstance.get(key.toString());
                    listOfAttributeValue.add(attrVal);
                    sum += attrVal;
                }
            }

            mean = sum / listOfAttributeValue.size();
            attrMean.put(key.toString(), mean);
            mapOfAggregatedAttributeValue.put(key.toString(), listOfAttributeValue);
        }

        //* - Find stddev for each attribute
        for (String key : maxAttrInstance.keySet()) {
            ArrayList tempAttrDataList = (ArrayList) mapOfAggregatedAttributeValue.get(key.toString());

            double attrMeanVal = (double) attrMean.get(key.toString());
            double meanDiff = 0;
            double meanSquaredDiffSum = 0;
            double stdDev = 0;

            for (int tempAttrDataListCounter = 0; tempAttrDataListCounter < tempAttrDataList.size(); tempAttrDataListCounter++) {
                double x = (double) tempAttrDataList.get(tempAttrDataListCounter);
                meanDiff = x - attrMeanVal;
                meanSquaredDiffSum += (meanDiff * meanDiff);
            }

            stdDev = Math.sqrt((meanSquaredDiffSum / tempAttrDataList.size()));
            attrStddev.put(key.toString(), stdDev);
        }

        listOfTrainingInstancesNormalized = new ArrayList();
        HashMap<String, Double> normalizedInstance, tempTrainingInstance;

        //* - Compute z score
        // The normalized value(z-score) will be stored in a new data structure: normalizedInstance, listOfInstanceNormalized
        for (int trainingInstanceCounter = 0; trainingInstanceCounter < listOfTrainingInstance.size(); trainingInstanceCounter++) {

            tempTrainingInstance = (HashMap<String, Double>) listOfTrainingInstance.get(trainingInstanceCounter);
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
                } else {
                    normalizedInstance.put(key.toString(), 3.0);
                }
            }
            listOfTrainingInstancesNormalized.add(normalizedInstance);
        }
        isNormalized = true;
    }

    public void testInstNormalization() {
        // This method performs the normalization for test instance
        // Z score normalization is required for every instance
        HashMap<String, Double> tempTestInstance = this.testInstance;
        testInstanceNormalized = new HashMap<String, Double>();

        for (String key : tempTestInstance.keySet()) {

            ArrayList tempAttrDataList = (ArrayList) mapOfAggregatedAttributeValue.get(key);
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

    /**
     * Perform classification or regression, sub-method: regression() and
     * classification()
     *
     * @param nearestNeighbours - HashMap contains
     * <indexOfNearestNeighbour,meanSquaredError>
     * @return Object - an object either in String or numeric
     */
    public Object predict() {

        /*  Performs classification or regression 
         * **************************************
         * - Check if all training instance labels are assigned 
         * - Checks if training and test instance both normalized 
         * - Calculate MSE for every training instance
         * - Find the K nearest neighbor using computed MSE list
         * - Pass nearest neighbours to classification() or regression()
         * ****************************************
         */
        // * - Check if all training instance labels are assigned 
        if (listOfTrainingInstanceLabel.size() != listOfTrainingInstance.size()) {
            System.err.println("Perhaps a label wasn't assigned to a previous instance.");
            return false;
        }

        // * - Checks if training and test instance both normalized 
        if (!isNormalized) {
            zScoreNormalization();
        }
        if (!isTestInstanceNormalized) {
            testInstNormalization();
        }

        // * - Calculate MSE for every training instance
        listOfInstanceMeanSquaredError = new ArrayList();
        HashMap<String, Double> maxAttrInstance = (HashMap<String, Double>) listOfTrainingInstancesNormalized.get(indexOfMaxAttributesCountInstance);
        HashMap<String, Double> tempTrainingInst = null;
        for (int trainingInstanceListCounter = 0; trainingInstanceListCounter < listOfTrainingInstancesNormalized.size(); trainingInstanceListCounter++) {
            double mse = 0;
            tempTrainingInst = (HashMap<String, Double>) listOfTrainingInstancesNormalized.get(trainingInstanceListCounter);
            for (String key : maxAttrInstance.keySet()) {
                if (!testInstanceNormalized.containsKey(key.toString())) {
                    testInstanceNormalized.put(key.toString(), -3);
                }
                double trainingInstValue = (double) tempTrainingInst.get(key.toString());
                double testInstValue = (double) testInstanceNormalized.get(key.toString());

                mse += Math.abs(testInstValue - trainingInstValue);
            }
            listOfInstanceMeanSquaredError.add(trainingInstanceListCounter, mse);
        }

        // * - Find the K nearest neighbor using computed MSE list
        nearestNeighbours = new HashMap<Integer, Double>();
        Object[] arrOfInstanceMeanSquaredError = listOfInstanceMeanSquaredError.toArray();

        for (int nn_counter = 0; nn_counter < this.nearestNeighborCount; nn_counter++) {
            double min_mse = Double.MAX_VALUE;
            int min_mse_index = -1;
            for (int k = 0; k < arrOfInstanceMeanSquaredError.length; k++) {
                double mse = (double) arrOfInstanceMeanSquaredError[k];
                if (mse < min_mse) {
                    min_mse = mse;
                    min_mse_index = k;
                }
            }
            arrOfInstanceMeanSquaredError[min_mse_index] = Double.MAX_VALUE;
            nearestNeighbours.put(min_mse_index, min_mse);
        }
        //* - Pass nearest neighbours to classification() or regression()
        return (this.isClassification) ? classify(nearestNeighbours) : regress(nearestNeighbours);
    }

    /**
     * Performs KNN classification, an extension of predict()
     *
     * @param nearestNeighbours a hashmap contains index of nearest neighbours
     * and its mean squared error
     * @return a label as a classification result
     */
    public String classify(HashMap nearestNeighbours) {

        /* Majority Voting Mechanism
         ********************************************
         *- Calculates the label frequency 
         *- Returns the label with highest frequency
         ********************************************/
        // *- Calculates the label frequency 
        HashMap<String, Integer> labelFrequency = new HashMap();
        for (Object key : nearestNeighbours.keySet()) {
            int instanceIndex = (int) key;
            String instanceLabel = (String) this.listOfTrainingInstanceLabel.get(instanceIndex);
            if (labelFrequency.containsKey(instanceLabel)) {
                int frequency = labelFrequency.get(instanceLabel);
                labelFrequency.put(instanceLabel, frequency + 1);
            } else {
                labelFrequency.put(instanceLabel, 1);
            }
        }

        //*- Returns the label with highest frequency
        String predictedLabel = "null";
        int maxFreq = 0;
        for (String key : labelFrequency.keySet()) {
            if (labelFrequency.get(key) > maxFreq) {
                maxFreq = labelFrequency.get(key);
                predictedLabel = key;
            }
        }
        return predictedLabel;
    }

    /**
     * Performs KNN regression, an extension of of predict()
     *
     * @param nearestNeighbours a hashmap contains index of nearest neighbours
     * and its mean squared error
     * @return a double as a result of regression
     */
    public double regress(HashMap nearestNeighbours) {

        /* Majority Voting Mechanism
         * *******************************************
         * - Find out and return the average of all neighbours
         ********************************************
         */
        
        //* - Find out and return the average of all neighbours
        double sum=0;
        for (Object key : nearestNeighbours.keySet()) {
            int instanceIndex = (int) key;
            double value=Double.parseDouble(this.listOfTrainingInstanceLabel.get(instanceIndex).toString()) ;
            sum+=value;
        }
        
        return sum/(nearestNeighbours.size());
    }
}
