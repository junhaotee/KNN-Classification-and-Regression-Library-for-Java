/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package KNNClassificationRegression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author dev
 */
public class KNNClassificationRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        KNN kNN = new KNN(1, true);

        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader("./iris.txt"));

            String line = reader.readLine();
            String[] attributeNames = line.split(",");
            line = reader.readLine();

            while (line != null) {
                kNN.createInstance();
                Object[] columns = line.split(",");
                for (int attr_counter = 0; attr_counter < columns.length - 1; attr_counter++) {
                    String attrName = attributeNames[attr_counter].toString();
                    double attrVal = Double.valueOf((String) columns[attr_counter]);
                    kNN.setInstanceAttr(attrName, attrVal);
                }
                kNN.setInstanceLabel(columns[columns.length - 1]);
                line = reader.readLine();
            }
            
            kNN.createTestInstance();
            kNN.setTestInstanceAttr(attributeNames[0],5.1);
            kNN.setTestInstanceAttr(attributeNames[1],3.5);
            kNN.setTestInstanceAttr(attributeNames[2],1.4);
            kNN.setTestInstanceAttr(attributeNames[3],0.2);
            
            
            System.out.println(kNN.predict()+"");
            reader.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        /*
        
         kNN.createInstance();
         kNN.setInstanceAttr("attr1",99999999.0);
         kNN.setInstanceAttr("attr2",3.0);
         kNN.setInstanceLabel("label-1");
         //System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
         kNN.createInstance();
         kNN.setInstanceAttr("attr1",7.0);
         kNN.setInstanceLabel("label-2");
         //System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        
         kNN.createInstance();
         kNN.setInstanceAttr("attr1",4.0);
         kNN.setInstanceAttr("attr2",9.0);
         kNN.setInstanceAttr("attr3",10.0);
         kNN.setInstanceAttr("attr4",1.0);
         kNN.setInstanceLabel("label-1");
         //System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        
         kNN.createInstance();
         kNN.setInstanceAttr("attr1",3.0);
         kNN.setInstanceAttr("attr2",1.0);
         kNN.setInstanceAttr("attr3",8.0);
         kNN.setInstanceAttr("attr4",9.0);
         kNN.setInstanceAttr("attr5",0.0);
         kNN.setInstanceLabel("label-1");
         //System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
         kNN.createTestInstance();
         kNN.setTestInstanceAttr("attr1",3.0);
         kNN.setTestInstanceAttr("attr2",1.0);
         kNN.setTestInstanceAttr("attr3",8.0);
         kNN.setTestInstanceAttr("attr4",9.0);
         kNN.setTestInstanceAttr("attr5",0.0);
         */
        //System.out.println("Predicted label::" + kNN.predict());
    }

}
