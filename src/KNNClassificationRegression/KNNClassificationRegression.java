/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package KNNClassificationRegression;

/**
 *
 * @author dev
 */
public class KNNClassificationRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        KNN kNN=new KNN(1,true);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr1",99999999.0);
        kNN.setInstanceAttr("attr2",3.0);
        kNN.setInstanceLabel("label-1");
        System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr1",7.0);
        kNN.setInstanceLabel("label-2");
        System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr1",4.0);
        kNN.setInstanceAttr("attr2",9.0);
        kNN.setInstanceAttr("attr3",10.0);
        kNN.setInstanceAttr("attr4",1.0);
        kNN.setInstanceLabel("label-1");
        System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr1",3.0);
        kNN.setInstanceAttr("attr2",1.0);
        kNN.setInstanceAttr("attr3",8.0);
        kNN.setInstanceAttr("attr4",9.0);
        kNN.setInstanceAttr("attr5",0.0);
        kNN.setInstanceLabel("label-1");
        System.out.println(kNN.getMaxAttributesInstanceCountIndex());
        
        kNN.createTestInstance();
        kNN.setTestInstanceAttr("attr1",3.0);
        kNN.setTestInstanceAttr("attr2",1.0);
        kNN.setTestInstanceAttr("attr3",8.0);
        kNN.setTestInstanceAttr("attr4",9.0);
        kNN.setTestInstanceAttr("attr5",0.0);
        
        kNN.predict();
        
    }
    
}
