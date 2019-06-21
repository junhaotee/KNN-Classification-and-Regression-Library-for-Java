package KNNClassificationRegression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/*********************************************
 * An example of main program running KNN class
 * *******************************************
 * @author junhaotee
 */
public class KNNClassificationRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        /* An example of KNN regression */
        KNN kNN = new KNN(2, false);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr"+1,2.0);
        kNN.setInstanceAttr("attr"+2,3.0);
        kNN.setInstanceAttr("attr"+3,4.0);
        kNN.setInstanceAttr("attr"+4,5.0);
        kNN.setInstanceAttr("attr"+5,6.0);
        kNN.setInstanceAttr("attr"+6,7.0);
        kNN.setInstanceLabel(20);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr"+1,2.0);
        kNN.setInstanceAttr("attr"+2,3.0);
        kNN.setInstanceAttr("attr"+3,4.0);
        kNN.setInstanceAttr("attr"+4,5.0);
        kNN.setInstanceAttr("attr"+5,6.0);
        kNN.setInstanceAttr("attr"+6,7.0);
        kNN.setInstanceLabel(20);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr"+1,22.0);
        kNN.setInstanceAttr("attr"+2,33.0);
        kNN.setInstanceAttr("attr"+3,44.0);
        kNN.setInstanceAttr("attr"+4,55.0);
        kNN.setInstanceAttr("attr"+5,66.0);
        kNN.setInstanceAttr("attr"+6,77.0);
        kNN.setInstanceLabel(50);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr"+1,222.0);
        kNN.setInstanceAttr("attr"+2,333.0);
        kNN.setInstanceAttr("attr"+3,444.0);
        kNN.setInstanceAttr("attr"+4,555.0);
        kNN.setInstanceAttr("attr"+5,666.0);
        kNN.setInstanceAttr("attr"+6,777.0);
        kNN.setInstanceLabel(500);
        
        kNN.createInstance();
        kNN.setInstanceAttr("attr"+1,1.0);
        kNN.setInstanceAttr("attr"+2,3.0);
        kNN.setInstanceAttr("attr"+3,8.0);
        kNN.setInstanceAttr("attr"+4,9.0);
        kNN.setInstanceAttr("attr"+5,1.0);
        kNN.setInstanceAttr("attr"+6,0.0);
        kNN.setInstanceLabel(50);
        
        kNN.createTestInstance();
        kNN.setTestInstanceAttr("attr"+1,1.0);
        kNN.setTestInstanceAttr("attr"+2,3.0);
        kNN.setTestInstanceAttr("attr"+3,8.0);
        kNN.setTestInstanceAttr("attr"+4,9.0);
        kNN.setTestInstanceAttr("attr"+5,1.0);
        kNN.setTestInstanceAttr("attr"+6,0.0);
        
        
        System.out.println(kNN.predict());
        
        
        
        
        
    }

}
