package com.deeplearning.predictions.singleSequence;

import javafx.util.Pair;
import org.apache.log4j.PropertyConfigurator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.deeplearning.network.LSTMNetwork;
import com.deeplearning.util.DrawingTool;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Log prediction
 * 
 * @author Liucy
 *
 */
public class LogPrediction {
	
	private static final Logger logger = LoggerFactory.getLogger(LogPrediction.class);
	
	static int batchSize = 64;
	static int exampleLength = 32; // time series length
	static int epochs = 100; // for training rounds
	static int VECTOR_SIZE = 1;
	static String datasetFilename = "cleanedLogData.csv";//"cleanedLogData.csv"; // data set file name
	static int firstTestItemNumber = 2048; //training data starts from 0 to this value (excluded)
	static int testItems = 150; // the sum of firstTestItemNumber and testItems shouldn't greater than whole item number in total
	
    public static void main (String[] args) throws IOException{
		PropertyConfigurator.configure(new File("log4j.properties").getAbsolutePath()); // load log configuration file
		logger.info("Application is starting!");
		
    	String filename = new ClassPathResource(datasetFilename).getFile().getAbsolutePath();
        // create dataset iterator
        logger.info("create log dataSet iterator...");
        SingleSeqDataSetIterator iterator = new SingleSeqDataSetIterator(filename, batchSize, exampleLength, firstTestItemNumber, testItems);
        logger.info("load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestData();
        
		trainAndTest(iterator, test, "");
    }
    
    private static void trainAndTest(SingleSeqDataSetIterator iterator, List<Pair<INDArray, INDArray>> test, String waveletID) throws IOException {
		// build lstm network
		logger.info("build lstm networks...");
		
		MultiLayerNetwork net = null;
		net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());

		// training
		for (int i = 0; i <= epochs; i++) {
			logger.info("training at epoch "+i);
			DataSet dataSet;
			while (iterator.hasNext()) {
				dataSet = iterator.next();
				net.fit(dataSet);
			}
			
            if(i%10==0 || i==epochs)
            {
            	logger.error("Testing...");
            	test(net, test, VECTOR_SIZE, iterator, waveletID, exampleLength, i);
            }
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
		}

		// save model
		logger.info("saving trained network model...");
		File locationToSave = new File("src/main/resources/LogLSTM.zip");
		ModelSerializer.writeModel(net, locationToSave, true);

        logger.info("load network model...");
//        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        
//        logger.info("Both the training and testing are finished!");
        logger.info("Both the training and testing are finished, system is exiting...");
        System.exit(0);
    }
    
    private static void test(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> test, int VECTOR_SIZE, SingleSeqDataSetIterator iterator, String waveletID, int exampleLength, int epochNum){
		// Testing
		logger.info("testing...");
        INDArray max = Nd4j.create(iterator.getMaxValue());
        INDArray min = Nd4j.create(iterator.getMinValue());
        INDArray[] predicts = new INDArray[test.size()];
        INDArray[] actuals = new INDArray[test.size()];
        
        // Calculate the MSE of results
        double[] mseValue=new double[VECTOR_SIZE];
        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = test.get(i).getValue();
            mseValue[0]+=Math.pow((actuals[i].getDouble(0,0) - predicts[i].getDouble(0,0)), 2);
        }
        double mseProgram = Math.sqrt(mseValue[0]/test.size());
        logger.info("MSE for [host, program id, severity] is: ["+mseProgram+", ");

        logger.info("Starting to print out values.");
		// draw chart for prediction and actual values
		  for (int i = 0; i < predicts.length; i++) 
			  logger.info("Prediction="+predicts[i] + ", Actual=" + actuals[i]);
		  logger.info("Drawing chart...");
		  drawAll(predicts, actuals, epochNum);
		  logger.info("Finished drawing...");
    }
    
    /**
     * plot all predictions
     * @param predicts
     * @param actuals
     * @param epochNum
     */
    private static void drawAll(INDArray[] predicts, INDArray[] actuals, int epochNum) {
    	String[] titles = { "Host", "Program ID", "Severity" };
        for (int j = 0; j < VECTOR_SIZE; j++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(j);
                actu[i] = actuals[i].getDouble(j);
            }
            DrawingTool.drawChart(pred, actu, titles[j], epochNum);
        }
    }
}
