package com.deeplearning.predictions.log;

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
import com.deeplearning.util.PropertiesUtil;
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
	
    public static void main (String[] args) throws IOException{
		logger.info("Application is starting!");
		PropertyConfigurator.configure(new File("log4j.properties").getAbsolutePath());
		// get batchSize from Application.properties file
		
    	String filename = new ClassPathResource(PropertiesUtil.getDatasetFilename()).getFile().getAbsolutePath();
        // create dataset iterator
        logger.info("create log dataSet iterator...");
        int batchSize = PropertiesUtil.getBatchSize();
        int exampleLength = PropertiesUtil.getExampleLength();
        int firstTestItemNumber = PropertiesUtil.getFirstTestItemNumber();
        int testItems = PropertiesUtil.getTestItems();
        int vectorSize = PropertiesUtil.getVectorSize();
        LogDataSetIterator iterator = new LogDataSetIterator(filename, batchSize, exampleLength, firstTestItemNumber, testItems, vectorSize);
        logger.info("load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestData();
        // train the model and test every 10 epochs
		trainAndTest(iterator, test);
    }
    
    private static void trainAndTest(LogDataSetIterator iterator, List<Pair<INDArray, INDArray>> test) throws IOException {
		// build lstm network
		logger.info("build LSTM networks...");
		MultiLayerNetwork net = null;
		net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());
		int epochs = PropertiesUtil.getEpochs();

		String fileName = "LogLSTM_" + PropertiesUtil.getWaveletType() + ".zip";
		File locationToSave = new File("savedModels/" + fileName);
		// if not use saved model, train new model
		if(!PropertiesUtil.getUseSavedModel()) {
			logger.info("starting to train LSTM networks with " +PropertiesUtil.getWaveletType()+ " wavelet...");
			// train
			for (int i = 0; i <= epochs; i++) {
				logger.info("training at epoch "+i);
				DataSet dataSet;
				while (iterator.hasNext()) {
					dataSet = iterator.next();
					net.fit(dataSet);
				}
				// test every 10 epochs, for monitoring training process usage
//				if(i%10==0 || i==epochs)
//				{
//					logger.error("Testing...");
//					test(net, test, PropertiesUtil.getVectorSize(), iterator, PropertiesUtil.getExampleLength(), i);
//				}
				iterator.reset(); // reset iterator
				net.rnnClearPreviousState(); // clear previous state
			}
			// save model to file
			logger.info("saving trained network model...");
			ModelSerializer.writeModel(net, locationToSave, true);
		} else {
			logger.info("loading network model...");
			net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		}
		// testing
		test(net, test, PropertiesUtil.getVectorSize(), iterator, PropertiesUtil.getExampleLength(), epochs);
		
        logger.info("Both the training and testing are finished, system is exiting...");
        System.exit(0);
    }
    
    private static void test(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> test, int VECTOR_SIZE, LogDataSetIterator iterator, int exampleLength, int epochNum){
		// Testing
		logger.info("testing...");
        INDArray max = Nd4j.create(iterator.getMaxValue());
        INDArray min = Nd4j.create(iterator.getMinValue());
        INDArray[] predicts = new INDArray[test.size()];
        INDArray[] actuals = new INDArray[test.size()];
        
        double[] mseValue=new double[VECTOR_SIZE];
        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = test.get(i).getValue();
            // Calculate the MSE of results
            mseValue[0]+=Math.pow((actuals[i].getDouble(0,0) - predicts[i].getDouble(0,0)), 2);
            mseValue[1]+=Math.pow((actuals[i].getDouble(0,1) - predicts[i].getDouble(0,1)), 2);
            mseValue[2]+=Math.pow((actuals[i].getDouble(0,2) - predicts[i].getDouble(0,2)), 2);
        }
        double mseHost = Math.sqrt(mseValue[0]/test.size());
        double mseProgram = Math.sqrt(mseValue[1]/test.size());
        double mseSeverity = Math.sqrt(mseValue[2]/test.size());
        logger.info("MSE for [severity, program id, host] is: ["+mseSeverity+", "+mseProgram+", "+mseHost + "]");

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
        for (int j = 0; j < PropertiesUtil.getVectorSize(); j++) {
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
