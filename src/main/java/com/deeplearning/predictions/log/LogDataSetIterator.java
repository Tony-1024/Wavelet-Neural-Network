package com.deeplearning.predictions.log;

import com.deeplearning.util.PropertiesUtil;
import com.deeplearning.wavelet.WaveletTransformProcessor;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Iterator for log data set
 * 
 * @author Liucy
 * 
 */
public class LogDataSetIterator {
	private static final Logger logger = LoggerFactory.getLogger(LogDataSetIterator.class);
	
	private final int VECTOR_SIZE;
	// for the data normalization
    private double[] minValue;
    private double[] maxValue;
    private int miniBatchSize;
    private int exampleLength;
    private List<LogBean> train;
    private List<Pair<INDArray, INDArray>> test;
    private LinkedList<Integer> offsetList = new LinkedList<>();

    public LogDataSetIterator (String filename, int miniBatchSize, int exampleLength, int firstTestItemNumber, int testItems, int VECTOR_SIZE) {
    	this.VECTOR_SIZE = VECTOR_SIZE;
    	minValue = new double[VECTOR_SIZE];
    	maxValue = new double[VECTOR_SIZE];
        List<LogBean> logDataList = readLogData(filename);
        
        if(firstTestItemNumber + testItems > logDataList.size()){
        	logger.info("the sum of training and testing number shouldn't greater than the size of dataset!");
        	return;
        }
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        train = logDataList.subList(0, firstTestItemNumber);
        // if not Non, use WNN for denoising, otherwise use traditional NN
        if("Non".compareToIgnoreCase(PropertiesUtil.getWaveletType())!=0)
        	WaveletTransformProcessor.processor(train);
        // build test data
        test = buildTestDataSet(logDataList.subList(firstTestItemNumber, firstTestItemNumber+testItems));
        initOffsetsList();
    }

    private void initOffsetsList () {
        offsetList.clear();
        int window = exampleLength + 1;
        for (int i = 0; i < train.size() - window; i++) {
            offsetList.add(i);
        }
    }

    public List<Pair<INDArray, INDArray>> getTestData () { return test; }

    public double[] getMaxValue () { return maxValue; }

    public double[] getMinValue () { return minValue; }
    
    public int inputColumns() { return VECTOR_SIZE; }

    public int totalOutcomes() { return VECTOR_SIZE; }

    public void reset() { initOffsetsList(); }

    public boolean hasNext() { return offsetList.size() > 0; }

    /**
     * get the next data for training
     * @return
     */
    public DataSet next() {
        if (offsetList.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(miniBatchSize, offsetList.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = offsetList.removeFirst();
            int endIdx = startIdx + exampleLength;
            LogBean curData = train.get(startIdx);
            LogBean nextData;
            for (int i = startIdx; i < endIdx; i++) {
                nextData = train.get(i + 1);
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getHostID() - minValue[0]) / (maxValue[0] - minValue[0]));
                input.putScalar(new int[] {index, 1, c}, (curData.getProgramID() - minValue[1]) / (maxValue[1] - minValue[1]));
                input.putScalar(new int[] {index, 2, c}, (curData.getSeverity() - minValue[2]) / (maxValue[2] - minValue[2]));
                
                label.putScalar(new int[] {index, 0, c}, (nextData.getHostID() - minValue[0]) / (maxValue[0] - minValue[0]));
                label.putScalar(new int[] {index, 1, c}, (nextData.getProgramID() - minValue[1]) / (maxValue[1] - minValue[1]));
                label.putScalar(new int[] {index, 2, c}, (nextData.getSeverity() - minValue[2]) / (maxValue[2] - minValue[2]));
                curData = nextData;
            }
            if (offsetList.size() == 0)
            	break;
        }
        return new DataSet(input, label);
    }

    /**
     * build test data with input and label
     * @param logDataList
     * @return
     */
    private List<Pair<INDArray, INDArray>> buildTestDataSet (List<LogBean> logDataList) {
    	int window = exampleLength + 1; // window is for predicting the following items as a time series
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < logDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			LogBean log = logDataList.get(j);
    			input.putScalar(new int[] {j - i, 0}, (log.getHostID() - minValue[0]) / (maxValue[0] - minValue[0]));
    			input.putScalar(new int[] {j - i, 1}, (log.getProgramID() - minValue[1]) / (maxValue[1] - minValue[1]));
    			input.putScalar(new int[] {j - i, 2}, (log.getSeverity() - minValue[2]) / (maxValue[2] - minValue[2]));
    		}
    		
    		INDArray label = Nd4j.create(new int[] {VECTOR_SIZE}, 'f');
    		LogBean log = logDataList.get(i + exampleLength);
    		label.putScalar(new int[] {0}, log.getHostID());
    		label.putScalar(new int[] {1}, log.getProgramID());
    		label.putScalar(new int[] {2}, log.getSeverity());
    		
    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }
    
    /**
     * read data from file
     * @param filename
     * @return
     */
	private List<LogBean> readLogData (String filename) {
        List<LogBean> logDataList = new ArrayList<>();
        try {
        	logger.debug("Reading data items..");
            @SuppressWarnings("resource")
			List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            for (int i = 0; i < maxValue.length; i++) {
                maxValue[i] = Double.MIN_VALUE;
                minValue[i] = Double.MAX_VALUE;
            }
            for (String[] arr : list) {
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length; i++) {
                    nums[i] = Double.valueOf(arr[i]);
                    if (nums[i] > maxValue[i]) 
                    	maxValue[i] = nums[i];
                    if (nums[i] < minValue[i]) 
                    	minValue[i] = nums[i];
                }
                logDataList.add(new LogBean(nums[0], nums[1], nums[2]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return logDataList;
    }
}
