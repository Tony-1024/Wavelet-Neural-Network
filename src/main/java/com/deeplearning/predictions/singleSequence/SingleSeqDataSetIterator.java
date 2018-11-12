package com.deeplearning.predictions.singleSequence;

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
public class SingleSeqDataSetIterator {
	private static final Logger logger = LoggerFactory.getLogger(SingleSeqDataSetIterator.class);
	
	private final int VECTOR_SIZE = 1;
	// for the data normalization
    private double[] minValue = new double[VECTOR_SIZE];
    private double[] maxValue = new double[VECTOR_SIZE];
    private int miniBatchSize;
    private int exampleLength;
    private List<SingleSeqBean> train;
    private List<Pair<INDArray, INDArray>> test;
    private LinkedList<Integer> offsetList = new LinkedList<>();

    public SingleSeqDataSetIterator (String filename, int miniBatchSize, int exampleLength, int firstTestItemNumber, int testItems) {
        List<SingleSeqBean> logDataList = readLogData(filename);
        
        if(firstTestItemNumber + testItems > logDataList.size()){
        	logger.info("the sum of training and testing number shouldn't greater than the size of dataset!");
        	return;
        }
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        train = logDataList.subList(0, firstTestItemNumber);
        // denoise
        SingleSeqWaveletProcessor.processor(train);
        
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
     * get next data for training
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
            SingleSeqBean curData = train.get(startIdx);
            SingleSeqBean nextData;
            for (int i = startIdx; i < endIdx; i++) {
                nextData = train.get(i + 1);
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getSeqID() - minValue[0]) / (maxValue[0] - minValue[0]));
                
                label.putScalar(new int[] {index, 0, c}, (nextData.getSeqID() - minValue[0]) / (maxValue[0] - minValue[0]));
                curData = nextData;
            }
            if (offsetList.size() == 0)
            	break;
        }
        return new DataSet(input, label);
    }

    /**
     * build test data with input and label
     */
    private List<Pair<INDArray, INDArray>> buildTestDataSet (List<SingleSeqBean> logDataList) {
    	int window = exampleLength + 1; // window is for predicting the following items as a time series
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < logDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			SingleSeqBean log = logDataList.get(j);
    			input.putScalar(new int[] {j - i, 0}, (log.getSeqID() - minValue[0]) / (maxValue[0] - minValue[0]));
    		}
    		
    		INDArray label = Nd4j.create(new int[] {VECTOR_SIZE}, 'f');
    		SingleSeqBean log = logDataList.get(i + exampleLength);
    		label.putScalar(new int[] {0}, log.getSeqID());
    		
    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }
    
    /**
     * read data from file
     */
	private List<SingleSeqBean> readLogData (String filename) {
        List<SingleSeqBean> logDataList = new ArrayList<>();
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
                    nums[0] = Double.valueOf(arr[2]);
                    if (nums[0] > maxValue[0]) 
                    	maxValue[0] = nums[0];
                    if (nums[0] < minValue[0]) 
                    	minValue[0] = nums[0];
                logDataList.add(new SingleSeqBean(nums[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return logDataList;
    }
}
