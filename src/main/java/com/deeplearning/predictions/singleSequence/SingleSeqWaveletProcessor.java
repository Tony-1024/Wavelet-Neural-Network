package com.deeplearning.predictions.singleSequence;

import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jwave.Transform;
import jwave.transforms.WaveletPacketTransform;
//import jwave.transforms.wavelets.daubechies.Daubechies3;
import jwave.transforms.wavelets.haar.Haar1;

/**
 * Wavelet denoise for Single column dataset
 * 
 * @author Liucy
 *
 */
public class SingleSeqWaveletProcessor {
	private static final Logger logger = LoggerFactory.getLogger(SingleSeqWaveletProcessor.class);

	public static void processor(List<SingleSeqBean> logList) {
        int len = logList.size();
        int level_decomposed = 3;
        double[] seqArr = new double[len];
        for(int i=0; i<len; i++)
        {
        	SingleSeqBean lb = logList.get(i);
        	seqArr[i] = lb.getSeqID();
        }
        
        Transform transform=new Transform(new WaveletPacketTransform(new Haar1()));
        
        // denoise
        double[][] seqArrHilb2D = transform.decompose(seqArr);
        // Get threshold
        double seqThreshold = getVisushinkThreshold(seqArrHilb2D[1]);
        
        // Use threshold to denoise
        thresholding(seqArrHilb2D[level_decomposed], seqThreshold, level_decomposed, "hard");
        
        double [] seqArrReco = transform.recompose(seqArrHilb2D, level_decomposed);
        for(int i=0; i<len; i++)
        {
        	SingleSeqBean lb = logList.get(i);
        	lb.setSeqID(seqArrReco[i]);
        }
        
        logger.info("");
        logger.info("Array Origianl : " + Arrays.toString(seqArr));
        logger.info("Array Recovered: " + Arrays.toString(seqArrReco));
        logger.info("");
        logger.info("");
	}
	
	/**
	 * Calculate the threshold value
	 * @param arrLevelHilb
	 * @param threshold
	 */
	private static void thresholding(double[] arrLevelHilb, double threshold, int level_decomposed, String thresholdType) {
		int approximationNum =(int) (arrLevelHilb.length/Math.pow(2, level_decomposed));
		// hard threshold
		if(thresholdType.equalsIgnoreCase("hard"))
		{
			// use threshold only for detail part, exclude from approximation part
			for (int i = approximationNum; i < arrLevelHilb.length; i++) {
				if (Math.abs(arrLevelHilb[i]) < threshold) {
					arrLevelHilb[i] = 0;
				}
			}
		} else if(thresholdType.equalsIgnoreCase("soft")) {
			// soft threshold
			for (int i = approximationNum; i < arrLevelHilb.length; i++) {
				if (Math.abs(arrLevelHilb[i]) < threshold) {
					arrLevelHilb[i] = 0;
				} else {
					if(arrLevelHilb[i] < 0)
						arrLevelHilb[i] = threshold - Math.abs(arrLevelHilb[i]);
					else
						arrLevelHilb[i] = Math.abs(arrLevelHilb[i] - threshold);
				}
			}
		}
	}
	
	/**
	 * According to Visushink threshold method (by Donoho and Johnstone)
	 * @param coefArray
	 * @return
	 */
	private static double getVisushinkThreshold(double[] coefArray) {
		double threshold=0;
		double sigma = 0;
		
		double[] tempArray = Arrays.copyOfRange(coefArray, 0, coefArray.length/2);
		int len = tempArray.length;
		
		for(int i=0; i<len; i++)
		{
			tempArray[i] = Math.abs(tempArray[i]);
		}
		Arrays.sort(tempArray);
		if(len%2==0 && len>=2)
			sigma =(tempArray[len/2-1] + tempArray[len/2])/2/0.6745;
		else
			sigma = tempArray[len/2]/0.6745;
		
		threshold = sigma * Math.sqrt(2.0 * Math.log(len));
		
		return threshold;
	}
}
