package com.deeplearning.wavelet;

import java.util.Arrays;

/**
 * Wavelet tools, e.g. calculate threshold value, threshold coefficients
 * 
 * @author Liucy
 *
 */
public class WaveletCoefHandler {
	/**
	 * Apply the threshold into coefficients, support hard and soft thresholds
	 * @param arrLevelHilb
	 * @param threshold
	 */
	public static void thresholding(double[] arrLevelHilb, double threshold, int level_decomposed, String thresholdType) {
		int approximationNum =(int) (arrLevelHilb.length/Math.pow(2, level_decomposed));
		// hard threshold
		if(thresholdType.equalsIgnoreCase("hard"))
		{
			// use threshold only for detail part, exclude from approximation part
			for (int i = approximationNum; i < arrLevelHilb.length; i++) {
				if (Math.abs(arrLevelHilb[i]) <= threshold) {
					arrLevelHilb[i] = 0;
				}
			}
		} else if(thresholdType.equalsIgnoreCase("soft")) {
			// soft threshold
			for (int i = approximationNum; i < arrLevelHilb.length; i++) {
				if (Math.abs(arrLevelHilb[i]) <= threshold) {
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
	 * Calculate the threshold value, Implement this method according to
	 * 'Visushink threshold method' (by Donoho and Johnstone)
	 * 
	 * @param coefArray
	 * @return
	 */
	public static double getVisushinkThreshold(double[] coefArray) {
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
