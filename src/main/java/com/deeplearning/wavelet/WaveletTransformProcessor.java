package com.deeplearning.wavelet;

import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.deeplearning.predictions.log.LogBean;
import com.deeplearning.util.PropertiesUtil;

import jwave.Transform;
import jwave.transforms.WaveletPacketTransform;
import jwave.transforms.wavelets.Wavelet;
import jwave.transforms.wavelets.daubechies.Daubechies3;
import jwave.transforms.wavelets.haar.Haar1;

/**
 * Wavelet feature extraction, decompose, thresholding, denoising, and
 * reconstruction
 * 
 * @author Liucy
 *
 */
public class WaveletTransformProcessor {
	private static final Logger logger = LoggerFactory.getLogger(WaveletTransformProcessor.class);

	public static void processor(List<LogBean> logList) {
		logger.info("Strating to denoise log data series with " + PropertiesUtil.getWaveletType());
        int len = logList.size();
        int level_decomposed = 3;
        // create arrays for wavelet transform
        double[] hostArr = new double[len];
        double[] programArr = new double[len];
        double[] severityArr = new double[len];
        for(int i=0; i<len; i++)
        {
        	LogBean lb = logList.get(i);
        	hostArr[i] = lb.getHostID();
        	programArr[i] = lb.getProgramID();
        	severityArr[i] = lb.getSeverity();
        }
        Wavelet waveletType = null;
        if("DB3".compareToIgnoreCase(PropertiesUtil.getWaveletType())==0)
        	waveletType=new Daubechies3();
        else if("Haar".compareToIgnoreCase(PropertiesUtil.getWaveletType())==0)
        	waveletType=new Haar1();
        
        Transform transform=new Transform(new WaveletPacketTransform(waveletType));
        
        // denoise
        // These values for de-noising the signal only works for 2^p length signals
        double[][] hostArrHilb2D = transform.decompose(hostArr);
        double[][] programArrHilb2D = transform.decompose(programArr);
        double[][] severityArrHilb2D = transform.decompose(severityArr);
 
        // log coefficients into file-for denoising effect analysis
/*      logger.info("Host Array Hilb2D : ");
        for(double[] darr:hostArrHilb2D)
        	logger.info(Arrays.toString(darr));
        logger.info("program Array Hilb2D : ");
        for(double[] darr:programArrHilb2D)
        	logger.info(Arrays.toString(darr));
        logger.info("severity Array Hilb2D : ");
        for(double[] darr:severityArrHilb2D)
        	logger.info(Arrays.toString(darr));*/
        
        // calculate the threshold, two methods can be used, use SURE here
//        double threshold = Math.sqrt(2 * Math.log(len * (Math.log(len) / Math.log(2))));
        
        // Get threshold by details coefficient from level 1
        double hostThreshold = WaveletCoefHandler.getVisushinkThreshold(hostArrHilb2D[1]);
        double programThreshold = WaveletCoefHandler.getVisushinkThreshold(programArrHilb2D[1]);
        double severityThreshold = WaveletCoefHandler.getVisushinkThreshold(severityArrHilb2D[1]);
        
        // Use threshold to denoise
        WaveletCoefHandler.thresholding(hostArrHilb2D[level_decomposed], hostThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(programArrHilb2D[level_decomposed], programThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(severityArrHilb2D[level_decomposed], severityThreshold, level_decomposed, "hard");
        
        double [] hostArrReco = transform.recompose(hostArrHilb2D, level_decomposed);
        double [] programArrReco = transform.recompose(programArrHilb2D, level_decomposed);
        double [] severityReco = transform.recompose(severityArrHilb2D, level_decomposed);
        
        for(int i=0; i<len; i++)
        {
        	LogBean lb = logList.get(i);
        	lb.setHostID(hostArrReco[i]);
        	lb.setProgramID(programArrReco[i]);
        	lb.setSeverity(severityReco[i]);
        }
        
        logger.info("Host Data Series Origianl: " + Arrays.toString(hostArr));
        logger.info("Host Data Series Denoised: " + Arrays.toString(hostArrReco));
        logger.info("");
        logger.info("Program Data Series Origianl: " + Arrays.toString(programArr));
        logger.info("Program Data Series Denoised: " + Arrays.toString(programArrReco));
        logger.info("");
        logger.info("Severity Data Series Origianl: " + Arrays.toString(severityArr));
        logger.info("Severity Data Series Denoised: " + Arrays.toString(severityReco));
        logger.info("");
	}
}
