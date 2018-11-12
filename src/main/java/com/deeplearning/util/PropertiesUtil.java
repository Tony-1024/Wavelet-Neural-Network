package com.deeplearning.util;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Read properties file
 * 
 * @author Liucy
 *
 */
public class PropertiesUtil {
	private static Properties cfg = new Properties();

	static{
		try {
			cfg.load(new FileInputStream("Application.properties"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static int getBatchSize() {
		return Integer.parseInt(cfg.getProperty("batchSize"));
	}
	
	public static int getExampleLength() {
		return Integer.parseInt(cfg.getProperty("exampleLength"));
	}
	
	public static int getEpochs() {
		return Integer.parseInt(cfg.getProperty("epochs"));
	}
	
	public static int getVectorSize() {
		return Integer.parseInt(cfg.getProperty("vectorSize"));
	}
	
	public static String getDatasetFilename() {
		return cfg.getProperty("datasetFilename");
	}
	
	public static int getFirstTestItemNumber() {
		return Integer.parseInt(cfg.getProperty("firstTestItemNumber"));
	}
	
	public static int getTestItems() {
		return Integer.parseInt(cfg.getProperty("testItems"));
	}
	
	public static double getLearningRate() {
		return Double.parseDouble(cfg.getProperty("learningRate"));
	}
	
	public static String getWaveletType() {
		return cfg.getProperty("waveletType");
	}
	
	public static boolean getUseSavedModel() {
		return Boolean.parseBoolean(cfg.getProperty("useSavedModel"));
	}
	
}
