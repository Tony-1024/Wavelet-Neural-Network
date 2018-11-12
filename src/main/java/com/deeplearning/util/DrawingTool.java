package com.deeplearning.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;

/**
 * Draw chart for both predicts and actual data
 * 
 * @author Liucy
 *
 */
public class DrawingTool {

	public static void drawChart (double[] predicts, double[] actuals, String title, int epochNum) {
		double[] dataIndex = new double[predicts.length];
		for (int i = 0; i < predicts.length; i++) {
			dataIndex[i] = i;
		}
		//calculate the maximum and minimum value of dataset
		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < predicts.length; i++) {
			if (max < (int) predicts[i]) max = (int) predicts[i];
			if (max < (int) actuals[i]) max = (int) actuals[i];
			
			if (min > (int) predicts[i]) min = (int) predicts[i];
			if (min > (int) actuals[i]) min = (int) actuals[i];
		}
		
		XYSeriesCollection dataSet = new XYSeriesCollection();
		addSeries(dataSet, dataIndex, predicts, "Predicts");
		addSeries(dataSet, dataIndex, actuals, "Actuals");
		// instantiate JFreeChart object
		final JFreeChart chart = ChartFactory.createXYLineChart(
				title, // title
				"Data Index", // xAxisLabel
				"Value", // yAxisLabel
				dataSet, // dataset
				PlotOrientation.VERTICAL, // orientation
				true, // legend
				true, // tooltips
				false // urls
		);
		XYPlot xyPlot = chart.getXYPlot();
		// set x axis
		final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
		domainAxis.setRange((int) dataIndex[0], (int) (dataIndex[dataIndex.length - 1] + 2));
		domainAxis.setVerticalTickLabels(true);
		// set y axis
		final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
		System.out.println("min: "+min+", max: "+max);
		if (min >= max) {
			min = max - 1;
			max = max + 1;
		}
		rangeAxis.setRange(min*0.96, max*1.04);
		// Constructs a panel to display the specified chart
		final ChartPanel panel = new ChartPanel(chart);
		final JFrame f = new JFrame();
		f.add(panel);
		f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
		// save chart to file system
		BufferedImage  bi = new BufferedImage(f.getWidth(), f.getHeight(), BufferedImage.TYPE_INT_ARGB);
		Graphics2D  g2d = bi.createGraphics();
		f.paint(g2d);
		try {
			ImageIO.write(bi, "PNG", new File("charts/"+PropertiesUtil.getWaveletType()+"_"+epochNum+"_"+title+".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		// set visible to false after saving
		f.setVisible(false);
	}

	private static void addSeries (final XYSeriesCollection dataSet, double[] x, double[] y, final String label){
		final XYSeries s = new XYSeries(label);
		for( int j = 0; j < x.length; j++ ) {
			s.add(x[j], y[j]);
		}
		dataSet.addSeries(s);
	}
}