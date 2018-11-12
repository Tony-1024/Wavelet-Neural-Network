package com.deeplearning.predictions.singleSequence;

/**
 * Java Bean for log
 * 
 * @author Liucy
 *
 */
public class SingleSeqBean {
	private double seqID;
	public SingleSeqBean(double seqID) {
		this.seqID = seqID;
	}

	public double getSeqID() {
		return seqID;
	}

	public void setSeqID(double seqID) {
		this.seqID = seqID;
	}
}