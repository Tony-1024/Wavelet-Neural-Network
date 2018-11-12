package com.deeplearning.predictions.log;

/**
 * Log Bean
 * 
 * @author Liucy
 * 
 */
public class LogBean {
//private long time;
private double hostID;
private double programID;
private double severity;

//public long getTime() {
//	return time;
//}
//public void setTime(long time) {
//	this.time = time;
//}
public LogBean(double hostID, double programID, double severity) {
	this.hostID = hostID;
	this.programID = programID;
	this.severity = severity;
}

public double getHostID() {
	return hostID;
}
public void setHostID(double hostID) {
	this.hostID = hostID;
}
public double getProgramID() {
	return programID;
}
public void setProgramID(double programID) {
	this.programID = programID;
}
public double getSeverity() {
	return severity;
}
public void setSeverity(double severity) {
	this.severity = severity;
}

@Override
public String toString()
{
	StringBuilder sb=new StringBuilder();
//	sb.append("Time="+this.time+", ");
	sb.append("Host ID=="+this.hostID+", ");
	sb.append("Program ID="+this.programID+", ");
	sb.append("Severity="+this.severity);
	return sb.toString();
}
}