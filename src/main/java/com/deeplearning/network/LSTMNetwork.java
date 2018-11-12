package com.deeplearning.network;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import com.deeplearning.util.PropertiesUtil;

/**
 * Build LSTM model 
 * 
 * @author Liucy
 *
 */
public class LSTMNetwork {
	
	private static final int seed = 12345;
	private static final int iterations = 1;
    private static final int layer1Size = 256;
    private static final int layer2Size = 256;
    private static final int denseLayerSize = 32;
    private static final int truncatedBPTTLength = 32;
    private static final double dropoutRatio = 0.2;
    private static final double rmsDecay = 0.95;

	/**
	 * build LSTM model with parameters, and use UIServer to get the visualized
	 * monitoring information when training the model
	 * 
	 * @param nIn: node size of input
	 * @param nOut: node size of output
	 * @return LSTM Model
	 */
    public static MultiLayerNetwork buildLSTMNetwork (int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(PropertiesUtil.getLearningRate()) // get learning rate from configuration file
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .rmsDecay(rmsDecay)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(layer1Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(layer1Size)
                        .nOut(layer2Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(2, new DenseLayer.Builder()
                		.nIn(layer2Size)
                		.nOut(denseLayerSize)
                		.activation(Activation.RELU)
                		.build())
                .layer(3, new RnnOutputLayer.Builder()
                        .nIn(denseLayerSize)
                        .nOut(nOut)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(truncatedBPTTLength)
                .tBPTTBackwardLength(truncatedBPTTLength)
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        
        //System.setProperty("org.deeplearning4j.ui.port", "9666");
        // instantiate the UIserver
        UIServer uiServer = UIServer.getInstance();

        // set the network info (), the storage location, set to memory here
        //or use: new FileStatsStorage(File) for the afterwards storing and saving
        StatsStorage statsStorage = new InMemoryStatsStorage();

        // add StatsListener to collect info
        net.setListeners(new StatsListener(statsStorage));
        
        // attach StatsStorage to uiServer, for the visualization of the content of StatsStorage
        uiServer.attach(statsStorage);
        
        //net.setListeners(new ScoreIterationListener(100));
        return net;
    }
}
