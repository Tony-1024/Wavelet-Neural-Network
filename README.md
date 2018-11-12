# Wavelet-Neural-Network
This project attempts to find an improved neural network model for log analysis by combining the wavelet transform and a neural network.
The log datasets are from a virtualized cloud platform. Based on the log dataset features, I design and implement the new model. 

1. Wavelet transform
By using wavelet transform, we can decompose a signal into a series of wavelets with different scales and positions. These wavelets are dilated and translated forms of a mother wavelet. There are several types of wavelet transform: continuous wavelet transform (CWT), discrete wavelet transform (DWT), and wavelet packet transform (WPT). WPT is very similar to DWT, the differences are that DWT only decompose the approximation coefficients, while in WPT, both the approximation and detail coefficients are decomposed.
![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/Wavelet%20Packet%20Decomposition%20Tree.JPG)
Wavelet Packet Decomposition Tree




The experimental results demonstrate the proposed model is more effective and accurate, it has a better ability for feature attraction and noise tolerance than conventional neural networks.

