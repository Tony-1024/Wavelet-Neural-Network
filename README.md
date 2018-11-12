# Wavelet-Neural-Network
This project attempts to find an improved neural network model for handling time series that contain non-linear, non-stationary, high noisy and chaotic characteristics. The model combines neural network with wavelet.

1. Wavelet transform

By using wavelet transform, we can decompose a signal into a series of wavelets with different scales and positions. These wavelets are dilated and translated forms of a mother wavelet. There are several types of wavelet transform: continuous wavelet transform (CWT), discrete wavelet transform (DWT), and wavelet packet transform (WPT). WPT is very similar to DWT, the differences are that DWT only decompose the approximation coefficients, while in WPT, both the approximation and detail coefficients are decomposed.

![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/Wavelet%20Packet%20Decomposition%20Tree.JPG)

Wavelet Packet Decomposition Tree

2. Wavelet Denoising

The process of wavelet denoising can be done in three steps: 1. Select a wavelet and a level, apply wavelet or wavelet packet decomposition to the noisy signal to get a set of coefficients. 2. Select an appropriate threshold value and apply thresholding to the detail part of coefficients. 3. Reconstruct the signal to get a denoised signal. Figure bwlow shows the process.

![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/Wavelet%20threshold%20denoising%20process.JPG)

Wavelet threshold denoising process

A number of wavelets can be used, such as Haar, Morlet, Daubechies, Coiflet, Symlets.

3. Wavelet Denoising Resule

Take Haar for example, below shows the comparation between original singal and denoised signal:

![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/Original%20and%20denoised%20(with%20Haar)%20severity%20signal.png)

Original and denoised (with Haar) severity signal

![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/Original%20and%20denoised%20(with%20Haar)%20severity%20signal%20in%20one%20chart.png)

Original and denoised (with Haar) severity signal in one chart

The experimental results demonstrate the proposed model is more effective and accurate, it has a better ability for feature attraction and noise tolerance than conventional neural networks.

![alt text](https://github.com/Tony-1024/Wavelet-Neural-Network/blob/master/images/The%20comparison%20of%20RMSE%20values.JPG)

The comparison of RMSE values

The result shows that lower RMSE values for wavelet neural network models when compared to a conventional network. Take DB3 for example, the MSE values of severity, program, and host predictions are 40.3%, 18.9%, and 16.4% less than the corresponding values of conventional neural network respectively.
