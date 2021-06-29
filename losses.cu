#ifndef LOSSES_H
#define LOSSES_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include "Matrix.cu"
#include <math.h>

/* ----------------------------
Loss class
---------------------------- */
class Loss{
private:
    std::string name;
    
public:
	Loss(std::string name_);	//Default constructor
	virtual ~Loss();
	
	std::string getName();
	virtual float call(Matrix &y_pred, Matrix &y_true) = 0;
	virtual void gradient(Matrix &y_pred, Matrix &y_true, Matrix &dY) = 0;
};

Loss::Loss(std::string name_) : name(name_) {}

Loss::~Loss(){}

std::string Loss::getName(){
	return name;
}

/* ----------------------------
MSE class and Kernels
---------------------------- */
__global__ void mseGradLossKernel(float *d_pred, float *d_true, float *d_grad, int size, int height);


class MSE : public Loss{
public:
	MSE();
    ~MSE();

	float call(Matrix &y_pred, Matrix &y_true);
	void gradient(Matrix &y_pred, Matrix &y_true, Matrix &dY);
};

MSE::MSE():Loss("MSE"){}

MSE::~MSE(){}

float MSE::call(Matrix &y_pred, Matrix &y_true){
	assert(y_pred.width == y_true.width);
	assert(y_pred.height == y_true.height);

	// Voy a hacerlo serial y despues ver que hago
	y_pred.copyDeviceToHost();
	y_true.copyDeviceToHost();

	// y_true: asumo que ya viene como one-hot-encoder
	// y_true deberian ser solo 0 y 1
	// widht: numero de clases
	// height: tamaño del bacth
	float cost = 0;
	for(int i=0; i < y_pred.size; ++i){
		cost += pow(y_pred.h_elem[i]-y_true.h_elem[i], 2.0);
	}
	cost = (cost / y_pred.height);

	return cost;
}

void MSE::gradient(Matrix &y_pred, Matrix &y_true, Matrix &dY){
	int dev;
	cudaGetDevice(&dev);
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
	
	// dim3 nThreads(256);
	dim3 nThreads(deviceProp.maxThreadsDim[0]);
	dim3 nBlocks((y_pred.size + nThreads.x - 1) / nThreads.x);
	if(nBlocks.x > deviceProp.maxGridSize[0]){
		nBlocks.x = deviceProp.maxGridSize[0];
	}
	
	mseGradLossKernel<<< nBlocks, nThreads >>>(y_pred.getDeviceData(), y_true.getDeviceData(), dY.getDeviceData(), y_true.size, y_true.height);
	cudaDeviceSynchronize();
}

__global__ void mseGradLossKernel(float *d_pred, float *d_true, float *d_grad, int size, int height){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_grad[i] = 2.0f * (d_pred[i] - d_true[i]) / height;

		i += blockDim.x * gridDim.x;
	}
}

#endif
