#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.h"

__device__ __host__ float sigmoid(int x);

__global__ void sigmoidKernel(float *d_in, float *d_out, int size);

/* ----------------------------
Activation class
---------------------------- */
class Activation{
private:
    std::string name;
    
public:
	Activation(std::string name_);	//Default constructor
	virtual ~Activation();

	std::string getName();
	virtual void call(Matrix &in, Matrix &out) = 0;
};

Activation::Activation(std::string name_) : name(name_) {}

Activation::~Activation(){}

std::string Activation::getName(){
    return name;
}

/* ----------------------------
Sigmoid class
---------------------------- */
class Sigmoid : public Activation{
public:
	Sigmoid();
    ~Sigmoid();

	void call(Matrix &in, Matrix &out);
};

Sigmoid::Sigmoid():Activation("Sigmoid") {}

Sigmoid::~Sigmoid(){}

void Sigmoid::call(Matrix &in, Matrix &out){
	int dev;
	cudaGetDevice(&dev);

	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

	// dim3 nThreads(256);
	dim3 nThreads(deviceProp.maxThreadsDim[0]);
	dim3 nBlocks((in.size + nThreads.x - 1) / nThreads.x);
	if(nBlocks.x > deviceProp.maxGridSize[0]){
		nBlocks.x = deviceProp.maxGridSize[0];
	}

	// sigmoidKernel<<< 1, 6 >>>(A.d_elem, A.size);
	sigmoidKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
}



__device__ __host__ float sigmoid(float x){
	return 1.0f / (1 + expf(-x));
}

__global__ void sigmoidKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = sigmoid(d_in[i]);
		i += blockDim.x*gridDim.x;
	}
}




#endif
