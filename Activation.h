#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.h"

__device__ __host__ float sigmoid(int x);

__global__ void sigmoidKernel(float* d_e, int size);


class Activation{
private:
    std::string name;
    
public:
	Activation(std::string name_);	//Default constructor
	virtual ~Activation();

	std::string getName();
	virtual void call(Matrix &A) = 0;
};

Activation::Activation(std::string name_) : name(name_) {}

Activation::~Activation(){}

std::string Activation::getName(){
    return name;
}


class Sigmoid : public Activation{
public:
	Sigmoid();
    ~Sigmoid();

	void call(Matrix &A);
};

Sigmoid::Sigmoid():Activation("Sigmoid") {}

Sigmoid::~Sigmoid(){}

void Sigmoid::call(Matrix &A){
	// sigmoidKernel<<< 1, 6 >>>(A.d_elem, A.size);
	sigmoidKernel<<< 1, 6 >>>(A.getDeviceData(), A.size);
}



__device__ __host__ float sigmoid(int x){
	return 1.0f / (1 + expf(-x));
}

__global__ void sigmoidKernel(float* d_e, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_e[i] = sigmoid(d_e[i]);
		i += blockDim.x*gridDim.x;
	}
}




#endif
