#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.cu"

// En cada gradiente agregue el termino d_in[i]
// No se si esta bien
// Se supone que me ahorro un paso con eso porque es el 
// gradiente de la activacion


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
	virtual void gradient(Matrix &in, Matrix &out) = 0;
};

Activation::Activation(std::string name_) : name(name_) {}

Activation::~Activation(){}

std::string Activation::getName(){
	return name;
}

/* ----------------------------
Sigmoid class and Kernels
---------------------------- */
__device__ __host__ float sigmoid(float x);
__global__ void sigmoidKernel(float *d_in, float *d_out, int size);
__global__ void sigmoidGradKernel(float *d_in, float *d_out, int size);

class Sigmoid : public Activation{
public:
	Sigmoid();
    ~Sigmoid();
	
	void call(Matrix &in, Matrix &out);
	void gradient(Matrix &in, Matrix &out);
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
	
	sigmoidKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

void Sigmoid::gradient(Matrix &in, Matrix &out){
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
	
	sigmoidGradKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

__device__ __host__ float sigmoid(float x){
	return 1.0f / (1 + expf(-x));
}

__global__ void sigmoidKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = sigmoid(d_in[i]);
		i += blockDim.x * gridDim.x;
	}
}

__global__ void sigmoidGradKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		float sig = d_in[i] * sigmoid(d_in[i]);
		d_out[i] = sig * (1.0f - sig);

		i += blockDim.x * gridDim.x;
	}
}

/* ----------------------------
Relu class and Kernels
---------------------------- */
__global__ void reluKernel(float *d_in, float *d_out, int size);
__global__ void reluGradKernel(float *d_in, float *d_out, int size);

class Relu : public Activation{
public:
	Relu();
    ~Relu();
	
	void call(Matrix &in, Matrix &out);
	void gradient(Matrix &in, Matrix &out);
};

Relu::Relu():Activation("Relu") {}

Relu::~Relu(){}

void Relu::call(Matrix &in, Matrix &out){
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
	
	reluKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

void Relu::gradient(Matrix &in, Matrix &out){
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
	
	reluGradKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

__global__ void reluKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = fmaxf(d_in[i], 0);
		i += blockDim.x * gridDim.x;
	}
}

__global__ void reluGradKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		if(d_in[i] > 0)
			d_out[i] = d_in[i] * 1.0;
		else
			d_out[i] = 0.0;

		i += blockDim.x * gridDim.x;
	}
}

/* ----------------------------
Linear class and Kernels
---------------------------- */
__global__ void linearKernel(float *d_in, float *d_out, int size);
__global__ void linearGradKernel(float *d_in, float *d_out, int size);

class Linear : public Activation{
public:
	Linear();
    ~Linear();
	
	void call(Matrix &in, Matrix &out);
	void gradient(Matrix &in, Matrix &out);
};

Linear::Linear():Activation("Linear") {}

Linear::~Linear(){}

void Linear::call(Matrix &in, Matrix &out){
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
	
	linearKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

void Linear::gradient(Matrix &in, Matrix &out){
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
	
	linearGradKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

__global__ void linearKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = d_in[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__ void linearGradKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = d_in[i] * 1;
		i += blockDim.x * gridDim.x;
	}
}

/* ----------------------------
Tanh class and Kernels
---------------------------- */
__global__ void tanhKernel(float *d_in, float *d_out, int size);
__global__ void tanhGradKernel(float *d_in, float *d_out, int size);

class Tanh : public Activation{
public:
	Tanh();
    ~Tanh();
	
	void call(Matrix &in, Matrix &out);
	void gradient(Matrix &in, Matrix &out);
};

Tanh::Tanh():Activation("Tanh") {}

Tanh::~Tanh(){}

void Tanh::call(Matrix &in, Matrix &out){
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
	
	tanhKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

void Tanh::gradient(Matrix &in, Matrix &out){
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
	
	tanhGradKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size);
	cudaDeviceSynchronize();
}

__global__ void tanhKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = tanhf(d_in[i]);
		i += blockDim.x * gridDim.x;
	}
}

__global__ void tanhGradKernel(float *d_in, float *d_out, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = d_in[i] * (1.0f - powf(tanhf(d_in[i]), 2.0f));
		i += blockDim.x * gridDim.x;
	}
}

/* ----------------------------
LeakyRelu class and Kernels
---------------------------- */
__global__ void leakyReluKernel(float *d_in, float *d_out, int size, float arg);
__global__ void leakyReluGradKernel(float *d_in, float *d_out, int size, float arg);

class LeakyRelu : public Activation{
private:
	float arg;
public:
	LeakyRelu(float arg=0.1);
    ~LeakyRelu();
	
	void call(Matrix &in, Matrix &out);
	void gradient(Matrix &in, Matrix &out);
};

LeakyRelu::LeakyRelu(float arg_):Activation("LeakyRelu") {arg = arg_;}

LeakyRelu::~LeakyRelu(){}

void LeakyRelu::call(Matrix &in, Matrix &out){
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
	
	leakyReluKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size, arg);
	cudaDeviceSynchronize();
}

void LeakyRelu::gradient(Matrix &in, Matrix &out){
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
	
	leakyReluGradKernel<<< nBlocks, nThreads >>>(in.getDeviceData(), out.getDeviceData(), in.size, arg);
	cudaDeviceSynchronize();
}

__global__ void leakyReluKernel(float *d_in, float *d_out, int size, float arg){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_out[i] = fmaxf(d_in[i], d_in[i]*arg);
		i += blockDim.x * gridDim.x;
	}
}

__global__ void leakyReluGradKernel(float *d_in, float *d_out, int size, float arg){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		if(d_in[i] > 0)
			d_out[i] = d_in[i] * 1.0;
		else
			d_out[i] = d_in[i] * 1.0 * arg;

		i += blockDim.x * gridDim.x;
	}
}

#endif
