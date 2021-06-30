#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <random>

__global__ void copyFromTo(float *from, float *to, int size);

class Matrix{
public:
	int height, width, size;
	float *h_elem, *d_elem;
	float weight;
	std::string dist;

	bool allocated;

// public:
	Matrix();
	Matrix(int height, int width, std::string dist = "uniform", float w = 1);
	~Matrix();

	void copyDeviceToHost();
	void copyHostToDevice();
	void print();
	void printDimensions();

	float* getHostData();
	float* getDeviceData();

	int getHeight();
	int getWidth();

	void initialize(int height, int width, std::string dist = "zeros", float w = 1);
	void copyDeviceDataFromAnother(Matrix &from);
};

Matrix::Matrix(){
	allocated = false;
}

Matrix::Matrix(int height, int width, std::string dist, float w)
			: height(height), width(width), size(width * height){
	weight = w;
	dist = dist;
	h_elem = new float[size];
	
	std::random_device rd;
	std::mt19937 mt(rd());

	if(dist == "normal"){
		// std::default_random_engine generator;
  		std::normal_distribution<float> distribution(0.0,weight);
		for(int i=0; i < size; ++i){
			h_elem[i] = distribution(mt);
		}
	}else if(dist == "uniform"){
		// std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-weight,1.0);
		for(int i=0; i < size; ++i){
			h_elem[i] = distribution(mt);
		}
	}else if(dist == "ones"){
		for(int i=0; i < size; ++i){
			h_elem[i] = 1.0f;
		}
	}else if(dist == "zeros"){
		for(int i=0; i < size; ++i){
			h_elem[i] = 0.0f;		}
	}else{
		throw std::invalid_argument("Invalid Weight initialization");
	}

	// Allocacion en device
	cudaMalloc(&d_elem, size * sizeof(float));
	cudaMemcpy( d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice);

	allocated = true;
}

Matrix::~Matrix(){
	delete [] h_elem;
	cudaFree(d_elem);
}

void Matrix::copyDeviceToHost(){
	cudaMemcpy(h_elem, d_elem, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::copyHostToDevice(){
	cudaMemcpy(d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice );
}

void Matrix::print(){
	for(int i=0; i < height; ++i){
		for(int j=0; j < width; ++j)
			std::cout << h_elem[i*width + j] << "\t";
		std::cout << std::endl;
	}
}

void Matrix::printDimensions(){
	std::cout << "(" << height << "," << width << ")";
}

float* Matrix::getHostData(){
	return h_elem;
}

float* Matrix::getDeviceData(){
	return d_elem;
}

int Matrix::getHeight(){return height;}

int Matrix::getWidth(){return width;}

void Matrix::initialize(int height_, int width_, std::string dist, float w){
	if (allocated){
		delete [] h_elem;
		cudaFree(d_elem);
		allocated = false;
	}

	height = height_;
	width = width_;
	size = width * height;
	weight = w;
	dist = dist;
	h_elem = new float[size];
	
	std::random_device rd;
	std::mt19937 mt(rd());

	if(dist == "normal"){
		// std::default_random_engine generator;
  		std::normal_distribution<float> distribution(0.0,weight);
		for(int i=0; i < size; ++i){
			h_elem[i] = distribution(mt);
		}
	}else if(dist == "uniform"){
		// std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-weight,1.0);
		for(int i=0; i < size; ++i){
			h_elem[i] = distribution(mt);
		}
	}else if(dist == "ones"){
		for(int i=0; i < size; ++i){
			h_elem[i] = 1.0f;
		}
	}else if(dist == "zeros"){
		for(int i=0; i < size; ++i){
			h_elem[i] = 0.0f;		}
	}else{
		throw std::invalid_argument("Invalid Weight initialization");
	}

	// Allocacion en device
	cudaMalloc(&d_elem, size * sizeof(float));
	cudaMemcpy( d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice);

	allocated = true;
}

void Matrix::copyDeviceDataFromAnother(Matrix &from){
	// Asumo dimensiones correctas
	int dev;
	cudaGetDevice(&dev);
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
	
	// dim3 nThreads(256);
	dim3 nThreads(deviceProp.maxThreadsDim[0]);
	dim3 nBlocks((from.size + nThreads.x - 1) / nThreads.x);
	if(nBlocks.x > deviceProp.maxGridSize[0]){
		nBlocks.x = deviceProp.maxGridSize[0];
	}
	
	copyFromTo<<< nBlocks, nThreads >>>(from.getDeviceData(), d_elem, from.size);
	cudaDeviceSynchronize();
	// Aca Host y Device son distintos
}




/* ----------------------------
Kernels
---------------------------- */


__global__ void copyFromTo(float *from, float *to, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		to[i] = from[i];

		i += blockDim.x * gridDim.x;
	}
}



#endif
