#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <random>

// Copia los elementos de to a from
__global__ void copyFromTo(float *from, float *to, int size);
// Copia los elementos de las filas idx de to a from
__global__ void copyFromToIdx(float *from, float *to, int *idx, int h, int w, int idx0);

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
	void copyDeviceDataFromBatch(Matrix &from, int *idx, int idx0);
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
	if(allocated){
		delete [] h_elem;
		cudaFree(d_elem);
	}
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

void Matrix::copyDeviceDataFromBatch(Matrix &from, int *idx, int idx0){
	// Asumo dimensiones correctas
	int dev;
	cudaGetDevice(&dev);
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

	// dim3 nThreads(256);
	// dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);

	// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	
	// dim3 nThreads(256);
	// dim3 nThreads(deviceProp.maxThreadsDim[0]);
	// dim3 nBlocks((from.size + nThreads.x - 1) / nThreads.x);
	// if(nBlocks.x > deviceProp.maxGridSize[0]){
	// 	nBlocks.x = deviceProp.maxGridSize[0];
	// }
	dim3 dimBlock(deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1]);
	dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);
	if(dimGrid.x > deviceProp.maxGridSize[0]){
		dimGrid.x = deviceProp.maxGridSize[0];
	}
	if(dimGrid.y > deviceProp.maxGridSize[1]){
		dimGrid.y = deviceProp.maxGridSize[1];
	}
	
	copyFromToIdx<<< dimGrid, dimBlock >>>(from.getDeviceData(), d_elem, idx, height, width, idx0);
	cudaDeviceSynchronize();
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

// from es mas grande
// las filas de to son iguales a el tamaño
// idx deberia tener h elementos

__global__ void copyFromToIdx(float *from, float *to, int *idx, int h, int w, int idx0){
	// lo tengo que hacer como columna
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row < h && col < w){
		to[row * w + col] = from[idx[idx0+row] * w + col];
	}
}



#endif
