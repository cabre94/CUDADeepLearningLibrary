#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdio.h>

__global__ void hello(){
    printf("Hola\n");
}

class Matrix{
public:
	int height, width, size;
	float *h_elem, *d_elem;

// public:
	Matrix(int width, int height);
	~Matrix();

	void copyDeviceToHost();
	void copyHostToDevice();
	void print();

	float* getDeviceData();
};

// __device__ __host__ float sigmoid(int x){
// __device__ __host__ float sigmoid(int x);

// __global__ void sigmoidKernel(float* d_e, int size);


Matrix::Matrix(int height, int width) : height(height), width(width), size(width * height){
	h_elem = new float[size];
	// float aux[3] = {-1, 0 , 1};
	for(size_t i=0; i < size; ++i){
		h_elem[i] = i;
		// h_elem[i] = aux[i%3];
	}

	// Allocacion en device
	cudaMalloc(&d_elem, size * sizeof(float));
	cudaMemcpy( d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice);
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

float* Matrix::getDeviceData(){
	return d_elem;
}

#endif
