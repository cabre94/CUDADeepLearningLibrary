#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <random>


class Matrix{
public:
	int height, width, size;
	float *h_elem, *d_elem;
	float weight;
	std::string dist;

// public:
	Matrix(int height, int width, std::string dist = "uniform", float w = 1);
	~Matrix();

	void copyDeviceToHost();
	void copyHostToDevice();
	void print();

	float* getDeviceData();
};


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
