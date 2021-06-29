#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdexcept>
#include "Matrix.cu"
#include "Activation.cu"

/* ----------------------------
Kernel
---------------------------- */

template<int BLOCK_SIZE> __global__ void
XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB);

/* ----------------------------
Layer class
---------------------------- */
class Layer{
private:
    std::string name;
    
public:
	Layer(std::string name_);	//Default constructor
	virtual ~Layer();
	
	std::string getName();
	// virtual void call(Matrix &in, Matrix &out) = 0;
	// virtual void gradient(Matrix &in, Matrix &out) = 0;
	virtual void printWeights() = 0;
	virtual int getWidth() = 0;
	virtual int getHeight() = 0;
	virtual std::string getActivation() = 0;

	// virtual void forward(Matrix &X) = 0;
};

Layer::Layer(std::string name_) : name(name_) {}

Layer::~Layer(){}

std::string Layer::getName(){return name;}


/* ----------------------------
Dense Layer
---------------------------- */

class Dense : public Layer{
private:
	Matrix W;
	Matrix b;
	Matrix Y; // Y = input*W + b -> 
	// Matrix Output;
	Activation *activation;
public:
	Dense(int input_shape, int output_shape, std::string act, std::string dist = "uniform", float w = 0.1);
    ~Dense();
	
	// void call(Matrix &in, Matrix &out);
	// void gradient(Matrix &in, Matrix &out);
	void printWeights();
	int getWidth();
	int getHeight();
	std::string getActivation();
};

Dense::Dense(int input_shape, int output_shape, std::string act, std::string dist, float w)
	:Layer("Dense"), W(input_shape,output_shape,dist,w), b(1,output_shape,dist,w) {
		if(act == "linear")
			activation = new Linear;
		else if(act == "relu")
			activation = new Relu;
		else if(act == "sigmoid")
			activation = new Sigmoid;
		else if(act == "tanh")
			activation = new Tanh;
		else if(act == "leakyRelu")
			activation = new LeakyRelu();
		else
			throw std::invalid_argument("Invalid activation");
	}

Dense::~Dense(){
	delete activation;
}

void Dense::printWeights(){
	float *ptr_W = W.getHostData();
	float *ptr_b = b.getHostData();
	for(int i=0; i < W.height; ++i){
		for(int j=0; j < W.width; ++j)
			std::cout << ptr_W[i*W.width + j] << "\t";
		std::cout << ptr_b[i] << "\t";
		std::cout << std::endl;
	}
}

int Dense::getWidth(){return W.width;}

int Dense::getHeight(){return W.height;}

std::string Dense::getActivation(){
	return activation->getName();
}


/* ----------------------------
Input Layer
---------------------------- */

class Input : public Layer{
private:
	int width, height; // salida, entrada (entrada no la se a esta altiura)
	Matrix Datos;
public:
	Input(int width, int height = -1);
    ~Input();
	
	// void call(Matrix &in, Matrix &out);
	// void gradient(Matrix &in, Matrix &out);
	void printWeights();
	int getWidth();
	int getHeight();
	std::string getActivation();
};

// Input::Input(int width, int height):Layer("Input"), width(width), height(-1){}

//#! Capaz cambie esto
Input::Input(int width, int height)
	: Layer("Input"), width(width), height(-1), Datos(height, width){}


Input::~Input(){}

void Input::printWeights(){
	std::cout << "Input Layer - Serian los datos" << std::endl;
	float *ptr_W = Datos.getHostData();
	for(int i=0; i < Datos.height; ++i){
		for(int j=0; j < Datos.width; ++j)
			std::cout << ptr_W[i*Datos.width + j] << "\t";
		std::cout << std::endl;
	}
}

int Input::getWidth(){return width;}

int Input::getHeight(){return height;}

std::string Input::getActivation(){return "None";}


// Kernel
// Kernel modified from https://www.programmersought.com/article/13436584263/
template<int BLOCK_SIZE> __global__ void
XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
	//Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	/* Divide the matrix into sub-matrices, apply the parallel calculation of the thread in the block
	to the multiplication of the sub-matrices, and finally add their values ​​to obtain an element value of C */
	int aBegin = by * BLOCK_SIZE * wA;	//The row coordinates of the sub-matrix of A
	int aStep = BLOCK_SIZE;				//The movement step size of A's sub-matrix column coordinates
	int aEnd = aBegin + wA - 1;			//Limit an end point

	int bBegin = bx * BLOCK_SIZE;
	int bStep = BLOCK_SIZE * wB;

	float Csub = 0;	//Define the element value of C at the corresponding position in the block (x,. y) (ty, tx)

	int subAw = BLOCK_SIZE;
	int subAh = BLOCK_SIZE;
	int subBh = BLOCK_SIZE;
	int subBw = BLOCK_SIZE;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
		//The number of columns in the last column of the A matrix is ​​less than BLOCK_SIZE
		if (a + aStep - 1 > aEnd){			
			subAw = aEnd - a + 1;
		}else{
			subAw = BLOCK_SIZE;
		}
		subBh = subAw;

		//The number of rows in the last row of the A matrix is ​​less than BLOCK_SIZE
		if ((by + 1) * BLOCK_SIZE > hA){
			subAh = hA - by * BLOCK_SIZE;
		}else{
			subAh = BLOCK_SIZE;
		}

		//The number of columns in the last column of the B matrix is ​​less than BLOCK_SIZE
		if ((bx + 1) * BLOCK_SIZE > wB){
			subBw = wB - bx * BLOCK_SIZE;
		}else{
			subBw = BLOCK_SIZE;
		}
		
		/* Develop shared memory in the block */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* Assign values ​​to the corresponding elements of the sub-matrix in the range of rows and columns */
		if (ty < subAh && tx < subAw){
			As[ty][tx] = A[a + ty * wA + tx];
		}
		if (ty < subBh && tx < subBw){
			Bs[ty][tx] = B[b + ty * wB + tx];
		}
		__syncthreads();

		//Unroll the loop to compile to speed up		
		#pragma unroll
		//The inner loop calculates the vector product of the corresponding row and column in each sub-matrix and adds it to the previously obtained value
		for (int k = 0; k < subAw; k++){
			//Satisfy the elements within the row and column constraints to calculate the product and sum
			if (ty < subAh && tx < subBw){
				Csub += As[ty][k] * Bs[k][tx];
			}			
		}
		__syncthreads();
	}

	//Satisfy the elements within the row and column constraints to calculate the product and sum
	if (ty < subAh && tx < subBw)
	{
		C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub;
	}	
}


#endif
