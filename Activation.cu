/*
date: 27-06-21
File: Activations.cu
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
*/

#include "Activation.h"

class Matrix{
private:
	int width;
	int height;
	size_t size;
	float* h_elem;
	float* d_elem;

public:
	Matrix(int width, int height);
	~Matrix();

	void copyDeviceToHost();

}:

Matrix::Matrix(int width, int height) : width(width), height(height), size(width * height){
	// size = width * height;

	h_elem = new float[size];
	int aux[3] = {-1, 0 , 1};
	for(size_t i=0; i < size; ++i){
		h_elem[i] = aux[i%3];
	}

	// Allocacion en device
	cudaMalloc(&d_elem, size * sizeof(float));
	cudaMemcpy( d_elem, a, size * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix::~Matrix(){
	delete [] h_elem;
	cudaFree(d_a);
}

void Matrix::copyDeviceToHost(){
	cudaMemcpy(h_elem, d_elem, size * sizeof(float), cudaMemcpyDeviceToHost);
}


/*
------------------------------
*/
class Activation{
private:
    std::string name;
    
public:
	__host__ __device__ Activation(std::string name_);	//Default constructor
	__host__ __device__ virtual ~Activation();

	Activation(const Activation &) = delete;	//Copy constructor
	Activation &operator=(const Activation &) = delete;	//Copy assignment
	Activation(Activation &&) = delete;	//Move constructor
	Activation &operator=(Activation &&) = delete;	// Move assignment

	__host__ __device__ std::string getName();
	__host__ __device__ void call() = 0;
};

__host__ __device__ Activation::Activation(std::string name_) : name(name_) {}

__host__ __device__ Activation::~Activation(){}

__host__ __device__ std::string Activation::getName(){
    return name;
}





class Sigmoid : public Activation{
public:
    __host__ __device__ Sigmoid(ActivationColour C);
    __host__ __device__ ~Sigmoid();

    void printActivation();
    __host__ __device__ std::string getName();
};

Sigmoid::Sigmoid(PieceColour C):Activation(C,PAWN,"Sigmoid") {}




int main(int argc, const char** argv) {

    hello<<<1, 10>>>();  // 1 bloque con 10 hilos
    cudaDeviceSynchronize();

    return 0;
}