/*
date: 27-06-21
File: Activations.cu
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
//TODO - Ver si puedo poner un getter que de el d_elem por referencia, asi lo puedo dejar privado
*/

// si prefiere trabajar con indices de fila y columna 
// estos macros son utiles:

// C[IDX2C(i,j,M)] == valor en fila i (=0,...,Width-1) columna (j=0,1,...Height-1), row-major-C
// #define  IDX2C(i,j,ld) (((j)*(ld))+( i )) 

// C[IDX2F(i,j,M)] == valor en fila i (=1,...,Width) columna (j=1,...Height), column-major-F
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 

#include "Activation.h"

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
};

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


------------------------------

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




// __device__ __host__ float sigmoid(int x){
__device__ __host__
float sigmoid(int x){
	return 1.0f / (1 + expf(-x));
}

__global__ void sigmoidKernel(float* d_e, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		d_e[i] = sigmoid(d_e[i]);
		i += blockDim.x*gridDim.x;
	}
}



int main(int argc, const char** argv) {

	Matrix A(3, 2);

	A.print();

	std::cout << std::endl;

	// dim3 nThreads(256); // CORREGIR
	// dim3 nBlocks((A.size + nThreads.x - 1) / nThreads.x); // CORREGIR

	// if(nBlocks.x>65535)
	// 	nBlocks.x=65535;	
	
	// sigmoidKernel<<< nBlocks, nThreads >>>(A, A.size);
	// sigmoidKernel<<< 1, 6 >>>(&A, A.size);
	sigmoidKernel<<< 1, 6 >>>(A.d_elem, A.size);
	cudaDeviceSynchronize();

	A.print();
	std::cout << std::endl;

	A.copyDeviceToHost();

	A.print();
	std::cout << std::endl;


    return 0;
}