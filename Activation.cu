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
	int height;
	int width;
	size_t size;
	float* h_elem;
	float* d_elem;

// public:
	Matrix(int width, int height);
	~Matrix();

	void print();

	void copyDeviceToHost();

};

Matrix::Matrix(int height, int width) : height(height), width(width), size(width * height){
	// size = width * height;

	h_elem = new float[size];
	int aux[3] = {-1, 0 , 1};
	for(size_t i=0; i < size; ++i){
		h_elem[i] = aux[i%3];
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

void Matrix::print(){
	for(int i=0; i < height; ++i){
		for(int j=0; j < width; ++j){
			std::cout << h_elem[i*height + j] << "\t";
		}
		std::cout << std::endl;
	}
}

/*
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

*/


// __device__ __host__ float sigmoid(int x){
__device__ __host__
float sigmoid(int x){
	return 1.0f / (1 + expf(-x));
}

// __device__ __host__ 
// float MiFuncion(int i){
//     // return sin(2*M_PI*i/10.0);
//     return expf(2*M_PI*i/10.0);
// }

__global__ void sigmoidKernel(Matrix A, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		A.d_elem[i] = sigmoid(A.d_elem[i]);
		// A.d_elem[i] = MiFuncion(A.d_elem[i]);
		// A.d_elem[i] = MiFuncion(i);
		i += blockDim.x*gridDim.x;
	}
}

// __global__ void Tabular(float *d_c, int n){
// 	// indice de thread mapeado a indice de array 
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;

// 	//COMPLETAR PARA QUE c[i]=MiFuncion(i)
// 	//ASEGURARSE DE QUE TODO EL ARRAY ESTE TABULADO CON LA GRILLA LANZADA
// 	//Y DE QUE NO SE ACCEDAN POSICIONES ILEGALES
// 	if(i < n)
// 		d_c[i] = MiFuncion(i);
// }


int main(int argc, const char** argv) {

	Matrix A(5, 2);

	A.print();



    // hello<<<1, 10>>>();  // 1 bloque con 10 hilos
    // cudaDeviceSynchronize();

    return 0;
}