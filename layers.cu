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

// template<int BLOCK_SIZE> __global__ void
// XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB);

template<int BLOCK_SIZE> __global__ void
XdotWplusBias(float* A, float* B, float* C, int wA, int wB, int hA, int hB, float *bias);

template<int BLOCK_SIZE> __global__ void
XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB);

template<int BLOCK_SIZE> __global__ void
transpose(float *odata, float *idata, int width, int height);



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

	virtual void forward(Matrix &X) = 0;
	virtual void backward(Matrix &X, Matrix &dX, Matrix &dX_T) = 0;
	
	virtual Matrix& getW() = 0;
	virtual Matrix& getdW() = 0;
	virtual Matrix& getOutput() = 0;
	virtual Matrix& getGradOutput() = 0;
	virtual Matrix& getOutput_T() = 0;

	virtual void updateW(float lr) = 0;
	// virtual void updateBias(float lr) = 0;
};

Layer::Layer(std::string name_) : name(name_) {}

Layer::~Layer(){}

std::string Layer::getName(){return name;}


/* ----------------------------
Dense Layer
---------------------------- */
__global__ void updateWKernel(float *W, float *dW, float lr, int size);

class Dense : public Layer{
private:
	Matrix W;
	Matrix W_T;
	Matrix dW;
	Matrix b;
	Matrix Y; // Y = input*W + b -> 
	Matrix dY; // Y = input*W + b -> 
	Matrix Y_T; // Y = input*W + b -> 
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

	void forward(Matrix &X);
	void backward(Matrix &X, Matrix &dX, Matrix &dX_T);

	Matrix& getW();
	Matrix& getdW();
	Matrix& getOutput();
	Matrix& getGradOutput();
	Matrix& getOutput_T();

	void updateW(float lr);
};

Dense::Dense(int input_shape, int output_shape, std::string act, std::string dist, float w)
	:Layer("Dense"), W(input_shape,output_shape,dist,w), W_T(output_shape,input_shape,"zeros",w), dW(input_shape,output_shape,"zeros",w), b(1,output_shape,dist,w) {
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

void Dense::forward(Matrix &X){
	// Tengo que hacer X*W
	const int block_size = 32;

	dim3 threads(block_size, block_size);
	dim3 grid((W.width -1) / threads.x + 1, (X.height - 1) / threads.y + 1);

	XdotWplusBias<block_size> <<<grid, threads >>> (
		X.getDeviceData(),
		W.getDeviceData(),
		Y.getDeviceData(),
		X.getWidth(),
		W.getWidth(),
		X.getHeight(),
		W.getHeight(),
		b.getDeviceData()
	);

	cudaDeviceSynchronize();

	// Sumarle el bias y guardarlo en Y
	activation->call(Y, Y);

	cudaDeviceSynchronize();
	// En este punto, Y deberia estar listo para la siguiente capa

	return;
}

void Dense::backward(Matrix &X, Matrix &dX, Matrix &X_T){
	// Supongo que MI dY salida ya la tengo acualizada.
	// Tengo que actualizar mi dW, mi db y modificar el dY de la otra (argumento)

	// dW
	// actualizo x_T
	const int block_size = 16;

    dim3 threads(block_size, block_size, 1);
	dim3 grid((X_T.width -1) / threads.x + 1, (X.height - 1) / threads.y + 1);
	// dim3 grid(X.width / block_size, Y.width / block_size, 1);

	transpose<block_size> <<<grid, threads >>> (
		X.getDeviceData(),
		X_T.getDeviceData(),
		X_T.getWidth(),
		X.getWidth()
	);
	cudaDeviceSynchronize();

	// Ahora a dY le aplico el gradiente de la activacion
	activation->gradient(dY,dY);

	// Ahora necesito calcular dw = x_t * dY
	// Tengo que hacer X*W
	const int block_size2 = 32;

	dim3 threads2(block_size2, block_size2);
	dim3 grid2((dY.width -1) / threads2.x + 1, (X_T.height - 1) / threads2.y + 1);

	XdotW<block_size2> <<<grid2, threads2 >>> (
		X_T.getDeviceData(),
		dY.getDeviceData(),
		dW.getDeviceData(),
		X_T.getWidth(),
		dY.getWidth(),
		X_T.getHeight(),
		dY.getHeight()
	);

	cudaDeviceSynchronize();

	// Ya esta dW
	// Ahora falta dX
	//Tranpongo W
	// actualizo x_T
	const int block_size3 = 16;

    dim3 threads3(block_size3, block_size3, 1);
	dim3 grid3((W_T.width -1) / threads3.x + 1, (W.height - 1) / threads3.y + 1);
	// dim3 grid(X.width / block_size, Y.width / block_size, 1);

	transpose<block_size3> <<<grid3, threads3 >>> (
		W.getDeviceData(),
		W_T.getDeviceData(),
		W_T.getWidth(),
		W.getWidth()
	);
	cudaDeviceSynchronize();

	// Ahora hago dX = dY W_T
	// Ahora necesito calcular dw = x_t * dY
	const int block_size4 = 32;

	dim3 threads4(block_size4, block_size4);
	dim3 grid4((W_T.width -1) / threads4.x + 1, (dY.height - 1) / threads4.y + 1);

	XdotW<block_size4> <<<grid4, threads4 >>> (
		dY.getDeviceData(),
		W_T.getDeviceData(),
		dX.getDeviceData(),
		dY.getWidth(),
		W_T.getWidth(),
		dY.getHeight(),
		W_T.getHeight()
	);

	cudaDeviceSynchronize();

}

Matrix& Dense::getW(){return W;}

Matrix& Dense::getdW(){return dW;};

Matrix& Dense::getOutput(){return Y;}

Matrix& Dense::getGradOutput(){return dY;}

Matrix& Dense::getOutput_T(){return Y_T;}

void Dense::updateW(float lr){
	int dev;
	cudaGetDevice(&dev);
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
	
	// dim3 nThreads(256);
	dim3 nThreads(deviceProp.maxThreadsDim[0]);
	dim3 nBlocks((W.size + nThreads.x - 1) / nThreads.x);
	if(nBlocks.x > deviceProp.maxGridSize[0]){
		nBlocks.x = deviceProp.maxGridSize[0];
	}
	
	updateWKernel<<< nBlocks, nThreads >>>(W.getDeviceData(), dW.getDeviceData(), lr, W.size);
	cudaDeviceSynchronize();
}


__global__ void updateWKernel(float *W, float *dW, float lr, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < size){
		W[i] = (W[i] - lr * dW[i]);
		i += blockDim.x * gridDim.x;
	}
}

/* ----------------------------
Input Layer
---------------------------- */

class Input : public Layer{
private:
	int width, height; // salida, entrada (entrada no la se a esta altiura)
	// Matrix Datos;
	Matrix W;	// Not needed
	Matrix dW;	// Not needed
	Matrix W_T;	// Not needed
	Matrix Y;
	Matrix dY;	// Not needed
	Matrix Y_T;	// Not needed
public:
	Input(int width, int height = -1);
    ~Input();
	
	// void call(Matrix &in, Matrix &out);
	// void gradient(Matrix &in, Matrix &out);
	void printWeights();
	int getWidth();
	int getHeight();
	std::string getActivation();

	void forward(Matrix &X);
	void backward(Matrix &X, Matrix &dX, Matrix &dX_T);

	Matrix& getW();
	Matrix& getdW();
	Matrix& getOutput();
	Matrix& getGradOutput();
	Matrix& getOutput_T();

	void updateW(float lr);
};

// Input::Input(int width, int height):Layer("Input"), width(width), height(-1){}

//#! Capaz cambie esto
Input::Input(int width, int height)
	: Layer("Input"), width(width), height(-1), Y(height, width), Y_T(width,height),  W(1,1), dW(1,1), W_T(1,1){}


Input::~Input(){}

void Input::printWeights(){
	std::cout << "Input Layer - Serian los datos" << std::endl;
	float *ptr_W = Y.getHostData();
	for(int i=0; i < Y.height; ++i){
		for(int j=0; j < Y.width; ++j)
			std::cout << ptr_W[i*Y.width + j] << "\t";
		std::cout << std::endl;
	}
}

int Input::getWidth(){return width;}

int Input::getHeight(){return height;}

std::string Input::getActivation(){return "None";}

void Input::forward(Matrix &X){
	// Esta funcion creo que deberia inicializar los datos a la matrix correspondiente
	// ACA ASUMO QUE YA TIENEN EL MISMO TAMAÑO
	// No hay que cagarla con los constructurores
	std::cout << "Unimplemted - Input Layer" << std::endl;

	if((X.getHeight() != Y.getHeight()) || (X.getWidth() != Y.getWidth())){
		Y.initialize(X.getHeight(), X.getWidth());
		//  Matrix::initialize(int height_, int width_, std::string dist, float w)
	}
	Y.copyDeviceDataFromAnother(X);
}

void Input::backward(Matrix &X, Matrix &dX, Matrix &dX_T){
	std::cout << "Unimplemted - Backward - Input Layer" << std::endl;
}

Matrix& Input::getW(){return W;}

Matrix& Input::getdW(){return dW;};

Matrix& Input::getOutput(){return Y;}

Matrix& Input::getGradOutput(){return dY;}

Matrix& Input::getOutput_T(){return Y_T;}

void Input::updateW(float lr){return;};



/* ----------------------------
Kernels
---------------------------- */

// Kernel modified from https://www.programmersought.com/article/13436584263/
template<int BLOCK_SIZE> __global__ void
XdotWplusBias(float* A, float* B, float* C, int wA, int wB, int hA, int hB, float *bias){
// XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
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
	if (ty < subAh && tx < subBw)	{
		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub;
		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[by*BLOCK_SIZE+ty]; //row
		C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[bx*BLOCK_SIZE+tx]; //col
	}	
}


// Kernel modified from https://www.programmersought.com/article/13436584263/
template<int BLOCK_SIZE> __global__ void
XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
// XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
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
	if (ty < subAh && tx < subBw)	{
		C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub;
		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[by*BLOCK_SIZE+ty]; //row
		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[bx*BLOCK_SIZE+tx]; //col
	}	
}




// template<int BLOCK_SIZE> __global__ void
// XTdotdY(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
// // XdotW(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
// 	//Block index
// 	int bx = blockIdx.x;
// 	int by = blockIdx.y;

// 	//Thread index
// 	int tx = threadIdx.x;
// 	int ty = threadIdx.y;

// 	/* Divide the matrix into sub-matrices, apply the parallel calculation of the thread in the block
// 	to the multiplication of the sub-matrices, and finally add their values ​​to obtain an element value of C */
// 	int aBegin = by * BLOCK_SIZE * wA;	//The row coordinates of the sub-matrix of A
// 	int aStep = BLOCK_SIZE;				//The movement step size of A's sub-matrix column coordinates
// 	int aEnd = aBegin + wA - 1;			//Limit an end point

// 	int bBegin = bx * BLOCK_SIZE;
// 	int bStep = BLOCK_SIZE * wB;

// 	float Csub = 0;	//Define the element value of C at the corresponding position in the block (x,. y) (ty, tx)

// 	int subAw = BLOCK_SIZE;
// 	int subAh = BLOCK_SIZE;
// 	int subBh = BLOCK_SIZE;
// 	int subBw = BLOCK_SIZE;

// 	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
// 		//The number of columns in the last column of the A matrix is ​​less than BLOCK_SIZE
// 		if (a + aStep - 1 > aEnd){			
// 			subAw = aEnd - a + 1;
// 		}else{
// 			subAw = BLOCK_SIZE;
// 		}
// 		subBh = subAw;

// 		//The number of rows in the last row of the A matrix is ​​less than BLOCK_SIZE
// 		if ((by + 1) * BLOCK_SIZE > hA){
// 			subAh = hA - by * BLOCK_SIZE;
// 		}else{
// 			subAh = BLOCK_SIZE;
// 		}

// 		//The number of columns in the last column of the B matrix is ​​less than BLOCK_SIZE
// 		if ((bx + 1) * BLOCK_SIZE > wB){
// 			subBw = wB - bx * BLOCK_SIZE;
// 		}else{
// 			subBw = BLOCK_SIZE;
// 		}
		
// 		/* Develop shared memory in the block */
// 		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
// 		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

// 		/* Assign values ​​to the corresponding elements of the sub-matrix in the range of rows and columns */
// 		if (ty < subAh && tx < subAw){
// 			As[ty][tx] = A[a + ty * wA + tx];
// 		}
// 		if (ty < subBh && tx < subBw){
// 			Bs[ty][tx] = B[b + ty * wB + tx];
// 		}
// 		__syncthreads();

// 		//Unroll the loop to compile to speed up		
// 		#pragma unroll
// 		//The inner loop calculates the vector product of the corresponding row and column in each sub-matrix and adds it to the previously obtained value
// 		for (int k = 0; k < subAw; k++){
// 			//Satisfy the elements within the row and column constraints to calculate the product and sum
// 			if (ty < subAh && tx < subBw){
// 				Csub += As[ty][k] * Bs[k][tx];
// 			}			
// 		}
// 		__syncthreads();
// 	}

// 	//Satisfy the elements within the row and column constraints to calculate the product and sum
// 	if (ty < subAh && tx < subBw)	{
// 		C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub;
// 		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[by*BLOCK_SIZE+ty]; //row
// 		// C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub + bias[bx*BLOCK_SIZE+tx]; //col
// 	}	
// }

template<int BLOCK_SIZE> __global__ void
transpose(float *odata, float *idata, int width, int height){
	__shared__ float block[BLOCK_SIZE][BLOCK_SIZE+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if((xIndex < width) && (yIndex < height))	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}
        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if((xIndex < height) && (yIndex < width))	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}








#endif
