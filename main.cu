/*
date: 28-06-21
File: main.cu
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
*/

// #include "Matrix/Matrix.h"
#include "Matrix.cu"
#include "Activation.cu"
#include "layers.cu"
#include "models.cu"
#include "losses.cu"

// #define BLOCK_SIZE 2
// #define TILE_DIM 2



// Kernel modified from https://www.programmersought.com/article/13436584263/
template<int BLOCK_SIZE> __global__ void
MatrixMulCUDA(float* A, float* B, float* C, int wA, int wB, int hA, int hB){
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


void testRed();

int main(int argc, const char** argv){
	const int block_size = 32;

	
	Matrix X(3, 2, "uniform");
	Matrix W(2, 4, "uniform");
	Matrix b(1, 4, "uniform");
	Matrix Y(3, 4, "zeros");

	
	// y_pred.copyDeviceToHost();
	// y_true.copyDeviceToHost();
	// dY.copyDeviceToHost();

	std::cout << "X" << std::endl; X.print(); std::cout << std::endl;
	std::cout << "W" << std::endl; W.print(); std::cout << std::endl;
	std::cout << "b" << std::endl; b.print(); std::cout << std::endl;
	std::cout << "Y" << std::endl; Y.print(); std::cout << std::endl;
	
	// int dev;
	// cudaGetDevice(&dev);
	
	// cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
	
	// // dim3 nThreads(256);
	// dim3 nThreads(deviceProp.maxThreadsDim[0]);
	// dim3 nBlocks((C.size + nThreads.x - 1) / nThreads.x);
	// if(nBlocks.x > deviceProp.maxGridSize[0]){
	// 	nBlocks.x = deviceProp.maxGridSize[0];
	// }
	
	// MatMulCublas(A, B, C);
	// AdotBKernel<<< 256, 1024 >>>(
	// 	A.getDeviceData(),
	// 	B.getDeviceData(),
	// 	C.getDeviceData(),
	// 	A.getWidth(),
	// 	A.getHeight(),
	// 	B.getWidth(),
	// 	B.getHeight()
	// );
	// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	dim3 threads(block_size, block_size);
	dim3 grid((W.width -1) / threads.x + 1, (X.height - 1) / threads.y + 1);
	// dim3 grid((dimsB.x -1) / threads.x + 1, (dimsA.y - 1) / threads.y + 1);
	
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



	Y.copyDeviceToHost();
	
	std::cout << "Y" << std::endl; Y.print(); std::cout << std::endl;
	
	
	
	
	
	return 0;
}




void testRed(){

	NeuralNetwork nn(2,3);
	nn.add("Dense",3,"linear");
	nn.add("Dense",2,"relu");

	nn.print();

	nn.printWeights();

}


// Matrix A(2,3);
// A.print();
// std::cout << std::endl << std::flush;
// A.initialize(4,4);
// A.print();





// void MatMulCublas(Matrix &A, Matrix &B, Matrix &C);


// int main(int argc, const char** argv){
	
	// 	Matrix A(3, 2, "uniform");
	// 	Matrix B(2, 4, "uniform");
	// 	Matrix C(3, 4, "zeros");
	
	// 	// y_pred.copyDeviceToHost();
	// 	// y_true.copyDeviceToHost();
	// 	// dY.copyDeviceToHost();
	
	// 	std::cout << "A" << std::endl; A.print(); std::cout << std::endl;
	// 	std::cout << "B" << std::endl; B.print(); std::cout << std::endl;
// 	std::cout << "C" << std::endl; C.print(); std::cout << std::endl;

// 	MatMulCublas(A, B, C);
// 	C.copyDeviceToHost();

// 	std::cout << "C" << std::endl; C.print(); std::cout << std::endl;





// 	return 0;
// }





// int main(int argc, const char** argv){
	
	// 	Matrix y_pred(3, 2, "uniform");
	// 	Matrix y_true(3, 2, "ones");
	// 	Matrix dY(3, 2, "zeros");
	
	// 	y_pred.copyDeviceToHost();
	// 	y_true.copyDeviceToHost();
	// 	dY.copyDeviceToHost();
	
	// 	std::cout << "y_pred" << std::endl; y_pred.print(); std::cout << std::endl;
	// 	std::cout << "y_true" << std::endl; y_true.print(); std::cout << std::endl;
	// 	std::cout << "dY" << std::endl; dY.print(); std::cout << std::endl;
	
	// 	Loss *loss;
	// 	loss = new MSE;
	
	
	
// 	float cost = loss->call(y_pred, y_true);
// 	std::cout << "Costo: " << cost << std::endl;

// 	loss->gradient(y_pred, y_true, dY);
// 	dY.copyDeviceToHost();

// 	std::cout << std::endl << "El gradiente: " << std::endl;
// 	dY.print();



// 	delete loss;





// 	return 0;
// }

















/*
int main(int argc, const char** argv) {
	
	std::string D = "uniform";
	
	Matrix A(3, 2, D);
	Matrix B(3, 2, D);
	
	Activation *activacion;
	// activacion = new Sigmoid;
	activacion = new Relu;
	// activacion = new Linear;
	// activacion = new Tanh;
	// activacion = new LeakyRelu(0.3);
	
	std::cout << "A" << std::endl;
	A.print();
	std::cout << std::endl;
	
	std::cout << "B" << std::endl;
	B.print();
	std::cout << std::endl;
	
	std::cout << "aplico " << activacion->getName() << " a A y guardo en B" << std::endl;
	activacion->call(A,B);
	A.copyDeviceToHost();
	B.copyDeviceToHost();
	
	// Ahora veo cuanto vale A y B
	std::cout << "A" << std::endl;
	A.print();
	std::cout << std::endl;
	
	std::cout << "B" << std::endl;
	B.print();
	std::cout << std::endl;
	
	std::cout << "aplico Grad " << activacion->getName() << " sigmoide a A (que sigue igual) y guardo en B" << std::endl;
	activacion->gradient(A,B);
	A.copyDeviceToHost();
	B.copyDeviceToHost();
	
	A.print();
	std::cout << std::endl;
	
	B.print();
	std::cout << std::endl;
	
	delete activacion;
	
	
    return 0;
}
*/







/*
void MatMulCublas(Matrix &A, Matrix &B, Matrix &C){
	// buena costumbre: hacer algo con los codigos de error
    cublasStatus_t stat;
	
    float  al = 1.0f;                 
    float bet = 0.0f;
    int m = C.width;
	
    cublasHandle_t manija;
    stat = cublasCreate(&manija);
	
    // Esto:
    //cudaMemcpy(A.elements, d_A.elements, size,cudaMemcpyDeviceToHost);
    //cudaMemcpy(B.elements, d_B.elements, size,cudaMemcpyDeviceToHost);
    //cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
    // Es equivalente a esto:
    // stat = cublasSetMatrix(m,m,sizeof(float),(A.elements),m,(d_A.elements) ,m);//a -> d_a
    // stat = cublasSetMatrix(m,m,sizeof(float),(B.elements),m,(d_B.elements) ,m);//b -> d_b
    // stat = cublasSetMatrix(m,m,sizeof(float),(C.elements),m,(d_C.elements) ,m);//c -> d_c
	
    // multiplication
    stat = cublasSgemm(manija, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &al, A.d_elem, m, B.d_elem, m, &bet, C.d_elem, m);

    // La variable stat se puede usar asi (recomendado para todas las llamadas...)
    if (stat != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "!!!! CUBLAS Sgemm error\n");
		exit(1);
    }
	
    // Hacer esto:
    //cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
    // es equivalente a esto:
    // stat = cublasGetMatrix(m,m,sizeof(float),(d_C.elements) ,m,(C.elements),m);	//d_c->c
	// C.copyDeviceToHost();
	
    // Free device memory
    cublasDestroy(manija); // liberamos las "variables ocultas" de cublas
}
*/



// __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
	
//     float CValue = 0;
	
//     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
//     int Col = blockIdx.x*TILE_DIM + threadIdx.x;

//     __shared__ float As[TILE_DIM][TILE_DIM];
//     __shared__ float Bs[TILE_DIM][TILE_DIM];
		 
//     for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {
			
//         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
// 			As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
// 		else
// 			As[threadIdx.y][threadIdx.x] = 0.0;

// 		if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
// 			Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
// 		else
// 			Bs[threadIdx.y][threadIdx.x] = 0.0;
		 
// 		__syncthreads();

// 		for (int n = 0; n < TILE_DIM; ++n)
// 			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
		
// 		__syncthreads();
//     }
	
//     if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
// }