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
#include "Matrix.h"
#include "Activation.h"






int main(int argc, const char** argv) {

	Matrix A(3, 2);
	Activation *activacion;
	activacion = new Sigmoid;

	A.print();

	std::cout << std::endl;

	// dim3 nThreads(256); // CORREGIR
	// dim3 nBlocks((A.size + nThreads.x - 1) / nThreads.x); // CORREGIR

	// if(nBlocks.x>65535)
	// 	nBlocks.x=65535;	
	
	// sigmoidKernel<<< nBlocks, nThreads >>>(A, A.size);
	// sigmoidKernel<<< 1, 6 >>>(&A, A.size);
	// sigmoidKernel<<< 1, 6 >>>(A.d_elem, A.size);
	activacion->call(A);
	cudaDeviceSynchronize();

	A.print();
	std::cout << std::endl;

	A.copyDeviceToHost();

	A.print();
	std::cout << std::endl;


    return 0;
}