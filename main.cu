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
// #include "Matrix.h"
#include "Activation.cu"






int main(int argc, const char** argv) {

	Matrix A(3, 2);
	Matrix B(3, 2);

	Activation *activacion;
	// activacion = new Sigmoid;
	// activacion = new Relu;
	// activacion = new Linear;
	// activacion = new Tanh;
	activacion = new LeakyRelu(0.3);

	std::cout << "A" << std::endl;
	A.print();
	std::cout << std::endl;

	std::cout << "B" << std::endl;
	B.print();
	std::cout << std::endl;

	std::cout << "aplico sigmoide a A y guardo en B" << std::endl;
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

	std::cout << "aplico Grad sigmoide a A (que sigue igual) y guardo en B" << std::endl;
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