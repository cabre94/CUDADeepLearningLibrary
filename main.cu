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
#include "layers.cu"
#include "models.cu"
#include "losses.cu"








int main(int argc, const char** argv){
	
	Matrix y_pred(3, 2, "uniform");
	Matrix y_true(3, 2, "ones");
	Matrix dY(3, 2, "zeros");

	y_pred.copyDeviceToHost();
	y_true.copyDeviceToHost();
	dY.copyDeviceToHost();

	std::cout << "y_pred" << std::endl; y_pred.print(); std::cout << std::endl;
	std::cout << "y_true" << std::endl; y_true.print(); std::cout << std::endl;
	std::cout << "dY" << std::endl; dY.print(); std::cout << std::endl;

	Loss *loss;
	loss = new MSE;



	float cost = loss->call(y_pred, y_true);
	std::cout << "Costo: " << cost << std::endl;

	loss->gradient(y_pred, y_true, dY);
	dY.copyDeviceToHost();

	std::cout << std::endl << "El gradiente: " << std::endl;
	dY.print();



	delete loss;





	return 0;
}














// int main(int argc, const char** argv){
	
// 	// Dense capa(2,3,"relu");

// 	// capa.printWeights();

// 	NeuralNetwork nn;
// 	nn.add(new Input(2,3));
// 	nn.add(new Dense(2,3,"linear"));
// 	nn.add(new Dense(2,3,"relu"));
// 	nn.add(new Dense(2,3,"sigmoid"));
// 	nn.add(new Dense(2,3,"tanh"));
// 	nn.add(new Dense(2,3,"leakyRelu"));

// 	nn.print();



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