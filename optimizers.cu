#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>
#include <cstdlib>      // std::rand, std::srand
#include "Matrix.cu"
// #include "models.cu"
#include "layers.cu"

class NeuralNetwork;
class Layer;
/* ----------------------------
Optimizer class
---------------------------- */
class Optimizer{
private:
	std::string name;

	float cost;
	float acc;
    
public:
	Optimizer();	//Default constructor
	virtual ~Optimizer();

	void call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs) = 0;

	void updateW(Layer *layer) = 0;

	float getCost();
	float getAcc();
	
};

Optimizer::Optimizer(std::string name_) : name(name_){}

Optimizer::~Optimizer(){}

float Optimizer::getCost(){return cost;}

float Optimizer::getAcc(){return acc;}



/* ----------------------------
SGD class
---------------------------- */
class SGD : public Optimizer{
private:
    float lr;
	
public:
	SGD(float lr);	//Default constructor
	virtual ~SGD();
	
	void call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs);
	
	void updateW(Layer *layer);
	
};

SGD::SGD(float lr) : Optimizer("SGD"), lr(lr) {}

SGD::~SGD(){}

void SGD::call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs){
	int nSamples = X.getHeight();		// # de datos
	int nBatch = int(nSamples / bs);	// # de batchs
	float loss_mean = 0;				// Loss de la epoca
	
	for(int from = 0, to = bs-1; to < nSamples; from+=bs, to+=bs){
	// for(int from = 0, to = nBatch-1; to < nSamples; from+=bs, to+=bs){
		// Copiar a la matrix del primer layer los datos
		std::vector<Layer*>::iterator itr;
		// itr = layers.begin();
		itr = NN.getLayers().begin();
		(*itr)->getOutput().copyDeviceDataFromBatch(X, d_idx, from);
		
		NN.forward();
		
		loss_mean += NN.getLoss()->call();
		// Calcular metrica
		
		//Backward
		
		// Actualizar W
		// Actualizar b
	}
	NN.getLoss().push_back(loss_mean/nBatch);
}
	
void SGD::updateW(Layer *layer){
	std::cout << "Falta implementar" << std::endl;
}
	
	
#endif
	