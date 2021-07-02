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
#include "models.cu"
#include "layers.cu"

class NeuralNetwork;
class Layer;
/* ----------------------------
Optimizer class
---------------------------- */
class Optimizer{
private:
	std::string name;
    float lr;

	float cost;
	float acc;
    
public:
	Optimizer(float lr);	//Default constructor
	virtual ~Optimizer();

	void call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs);

	void updateW(Layer *layer);

	float getCost();
	float getAcc();
	
};

Optimizer::Optimizer(float lr) : lr(lr) {}

Optimizer::~Optimizer(){}

void Optimizer::call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs){
	int nSamples = X.getHeight();
	// int nBatch = int(nSamples / bs);
	for(int from = 0, to = bs-1; to < nSamples; from+=bs, to+=bs){
	// for(int from = 0, to = nBatch-1; to < nSamples; from+=bs, to+=bs){
		// Copiar a la matrix del primer layer los datos
		std::vector<Layer*>::iterator itr;
		// itr = layers.begin();
		itr = NN.getLayers().begin();
		(*itr)->getOutput().copyDeviceDataFromBatch(X, d_idx, from);

		NN.forward();
	}
}

void Optimizer::updateW(Layer *layer){
	std::cout << "Falta implementar" << std::endl;
}

float Optimizer::getCost(){return cost;}

float Optimizer::getAcc(){return acc;}

#endif
