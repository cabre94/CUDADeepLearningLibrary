#ifndef MODELS_H
#define MODELS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.cu"
#include "Activation.cu"
#include "layers.cu"
#include "losses.cu"
#include "optimizers.cu"
#include "metrics.cu"


class NeuralNetwork{
private:
    std::vector<Layer*> layers;
	// optimizador
	// loss
	// metrica
    
public:
	NeuralNetwork();	//Default constructor
	~NeuralNetwork();

	void add(Layer *layer);
	// void getLayer(int idx);
	void fit();
	void predict();
	void forward();
	void backward();

	void print();
};

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::~NeuralNetwork(){
	std::vector<Layer*>::iterator itr;
	for(itr = layers.begin(); itr != layers.end(); ++itr){
		delete (*itr);
	}
}

void NeuralNetwork::add(Layer *layer){
	layers.push_back(layer);
}

void NeuralNetwork::fit(){
	std::cout << "Fit method unimplemented" << std::endl;
	return;
}

void NeuralNetwork::predict(){
	std::cout << "Predict method unimplemented" << std::endl;
	return;
}

void NeuralNetwork::forward(){
	std::cout << "Forward method unimplemented" << std::endl;
	return;
}

void NeuralNetwork::backward(){
	std::cout << "Backward method unimplemented" << std::endl;
	return;
}

void NeuralNetwork::print(){
	std::cout << "Neural Network Architecture" << std::endl;
	std::vector<Layer*>::iterator itr;
	for(itr = layers.begin(); itr != layers.end(); ++itr){
		std::cout << (*itr)->getName() << " - ";
		std::cout << "(" << (*itr)->getHeight() << ",";
		std::cout << (*itr)->getWidth()<< ")" << " - ";
		std::cout << (*itr)->getActivation() << std::endl;
	}
}









#endif
