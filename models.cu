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
    
public:
	NeuralNetwork();	//Default constructor
	~NeuralNetwork();
	
	// std::string getName();
	// virtual void call(Matrix &in, Matrix &out) = 0;
	// virtual void gradient(Matrix &in, Matrix &out) = 0;
	void add(Layer *layer);
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
