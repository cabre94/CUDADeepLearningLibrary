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
	NeuralNetwork(int width, int height);	//Default constructor
	~NeuralNetwork();

	// void add(Layer *layer);
	void add(std::string type, int nn, std::string act, std::string dist = "uniform", float w = 0.1);
	// void getLayer(int idx);
	void fit();
	void predict();
	void forward();
	void backward();

	void print();
	void printWeights();
};

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(int width, int height){
	Layer *layer = new Input(width,height);
	layers.push_back(layer);
}

NeuralNetwork::~NeuralNetwork(){
	std::vector<Layer*>::iterator itr;
	for(itr = layers.begin(); itr != layers.end(); ++itr){
		delete (*itr);
	}
}

// void NeuralNetwork::add(Layer *layer){
// 	layers.push_back(layer);
// }

void NeuralNetwork::add(std::string type, int nn, std::string act, std::string dist, float w){
	Layer *layer;

	if(type == "dense" || type == "Dense"){
		Layer *last_layer = layers.back();
		int input_shape = last_layer->getWidth();
		// layer = new Dense(nn,input_shape, act, dist, w);
		layer = new Dense(input_shape, nn, act, dist, w);
	}
	else
		throw std::invalid_argument("Invalid layer");

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

void NeuralNetwork::printWeights(){
	std::vector<Layer*>::iterator itr;
	for(itr = layers.begin(); itr != layers.end(); ++itr){
		(*itr)->printWeights();
		std::cout << std::endl << std::endl;
		// std::cout << (*itr)->getName() << " - ";
		// std::cout << "(" << (*itr)->getHeight() << ",";
		// std::cout << (*itr)->getWidth()<< ")" << " - ";
		// std::cout << (*itr)->getActivation() << std::endl;
	}
}








#endif
