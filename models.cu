#ifndef MODELS_H
#define MODELS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>       // std::vector
#include "Matrix.cu"
#include "Activation.cu"
#include "layers.cu"
#include "losses.cu"
#include <algorithm>    // std::random_shuffle
// #include "optimizers.cu"
#include "metrics.cu"

class Optimizer;

class NeuralNetwork{
public:
// private:
    std::vector<Layer*> layers;
	// optimizador
	Loss *loss; // loss
	// metrica
	int batch_size;
	std::vector<float> loss_log, val_loss_log;
	std::vector<float> acc_log, val_acc_log;

	Matrix y, val_y;
    
public:
	NeuralNetwork();	//Default constructor
	NeuralNetwork(int width, int height);	//Default constructor
	~NeuralNetwork();

	// void add(Layer *layer);
	void setLoss(std::string l="MSE");
	void add(std::string type, int nn, std::string act, std::string dist = "uniform", float w = 0.1);
	// void getLayer(int idx);
	// void fit(int epochs, int batch_size_ = 1);
	void fit(Matrix &X, Matrix &Y, int epochs, int batch_size_ = 1);
	void predict();
	void forward(Matrix &X);
	void backward();

	void print();
	void printWeights();
	void printAllDimensions();

	void setBatchSize(int batch_size);

	Loss* getLoss();
	std::vector<Layer*>& getLayers();

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

void NeuralNetwork::setLoss(std::string l){
	if(l == "MSE")
		loss = new MSE;
	else
		throw std::invalid_argument("Invalid activation");
}

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



void NeuralNetwork::fit(Matrix &X, Matrix &Y, int epochs, int batch_size_){
	setBatchSize(batch_size_);
	// Setear loss
	// Setear  batth
	// Setear optimizador (este deberia tener el lr)
	// Setear metrica
	// - Training 
	// For sobre epocas
	// 		como hago para quedarme con un pedazo de datos?
	// 		- Le puedo pasar la matrix y los indices en donde se tiene que quedar (deberian ser random)
	//		Forward con batch (optimizador)
	//		Backward con batch (optimizador)
	//		Calcular loss y metrica y appendear
	//	Calcular metricas para val_data

	int nSamples = X.getHeight();		// Numero de datos
	// int nBatch = int(nSamples / batch_size);	// Numero de Batchs
	int *idx = new int[nSamples];		// Indices
	for(int i=0; i < nSamples; ++i){
		idx[i] = i;
	}
	// Alloco los indices en device
	float *d_idx;
	cudaMalloc(&d_idx, nSamples * sizeof(int));

	srand(time(0));

	for(int e = 1; e <= epochs; ++e){
		// shuffleo los indices en host
		std::random_shuffle(&idx[0], &idx[nSamples]);
		// lo paso a device
		cudaMemcpy(d_idx, idx, nSamples * sizeof(int), cudaMemcpyHostToDevice);

		continue;
	}

	delete [] idx;
	cudaFree(d_idx);
	return;
}

void NeuralNetwork::predict(){
	std::cout << "Predict method unimplemented" << std::endl;
	return;
}

void NeuralNetwork::forward(Matrix &X){
	std::vector<Layer*>::iterator l, l_prev;
	l = layers.begin();
	l_prev = layers.begin();

	// Tomo la matriz de entrada y la "guardo" en el layer input
	// (*l)->forward(X);	CREO QUE NO HACE FALTA
	l++;
	
	// Actualizo iterativamente la salida de cada capa
	while(l != layers.end()){
		(*l)->forward((*l_prev)->getOutput());
		l++;
		l_prev++;
	}
	// Creo con eso ya estaria, no? La ultima capa ya deberia tener su
	// salida y con eso deberia poder calcular el costo.
}

void NeuralNetwork::backward(){
	std::cout << "Backward method unimplemented" << std::endl;
	// calcular el costo con la ultima capa usando el y_true
	// Actualizar el gradiente de la ultima capa
	// Iterativamente calcular los dW y dY de cada capa
	// Actualizar W usando los dW calculados
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
	}
}

void NeuralNetwork::printAllDimensions(){
	std::vector<Layer*>::iterator itr;

	for(itr = layers.begin(); itr != layers.end(); ++itr){
		std::cout << (*itr)->getName() << " - ";
		// W
		std::cout << "W" << " - ";
		(*itr)->getW().printDimensions();
		// Y
		std::cout << "Y" << " - ";
		(*itr)->getOutput().printDimensions();
		// dY
		std::cout << "dy" << " - ";
		(*itr)->getGradOutput().printDimensions();
		std::cout << std::endl;
	}

}

void NeuralNetwork::setBatchSize(int batch_size_){
	batch_size = batch_size_;
	std::vector<Layer*>::iterator itr;

	for(itr = layers.begin(); itr != layers.end(); ++itr){
		int out_dim = (*itr)->getWidth();

		(*itr)->getOutput().initialize(batch_size, out_dim);
		(*itr)->getGradOutput().initialize(batch_size, out_dim);
	}
}

Loss* NeuralNetwork::getLoss(){return loss;}

std::vector<Layer*>& NeuralNetwork::getLayers(){return layers;}









#endif
