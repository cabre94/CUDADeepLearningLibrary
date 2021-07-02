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
private:
    std::vector<Layer*> layers;
	// optimizador
	Loss *loss; // loss
	Optimizer *opt;
	// metrica
	int batch_size;
	std::vector<float> loss_log, val_loss_log;
	std::vector<float> acc_log, val_acc_log;

	Matrix Y_batch, val_Y_batch;

	bool loss_seted = false;
	bool opt_seted = false;

public:
	NeuralNetwork();	//Default constructor
	NeuralNetwork(int width, int height);	//Default constructor
	~NeuralNetwork();

	// void add(Layer *layer);
	void setLoss(std::string l="MSE");
	void setOptimizer(std::string opt="SGD", float lr=1e-2);

	void add(std::string type, int nn, std::string act, std::string dist = "uniform", float w = 0.1);
	// void getLayer(int idx);
	// void fit(int epochs, int batch_size_ = 1);
	void fit(Matrix &X, Matrix &Y, int epochs, int batch_size_ = 1);
	Matrix& predict();
	void forward(Matrix &X);
	void backward();

	void print();
	void printWeights();
	void printAllDimensions();

	void setBatchSize(int batch_size);

	Loss* getLossFunction();
	std::vector<Layer*>& getLayers();

	std::vector<float>& getLoss(){return loss_log;}
	std::vector<float>& getValLoss(){return val_loss_log;}
	std::vector<float>& getAcc(){return acc_log;}
	std::vector<float>& getValAcc(){return val_acc_log;}
	
	Matrix& getYbatch(){return Y_batch;}
	Matrix& getValYbatch(){return val_Y_batch;}
};


/* ----------------------------
Optimizer class
---------------------------- */

class Optimizer{
private:
	std::string name;

	float cost;
	float acc;
    
public:
	Optimizer(std::string);	//Default constructor
	virtual ~Optimizer();

	virtual void call(Matrix &X, Matrix &Y, NeuralNetwork &NN, int *d_idx, int bs) = 0;

	virtual void updateW(Layer *layer) = 0;

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
		continue;
		std::vector<Layer*>::iterator itr;
		
		// Copiar a la matrix del primer layer los datos
		itr = NN.getLayers().begin();
		(*itr)->getOutput().copyDeviceDataFromBatch(X, d_idx, from);
		
		// Ahora lo mismo para el Y_batch
		NN.getYbatch().copyDeviceDataFromBatch(Y, d_idx, from);
		
		
		// Forward
		NN.forward((*itr)->getOutput());
		
		// Calculo la loss
		loss_mean += NN.getLossFunction()->call((*itr)->getOutput(), NN.getYbatch());
		
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
	





NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(int width, int height){
	Layer *layer = new Input(width,height);
	layers.push_back(layer);
}

NeuralNetwork::~NeuralNetwork(){
	std::vector<Layer*>::iterator itr;
	if(!layers.empty()){
		for(itr = layers.begin(); itr != layers.end(); ++itr){
			delete (*itr);
		}
	}
	if(loss_seted){
		delete loss;
	}
	if(opt_seted){
		delete opt;
	}
}

void NeuralNetwork::setLoss(std::string l){
	if(l == "MSE")
		loss = new MSE;
	else
		throw std::invalid_argument("Invalid activation");
	loss_seted = true;
}

void NeuralNetwork::setOptimizer(std::string opt_, float lr){
	if(opt_ == "SGD")
		opt = new SGD(lr);
	else
		throw std::invalid_argument("Invalid activation");
	opt_seted = true;
}

void NeuralNetwork::add(std::string type, int nn, std::string act, std::string dist, float w){
	Layer *layer;

	if(type == "dense" || type == "Dense"){
		Layer *last_layer = layers.back();
		int input_shape = last_layer->getWidth();
		// layer = new Dense(nn,input_shape, act, dist, w);
		layer = new Dense(input_shape, nn, act, dist, w);
	}else
		throw std::invalid_argument("Invalid layer");

	layers.push_back(layer);
}

void NeuralNetwork::fit(Matrix &X, Matrix &Y, int epochs, int batch_size_){
	// setBatchSize(batch_size_);
	setBatchSize(X.height);
	// printAllDimensions();
	int nSamples = X.getHeight();		// Numero de datos

	std::vector<Layer*>::iterator itr = layers.begin();
	float loss_epoch;


	for(int e = 1; e <= epochs; ++e){
		// Mando los datos al primer layer
		(*itr)->getOutput().copyDeviceDataFromAnother(X);
		// Mando los datos al Y_batch
		Y_batch.copyDeviceDataFromAnother(Y);
		
		// // Forward
		forward((*itr)->getOutput());
		
		// // Calculo la loss
		loss_epoch = loss->call((*itr)->getOutput(), Y_batch);

		// Calcular metrica
		
		//Backward
		// Primero actualizo el gradiente de la ultima capa
		Layer *last_layer = layers.back();
		
		// Actualizar W
		// Actualizar b

		loss_log.push_back(loss_epoch);
		// acc_log.push_back(loss_epoch);
		// val_loss_log.push_back(loss_epoch);
		// val_acc_log.push_back(loss_epoch);

	}













	std::cout << "Hola" << std::endl;
	// delete [] idx;
	// cudaFree(d_idx);
}

/*
void fit(Matrix &X, Matrix &Y, int epochs, int batch_size_){
	setBatchSize(batch_size_);
	printAllDimensions();
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
	int *d_idx;
	cudaMalloc(&d_idx, nSamples * sizeof(int));

	srand(time(0));

	for(int e = 1; e <= epochs; ++e){
		// shuffleo los indices en host
		std::random_shuffle(&idx[0], &idx[nSamples]);
		// lo paso a device
		cudaMemcpy(d_idx, idx, nSamples * sizeof(int), cudaMemcpyHostToDevice);

		opt->call(X, Y, *this, d_idx, batch_size);

	}

	delete [] idx;
	cudaFree(d_idx);
	return;
}*/

Matrix& NeuralNetwork::predict(){
	// Asumo que ya hice el forward por ahora
	Layer *last_layer = layers.back();
	return last_layer->getOutput();
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
	//  Y_batch, val_Y_batch;
	Layer *last_layer = layers.back();
	Y_batch.initialize(batch_size, last_layer->getOutput().getWidth());
	val_Y_batch.initialize(batch_size, last_layer->getOutput().getWidth());
}

Loss* NeuralNetwork::getLossFunction(){return loss;}

std::vector<Layer*>& NeuralNetwork::getLayers(){return layers;}









#endif
