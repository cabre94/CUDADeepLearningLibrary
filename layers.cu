#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdexcept>
#include "Matrix.cu"
#include "Activation.cu"

/* ----------------------------
Layer class
---------------------------- */
class Layer{
private:
    std::string name;
    
public:
	Layer(std::string name_);	//Default constructor
	virtual ~Layer();
	
	std::string getName();
	// virtual void call(Matrix &in, Matrix &out) = 0;
	// virtual void gradient(Matrix &in, Matrix &out) = 0;
	virtual void printWeights() = 0;
};

Layer::Layer(std::string name_) : name(name_) {}

Layer::~Layer(){}

std::string Layer::getName(){return name;}


/* ----------------------------
Dense Layer
---------------------------- */

class Dense : public Layer{
private:
	Matrix W;
	Matrix b;
	Activation *activation;
public:
	Dense(int width, int height, std::string act, std::string dist = "uniform", float w = 0.1);
    ~Dense();
	
	// void call(Matrix &in, Matrix &out);
	// void gradient(Matrix &in, Matrix &out);
	void printWeights();
};

Dense::Dense(int width, int height, std::string act, std::string dist, float w)
	:Layer("Dense"), W(width,height,dist,w), b(1,height,"ones",0) {
		if(act == "linear")
			activation = new Linear;
		else if(act == "relu")
			activation = new Relu;
		else if(act == "sigmoid")
			activation = new Sigmoid;
		else if(act == "tanh")
			activation = new Tanh;
		else if(act == "leakyRelu")
			activation = new LeakyRelu();
		else
			throw std::invalid_argument("Invalid activation");
	}

Dense::~Dense(){
	delete activation;
}

void Dense::printWeights(){
	float *ptr_W = W.getHostData();
	float *ptr_b = b.getHostData();
	for(int i=0; i < W.height; ++i){
		for(int j=0; j < W.width; ++j)
			std::cout << ptr_W[i*W.width + j] << "\t";
		std::cout << ptr_b[i] << "\t";
		std::cout << std::endl;
	}
}


/* ----------------------------
Input Layer
---------------------------- */

class Input : public Layer{
private:
	int out_width, out_height;
public:
	Input(int width, int height);
    ~Input();
	
	// void call(Matrix &in, Matrix &out);
	// void gradient(Matrix &in, Matrix &out);
	void printWeights();
};

Input::Input(int width, int height):Layer("Input"), out_width(width), out_height(height){}

Input::~Input(){}

void Input::printWeights(){
	std::cout << "Input Layer?" << std::endl;
}






#endif
