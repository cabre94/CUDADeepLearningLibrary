#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.cu"


/* ----------------------------
Optimizer class
---------------------------- */
class Optimizer{
private:
    float lr;
    
public:
	Optimizer(float lr);	//Default constructor
	virtual ~Optimizer();
	
	// std::string getName();
	// virtual void call(Matrix &in, Matrix &out) = 0;
	// virtual void gradient(Matrix &in, Matrix &out) = 0;
	// virtual void printWeights() = 0;
	// virtual int getWidth() = 0;
	// virtual int getHeight() = 0;
	// virtual std::string getActivation() = 0;


	
};

Optimizer::Optimizer(float lr) : lr(lr) {}

Optimizer::~Optimizer(){}




#endif
