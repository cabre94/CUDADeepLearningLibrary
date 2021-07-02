#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "Matrix.cu"


/* ----------------------------
Metric class
---------------------------- */
class Metric{
private:
    std::string name;
    
public:
	Metric(std::string name_);	//Default constructor
	virtual ~Metric();
	
	std::string getName();
	virtual float call(Matrix &y_pred, Matrix &y_true) = 0;
};

Metric::Metric(std::string name_) : name(name_) {}

Metric::~Metric(){}

std::string Metric::getName(){return name;}

/* ----------------------------
MSE XOR
---------------------------- */
class MSE_XOR : public Metric{
public:
	MSE_XOR();
    ~MSE_XOR();

	float call(Matrix &y_pred, Matrix &y_true);
};

MSE_XOR::MSE_XOR():Metric("MSE_XOR"){}

MSE_XOR::~MSE_XOR(){}

float MSE_XOR::call(Matrix &y_pred, Matrix &y_true){
	// Voy a hacerlo serial y despues ver que hago
	y_pred.copyDeviceToHost();
	y_true.copyDeviceToHost();

	float acc = 0;
	for(int i=0; i < y_pred.size; ++i){
		acc += float(y_pred.h_elem[i]==y_true.h_elem[i]);
	}
	acc = (acc / y_pred.height);
	return acc;
}

#endif
