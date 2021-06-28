#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdio.h>

__global__ void hello(){
    printf("Hola\n");
}

class Matrix{
public:
	int height, width, size;
	float *h_elem, *d_elem;

// public:
	Matrix(int width, int height);
	~Matrix();

	void copyDeviceToHost();
	void copyHostToDevice();
	void print();
};

// __device__ __host__ float sigmoid(int x){
// __device__ __host__ float sigmoid(int x);

// __global__ void sigmoidKernel(float* d_e, int size);


Matrix::Matrix(int height, int width) : height(height), width(width), size(width * height){
	h_elem = new float[size];
	// float aux[3] = {-1, 0 , 1};
	for(size_t i=0; i < size; ++i){
		h_elem[i] = i;
		// h_elem[i] = aux[i%3];
	}

	// Allocacion en device
	cudaMalloc(&d_elem, size * sizeof(float));
	cudaMemcpy( d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix::~Matrix(){
	delete [] h_elem;
	cudaFree(d_elem);
}

void Matrix::copyDeviceToHost(){
	cudaMemcpy(h_elem, d_elem, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::copyHostToDevice(){
	cudaMemcpy(d_elem, h_elem, size * sizeof(float), cudaMemcpyHostToDevice );
}

void Matrix::print(){
	for(int i=0; i < height; ++i){
		for(int j=0; j < width; ++j)
			std::cout << h_elem[i*width + j] << "\t";
		std::cout << std::endl;
	}
}

/*
------------------------------

class Activation{
private:
    std::string name;
    
public:
	__host__ __device__ Activation(std::string name_);	//Default constructor
	__host__ __device__ virtual ~Activation();

	Activation(const Activation &) = delete;	//Copy constructor
	Activation &operator=(const Activation &) = delete;	//Copy assignment
	Activation(Activation &&) = delete;	//Move constructor
	Activation &operator=(Activation &&) = delete;	// Move assignment

	__host__ __device__ std::string getName();
	__host__ __device__ void call() = 0;
};

__host__ __device__ Activation::Activation(std::string name_) : name(name_) {}

__host__ __device__ Activation::~Activation(){}

__host__ __device__ std::string Activation::getName(){
    return name;
}





class Sigmoid : public Activation{
public:
    __host__ __device__ Sigmoid(ActivationColour C);
    __host__ __device__ ~Sigmoid();

    void printActivation();
    __host__ __device__ std::string getName();
};

Sigmoid::Sigmoid(PieceColour C):Activation(C,PAWN,"Sigmoid") {}

*/


// __device__ __host__ float sigmoid(int x){

/*
enum PieceType{KING, QUEEN, BISHOP, KNIGHT, ROOK, PAWN, CHAMPION, MAGICIAN};


#include "Board.hpp"


class Board;

enum PieceColour{NONE, WHITE, BLACK};
enum State{CHECK, CHECKMATE, NORMAL};       //!
// enum PieceType{KING, QUEEN, BISHOP, KNIGHT, ROOK, PAWN};

// enum MoveType{HORIZONTAL, VERTICAL, DIAGONAL, L, FORWARD, ONESTEP};


class Piece{
protected:
    PieceColour colour;
    PieceType type;
    std::string name;
    
public:
    Piece(PieceColour C, PieceType T, std::string N);	//Default constructor
    virtual ~Piece();

    Piece(const Piece &) = delete;	//Copy constructor
    Piece &operator=(const Piece &) = delete;	//Copy assignment
    Piece(Piece &&) = delete;	//Move constructor
    Piece &operator=(Piece &&) = delete;	// Move assignment

    PieceType getType();
    PieceColour getColour();
    std::string getName();

    virtual std::set<std::string> getPossibleMoves(Board *board, std::string from) = 0;

    virtual void printPiece() = 0;

};

#include "Pawn.hpp"
#include "Rook.hpp"
#include "Knight.hpp"
#include "Bishop.hpp"
#include "Queen.hpp"
#include "King.hpp"
#include "Magician.hpp"
#include "Champion.hpp"

*/

#endif
