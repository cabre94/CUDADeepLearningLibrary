/*
date: 27-06-21
File: Activations.cu
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
*/

#include "Activation.h"

class Activation{
private:
    std::string name;
    
public:
    __device__ Activation();	//Default constructor
    __device__ virtual ~Activation();

    Activation(const Activation &) = delete;	//Copy constructor
    Activation &operator=(const Activation &) = delete;	//Copy assignment
    Activation(Activation &&) = delete;	//Move constructor
    Activation &operator=(Activation &&) = delete;	// Move assignment

    __host__ __device__ std::string getName();

};

__device__ Activation::Activation(std::string name)

Piece::Piece(PieceColour C, PieceType T, std::string N) : colour(C), type(T), name(N) {}

Piece::~Piece(){}

PieceType Piece::getType(){
    return type;
}

PieceColour Piece::getColour(){
    return colour;
}

std::string Piece::getName(){
    return name;
}





class Sigmoid : public Activation{
public:
    Sigmoid(ActivationColour C);
    ~Sigmoid();

    void printActivation();
    std::string getName();

    std::set<std::string> getPossibleMoves(Board  *board, std::string from);
};

Sigmoid::Sigmoid(PieceColour C):Activation(C,PAWN,"Sigmoid") {}




int main(int argc, const char** argv) {

    hello<<<1, 10>>>();  // 1 bloque con 10 hilos
    cudaDeviceSynchronize();

    return 0;
}