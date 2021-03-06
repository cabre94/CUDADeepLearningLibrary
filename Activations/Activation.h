#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <string>
#include <stdio.h>

__global__ void hello(){
    printf("Hola\n");
}


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
