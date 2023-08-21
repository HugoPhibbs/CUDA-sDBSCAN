#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

#include <sstream> // stringstream
#include <time.h> // for time(0) to generate different random number

/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

float computeDist(const Ref<VectorXf>, const Ref<VectorXf>);
float computeChi2(const Ref<VectorXf>, const Ref<VectorXf>);

void embedChiSquare(const Ref<VectorXf>, Ref<VectorXf>);
void embedJS(const Ref<VectorXf>, Ref<VectorXf>);

/* Generate Hadamard matrix
*/
void bitHD3Generator(int, boost::dynamic_bitset<> &);
void gaussGenerator(int, int, float, float, Ref<MatrixXf>);
void cauchyGenerator(int, int, float, float, Ref<MatrixXf>);

// Saving
void outputDbscan(const IVector &, string);
void outputOptics(const IVector &, const FVector &, const IVector &, string);


#endif // UTILITIES_H_INCLUDED
