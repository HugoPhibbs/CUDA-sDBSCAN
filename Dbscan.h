#pragma once

#include "Header.h"
#include "Utilities.h"

void FourierEmbed_L2();
void parRandomProjection();
void seqRandomProjection();

void parDbscanIndex_L2();
//void parDbscanIndex_Sort(); // Only for L2 testing: slower than using PQ

void parDbscanIndex_Metric();
void parDbscanIndex_NonMetric();

void findCorePoints_Asym();
void findCorePoints();


void formCluster(int &, IVector &);

void fastDbscan();
void memoryDbscan();


void findCoreDist_Asym();
void findCoreDist();
//void findCoreDist_Symmetric_L2_Embed(); // Only for testing

void formOptics(IVector & , IVector &, FVector &);
void formOptics_scikit(IVector &, FVector &);

void fastOptics();
void memoryOptics();






