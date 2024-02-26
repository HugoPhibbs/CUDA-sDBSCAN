#pragma once

#include "Header.h"
#include "Utilities.h"



//void parDbscanIndex_L2(); // Implement FWHT for Fourier embeddings
//void parDbscanIndex_Sort(); // Only for L2 testing: slower than using PQ

//void parDbscanIndex_Metric();
//void parDbscanIndex_NonMetric();

void parDbscanIndex();
void seqDbscanIndex(); // Faster and memory-efficient if using single-thread

void findCorePoints_Asym();
void formCluster_Asym(int &, IVector &); // Under construction

void findCorePoints();
void findCorePoints_uDbscan(); // based on sampling uDbscan++
void findCorePoints_sngDbscan(); // based on sampling

void formCluster(int &, IVector &);

void seq_sDbscan();

void sDbscan();
void sngDbscan();

void clusterNoise(IVector&, int);







