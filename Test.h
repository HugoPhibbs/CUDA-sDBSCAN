#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"
#include "Dbscan.h"


void FourierEmbed_L2();
void FourierEmbed_Nonmetric();

void parRandomProjection();
void seqRandomProjection();

void test_sDbscan(int);
void test_sDbscan_L2(int);
void test_sDbscan_Asym(int);
void test_sOptics_Asym();

#endif
