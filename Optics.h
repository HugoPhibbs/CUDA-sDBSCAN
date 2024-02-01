#pragma once

#include "Header.h"
#include "Utilities.h"
#include "Dbscan.h"

void findCoreDist();
//void findCoreDist_Symmetric_L2_Embed(); // Only for testing

void formOptics(IVector & , IVector &, FVector &);
void formOptics_scikit(IVector &, FVector &);


void sOptics();






