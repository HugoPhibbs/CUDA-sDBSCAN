#include <iostream>
#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

//#include "Test.h"
#include "Dbscan.h"

#include <time.h> // for time(0) to generate different random number
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday
#include <stdio.h>
#include <unistd.h>

#include <omp.h>

int main(int nargs, char** args)
{
    srand(time(NULL)); // should only be called once for random generator

//    cout << "RAM before loading data" << endl;
//    getRAM();

    /************************************************************************/
	int iType = loadInput(nargs, args);

//    cout << "RAM after loading data" << endl;
//    getRAM();

	/************************************************************************/
	/* Approaches                                             */
	/************************************************************************/

    chrono::steady_clock::time_point begin, end;

    /************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/
	switch (iType)
	{
        case 1:
        {
            begin = chrono::steady_clock::now();

            fastDbscan(); // speed friendly
//            memoryDbscan(); // memory friendly

            end = chrono::steady_clock::now();
            cout << "DBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;

        }

        case 2:
        {

            begin = chrono::steady_clock::now();

            fastOptics(); // for speed
//            memoryOptics(); // for memory

            end = chrono::steady_clock::now();
            cout << "OPTICS Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
            cout << endl;
            break;
        }

    }
}

