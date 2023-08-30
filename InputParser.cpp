#include "InputParser.h"
#include "Header.h"

#include <stdlib.h>     /* atoi */
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream

#include <vector>
#include <string.h> // strcmp

int loadInput(int nargs, char** args)
{
    if (nargs < 6)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numPts") == 0)
        {
            PARAM_DATA_N = atoi(args[i + 1]);
            cout << "Number of rows/points of X: " << PARAM_DATA_N << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of rows/points is missing !" << endl;
        exit(1);
    }

    // Dimension
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numDim") == 0)
        {
            PARAM_DATA_D = atoi(args[i + 1]);
            cout << "Number of columns/dimensions: " << PARAM_DATA_D << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: Number of columns/dimensions is missing !" << endl;
        exit(1);
    }

    /** NOTE: Reading IO is not safe for OpenMP **/

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
//    cout << "Read row-wise X, it will be converted to col-major Eigen matrix of size D x N..." << endl;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--X") == 0)
        {
            FILE *f = fopen(args[i + 1], "r");
            if (!f)
            {
                cerr << "Error: Data file does not exist !" << endl;
                exit(1);
            }

            FVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0);

            // Each line is a vector of D dimensions
            for (int n = 0; n < PARAM_DATA_N; ++n)
            {
                for (int d = 0; d < PARAM_DATA_D; ++d)
                {
                    fscanf(f, "%f", &vecTempX[n * PARAM_DATA_D + d]);
                    // cout << vecTempX[n + d * PARAM_DATA_N] << " ";
                }
                // cout << endl;
            }

            // Matrix_X is col-major
            MATRIX_X = Map<MatrixXf>(vecTempX.data(), PARAM_DATA_D, PARAM_DATA_N);
    //        MATRIX_X.transpose();
    //        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

            /**
            Print the first col (1 x N)
            Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
            **/
    //        cout << MATRIX_X.col(0) << endl << endl;
    //        cout << "In memory (col-major):" << endl;
    //        for (int n = 0; n < 10; n++)
    //            cout << *(MATRIX_X.data() + n) << "  ";
    //        cout << endl << endl;

            break;
        }
    }


    // Algorithm
    int iAlgType = 0;

    // DBSCAN or OPTICS
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--alg") == 0)
        {
            if (strcmp(args[i + 1], "Dbscan") == 0)
            {
                iAlgType = 1;
                cout << "Fast DBSCAN... " << endl;
            }
            else if (strcmp(args[i + 1], "Optics") == 0)
            {
                iAlgType = 2;
                cout << "Fast OPTICS... " << endl;
            }
            else
            {
                cerr << "Error: The algorithm is not supported !" << endl;
                exit(1);
            }

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Cannot find the algorithm !" << endl;
        exit(1);
    }

    // Eps
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--eps") == 0)
        {
            PARAM_DBSCAN_EPS = atof(args[i + 1]);
            cout << "Radius eps: " << PARAM_DBSCAN_EPS << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Eps is missing !" << endl;
        exit(1);
    }

    // MinPts
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--minPts") == 0)
        {
            PARAM_DBSCAN_MINPTS = atoi(args[i + 1]);
            cout << "MinPts: " << PARAM_DBSCAN_MINPTS << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: MinPts is missing !" << endl;
        exit(1);
    }

    // Kernel embedding
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numEmbed") == 0)
        {
            PARAM_KERNEL_EMBED_D = atoi(args[i + 1]);
            cout << "Number of kernel embedded dimensions: " << PARAM_KERNEL_EMBED_D << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_KERNEL_EMBED_D = PARAM_DATA_D;
        cout << "Default number of kernel embedded dimensions = numDim = " << PARAM_KERNEL_EMBED_D << endl;
    }

    // Random projection
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--numProj") == 0)
        {
            PARAM_NUM_PROJECTION = atoi(args[i + 1]);
            cout << "Number of projections: " << PARAM_NUM_PROJECTION << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        int iTemp = ceil(log2(1.0 * PARAM_DATA_D));
        PARAM_NUM_PROJECTION = max(512, 1 << iTemp);
        cout << "Default number of projections: " << PARAM_NUM_PROJECTION << endl;
    }

    // Top-K close/far random vectors
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--topKProj") == 0)
        {
            PARAM_PROJECTION_TOP_K = atoi(args[i + 1]);
            cout << "Top-k closest/furthest vectors: " << PARAM_PROJECTION_TOP_K << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        PARAM_PROJECTION_TOP_K = 20;
        cout << "Default top-k closest/furthest vectors: " << PARAM_PROJECTION_TOP_K << endl;
    }

    // Distance measurement
    PARAM_DISTANCE = 2;
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--dist") == 0)
        {
            if (strcmp(args[i + 1], "Cosine") == 0)
            {
                PARAM_DISTANCE = 0;
                cout << "Cosine distance - no kernel embedding" << endl;
            }
            else if (strcmp(args[i + 1], "L1") == 0)
            {
                PARAM_DISTANCE = 1;
                cout << "L1 distance" << endl;
            }
            else if (strcmp(args[i + 1], "L2") == 0)
            {
                PARAM_DISTANCE = 2;
                cout << "L2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "Chi2") == 0)
            {
                PARAM_DISTANCE = 3;
                cout << "Chi2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "JS") == 0)
            {
                PARAM_DISTANCE = 4;
                cout << "Jensen-Shannon distance" << endl;
            }
            else
            {
                cout << "Default L2 distance" << endl;
                PARAM_DISTANCE = 2;
            }

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cout << "Default L2 distance" << endl;
        PARAM_DISTANCE = 2;
    }

    // Scale gamma of kernel L2 and L1
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--sigma") == 0)
        {
            PARAM_KERNEL_SIGMA = atof(args[i + 1]);
            cout << "Sigma: " << PARAM_KERNEL_SIGMA << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        if (PARAM_DISTANCE == 1)
        {
            PARAM_KERNEL_SIGMA = PARAM_DBSCAN_EPS;
            cout << "Default sigma = eps for L1: " << PARAM_KERNEL_SIGMA << endl;
        }
        else if (PARAM_DISTANCE == 2)
        {
            PARAM_KERNEL_SIGMA = 2 * PARAM_DBSCAN_EPS;
            cout << "Default sigma = 2*eps for L2: " << PARAM_KERNEL_SIGMA << endl;
        }
    }

    // Sampling ratio used on Chi2 and JS - TPAMI 12 (interval_sampling in scikit-learn
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--samplingRatio") == 0)
        {
            PARAM_KERNEL_INTERVAL_SAMPLING = atof(args[i + 1]);
            cout << "Sampling ratio for divergence distance: " << PARAM_KERNEL_INTERVAL_SAMPLING << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess && ((PARAM_DISTANCE == 3) || (PARAM_DISTANCE == 4)))
    {
        PARAM_KERNEL_INTERVAL_SAMPLING = 0.4;
        cout << "Default sampling ratio for divergence distance: " << PARAM_KERNEL_INTERVAL_SAMPLING << endl;
    }

    // Output
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--output") == 0)
        {
            PARAM_OUTPUT_FILE = args[i + 1];

            if (PARAM_OUTPUT_FILE.empty())
                PARAM_INTERNAL_SAVE_OUTPUT = false;
            else
                PARAM_INTERNAL_SAVE_OUTPUT = true;

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
        PARAM_INTERNAL_SAVE_OUTPUT = false;

    return iAlgType;
}

