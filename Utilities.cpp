#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream
#include <sstream>

/**
Generate dynamic bitset for HD3HD2HD1
We need to generate two bitsets for 2 LSH functions
**/
void bitHD3Generator(int p_iNumBit, boost::dynamic_bitset<> & bitHD)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD[d] = unifDist(generator) & 1;
    }

}

/**
Generate random N(0, 1) from a normal distribution using C++ function

Input:
- p_iNumRows x p_iNumCols: Row x Col

Output:
- MATRIX_G: vector contains normal random variables

**/
MatrixXf gaussGenerator(int p_iNumRows, int p_iNumCols, float mean, float stddev)
{
    MatrixXf MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<float> normDist(mean, stddev);

//    MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_G(r, c) = normDist(generator);

    return MATRIX_G;
}

MatrixXf cauchyGenerator(int p_iNumRows, int p_iNumCols, float x0, float gamma)
{
    MatrixXf MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    cauchy_distribution<float> cauchyDist(x0, gamma); // {x0 /* a */, 𝛾 /* b */}

//    MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_C(r, c) = cauchyDist(generator);

    return MATRIX_C;
}

void outputDbscan(const IVector & p_Labels, string p_sOutputFile)
{
//	cout << "Outputing File..." << endl;
    ofstream myfile(p_sOutputFile);

    //cout << p_matKNN << endl;

    for (auto const& i : p_Labels)
    {
        myfile << i << '\n';
    }

    myfile.close();
//	cout << "Done" << endl;
}

void outputOptics(const IVector & p_vecOrder, const FVector & p_vecDist, const IVector & p_vecLabels, string p_sOutputFile)
{
//	cout << "Outputing File..." << endl;

    ofstream myfile(p_sOutputFile);


    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        myfile << p_vecOrder[n] << " " << p_vecDist[n] << " " << p_vecLabels[n] << '\n';
    }

    myfile.close();
//	cout << "Done" << endl;
}

void embedChiSquare(const Ref<VectorXf> p_vecPoint, Ref<VectorXf> p_vecEmbed)
{
    int iComponent = (PARAM_KERNEL_EMBED_D / PARAM_DATA_D) - 1; // kappa_1, kappa_2, ...
    iComponent /= 2; // since we take cos and sin

//    cout << "Number of components: " << iComponent << endl;

    // adding sqrt(x L kappa(0)
    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        // Only deal with non zero
        if (p_vecPoint[d] > 0)
            p_vecEmbed[d] = sqrt(p_vecPoint[d] * PARAM_KERNEL_INTERVAL_SAMPLING);
    }

    // adding other component
    for (int i = 1; i <= iComponent; ++i)
    {
        // We need the first D for kappa_0, 2D for kappa_1, 2D for kappa_2, ...
        int iBaseIndex = PARAM_DATA_D + (i - 1) * 2 * PARAM_DATA_D;

        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            if (p_vecPoint[d] > 0)
            {
                float fFactor = sqrt(2 * p_vecPoint[d] * PARAM_KERNEL_INTERVAL_SAMPLING / cosh(PI * i * PARAM_KERNEL_INTERVAL_SAMPLING));

                p_vecEmbed[iBaseIndex + d] = fFactor * cos(i * PARAM_KERNEL_INTERVAL_SAMPLING * log(p_vecPoint[d]));
                p_vecEmbed[iBaseIndex + PARAM_DATA_D + d] = fFactor * sin(i * PARAM_KERNEL_INTERVAL_SAMPLING * log(p_vecPoint[d]));
            }
        }
    }
}

void embedJS(const Ref<VectorXf> p_vecPoint, Ref<VectorXf> p_vecEmbed)
{
    int iComponent = (PARAM_KERNEL_EMBED_D / PARAM_DATA_D) - 1; // kappa_1, kappa_2, ...
    iComponent /= 2; // since we take cos and sin

    // adding sqrt(x L kappa(0)
    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        // Only deal with non zero
        if (p_vecPoint[d] > 0)
            p_vecEmbed[d] = sqrt(p_vecPoint[d] * PARAM_KERNEL_INTERVAL_SAMPLING * 2 / log(4));
    }

    // adding other component
    for (int i = 1; i <= iComponent; ++i)
    {
        // We need the first D for kappa_0, 2D for kappa_1, 2D for kappa_2, ...
        int iBaseIndex = PARAM_DATA_D + (i - 1) * 2 * PARAM_DATA_D;

        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            if (p_vecPoint[d] > 0)
            {
                // this is kappa(jL)
                float fFactor = 2 / (log(4) * (1 + 4 * (i * PARAM_KERNEL_INTERVAL_SAMPLING) * (i * PARAM_KERNEL_INTERVAL_SAMPLING)) * cosh(PI * i * PARAM_KERNEL_INTERVAL_SAMPLING));

                // This is sqrt(2X hkappa)
                fFactor = sqrt(2 * p_vecPoint[d] * PARAM_KERNEL_INTERVAL_SAMPLING * fFactor);

                p_vecEmbed[iBaseIndex + d] = fFactor * cos(i * PARAM_KERNEL_INTERVAL_SAMPLING * log(p_vecPoint[d]));
                p_vecEmbed[iBaseIndex + PARAM_DATA_D + d] = fFactor * sin(i * PARAM_KERNEL_INTERVAL_SAMPLING * log(p_vecPoint[d]));
            }
        }
    }
}

/** Useful for dense vector
**/
float computeDist(const Ref<VectorXf> p_vecX, const Ref<VectorXf> p_vecY)
{
    if (PARAM_DISTANCE == 0)
        return 1 - p_vecX.dot(p_vecY);
    if (PARAM_DISTANCE == 1)
        return (p_vecX - p_vecY).cwiseAbs().sum();
    else if (PARAM_DISTANCE == 2)
        return (p_vecX - p_vecY).norm();
    else if (PARAM_DISTANCE == 3) // ChiSquare
    {
        // hack for vectorize to ensure no zero element
        VectorXf vecX = p_vecX;
        VectorXf vecY = p_vecY;

        vecX.array() += EPSILON;
        vecY.array() += EPSILON;

        VectorXf temp = vecX.cwiseProduct(vecY); // x * y
        temp = temp.cwiseQuotient(vecX + vecY); // (x * y) / (x + y)
        temp.array() *= 2.0; // 2(x * y) / (x + y)

        return 1.0 - temp.sum();
    }

    else if (PARAM_DISTANCE == 4) // Jensen Shannon
    {
        // hack for vectorize
        VectorXf vecX = p_vecX;
        VectorXf vecY = p_vecY;

        vecX.array() += EPSILON;
        vecY.array() += EPSILON;

        VectorXf vecTemp1 = (vecX + vecY).cwiseQuotient(vecX); // (x + y) / x
        vecTemp1 = vecTemp1.array().log() / log(2.0); // log2( (x+y) / x))
        vecTemp1 = vecTemp1.cwiseProduct(vecX); // x * log2( (x+y) / x))

//        cout << vecTemp1.sum() / 2 << endl;

        VectorXf vecTemp2 = (vecX + vecY).cwiseQuotient(vecY);
        vecTemp2 = vecTemp2.array().log() / log(2.0);
        vecTemp2 = vecTemp2.cwiseProduct(vecY);

//        cout << vecTemp2.sum() / 2 << endl;

        return 1.0 - (vecTemp1 + vecTemp2).sum() / 2.0;
    }
    else
    {
        cout << "Error: The distance is not support" << endl;
        return 0;
    }
}

/** Faster with sparse representation
**/
float computeChi2(const Ref<VectorXf> vecX, const Ref<VectorXf> vecY)
{
    float dist = 0.0;
    for (int d = 0; d < vecX.size(); ++d)
    {
        if ((vecX(d) > 0) && (vecY(d) > 0))
            dist += 2 * vecX(d) * vecY(d) / (vecX(d) + vecY(d));
    }

    return 1.0 - dist;

}
