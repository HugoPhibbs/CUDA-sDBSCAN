//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include "Header.h"
#include "Utilities.h"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <af/cuda.h>
#include <arrayfire.h>
#include <matx.h>
#include <Eigen/Dense>

class GsDBSCAN {
private :
    af::array X;
    int minPts;
    int k;
    int m;
    float eps;
    bool skip_pre_checks;
    int n;
    int d;
    int D;
    float sigma;
    int seed;
    string distanceMetric;
    bool clusterNoise;
    int fhtDim;
    float batchAlpha;
    int nRotate;

    boost::dynamic_bitset<> bitHD3;

public:
    // Constructor if needed
    GsDBSCAN(const af::array &X, int D, int minPts, int k, int m, float eps, bool skip_pre_checks, float sigma, int seed, string distanceMetric, float batchAlpha, int fhtDim, int nRotate, bool clusterNoise);

    // Methods corresponding to the functions
    void performGsDbscan();

    void static preChecks(af::array &X, int D, int minPts, int k, int m, float eps);

    std::tuple<af::array, af::array> static preProcessing(af::array &X, int D, int k, int m);

    af::array static randomProjections(af::array &X, boost::dynamic_bitset<> bitHD3, int D, int k, int m, string distanceMetric, float sigma, int seed, int fhtDim, int nRotate);

    std::tuple<af::array, af::array> static constructABMatrices(const af::array& projections, int k, int m);

    af::array static findDistances(af::array &X, af::array &A, af::array &B, float alpha = 1.2);

    matx::tensor_t<matx::matxFp16, 2> static findDistancesMatX(matx::tensor_t<matx::matxFp16, 2> &X_t, matx::tensor_t<int32_t, 2> &A_t, matx::tensor_t<int32_t, 2> &B_t, float alpha = 1.2, int batchSize=-1);

    int static findDistanceBatchSize(float alpha, int n, int d, int k, int m);

    static af::array assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps, int blockSize=1024);

    af::array static constructQueryVectorDegreeArray(af::array &distances, float eps);

    af::array static processQueryVectorDegreeArray(af::array &E);

    tuple<vector<int>, int> static formClusters(af::array &adjacencyList, af::array &V, af::array &E, int n, int minPts, bool clusterNoise);

    template<typename eigenType, typename matxType>
    static matx::tensor_t<matxType , 2> eigenMatToMatXTensor(Matrix<eigenType, Eigen::Dynamic, Eigen::Dynamic> &matEigen, matx::matxMemorySpace_t matXMemorySpace = matx::MATX_MANAGED_MEMORY) {
        eigenType *eigenData = matEigen.data();
        int numElements = matEigen.rows() * matEigen.cols();

        matxType* deviceArray;
        size_t size = sizeof(eigenType) * numElements;

        cudaError_t err;

        if (matXMemorySpace == matx::MATX_MANAGED_MEMORY) {
            err = cudaMallocManaged(&deviceArray, size);
        } else {
            // Assume device memory, yes isn't all that robust, but for our use cases, should be ok.
            err = cudaMalloc(&deviceArray, size);
        }

        if (err != cudaSuccess) {
            std::cerr << "Error allocating memory: " << cudaGetErrorString(err) << std::endl;
            return matx::tensor_t<matxType, 2>();  // Empty tensor
        }

        err = cudaMemcpy(deviceArray, eigenData, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            std::cerr << "Error copying data to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(deviceArray);
            return matx::tensor_t<matxType, 2>(); // Empty tensor
        }

        auto tensor = matx::make_tensor<matxType>(deviceArray, {matEigen.rows(), matEigen.cols()}, matXMemorySpace);

        return tensor;
    }

    template<typename T>
    static T* cudaDeviceArrayToHostArray(const T* deviceArray, size_t numElements) {
        T* hostArray = new T[numElements];

        cudaError_t err = cudaMemcpy(hostArray, deviceArray, numElements * sizeof(T), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            std::cerr << "Error copying data from device to host: " << cudaGetErrorString(err) << std::endl;
            delete[] hostArray;
            return nullptr;
        }

        return hostArray;
    }

    template<typename afType, typename matXType>
    static matx::tensor_t<matXType, 2> afArrayToMatXTensor(af::array &afArray, matx::matxMemorySpace_t matXMemorySpace = matx::MATX_MANAGED_MEMORY) {
        // For simplicity, this only does 2D tensors

        afType *afData = afArray.device<afType>();

        auto matxTensor = matx::make_tensor<matXType>(afData, {afArray.dims()[0], afArray.dims()[1]}, matXMemorySpace);

        return matxTensor;
    }

    template <typename T>
    static T* hostToManagedArray(const T* hostData, size_t numElements) {
        T* managedArray;
        size_t size = numElements * sizeof(T);

        // Allocate managed memory
        cudaError_t err = cudaMallocManaged(&managedArray, size);
        if (err != cudaSuccess) {
            std::cerr << "Error allocating managed memory: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }

        // Copy data from host to managed memory
        err = cudaMemcpy(managedArray, hostData, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Error copying data to managed memory: " << cudaGetErrorString(err) << std::endl;
            cudaFree(managedArray);
            return nullptr;
        }

        return managedArray;
    }

    cudaStream_t static getAfCudaStream();

    // Destructor if needed
    ~GsDBSCAN() = default;
};

#endif //DBSCANCEOS_GSDBSCAN_H
