//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_UTILS_H
#define SDBSCAN_UTILS_H

#include <matx.h>
#include <Eigen/Dense>
#include <arrayfire.h>
#include <af/cuda.h>

/*
 * This file contains util functions that don't belong in a single file
 */

namespace GsDBSCAN {
    template<typename eigenType, typename matxType>
    inline static matx::tensor_t<matxType , 2> eigenMatToMatXTensor(Eigen::Matrix<eigenType, Eigen::Dynamic, Eigen::Dynamic> &matEigen, matx::matxMemorySpace_t matXMemorySpace = matx::MATX_MANAGED_MEMORY) {
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
    inline static T* cudaDeviceArrayToHostArray(const T* deviceArray, size_t numElements) {
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
    inline static matx::tensor_t<matXType, 2> afArrayToMatXTensor(af::array &afArray, matx::matxMemorySpace_t matXMemorySpace = matx::MATX_MANAGED_MEMORY) {
        // For simplicity, this only does 2D tensors

        afType *afData = afArray.device<afType>();

        auto matxTensor = matx::make_tensor<matXType>(afData, {afArray.dims()[0], afArray.dims()[1]}, matXMemorySpace);

        return matxTensor;
    }

    template <typename T>
    inline static T* hostToManagedArray(const T* hostData, size_t numElements) {
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

    inline void printCudaMemoryUsage() {
        size_t free_mem, total_mem;
        cudaError_t error;

        cudaDeviceSynchronize();

        // Get memory information
        error = cudaMemGetInfo(&free_mem, &total_mem);
        if (error != cudaSuccess) {
            std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }

        // Convert bytes to gigabytes
        double free_mem_gb = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
        double total_mem_gb = static_cast<double>(total_mem) / (1024.0 * 1024.0 * 1024.0);
        double used_mem_gb = total_mem_gb - free_mem_gb;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Memory Usage: " << used_mem_gb << " GB used, "
                  << free_mem_gb << " GB free, " << total_mem_gb << " GB total" << std::endl;
    }

    /**
     * Gets the CUDA stream from ArrayFire
     *
     * For easy testing of other functions
     *
     * See https://arrayfire.org/docs/interop_cuda.htm for info on this
     *
     * @return the CUDA stream
     */
    inline cudaStream_t getAfCudaStream() {
        int afId = af::getDevice();
        int cudaId = afcu::getNativeId(afId);
        return afcu::getStream(cudaId);
    }
}

#endif //SDBSCAN_UTILS_H
