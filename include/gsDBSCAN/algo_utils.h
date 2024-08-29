//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_ALGO_UTILS_H
#define SDBSCAN_ALGO_UTILS_H

#include <matx.h>
#include <arrayfire.h>
#include <af/cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <execinfo.h>
#include "../../include/rapidcsv.h"

/*
 * This file contains util functions that don't belong in a single file
 */


namespace GsDBSCAN::algo_utils {

    // Yes I shamelessly copied these from TestUtils

    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

    inline Time timeNow() {
        return std::chrono::high_resolution_clock::now();
    }


    inline int duration(Time start, Time stop) {
        return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }

    inline int durationSecs(Time start, Time stop) {
        return std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
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

    inline void printStackTrace() {
        void *array[10];
        size_t size;
        size = backtrace(array, 10);

        char **symbols = backtrace_symbols(array, size);
        std::cerr << "Stack trace:" << std::endl;
        for (size_t i = 0; i < size; i++) {
            std::cerr << symbols[i] << std::endl;
        }
        free(symbols);
    }

    inline void throwCudaError(const std::string &msg, cudaError_t err) {
        std::cout<<"An error occurred"<<std::endl;
        printStackTrace();
        std::cout<<"\n"<<std::endl;
        throw std::runtime_error(msg + ": " + std::string(cudaGetErrorString(err)));
    }

    template<typename T>
    inline T *copyHostToDevice(T *hostData, const size_t numElements, bool managedMemory = false) {
        T *deviceArray;
        size_t size = numElements * sizeof(T);

        cudaError_t err;

        if (managedMemory) {
            err = cudaMallocManaged(&deviceArray, size);
        } else {
            err = cudaMalloc(&deviceArray, size);
        }

        if (err != cudaSuccess) {
            throwCudaError("Error copying memory from device to host", err);;
        }

        err = cudaMemcpy(deviceArray, hostData, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            cudaFree(deviceArray);
            throwCudaError("Error copying memory from host to device", err);
        }

        return deviceArray;
    }

    template<typename T>
    inline T *copyDeviceToHost(T *deviceArray, const size_t numElements, cudaStream_t stream = nullptr) {
        cudaError_t err;

        T *hostArray = new T[numElements];

        if (stream != nullptr) {
            err = cudaMemcpyAsync(hostArray, deviceArray, numElements * sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else {
            err = cudaMemcpy(hostArray, deviceArray, numElements * sizeof(T), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();  // Synchronize the entire device if no stream is provided
        }

        // Check for errors
        if (err != cudaSuccess) {
            cudaFree(deviceArray);  // Free the device memory to prevent leaks
            delete[] hostArray;
            throwCudaError("Error copying memory from device to host", err);
        }

        return hostArray;  // Return true if successful
    }

    inline void printCudaMemoryUsage() {
        size_t free_mem, total_mem;
        cudaError_t error;

        cudaDeviceSynchronize();

        // Get memory information
        error = cudaMemGetInfo(&free_mem, &total_mem);
        if (error != cudaSuccess) {
            throw std::runtime_error("cudaMemGetInfo failed: " + std::string(cudaGetErrorString(error)));
        }

        // Convert bytes to gigabytes
        double free_mem_gb = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
        double total_mem_gb = static_cast<double>(total_mem) / (1024.0 * 1024.0 * 1024.0);
        double used_mem_gb = total_mem_gb - free_mem_gb;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Memory Usage: " << used_mem_gb << " GB used, "
                  << free_mem_gb << " GB free, " << total_mem_gb << " GB total" << std::endl;
    }

    template<typename T>
    __global__ void colMajorToRowArrayKernel(T *colMajorArray, T *rowMajorArray) {
        /*
         * Launch kernel with one block per row of the matrix
         * Each block has one thread per entry along the rows of the matrix
         *
         * Therefore, the number of blocks is numRows and the number of threads per block is numCols
         */
        int rowMajorIdx = blockDim.x * blockIdx.x + threadIdx.x;
        int colMajorIdx = gridDim.x * (rowMajorIdx % blockDim.x) + rowMajorIdx / blockDim.x;
        rowMajorArray[rowMajorIdx] = colMajorArray[colMajorIdx];
    }

    template<typename T>
    inline T *colMajorToRowMajorMat(T *colMajorMat, size_t numRows, size_t numCols, cudaStream_t stream = nullptr) {
        // We assume that colMajorMat is on the GPU
        T *rowMajorMat;
        size_t size = numRows * numCols * sizeof(T);

        cudaError_t err;
        if (stream != nullptr) {
            err = cudaMallocAsync(&rowMajorMat, size, stream);
        } else {
            err = cudaMalloc(&rowMajorMat, size);
        }

        if (err != cudaSuccess) {
            throwCudaError("Error allocating memory", err);
        }

        if (stream != nullptr) {
            colMajorToRowArrayKernel<<<numRows, numCols, 0, stream>>>(colMajorMat, rowMajorMat);
            cudaStreamSynchronize(stream);
        } else {
            colMajorToRowArrayKernel<<<numRows, numCols>>>(colMajorMat, rowMajorMat);
            cudaDeviceSynchronize();
        }

        return rowMajorMat;
    }

    template<typename afType, typename matXType>
    inline static matx::tensor_t<matXType, (int) 2>
    afMatToMatXTensor(af::array &afArray, matx::matxMemorySpace_t matXMemorySpace = matx::MATX_MANAGED_MEMORY) {
        afType *afColMajorArray = afArray.device<afType>(); // In col major

        int rows = afArray.dims()[0];
        int cols = afArray.dims()[1];

//        auto *afRowMajorArray = colMajorToRowMajorMat(afColMajorArray, rows, cols, getAfCudaStream());
        matXType *afRowMajorArray = colMajorToRowMajorMat<afType>(afColMajorArray, rows, cols);

        afArray.unlock();

        auto matxTensor = matx::make_tensor<matXType>(afRowMajorArray, {rows, cols}, matXMemorySpace);

        return matxTensor;
    }

    template<typename T, int nDims>
    inline T *matxTensorToHost(matx::tensor_t<T, nDims> tensor, int numElements, cudaStream_t stream = nullptr) {
        // Honestly can't figure out how to use this function. It complains about the type of nDims.
        T *matxTensor_d = tensor.Data();
        return copyDeviceToHost(matxTensor_d, numElements, stream);
    }

    template<typename T>
    inline T *allocateCudaArray(size_t length, bool managedMemory = false, bool fill = true, T fillValue = 0) {
        T *array;
        size_t size = sizeof(T) * length;

        // Allocate memory on the device
        cudaError_t err;

        if (managedMemory) {
            err = cudaMallocManaged(&array, size);
        } else {
            err = cudaMalloc(&array, size);
        }

        if (err != cudaSuccess) {
            throwCudaError("Error allocating memory for array", err);
        }

        if (fill) {
            // Zero out the memory
            err = cudaMemset(array, fillValue, size);  // Use array (not &array) for cudaMemset

            if (err != cudaSuccess) {
                cudaFree(array);  // Free the memory if memset fails
                throwCudaError("Error setting memory for array", err);
            }
        }

        return array;
    }

    template <typename T>
    inline T valueAtIdxDeviceToHost(const T* deviceArray, const int idx) {
        T value;
        cudaMemcpy(&value, deviceArray + idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }
}


#endif //SDBSCAN_ALGO_UTILS_H
