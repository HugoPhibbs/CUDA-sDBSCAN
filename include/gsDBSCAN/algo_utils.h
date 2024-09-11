//
// Created by hphi344 on 9/08/24.
//

#ifndef SDBSCAN_ALGO_UTILS_H
#define SDBSCAN_ALGO_UTILS_H

#include <tuple>
#include <execinfo.h>
#include "../pch.h"

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
        std::cout << "An error occurred" << std::endl;
        printStackTrace();
        std::cout << "\n" << std::endl;
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
        // We assume that colMajorMat is on the GPU, and that numCols<<numRows. I.e. dataset is stored in column major format
        T *rowMajorMat = allocateCudaArray<T>(numRows * numCols, false, false);

        if (stream != nullptr) {
            colMajorToRowArrayKernel<<<numRows, numCols, 0, stream>>>(colMajorMat, rowMajorMat);
            cudaStreamSynchronize(stream);
        } else {
            colMajorToRowArrayKernel<<<numRows, numCols>>>(colMajorMat, rowMajorMat);
            cudaDeviceSynchronize();
        }

        return rowMajorMat;
    }

    template<typename T>
    __global__ void rowMajorToColArrayKernel(T *colMajorArray, T *rowMajorArray) {
        /*
         * Launch kernel with one block per row of the matrix
         * Each block has one thread per entry along the rows of the matrix
         *
         * Therefore, the number of blocks is numRows and the number of threads per block is numCols
         */
        int rowMajorIdx = blockDim.x * blockIdx.x + threadIdx.x;
        int colMajorIdx = gridDim.x * (rowMajorIdx % blockDim.x) + rowMajorIdx / blockDim.x;
        colMajorArray[colMajorIdx] = rowMajorArray[rowMajorIdx];
    }

    template<typename T>
    inline T *rowMajorToColMajorMat(T *rowMajorMat, size_t numRows, size_t numCols, cudaStream_t stream = nullptr) {
        // We assume that colMajorMat is on the GPU, and that numCols<<numRows. I.e. dataset is stored in column major format
        T *colMajorMat = allocateCudaArray<T>(numRows * numCols, false, false);

        if (stream != nullptr) {
            rowMajorToColArrayKernel<<<numRows, numCols, 0, stream>>>(colMajorMat, rowMajorMat);
            cudaStreamSynchronize(stream);
        } else {
            rowMajorToColArrayKernel<<<numRows, numCols>>>(colMajorMat, rowMajorMat);
            cudaDeviceSynchronize();
        }

        return colMajorMat;
    }

    template<typename T>
    inline T valueAtIdxDeviceToHost(const T *deviceArray, const int idx) {
        T value;
        cudaMemcpy(&value, deviceArray + idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    template<typename ArrayType, typename torch::Dtype TorchType>
    inline torch::Tensor torchTensorFromDeviceArray(ArrayType *array, int rows, int cols) {
        auto options = torch::TensorOptions().dtype(TorchType).device(torch::kCUDA);
        torch::Tensor tensor = torch::from_blob(array, {rows, cols}, options);
        return tensor;
    }

    template<typename T>
    inline auto torchTensorToMatX(torch::Tensor tensor) {
        int rows = tensor.size(0);
        int cols = tensor.size(1);

        // TODO not sure what to do here if i have a Torch tensor in f16 and want to convert it to a matx f16
        return matx::make_tensor<T>(tensor.data_ptr<T>(), {rows, cols},
                                    matx::MATX_DEVICE_MEMORY);
    }
}


#endif //SDBSCAN_ALGO_UTILS_H
