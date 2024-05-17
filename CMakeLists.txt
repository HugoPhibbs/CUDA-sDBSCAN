cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)

# Add Google Test subdirectory
add_subdirectory(lib/googletest)

# Add Eigen include directory
include_directories(${PROJECT_SOURCE_DIR}/lib/eigen-3.4.0)

# Set the path to the CUDA toolkit. Replace this with your path if necessary.
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.3")
set(CUDA_TOOLKIT_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_TOOLKIT_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

# Include CUDA
include_directories(${CUDA_TOOLKIT_INCLUDE_DIR})

# Link CUDA explicitly
link_directories(${CUDA_TOOLKIT_LIB_DIR})
set(CUDA_LIBS
        ${CUDA_TOOLKIT_LIB_DIR}/libcudart.so
        ${CUDA_TOOLKIT_LIB_DIR}/libcublas.so
        ${CUDA_TOOLKIT_LIB_DIR}/libcusolver.so
)

# Include ArrayFire
set(ArrayFire_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/ArrayFire-3.9.0-Linux/share/ArrayFire/cmake)
find_package(ArrayFire REQUIRED)
include_directories(${ArrayFire_INCLUDE_DIRS})

if(ArrayFire_CUDA_FOUND)
    add_definitions(-DAF_CUDA)
    list(APPEND CUDA_LIBS ${ArrayFire_LIBRARIES})
elseif(ArrayFire_OpenCL_FOUND)
    add_definitions(-DAF_OPENCL)
    list(APPEND CUDA_LIBS ${ArrayFire_LIBRARIES})
elseif(ArrayFire_CPU_FOUND)
    add_definitions(-DAF_CPU)
    list(APPEND CUDA_LIBS ${ArrayFire_LIBRARIES})
endif()

# Include Google Test in the include path
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Define sources for the main project executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/fast_copy.c
        src/dbscan/sngDBSCAN.cpp
        include/sngDBSCAN.h
        src/dbscan/DBSCAN.cpp
        src/dbscan/sDBSCAN.cpp
        include/sDBSCAN.h
        src/dbscan/uDBSCAN.cpp
        include/uDBSCAN.h
        src/dbscan/gsDBSCAN.cpp
        include/gsDBSCAN.h
        src/dbscan/preprocessing.cpp
        include/preprocessing.h
        src/fht/fht.c
)

# Define the test executable
add_executable(run_gs_dbscan_tests
        test/gsDBSCANTest.cpp
        src/dbscan/gsDBSCAN.cpp  # Assuming test depends on these source files
)

# Link Google Test libraries to the test executable
target_link_libraries(run_gs_dbscan_tests gtest gtest_main)

# Link CUDA and ArrayFire libraries explicitly
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBS} ${ArrayFire_LIBRARIES})
target_link_libraries(run_gs_dbscan_tests PRIVATE ${CUDA_LIBS} ${ArrayFire_LIBRARIES})