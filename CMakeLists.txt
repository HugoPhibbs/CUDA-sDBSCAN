cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)

# Add Google Test subdirectory
add_subdirectory(lib/googletest)

# Add Eigen include directory
include_directories(${PROJECT_SOURCE_DIR}/lib/eigen-3.4.0)

# Set the path to the CUDA toolkit. Replace this with your path if necessary.
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.3")

# Include CUDA
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# Include ArrayFire
set(ArrayFire_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/ArrayFire-3.9.0-Linux/share/ArrayFire/cmake)
find_package(ArrayFire REQUIRED)

if(ArrayFire_CUDA_FOUND)
    # Add definitions and sources for the CUDA backend
    add_definitions(-DAF_CUDA)
elseif(ArrayFire_OpenCL_FOUND)
    # Or for the OpenCL backend
    add_definitions(-DAF_OPENCL)
elseif(ArrayFire_CPU_FOUND)
    # Fallback to CPU backend if needed
    add_definitions(-DAF_CPU)
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
        src/fht/fht.c
)

# Define the test executable
add_executable(run_gs_dbscan_tests
        test/Test.cpp
        test/gsDBSCANTest.cpp
        src/dbscan/gsDBSCAN.cpp  # Assuming test depends on these source files
        src/fht/fht.c            # Include other dependencies as needed
)

# Link Google Test libraries to the test executable
target_link_libraries(run_gs_dbscan_tests gtest gtest_main)

# Link Arrayfire
target_link_libraries(${PROJECT_NAME} ArrayFire::af)
