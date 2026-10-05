cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /home/cuda12.4/bin/nvcc)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_example LANGUAGES CXX CUDA)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architecture: "native" builds for the GPU in this machine.
# Override from the command line, e.g. -DCMAKE_CUDA_ARCHITECTURES="75;86"


# Default to Release if no build type is given
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(main main.cu)

# Needed if you split kernels across multiple .cu files and call device
# functions between them
set_target_properties(cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_example PRIVATE CUDA::cudart)