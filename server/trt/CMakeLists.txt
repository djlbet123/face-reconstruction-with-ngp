cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
project(rvm)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/root/TensorRT/include/)
link_directories(/root/TensorRT/lib/)

include_directories(${PROJECT_SOURCE_DIR}/include/)
AUX_SOURCE_DIRECTORY(src  SRC)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rvm ${SRC})
target_link_libraries(rvm nvinfer)
target_link_libraries(rvm cudart)
target_link_libraries(rvm ${OpenCV_LIBS})