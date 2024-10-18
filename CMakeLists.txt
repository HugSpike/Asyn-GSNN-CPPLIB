cmake_minimum_required(VERSION 3.20.2)
project(Spike)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER "/home/ff/gcc-9.5.0/install/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/ff/gcc-9.5.0/install/bin/g++")


# 手动指定 Python3 的路径
set(Python3_EXECUTABLE "/home/ff/anaconda3/envs/spike/bin/python3.9")
set(Python3_INCLUDE_DIR "/home/ff/anaconda3/envs/spike/include/python3.9")
set(Python3_LIBRARY "/home/ff/anaconda3/envs/spike/lib/libpython3.9.so")
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})


include_directories("/home/ff/files/libkineto/include")

# 设置cuda
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDAToolkit REQUIRED)
link_directories("/usr/local/cuda/lib64")

find_package(OpenMP)

include_directories("/home/ff/anaconda3/envs/spike/include/python3.9")

# 设置 LibTorch 路径
set(Torch_DIR "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories("/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/include")
include_directories("/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/include/torch/csrc/api/include")

include_directories("/home/ff/code/cpp_files_cpu")
# # 设置 pybind11 路径
# set(pybind11_DIR "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
# find_package(pybind11 REQUIRED)
# include_directories("/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/pybind11/include")


# pybind11_add_module(AsynchronousGraphSpiNet MODULE
#         AsynchronousConvLayer.cpp
#         AsynchronousGraphSpiNet.cpp
#         Util.cpp
#         SpikeGen.cpp
#         AsynchronousGraphSpiNet.h
#         AsynchronousConvLayer.h
#         MembranePotential.h
#         ThreadPool.h
#         Util.h
#         SpikeGen.h
#         Spike.h
# )
# target_link_libraries(
#         AsynchronousGraphSpiNet PRIVATE
#         ${TORCH_LIBRARIES}
#         Python3::Python
#         ${CUDA_CUDART_LIBRARY}
#         # pybind11::pybind11
#         ${OpenMP_CXX_FLAGS}
#         OpenMP::OpenMP_CXX
#         -pthread
# )
add_executable(test
        test2.cpp
        AsynchronousConvLayer.cpp
        AsynchronousGraphSpiNet.cpp
        Util.cpp
        SpikeGen.cpp
        AsynchronousGraphSpiNet.h
        AsynchronousConvLayer.h
        MembranePotential.h
        ThreadPool.h
        Util.h
        SpikeGen.h
        Spike.h
)
target_link_libraries(test PUBLIC "${TORCH_LIBRARIES}" Python3::Python -pthread OpenMP::OpenMP_CXX)
# 只为 test 可执行文件添加 -pg 标志
target_compile_options(test PRIVATE -pg)
target_link_options(test PRIVATE -pg)

