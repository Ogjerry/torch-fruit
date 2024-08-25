cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cam)

# Specify the path to the Torch installation
set(CMAKE_PREFIX_PATH "/home/zirui/pytorch/torch")  # Adjust to your actual libtorch installation path


include_directories(/usr/local/cuda-12.1/include)
link_directories(/usr/local/cuda-12.1/lib64 /usr/local/cuda-12.1/targets/x86_64-linux/lib)


# Find the required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# Set the C++ flags for PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Manually specify the CUDA architecture (Compute Capability 5.0 for GeForce 940MX)
set(CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_ARCHITECTURES "86")
set(TCNN_CUDA_ARCHITECTURES "86")


set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/local/cuda-12/bin/nvcc)


# Add executable target
add_executable(cam cam.cc)
add_executable(mv make_validation.cc)
add_executable(fp forward_prop.cc utils.hpp Net.hpp)


# Link libraries
target_link_libraries(cam "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_link_libraries(mv "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_link_libraries(fp "${TORCH_LIBRARIES}" ${OpenCV_LIBS} OpenMP::OpenMP_CXX)



# Ensure that CMake knows where to find the headers and libraries
target_include_directories(cam PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(mv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(fp PRIVATE ${OpenCV_INCLUDE_DIRS})


# Set the rpath to find the shared libraries
set_target_properties(cam PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
set_target_properties(mv PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
set_target_properties(fp PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")


# Add CUDA flags for specified architectures
if(CUDA_ARCHITECTURES)
  set_property(TARGET cam PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
  set_property(TARGET mv PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
  set_property(TARGET fp PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endif()