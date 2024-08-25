# Compiler
CXX := g++
NVCC := /usr/local/cuda-12/bin/nvcc

# Compiler flags
CXXFLAGS = -std=c++20

# Include and library paths for OpenCV
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc

# Include and libs paths for libtorch
LIBTORCH_PATH := /home/zirui/pytorch/torch
TORCH_INC_FLAGS := -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
TORCH_LD_FLAGS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -L/usr/local/cuda-12/lib64 -lcudart -lm

# Source files
SRCS = forward_prop.cc

# Object files
OBJS = $(SRCS:.cc=.o)

# Executable
TARGET := fp

# Default target
all: $(TARGET)

# Link the object files to create the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(TORCH_LD_FLAGS) $(OPENCV_LIBS)

# Compile the source files into object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(TORCH_INC_FLAGS) $(OPENCV_INCLUDE) -c $< -o $@

# RUN
run:
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
