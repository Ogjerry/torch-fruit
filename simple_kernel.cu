#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <NvInferRuntimePlugin.h>
#include <cudnn.h>


using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} logger;

int main() {
    // Create builder
    IBuilder* builder = createInferBuilder(logger);
    IBuilderConfig* config = builder -> createBuilderConfig();
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return -1;
    }

    // Create network
    INetworkDefinition* network = builder->createNetworkV2();
    if (!network) {
        std::cerr << "Failed to create TensorRT network" << std::endl;
        delete network;
        builder->reset();
        return -1;
    }
    // Add input tensor
    ITensor* inputTensor = network->addInput("input", DataType::kFLOAT, Dims3{3, 224, 224});
    if (!inputTensor) {
        std::cerr << "Failed to create input tensor" << std::endl;
        delete network;
        builder->reset();
        return -1;
    }

    // Example: Add a convolutional layer
    Weights conv1Weights{DataType::kFLOAT, nullptr, 64 * 3 * 3 * 3};
    Weights conv1Bias{DataType::kFLOAT, nullptr, 64};
    IConvolutionLayer* conv1 = network ->addConvolutionNd(*inputTensor, 64, DimsHW{3, 3}, conv1Weights, conv1Bias);
    if (!conv1) {
        std::cerr << "Failed to create convolution layer" << std::endl;
        delete network;
        builder->reset();
        return -1;
    }
    conv1->setStrideNd(DimsHW{1, 1});

    // Mark output tensor
    ITensor* outputTensor = conv1->getOutput(0);
    network->markOutput(*outputTensor);

    // Build the engine
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);  // 1 MB
    ICudaEngine* engine = builder ->(*network);
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        delete network;
        builder->reset();
        return -1;
    }

    // Clean up the network as it is no longer needed
    delete network;
}