/*
 * Assuming the input images are 100 x 100 pixels with 3 channels (as deduced from your reshaping size, 100 * 100 * 3), let's calculate the output size of each layer:

bot(): take round-down value

size conv(input_size, kernel, padding, stride):
    size = bot((input_size + padding * 2 - kernel) / stride + 1)

size pool(input_size, stride, padding)
    size = bot((input_size - kernel) / stride + 1)


 * Conv1 Output: With padding of 2, stride of 1, and a kernel size of 5, the output size remains bot((100 + 2 * 2 - 5) / 1 + 1) 100 x 100.
   Pool1: kernel = 2, stride = 2, no padding then pooled to bot((100 - 2) / 2 + 1) = 50 x 50
 
 * Conv2 Output: With padding of 2, stride of 2, and a kernel size of 5, the output size after convolution bot((50 + 2 * 2 - 5) / 2 + 1) = 25 x 25
   Pool2: kernel = 2, stride = 2, no padding then pooled to bot((25 - 2) / 2 + 1) 12 x 12
 
 * Conv3 Output: With padding of 2, stride of 1, and a kernel size of 5, Output size after convolution remains bot((12 + 2 * 2 - 5) / 1 + 1) 12 x 12.
   Pool3: kernel = 2, stride = 1, no padding then pooled to bot((12 - 2) / 1 + 1) 6 x 6
 
 * Conv4 Output: With padding of 2, stride of 2, and a kernel size of 5, Output size after convolution remains bot((6 - 2) / 2 + 1) 3 x 3.
   Pool4: kernel = 2, stride = 2, no padding then pooled to bot((3 - 2) / 2 + 1) 1 x 1
 
 * Since conv4 has 128 channels, the total number of features going into fc1 should be 128 * 1 * 1 = 128.
 * 

*/


#include <torch/torch.h>




#define DROPOUT_RATE 0.2
#define NUM_CLASSES 131
#define BATCH_SIZE 50




struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1, conv2, conv3, conv4;
    torch::nn::MaxPool2d pool1, pool2, pool3, pool4;
    //torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;
    torch::nn::Linear fc1, fc2, fc3;
    torch::nn::Dropout dropout;

    NetImpl(int num_classes = NUM_CLASSES, torch::Device device = torch::kCUDA)
        : // convolutional layers
          conv1(torch::nn::Conv2dOptions(3, 16, 5).padding(2).stride(1)),
          pool1(torch::nn::MaxPool2dOptions(2).stride(2)),
          conv2(torch::nn::Conv2dOptions(16, 32, 5).padding(2).stride(2)),
          pool2(torch::nn::MaxPool2dOptions(2).stride(2)),
          conv3(torch::nn::Conv2dOptions(32, 64, 5).padding(2).stride(1)),
          pool3(torch::nn::MaxPool2dOptions(2).stride(2)),
          conv4(torch::nn::Conv2dOptions(64, 128, 5).padding(2).stride(2)),
          pool4(torch::nn::MaxPool2dOptions(2).stride(2)),


          //bn1(16), bn2(32), bn3(64), bn4(128),
          fc1(128, 1024),
          fc2(1024, 256),
          fc3(256, NUM_CLASSES),
          dropout(DROPOUT_RATE)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);

        register_module("pool1", pool1);
        register_module("pool2", pool2);
        register_module("pool3", pool3);
        register_module("pool4", pool4);


        //register_module("bn1", bn1);
        //register_module("bn2", bn2);
        //register_module("bn3", bn3);
        //register_module("bn4", bn4);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("dropout", dropout);

        // Initialize weights
        torch::nn::init::kaiming_normal_(conv1->weight, 0.0, torch::kFanOut, torch::kReLU);
        torch::nn::init::kaiming_normal_(conv2->weight, 0.0, torch::kFanOut, torch::kReLU);
        torch::nn::init::kaiming_normal_(conv3->weight, 0.0, torch::kFanOut, torch::kReLU);
        torch::nn::init::kaiming_normal_(conv4->weight, 0.0, torch::kFanOut, torch::kReLU);
        // Correct usage of Kaiming initialization for a ReLU activation layer in LibTorch
        torch::nn::init::kaiming_uniform_(fc1->weight, 0.0, torch::kFanIn, torch::kReLU);
        torch::nn::init::kaiming_uniform_(fc2->weight, 0.0, torch::kFanIn, torch::kReLU);
        torch::nn::init::kaiming_uniform_(fc3->weight, 0.0, torch::kFanIn);



        // Move all parameters to the specified device
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(pool1->forward(conv1->forward(x)));
        x = torch::relu(pool2->forward(conv2->forward(x)));
        x = torch::relu(pool3->forward(conv3->forward(x)));
        x = torch::relu(pool4->forward(conv4->forward(x)));
    
        //if ((x.size(0) == BATCH_SIZE) == false) std::cout << "x.size(0): " << x.size(0) << std::endl;  // Ensure the batch size is as expected
        //assert(x.size(0) == BATCH_SIZE);
        x = x.reshape({x.size(0), -1});  // Flatten
        //std::cout << "Shape before fc1: " << x.sizes() << std::endl;

        //std::cout <<"shape after flatten: " << x.sizes() << std::endl;
        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);
        x = torch::relu(fc2->forward(x));
        x = dropout->forward(x);
        x = fc3->forward(x);

    return x;
}

};

TORCH_MODULE(Net); // This macro creates a module holder for NetImpl



