#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <iostream>
#include <random>
#include <omp.h>
#include <cmath>

namespace utils {



// Function to initialize tensor with truncated normal distribution
torch::Tensor trunc_normal_(torch::Tensor tensor, double mean, double std, torch::Device device) {
    if (tensor.device().type() == torch::kCUDA) {
        std::cerr << "trunc_normal_ function expects tensor to be on CPU for safety." << std::endl;
        return tensor;
    }

    if (tensor.scalar_type() != torch::kFloat) {
        std::cerr << "trunc_normal_ function expects float tensor." << std::endl;
        return tensor;
    }

    tensor = tensor.contiguous();
    auto data_ptr = tensor.data_ptr<float>();
    auto num_elements = tensor.numel();

    #pragma omp parallel
    {
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<double> distribution(mean, std);
        
        #pragma omp for
        for (int i = 0; i < num_elements; i++) {
            double val;
            int max_tries = 100;
            int tries = 0;
            do {
                val = distribution(generator);
                tries++;
            } while ((val < mean - 2 * std || val > mean + 2 * std) && tries < max_tries);

            if (tries == max_tries) {
                val = mean; // Fallback to mean if no suitable value is found
            }

            data_ptr[i] = static_cast<float>(val);
        }
    }

    return tensor.to(device);
}






// Function to create a convolutional layer with initialization
torch::Tensor conv(torch::Tensor input, 
                   const std::string& name, 
                   int kernel_width, 
                   int kernel_height, 
                   int num_out_activation_maps, 
                   int stride_horizontal = 1, 
                   int stride_vertical = 1, 
                   std::function<torch::Tensor(const torch::Tensor&)> activation_fn = torch::relu,
                   torch::Device device = torch::kCUDA) 
{
    int prev_layer_output = input.size(1); // number of channels

    auto options = torch::nn::Conv2dOptions(prev_layer_output, num_out_activation_maps, {kernel_height, kernel_width})
                    .stride({stride_vertical, stride_horizontal})
                    .padding({(kernel_height - 1) / 2, (kernel_width - 1) / 2})
                    .bias(true);
    
    torch::nn::Conv2d conv_layer(options);
    conv_layer->to(torch::kCPU); // Initialize on CPU
    trunc_normal_(conv_layer->weight, 0, 0.05, torch::kCPU);
    torch::nn::init::constant_(conv_layer->bias, 0.0);
    conv_layer->to(device); // Move to GPU after all initializations

    auto conv_res = conv_layer->forward(input.to(device));
    return activation_fn(conv_res);
}









// Function to create a fully connected layer with initialization
torch::Tensor full_connected(torch::Tensor input, 
                             const std::string& name,
                             int output_neurons, 
                             std::function<torch::Tensor(const torch::Tensor&)> activation_fn = torch::relu,
                             torch::Device device = torch::kCUDA) 
{
    int n_in = input.size(1);
    auto options = torch::nn::LinearOptions(n_in, output_neurons).bias(true);
    
    auto fc_layer = torch::nn::Linear(options);
    fc_layer->to(torch::kCPU);  // Initialize on CPU
    trunc_normal_(fc_layer->weight, 0, 0.05, torch::kCPU);
    torch::nn::init::constant_(fc_layer->bias, 0.0);
    fc_layer->to(device); // Ensure the entire layer is moved to GPU

    auto fc_result = fc_layer->forward(input.to(device));
    return activation_fn(fc_result);
}
















} // namespace utils








typedef struct ImageData {
    cv::Mat image;
    std::string label;
    std::filesystem::path path;
} ImageData;





// Function to read images and labels from a directory
std::vector<ImageData> readImagesFromDirectory(const std::string& baseDir) {
    std::vector<ImageData> imageData;
    std::filesystem::path dirPath(baseDir);

    // Check if directory exists
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << "Directory does not exist or is not accessible: " << baseDir << std::endl;
        return imageData; // Return empty vector
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); // Normalize the extension to lower case

            if (ext == ".jpg" || ext == ".jpeg") {
                std::string label = entry.path().parent_path().filename().string();
                cv::Mat image = cv::imread(entry.path().string());

                if (!image.empty()) {
                    imageData.push_back({image, label, entry.path()});
                } else {
                    std::cerr << "Failed to read image: " << entry.path() << " (could be corrupted or unsupported format)" << std::endl;
                }
            }
        }
    }
    return imageData;
}



















void loss_plot(const std::string &filename, int iteration, double loss, const std::string& loss_type) {
    // Open the file in append mode
    std::ofstream csv(filename, std::ios_base::app);
    if (!csv.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    // Check if file is empty and write the column names
    std::ifstream in(filename);
    if (in.peek() == std::ifstream::traits_type::eof()) {
        // File is empty
        csv << "Iteration," << loss_type << "\n";
    }
    in.close();

    // Append the loss values in one row
    csv << iteration << "," << loss << "\n";
    csv.close();
}













// save actual labels and predicted labels
void predict_save(const std::string& file_name, const std::vector<int> predictions, const std::vector<ImageData>& imagesData, const std::map<std::string, int>& label_map) {
    std::ofstream csv(file_name);
    if (!csv.is_open()) {
        std::cerr << "Error: could not open file " << file_name << std::endl;
        return;
    }

    csv << "Image,Actual_Label,Predicted_Label\n";

    int correct_predictions = 0;
    int total_samples = predictions.size();
    int num_classes = label_map.size();
    std::vector<int> true_positive(num_classes, 0);
    std::vector<int> false_positive(num_classes, 0);
    std::vector<int> false_negative(num_classes, 0);

    for (size_t i = 0; i < predictions.size(); i++) {
        int actual_label_index = label_map.at(imagesData[i].label);
        int predicted_label = predictions[i];

        // writing actual / prediction to the csv
        csv << "Image_" << i << "," << actual_label_index << "," << predicted_label << "\n";

        if (actual_label_index == predicted_label) {
            correct_predictions++;
            true_positive[actual_label_index]++;
        } else {
            false_positive[predicted_label]++;
            false_negative[actual_label_index]++;
        }
    }

    csv.close();

    double accuracy = static_cast<double>(correct_predictions) / total_samples;

    double precision_sum = 0.0;
    double recall_sum = 0.0;
    for (int i = 0; i < num_classes; i++) {
        int tp = true_positive[i];
        int fp = false_positive[i];
        int fn = false_negative[i];

        double precision = (tp + fp == 0) ? 0 : static_cast<double>(tp) / (tp + fp);
        double recall = (tp + fn == 0) ? 0 : static_cast<double>(tp) / (tp + fn);

        precision_sum += precision;
        recall_sum += recall;
    }

    double precision = precision_sum / num_classes;
    double recall = recall_sum / num_classes;
    double f1_score = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

    std::cout << "Precision: " << precision * 100 << "%\n";
    std::cout << "Recall: " << recall * 100 << "%\n";
    std::cout << "F1-Score: " << f1_score * 100 << "%\n";
}








#endif // UTILS_H
