#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <random>
#include <numeric>


// libtorch + opencv + cuda
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/utils.h>
#include <torch/script.h>
#include <torch/data.h>




// defined functions
#include "utils.hpp"
#include "Net.hpp"




//#define BATCH_SIZE 50
#define EPOCHS 30
#define IMAGE_SIZE 100
#define NUM_CLASSES 131
#define LEARNING_RATE_DECAY 0.4
#define DROPOUT 0.2



const double init_learning_rate = 0.001;
const double min_learning_rate = 0.00001;






// Random number generator initialization
std::default_random_engine generator;










// Function to augment the image
cv::Mat augment_image(const cv::Mat& img) {
    cv::Mat augmented_img = img.clone();
    std::uniform_int_distribution<int> flip_dist(0, 1);
    std::uniform_real_distribution<float> hue_dist(0.0, 0.02);
    std::uniform_real_distribution<float> sat_dist(0.9, 1.2);
    std::uniform_real_distribution<float> rotate_dist(-5.0, 5.0);
    std::uniform_real_distribution<float> scale_dist(0.9, 1.1);

    // Random flip
    if (flip_dist(generator)) {
        cv::flip(augmented_img, augmented_img, 1);
    }
    if (flip_dist(generator)) {
        cv::flip(augmented_img, augmented_img, 0);
    }

    // Random rotation and scaling
    float angle = rotate_dist(generator);
    float scale = scale_dist(generator);
    cv::Point2f center(augmented_img.cols / 2.0, augmented_img.rows / 2.0);
    cv::Mat rot_matrix = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(augmented_img, augmented_img, rot_matrix, augmented_img.size());

    // Random changes in hue and saturation
    cv::cvtColor(augmented_img, augmented_img, cv::COLOR_BGR2HSV);
    augmented_img.convertTo(augmented_img, CV_32F);

    float hue_shift = hue_dist(generator) * 180;
    float sat_scale = sat_dist(generator);

    for (int i = 0; i < augmented_img.rows; i++) {
        for (int j = 0; j < augmented_img.cols; j++) {
            augmented_img.at<cv::Vec3f>(i, j)[0] += hue_shift;
            augmented_img.at<cv::Vec3f>(i, j)[1] *= sat_scale;

            augmented_img.at<cv::Vec3f>(i, j)[0] = std::max(0.0f, std::min(180.0f, augmented_img.at<cv::Vec3f>(i, j)[0]));
            augmented_img.at<cv::Vec3f>(i, j)[1] = std::max(0.0f, std::min(255.0f, augmented_img.at<cv::Vec3f>(i, j)[1]));
        }
    }

    augmented_img.convertTo(augmented_img, CV_8U);
    cv::cvtColor(augmented_img, augmented_img, cv::COLOR_HSV2BGR);

    return augmented_img;
}






// image normalization in terms of R, G, B
// Function to convert OpenCV image to a normalized Torch tensor
// FAIL EXPERIMENT
cv::Mat normalize_image(const cv::Mat& img, const cv::Scalar& mean, const cv::Scalar& std) {
    cv::Mat normed_img;

    // Subtract mean and divide by std
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    channels[0].convertTo(channels[0], CV_32F, 1.0 / std[0], -mean[0] / std[0]);
    channels[1].convertTo(channels[1], CV_32F, 1.0 / std[1], -mean[1] / std[1]);
    channels[2].convertTo(channels[2], CV_32F, 1.0 / std[2], -mean[2] / std[2]);
    cv::merge(channels, normed_img);

    
    return normed_img;
}







// Function to convert image to tensor
torch::Tensor matToTensor(const cv::Mat& image) {
    // Convert image to float and normalize it to range [0, 1]
    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32F);

    // Ensure the data is in [channels, height, width] format expected by PyTorch
    auto tensor = torch::from_blob(
        imgFloat.data, 
        {imgFloat.rows, imgFloat.cols, imgFloat.channels()}, // Original shape
        torch::kFloat32
    ).clone();

    // permute to [channels, height, width]
    tensor = tensor.permute({2, 0, 1});

    tensor = tensor.unsqueeze(0);

    return tensor;
}









// visualize normed images
cv::Mat convertForDisplay(const cv::Mat& normalizedImg) {
    cv::Mat displayImg;
    // Assuming the image was normalized to zero mean and unit variance
    normalizedImg.convertTo(displayImg, CV_8UC3, 255.0, 128.0);
    return displayImg;
}

void displayImage(const cv::Mat& image, const std::string& windowName) {
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
    cv::waitKey(0); // Wait for a keystroke in the window
    cv::destroyWindow(windowName);
}












// Function to save the model
void save_model(const std::string& filename, const Net& model) {
    torch::serialize::OutputArchive output_archive;
    // model.save(output_archive);
    model -> save(output_archive);
    output_archive.save_to(filename);
}

void load_model(const std::string& filename, Net& model) {
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(filename);
    //model.load(input_archive);
    model -> load(input_archive);
}









int label_to_index(const std::string& label, const std::map<std::string, int>& label_map) {
    auto it = label_map.find(label);
    if (it != label_map.end()) {
        return it->second;
    } else {
        std::cerr << "Label not found in map: " << label << std::endl;
        return -1;
    }
}








void checkDataConsistency(const std::map<std::string, int>& labelMap, const std::vector<std::string>& directories) {
    for (const auto& dir : directories) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            std::string label = entry.path().filename().string();
            if (labelMap.find(label) == labelMap.end()) {
                std::cerr << "Unrecognized label in dataset: " << label << std::endl;
            }
        }
    }
}








std::map<std::string, int> createLabelMap(const std::vector<std::string>& directories) {
    std::map<std::string, int> labelMap;
    int labelIndex = 0;

    for (const auto& dir : directories) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            std::string label = entry.path().filename().string();
            if (labelMap.find(label) == labelMap.end()) {
                labelMap[label] = labelIndex++;
            }
        }
    }
    return labelMap;
}








// Decay function adapted for the RMSprop optimizer
template <typename Optimizer>
inline void decay(Optimizer& optimizer, double rate) {
    if (rate <= 0.0 || rate >= 1.0) {
        std::cerr << "Warning: Learning rate decay factor should be between 0 and 1." << std::endl;
        return;
    }
    double min_lr = min_learning_rate;
    for (auto & group : optimizer.param_groups()) {
        auto & options = static_cast<torch::optim::RMSpropOptions&>(group.options());
        double new_lr = std::max(min_lr, options.get_lr() * rate);
        options.set_lr(new_lr);
        std::cout << "Adjusted learning rate to: " << new_lr << std::endl;
    }
}








void train(Net& net, std::vector<ImageData>& train_data, std::vector<ImageData>& val_data, int epochs, torch::Device device,  const std::map<std::string, int>& label_map, const size_t batch_size) {
    net->train();
    torch::optim::RMSpropOptions rms_options(init_learning_rate);
    torch::optim::RMSprop optimizer(net->parameters(), rms_options);
    double best_val_loss = std::numeric_limits<double>::infinity();
    int patience = 10;
    int epochs_since_last_improvement = 0;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
        double running_loss = 0.0;
        int batch_index = 0;

        // Training loop
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            std::vector<torch::Tensor> batch_imgs;
            std::vector<int> batch_labels;

            for (size_t j = i; j < i + batch_size && j < train_data.size(); ++j) {
                cv::Mat img = augment_image(train_data[j].image);
                torch::Tensor img_tensor = matToTensor(img).to(device);
                batch_imgs.push_back(img_tensor);

                std::string label = train_data[j].label;
                if (label_map.find(label) == label_map.end()) {
                    std::cerr << "Label not found in map: " << label << std::endl;
                    continue; // Skip this image if the label is not found
                }

                int label_index = label_map.at(label);
                batch_labels.push_back(label_index);
            }

            if (batch_labels.empty()) continue;

            torch::Tensor data = torch::cat(batch_imgs, 0).to(device);
            torch::Tensor target = torch::tensor(batch_labels, torch::kInt64).to(device);

            optimizer.zero_grad();
            torch::Tensor output = net->forward(data);
            torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);
            loss.backward();
            // After loss.backward()
            double max_grad_norm = 1000.0; // Set a sensible threshold
            double total_norm = torch::nn::utils::clip_grad_norm_(net->parameters(), max_grad_norm);
            if (total_norm > max_grad_norm) {
                std::cout << "Clipping gradient: Norm before clipping: " << total_norm << std::endl;
            }
            optimizer.step();


            // Detach tensors and clear GPU memory
            data = data.cpu().detach_();
            target = target.cpu().detach_();
            output = output.cpu().detach_();
            loss = loss.cpu().detach_();

            running_loss += loss.item<double>();
            batch_index++;
        }

        double avg_loss = running_loss / std::max(batch_index, 1); // avoid division by 0
        std::cout << "Training loss: " << avg_loss << std::endl;

        // Validation loop
        net->eval();
        double val_running_loss = 0.0;
        int val_batch_index = 0;

        {
            torch::NoGradGuard no_grad; // Disable gradient calculation during validation
            for (size_t i = 0; i < val_data.size(); i += batch_size) {
                std::vector<torch::Tensor> batch_imgs;
                std::vector<int> batch_labels;

                for (size_t j = i; j < i + batch_size && j < val_data.size(); ++j) {
                    cv::Mat img = val_data[j].image;
                    torch::Tensor img_tensor = matToTensor(img).to(device);
                    batch_imgs.push_back(img_tensor);

                    std::string label = val_data[j].label;
                    if (label_map.find(label) == label_map.end()) {
                        std::cerr << "Label not found in map: " << label << std::endl;
                        continue; // Skip this image if the label is not found
                    }

                    int label_index = label_map.at(label);
                    batch_labels.push_back(label_index);
                }

                if (batch_labels.empty()) continue;

                torch::Tensor data = torch::cat(batch_imgs, 0).to(device);
                torch::Tensor target = torch::tensor(batch_labels, torch::kInt64).to(device);

                torch::Tensor output = net->forward(data);
                torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

                val_running_loss += loss.item<double>();

                // Detach tensors and clear GPU memory
                data = data.cpu().detach_();
                target = target.cpu().detach_();
                output = output.cpu().detach_();
                loss = loss.cpu().detach_();

                val_batch_index++;
            }
        }

        double avg_val_loss = val_running_loss / val_batch_index;
        std::cout << "Validation loss: " << avg_val_loss << std::endl;

        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            epochs_since_last_improvement = 0;
            // save model checkpoint
            ///torch::save(net, "curbest.pt");
            //save_model("curbest.pt", net);
        } else {
            epochs_since_last_improvement++;
            if (epochs_since_last_improvement >= patience) {
                decay(optimizer, LEARNING_RATE_DECAY); // Apply decay
                std::cout << "Reduced learning rate to " << optimizer.param_groups()[0].options().get_lr() << std::endl;
                epochs_since_last_improvement = 0; // Reset counter after decay
            }
        }

        net->train(); // Ensure the model is back in training mode after validation
    }

    // After the training loop save
    std::cout << "Final model training complete. Saving final model state..." << std::endl;
    save_model("final_model.pt", net);
}










// Training loop + Cross Validation //

void train_t(Net& model, std::vector<ImageData>& train_data, std::vector<ImageData>& val_data, int epochs, torch::Device device, const std::map<std::string, int>& label_map, size_t batch_size) {
    model->to(device);
    torch::optim::RMSprop optimizer(model->parameters(), torch::optim::RMSpropOptions(init_learning_rate));
    auto criterion = torch::nn::CrossEntropyLoss();
    double best_val_loss = std::numeric_limits<double>::infinity();
    int patience = 3;
    
    int epochs_since_last_improvement = 0;

    size_t num_batches = train_data.size() / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        double running_loss = 0.0;
        size_t processed_batches = 0;

        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            optimizer.zero_grad();
            std::vector<torch::Tensor> data_tensors;
            std::vector<int> target_labels;

            for (size_t j = i; j < i + batch_size && j < train_data.size(); ++j) {
                data_tensors.push_back(matToTensor(train_data[j].image));
                target_labels.push_back(label_map.at(train_data[j].label));
            }

            if (!data_tensors.empty()) { // Ensure there is data to process
                auto data = torch::cat(data_tensors).to(device);
                auto targets = torch::tensor(target_labels, torch::kInt64).to(device);
                //std::cout << "Data shape: " << data.sizes() << ", Target shape: " << targets.sizes() << std::endl;
                auto output = model->forward(data);
                //std::cout << "Output shape: " << output.sizes() << std::endl;
                auto loss = criterion(output, targets);
                loss.backward();


                // gradient checking
                // After loss.backward()
                double max_grad_norm = 500.0; // Set a sensible threshold
                double total_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), max_grad_norm);
                if (total_norm > max_grad_norm) {
                    std::cout << "Clipping gradient: Norm before clipping: " << total_norm << std::endl;
                }



                optimizer.step();
                running_loss += loss.item<double>();

                // store loss for plot
                //loss_plot("tloss.csv", i, loss.item<double>(), "train");

                output.detach_();
                loss.detach_();
                processed_batches++;
            }
        }

        std::cout << "Epoch: " << epoch + 1 << " Training loss: " << running_loss / processed_batches << std::endl;
        loss_plot("tloss.csv", epoch, running_loss / processed_batches, "train");

        // Validation step
        model->eval();
        torch::NoGradGuard no_grad;
        double val_loss = 0.0;
        size_t val_num_batches = val_data.size() / batch_size;
        for (size_t batch = 0; batch < val_num_batches; ++batch) {
            std::vector<torch::Tensor> data_tensors;
            std::vector<int> target_labels;
            for (size_t i = batch * batch_size; i < (batch + 1) * batch_size && i < val_data.size(); ++i) {
                data_tensors.push_back(matToTensor(val_data[i].image));
                target_labels.push_back(label_map.at(val_data[i].label));
            }


            if (!data_tensors.empty()) {
                auto data = torch::cat(data_tensors).to(device);
                auto targets = torch::tensor(target_labels, torch::kInt64).to(device);
                //std::cout << "Data shape: " << data.sizes() << ", Target shape: " << targets.sizes() << std::endl;

                auto output = model->forward(data);
                //std::cout << "Output shape: " << output.sizes() << std::endl;

                auto loss = criterion(output, targets);
                val_loss += loss.item<double>();
            }
        }

        std::cout << "Epoch: " << epoch + 1 << " Validation loss: " << val_loss / val_num_batches << std::endl;
        //loss_plot("loss.csv", epoch, val_loss / val_num_batches, "val_loss");
        loss_plot("vloss.csv", epoch, val_loss / val_num_batches, "val");

        // Decay the learning rate if needed
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            epochs_since_last_improvement = 0;
        } else {
            epochs_since_last_improvement++;
            if (epochs_since_last_improvement >= patience) { // No improvement in 3 epochs
                decay(optimizer, LEARNING_RATE_DECAY);
                std::cout << "Learning rate decayed, new rate = " << static_cast<torch::optim::RMSpropOptions&>(optimizer.param_groups()[0].options()).lr() << std::endl;
                epochs_since_last_improvement = 0;
            }
        }
        model -> train();
    }

    // After the training loop save
    std::cout << "Final model training complete. Saving final model state..." << std::endl;
    save_model("final_model.pt", model);
}









void cross_validation(Net& model, std::vector<ImageData>& all_data, int epochs, torch::Device device, const std::map<std::string, int>& label_map, size_t batch_size, int kfold = 5) {
    // randomly shuffle the data to ensure randomness in folds
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_data.begin(), all_data.end(), g);

    int fold_size = all_data.size() / kfold;

    // cross validation scores
    std::vector<double> val_sc;

    for (int fold = 0; fold < kfold; fold++) {
        std::cout << "Training on fold " << fold + 1 << "/" << kfold << std::endl;

        // split data into training and validation for this fold
        std::vector<ImageData> training, validation;
        
        for (int i = 0; i < all_data.size(); i++) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
                validation.push_back(all_data[i]);
            } else {
                training.push_back(all_data[i]);
            }
        }
        // train the model using the training set for this fold
        train_t(model, training, validation, epochs, device, label_map, batch_size);
    }
}

// Training loop + Cross Validation //








std::vector<int> predict(Net& net, const std::vector<ImageData>& data, torch::Device device, const std::map<std::string, int>& label_map) {
    if (torch::cuda::is_available()) {
        std::cout << "predict/eval on GPU" << std::endl;
    } else {
        std::cout << "predict/eval on CPU" << std::endl;
    }
    net->eval();
    std::vector<int> predictions;
    int correct = 0;
    int total = data.size();
    std::vector<int> true_labels(total);

    for (size_t i = 0; i < data.size(); i++) {
        torch::Tensor img_tensor = matToTensor(data[i].image).to(device);
        torch::Tensor output = net->forward(img_tensor);
        torch::Tensor probs = torch::softmax(output, 1);
        int pred = probs.argmax(1).item<int>();
        predictions.push_back(pred);

        int actual_label_index = label_map.at(data[i].label);
        true_labels[i] = actual_label_index;

        if (pred == actual_label_index) correct++;
    }

    double accuracy = static_cast<double>(correct) / total;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return predictions;
}











int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        cudaDeviceReset();
        int device_count = torch::cuda::device_count();
        std::cout << "CUDA is available! Number of CUDA devices: " << device_count << std::endl;
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            std::cout << "Device " << i << ": " << props.name << std::endl;
            std::cout << "  Compute capability: " << props.major << "." << props.minor << std::endl;
            std::cout << "  Total memory: " << props.totalGlobalMem << std::endl;
        }
        device = torch::kCUDA;
    } else {
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
    }




    // read validation and training data
    std::string baseDirectory = "/home/zirui/Desktop/cu/CV/fruits-360_dataset/fruits-360/Training";
    std::string testDirectory = "/home/zirui/Desktop/cu/CV/fruits-360_dataset/fruits-360/Test";

    std::vector<ImageData> imagesData = readImagesFromDirectory(baseDirectory);
    std::vector<ImageData> testData = readImagesFromDirectory(testDirectory);

    std::cout << "Number of training images read: " << imagesData.size() << std::endl;
    std::cout << "Number of testing images read: " << testData.size() << std::endl;

    //cv::Scalar sumBGR(0, 0, 0), sumSquaredBGR(0, 0, 0);
    //int total_pixels = 0;

    //for (const auto &data : imagesData) {
    //    cv::Mat img_float;
    //    data.image.convertTo(img_float, CV_32F, 1.0 / 255);
    //    cv::Scalar img_sum = cv::sum(img_float);
    //    cv::Scalar img_sum_sqr = cv::sum(img_float.mul(img_float));

    //    sumBGR += img_sum;
    //    sumSquaredBGR += img_sum_sqr;
    //    total_pixels += img_float.rows * img_float.cols;
    //}

    // cv::Scalar mean_BGR = sumBGR / total_pixels;
    // cv::Scalar std_BGR;
    // for (int i = 0; i < 3; i++) {
    //     std_BGR[i] = std::sqrt(sumSquaredBGR[i] / total_pixels - mean_BGR[i] * mean_BGR[i]);
    // }
    // std::cout << "Mean (B, G, R): " << mean_BGR << std::endl;
    // std::cout << "Standard Deviation (B, G, R): " << std_BGR << std::endl;



    // experiment with mean & std of R, G, B
    //cv::Scalar mean_accumulator_train(0, 0, 0), std_accumulator_train(0, 0, 0);
    //std::vector<ImageData> train_d, val_d;
    //std::vector<torch::Tensor> tensor_t, tensor_v;

    //for (const auto& img: imagesData) {
    //    cv::Scalar tmean, tstd;
    //    cv::meanStdDev(img.image, tmean, tstd); // seperate R, G, B and calculate their mean & std
    //    mean_accumulator_train += tmean;
    //    std_accumulator_train += tstd.mul(tstd);
    //}





    ///for (const auto& img: valData) {
    ///    cv::Scalar vmean, vstd;
    ///    cv::meanStdDev(img.image, vmean, vstd);
    ///    mean_accumulator_val += vmean;
    ///    std_accumulator_val += vstd;
    ///}

    //cv::Scalar t_mean = mean_accumulator_train / static_cast<double>(imagesData.size());
    //cv::Scalar t_std(
    //sqrt(std_accumulator_train[0] / static_cast<double>(imagesData.size())),
    //sqrt(std_accumulator_train[1] / static_cast<double>(imagesData.size())),
    //sqrt(std_accumulator_train[2] / static_cast<double>(imagesData.size()))
    //);

    //std::cout << "(R, G, B) train mean: " << t_mean <<std::endl;
    //std::cout << "(R, G, B) train std: " << t_std <<std::endl;







    //for (const auto & img : imagesData) {
    //    cv::Mat normed_i = normalize_image(img.image, t_mean, t_std);
    //    cv::Mat displayImg = convertForDisplay(normed_i);
    //    displayImage(displayImg, "Normalized Image");
    //    //torch::Tensor tensor = matToTensor(normed_i);
    //    //tensor_t.push_back(tensor);
    //    train_d.push_back({normed_i, img.label, img.path});
    //}
    //for (const auto & img : valData) {
    //    cv::Mat normed_i = normalize_image(img.image, t_mean, t_std);
    //    //torch::Tensor tensor = matToTensor(normed_i);
    //    //tensor_v.push_back(tensor);
    //    val_d.push_back({normed_i, img.label, img.path});
    //}




    std::map<std::string, int> label_map;
    int label_index = 0;
    for (const auto& data : imagesData) {
        if (label_map.find(data.label) == label_map.end()) {
            label_map[data.label] = label_index++;
            std::cout << "Added label to map: " << data.label << " with index " << label_map[data.label] << std::endl;
        }
    }

    Net net(NUM_CLASSES, device);  // Pass device to SimpleNet constructor
    //net.to(device);  // Ensure the model is on the correct device
    net->to(device);  // Ensure the model is on the correct device

    cross_validation(net, imagesData, EPOCHS, device, label_map, BATCH_SIZE, 2);

    std::vector<int> predicts = predict(net, testData, device, label_map);

    std::string csv_name = "predictions.csv";
    predict_save(csv_name, predicts, testData, label_map);

    return 0;
}
