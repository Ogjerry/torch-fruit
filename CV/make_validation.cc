#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <random>
#include <regex>


#include <opencv4/opencv2/opencv.hpp>


#include "utils.hpp"





void mk_validation(const std::string& base_dir, const std::string& destination_dir, size_t num_images) {
    auto images = readImagesFromDirectory(base_dir);
    
    // Shuffle the vector of image data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(images.begin(), images.end(), g);
    
    // Limit the number of images to `num_images`
    if (num_images < images.size()) {
        images.resize(num_images);
    }

    // Move the files to the new directory
    for (const auto& img_data : images) {
        std::filesystem::path new_location = std::filesystem::path(destination_dir) / img_data.label / img_data.path.filename();
        std::filesystem::create_directories(new_location.parent_path());  // Make sure the target directory exists
        std::filesystem::copy(img_data.path, new_location, std::filesystem::copy_options::overwrite_existing);
    }
}





int main(int argc, char const *argv[])
{
    std::string base_dir = "/home/zirui/Desktop/cu/CV/fruits-360_dataset/fruits-360/Test";
    std::filesystem::path destination_dir = "/home/zirui/Desktop/cu/CV/fruits-360_dataset/fruits-360/Validation";
    size_t num = 5000;

    mk_validation(base_dir, destination_dir, num);
    //std::cout << "Number of images moved for validation: " << validation.size() << std::endl;

    return 0;
}
