#ifndef CALCULATOR_HPP // Include guard to prevent multiple inclusions
#define CALCULATOR_HPP


#include <string>
#include <iostream>
#include <unordered_map>

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/dataloader.h>
#include <torch/data/samplers/random.h>
#include <torch/data/samplers/sequential.h>
#include <torch/data/transforms/tensor.h> //contains Normalize
#include <torch/data/transforms/stack.h>


// using MNISTTrainDataLoader = std::shared_ptr<torch::data::DataLoaderImpl<
//     torch::data::TransformedDataset<
//         torch::data::TransformedDataset<
//             torch::data::datasets::MNIST,
//             torch::data::transforms::Normalize<double>
//         >,
//         torch::data::transforms::Stack<>
//     >,
//     torch::data::samplers::RandomSampler,
//     torch::data::samplers::DefaultBatchSampler
// >>;

// using MNISTTestDataLoader = std::shared_ptr<torch::data::DataLoader<
//     torch::data::TransformedDataset<
//         torch::data::TransformedDataset<
//             torch::data::datasets::MNIST,
//             torch::data::transforms::Normalize<double>
//         >,
//         torch::data::transforms::Stack<>
//     >,
//     torch::data::samplers::SequentialSampler, // Often use SequentialSampler for testing
//     torch::data::samplers::DefaultBatchSampler
// >>;

template <typename T_TrainDataLoader, typename T_TestDataLoader>
class Main_Trainer {
public:
    
    // Constructor declaration and definition             
    Main_Trainer(std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options,
    T_TrainDataLoader train_loader,
    T_TestDataLoader test_loader,
        std::string save_folder,
        int batch_size,
        bool use_affinity,
        int epochs)
        : options_(options),
          train_loader_(std::move(train_loader)),
          test_loader_(std::move(test_loader)),
          save_folder_(save_folder),
          epochs_(epochs),
          batch_size_(batch_size),
          use_affinity_(use_affinity) {
        std::cout << "Entered Constructor " << std::endl;
    }
    

    void all_variables() {
        std::cout << "All variables: " << std::endl;
        std::cout << "Save Folder: " << save_folder_ << std::endl;
        std::cout << "Batch Size: " << batch_size_ << std::endl;
        std::cout << "Use Affinity: " << use_affinity_ << std::endl;
        std::cout << "Epochs: " << epochs_ << std::endl;
    }
    

private:
    std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options_;
    T_TrainDataLoader train_loader_;
    T_TestDataLoader test_loader_;
    std::string save_folder_;
    int epochs_;
    int batch_size_;
    bool use_affinity_;


};

#endif