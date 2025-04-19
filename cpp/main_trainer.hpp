#ifndef CALCULATOR_HPP // Include guard to prevent multiple inclusions
#define CALCULATOR_HPP


#include <string>
#include <iostream>
#include <unordered_map>

#include <torch/torch.h>


using MNISTTrainDataLoader = std::shared_ptr<torch::data::DataLoader<
    torch::data::TransformedDataset<
        torch::data::TransformedDataset<
            torch::data::datasets::MNIST,
            torch::data::transforms::Normalize<double>
        >,
        torch::data::transforms::Stack<>
    >,
    torch::data::samplers::RandomSampler,
    torch::data::samplers::DefaultBatchSampler
>>;

using MNISTTestDataLoader = std::shared_ptr<torch::data::DataLoader<
    torch::data::TransformedDataset<
        torch::data::TransformedDataset<
            torch::data::datasets::MNIST,
            torch::data::transforms::Normalize<double>
        >,
        torch::data::transforms::Stack<>
    >,
    torch::data::samplers::SequentialSampler, // Often use SequentialSampler for testing
    torch::data::samplers::DefaultBatchSampler
>>;


class Main_Trainer {
public:
    Main_Trainer(std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options,
                 MNISTTrainDataLoader train_loader,
                 MNISTTestDataLoader test_loader,
                 std::string save_folder,
                 int batch_size,
                 bool use_affinity,
                 int epochs);
    
    void all_variables();
private:
    std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options;
    MNISTTrainDataLoader train_loader;
    MNISTTestDataLoader test_loader;
    std::string save_folder;
    int epochs;
    int batch_size;
    bool use_affinity;


};

#endif