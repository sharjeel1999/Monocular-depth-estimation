#ifndef main_trainer_hpp // Include guard to prevent multiple inclusions
#define main_trainer_hpp


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

#include "./networks/resnet_encoder_pre.hpp"

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

void collect_module_names_recursive(
    const torch::jit::script::Module& module,
    const std::string& prefix,
    std::vector<std::string>& module_names) {

    // Iterate through named modules (direct children)
    for (const auto& named_module : module.named_modules()) {
        const std::string& name = named_module.name;
        const torch::jit::script::Module& submodule = named_module.value;

        // Construct the full hierarchical name
        std::string full_name = prefix.empty() ? name : prefix + "." + name;

        // Add the full name to the list
        module_names.push_back(full_name);

        // Recursively call for the submodule, unless it's the same as the current module
        // (named_modules() of the top-level module includes itself with an empty name)
        if (!name.empty()) {
             collect_module_names_recursive(submodule, full_name, module_names);
        }
    }
}

// Function to get all module names
std::vector<std::string> get_all_module_names(const torch::jit::script::Module& module) {
    std::vector<std::string> module_names;
    // Start the recursive collection with an empty prefix
    collect_module_names_recursive(module, "", module_names);
    return module_names;
}


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
        
        // device_ = torch::Device('cpu');
        

        // Both of the below lines are equivalent methods to load the model seem to work the exact same way... should i make cpp for each module??
        torch::jit::script::Module encoder_ = std::make_shared<networks::ResnetEncoder_pre>("/home/sharjeel/Desktop/repositories/Depth_estimation_cpp/cpp/networks_jit_saves/resnet50_fpn_backbone.pt")->backbone_;
        // encoder_ = torch::jit::load("/home/sharjeel/Desktop/repositories/Depth_estimation_cpp/cpp/networks_jit_saves/resnet50_fpn_backbone.pt");
        
        std::vector<std::string> mn = get_all_module_names(encoder_);
        std::cout << "Module names: " << std::endl;
        for (const auto& name : mn) {
            std::cout << name << std::endl;
        }
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

    int num_input_frames_;
    int num_pose_frames_;
    int num_scales_;

    // torch::Device device_
    std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> models_;
    std::shared_ptr<torch::optim::Optimizer> model_optimizer_;

    torch::jit::script::Module encoder_;

};

#endif