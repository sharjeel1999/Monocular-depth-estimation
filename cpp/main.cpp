
#include "main_trainer.hpp"


#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/dataloader.h>
#include <torch/data/samplers/random.h>
#include <torch/data/samplers/sequential.h>
// #include <torch/data/transforms/normalize.h>
#include <torch/data/transforms/tensor.h> //contains Normalize
#include <torch/data/transforms/stack.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <memory>


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
    torch::data::samplers::SequentialSampler,
    torch::data::samplers::DefaultBatchSampler
>>;


int main() {

    const std::string mnist_data_root = "/path/to/your/mnist_data";

    std::string save_folder = "saved_models";
    int batch_size = 64;
    bool use_affinity = true;
    int epochs = 10;


    std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options;
    options["learning_rate"] = 0.001;
    options["optimizer_type"] = std::string("Adam");
    options["num_classes"] = 10;


    std::cout << "Loading MNIST dataset from: " << mnist_data_root << std::endl;

    auto train_dataset = torch::data::datasets::MNIST(mnist_data_root, torch::data::datasets::MNIST::Mode::kTrain)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST(mnist_data_root, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        train_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
    );

    auto test_loader = torch::data::make_data_loader(
        test_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
    );

    std::cout << "MNIST DataLoaders created." << std::endl;

    std::cout << "\nCreating Main_Trainer object..." << std::endl;

    Main_Trainer trainer(options,
                         train_loader,
                         test_loader,
                         save_folder,
                         batch_size,
                         use_affinity,
                         epochs);

    std::cout << "Main_Trainer object successfully created." << std::endl;

    trainer.all_variables();
    std::cout << "All variables printed." << std::endl;
    return 0;
}
