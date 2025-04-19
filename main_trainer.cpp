#include <string>
#include <iostream>
#include <unordered_map>

#include <torch/torch.h>

class Main_Trainer {
public:
    Main_Trainer(auto options, auto train_loader,
                 auto test_loader,
                 std::string save_folder,
                 int batch_size,
                 bool use_affinity,
                 int epochs)
        : train_loader(train_loader),
          test_loader(test_loader),
          save_folder(save_folder),
          epochs(epochs),
          batch_size(batch_size),
          use_affinity(use_affinity) {

            std::cout << "Entered Constructor " << std::endl;
          }
    
    void all_variables() {
        std::cout << "All variables: " << std::endl;
        std::cout << "Save Folder: " << save_folder << std::endl;
        std::cout << "Batch Size: " << batch_size << std::endl;
        std::cout << "Use Affinity: " << use_affinity << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
    }

private:
    std::unordered_map<std::string, std::variant<int, double, bool, std::string>> options;
    std::shared_ptr<torch::data::DataLoader<>> train_loader;
    std::shared_ptr<torch::data::DataLoader<>> test_loader;
    std::string save_folder;
    int epochs;
    int batch_size;
    bool use_affinity;


};
    // ~Main_Trainer();

    // void train();
    // void test();
    // void save_model();
    // void load_model();