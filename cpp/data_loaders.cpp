#include <torch/torch.h>
#include <vector>
#include <iostream>


class Prepare_dataset_test : public torch::data::Dataset<Prepare_dataset_test, torch::data::Example<>> {
public:
Prepare_dataset_test(std::string data_path)
        : data_path_(data_path) {
        torch::Tensor data = torch::load(data_path_)

        auto to_tensor = torch::data::transforms::ToTensor<>();

        std::vector<double> normalize_mean = {0.485, 0.456, 0.406};
        std::vector<double> normalize_std = {0.227, 0.224, 0.225};
        auto normalize_transform_cpp = torch::data::transforms::Normalize<>(normalize_mean, normalize_std);
    }

    torch::data::Example<> get(size_t index) override {
        // --- Data Loading Logic ---
        
        return {data, target};
    }


    torch::optional<size_t> size() const override {
        return data.sizes()[0];
    }

private:
    // Member variables to store dataset properties
    size_t num_examples_;
    int data_size_;
    int target_size_;

    std::string data_path_;

    std::vector<int64_t> full_res_shape;
    std::unordered_map<std::string, int> side_map;
};


