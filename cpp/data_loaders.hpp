#ifndef CUSTOM_DATASET_HPP
#define CUSTOM_DATASET_HPP

#include <torch/torch.h>
#include <vector>
#include <string>

namespace data_loaders {

    class Prepare_dataset_test : public torch::data::Dataset<Prepare_dataset_test, torch::data::Example<>> {
        public:
            Prepare_dataset_test(std::string data_path);
            torch::Tensor norm_image(torch::Tensor image);
            std::unordered_map<std::string, std::variant<torch::Tensor, int>> get(size_t index) override;
            torch::optional<size_t> size() const override;
        private:
            std::string data_path_;
            torch::Tensor data;
            int height = 192;
            int width = 640;
            torch::Tensor K_;
    };
 #endif