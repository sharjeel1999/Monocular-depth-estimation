#include 'data_loaders.hpp'

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>

namespace data_loaders {

    Prepare_dataset_test::Prepare_dataset_test(std::string data_path)
            : data_path_(data_path) {
            torch::Tensor data = torch::load(data_path_)

            auto to_tensor = torch::data::transforms::ToTensor<>();

            std::vector<double> normalize_mean = {0.485, 0.456, 0.406};
            std::vector<double> normalize_std = {0.227, 0.224, 0.225};
            auto normalize_transform_cpp = torch::data::transforms::Normalize<>(normalize_mean, normalize_std);

            int height = 192;
            int width = 640;

            torch::Tensor K_ = torch::Tensor([[0.58, 0, 0.5, 0],
                                                [0, 1.92, 0.5, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]]);
        }

    torch::Tensor Prepare_dataset_test::norm_image(torch::Tensor image) {
        // Min-max normalization
        auto min_val = image.min();
        auto max_val = image.max();
        return (image - min_val) / (max_val - min_val);
    }


    std::unordered_map<std::string, std::variant<torch::Tensor, int>> Prepare_dataset_test::get(size_t index) override {
        image_m1,  image, image_p1, depth_gt = data[index, :];

        std::unordered_map<std::string, std::variant<torch::Tensor, int>> inputs;
        inputs["image_-1"] = image_m1;
        inputs["image_0"] = image;
        inputs["image_1"] = image_p1;

        torch::Tensor K = K_.clone();
        K.index({0}).mul_(width / 2);
        K.index({1}).mul_(height / 2);
        inputs["K"] = K;
        
        // Step 2: Convert Torch tensor to OpenCV Mat
        cv::Mat cv_K(K.size(0), K.size(1), CV_32F, K.data_ptr<float>());

        // Step 3: Compute pseudoinverse using OpenCV's cv::invert with DECOMP_SVD
        cv::Mat cv_inv_K;
        cv::invert(cv_K, cv_inv_K, cv::DECOMP_SVD);

        // Step 4: Convert the resulting Mat back to Torch Tensor
        torch::Tensor inv_K = torch::from_blob(cv_inv_K.data, {cv_K.rows, cv_K.cols}, torch::kFloat32).clone();

        inputs['inv_K'] = inv_K;

        inputs["depth_gt"] = depth_gt;
        
        return inputs;
    }


    torch::optional<size_t> Prepare_dataset_test::size() const override {
        return data.sizes()[0];
    }

}
