#ifndef resnet_encoder_pre_hpp
#define resnet_encoder_pre_hpp

#include <torch/torch.h>
#include <torch/script.h> // Required for torch::jit::Module
#include <string>
#include <vector>
#include <map>

namespace networks {

    class ResnetEncoder_pre : public torch::nn::Module {
    public:
        ResnetEncoder_pre(const std::string& script_module_path);
        std::vector<torch::Tensor> forward(torch::Tensor input_image);

    private:
        torch::jit::script::Module backbone_;
    };

}

#endif