#include "resnet_encoder_pre.hpp"
#include <iostream>

namespace networks {

    ResnetEncoder_pre::ResnetEncoder_pre(
        const std::string& script_module_path) 
        : torch::nn::Module("ResnetEncoder_pre") // Call base class constructor with module name
    {
        std::cout << "Loading pre-trained backbone from: " << script_module_path << std::endl;
        try {
            backbone_ = torch::jit::load(script_module_path);
            std::cout << "Backbone loaded successfully." << std::endl;

            // backbone_.to(at::kCUDA);
            // backbone_.eval();

        } catch (const c10::Error& e) {
            std::cerr << "Error loading the script module: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during script module loading: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<torch::Tensor> ResnetEncoder_pre::forward(torch::Tensor input_image) {

        torch::jit::IValue output_ivalue = backbone_.forward({input_image});


        std::map<std::string, torch::Tensor> base_out_map;
        try {
            // base_out_map = output_ivalue.to<std::map<std::string, torch::Tensor>>();
            // Alternative using toGenericDict() and casting:
            auto generic_dict = output_ivalue.toGenericDict();
            base_out_map["0"] = generic_dict.at("0").toTensor();

        } catch (const c10::Error& e) {
            std::cerr << "Error casting backbone output to map: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during output cast: " << e.what() << std::endl;
            throw;
        }


        std::vector<torch::Tensor> features;
        features.reserve(5); // Reserve space for 5 tensors

        // Access keys and append tensors
        // Use .at() which throws an exception if key is not found
        try {
            features.push_back(base_out_map.at("0"));
            features.push_back(base_out_map.at("1"));
            features.push_back(base_out_map.at("2"));
            features.push_back(base_out_map.at("3"));
            // Note: 'pool' key might be specific to the Python FPN backbone implementation.
            // Verify the exact keys output by the TorchScript module if you encounter issues.
            // You might need to print map keys during testing.
            features.push_back(base_out_map.at("pool"));

        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Key missing in backbone output map. Check keys ('0','1','2','3','pool'): " << e.what() << std::endl;
            throw; // Re-throw as this is a structural error
        } catch (const c10::Error& e) {
            std::cerr << "Error accessing tensors from backbone output map: " << e.what() << std::endl;
            throw;
        }


        return features;
    }

}