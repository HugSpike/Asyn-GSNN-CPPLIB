
#include "AsynchronousGraphSpiNet.h"
#include <fstream>       // For std::ifstream
#include <vector>        // For std::vector
#include <string>        // For std::string
#include <iterator>      // For std::istreambuf_iterator
#include <torch/torch.h> // For torch::Tensor and torch::pickle_load
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

torch::Tensor get_the_tensor(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    torch::Tensor x = torch::pickle_load(bytes).toTensor();
    return x;
}

int main()
{
    torch::Tensor node_feature;
    torch::Tensor edge_index;
    torch::Tensor pos;
    torch::Tensor edge_weights;
    torch::Tensor spike_node;
    torch::Tensor raw_features;
    torch::Tensor node_indices;
    if (torch::cuda::is_available())
    {
        node_feature = get_the_tensor("/home/ff/code/cpp_files_cpu/data/node_feature_cpu.pt");
        edge_weights = get_the_tensor("/home/ff/code/cpp_files_cpu/data/edge_weights_cpu.pt");
        pos = get_the_tensor("/home/ff/code/cpp_files_cpu/data/pos_cpu.pt");
        edge_index = get_the_tensor("/home/ff/code/cpp_files_cpu/data/edge_index_cpu.pt").toType(torch::kInt);
        spike_node = get_the_tensor("/home/ff/code/cpp_files_cpu/data/spike_node_cpu.pt").toType(torch::kInt);
        raw_features = get_the_tensor("/home/ff/code/cpp_files_cpu/data/raw_features_cpu.pt");
        node_indices = get_the_tensor("/home/ff/code/cpp_files_cpu/data/node_indices_cpu.pt").toType(torch::kInt);
    }

    std::ifstream i("/home/ff/code/cpp_files_cpu/config.json");
    json config;
    i >> config;
    
    // Reading each parameter individually
    double v_rest1 = config["v_rest1"];
    double v_r1 = config["v_r1"];
    double v_rest2 = config["v_rest2"];
    double v_r2 = config["v_r2"];
    int sample_num1 = config["sample_num1"];
    int sample_num2 = config["sample_num2"];
    double dropout_rate = config["dropout_rate"];
    double speed = config["speed"];
    double v_threshold_init1 = config["v_threshold_init1"];
    double v_threshold_init2 = config["v_threshold_init2"];
    double refractory_period_init = config["refractory_period_init"];
    double decay_alpha = config["decay_alpha"];
    int batchSize = config["batchSize"];
    int time_step_init = config["time_step_init"];
    double max_rate_init = config["max_rate_init"];
    double beta_init = config["beta_init"];
    int numClass = config["num_class"];


    auto input_features = node_feature.size(1);
    auto transformed_features = node_feature.size(1);
    try
    {
        AsynchronousGraphSpiNet net = AsynchronousGraphSpiNet(
            v_rest1, v_r1,
            v_rest2, v_r2,
            sample_num1, sample_num2, dropout_rate,
            speed, v_threshold_init1, v_threshold_init2,
            refractory_period_init, decay_alpha,
            batchSize,
            time_step_init, max_rate_init,
            beta_init,
            input_features,
            transformed_features,
            numClass,
            edge_weights);
        
        net.getConv1()->init_param(node_indices);

        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor predicted_label = net.forward(
            node_feature,
            edge_index,
            pos,
            spike_node,
            raw_features,
            node_indices
        );
        std::cout<<predicted_label<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "执行时间: " << duration.count() << " 毫秒" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}