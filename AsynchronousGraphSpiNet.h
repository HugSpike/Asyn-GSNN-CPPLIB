//
// Created by fengfan on 09/01/2024.
//

#ifndef SPIKE_ASYNCHRONOUSGRAPHSPINET_H
#define SPIKE_ASYNCHRONOUSGRAPHSPINET_H

#include "AsynchronousConvLayer.h"
#include "Util.h"

namespace py = pybind11;

template <typename K, typename V>
void printMap(const std::map<K, V> &m)
{
    std::cout << "Map contains:\n";
    for (const auto &[key, value] : m)
    {
        std::cout << "Key: " << key << ", Value: " << value << '\n';
    }
    std::cout << std::endl;
}

class AsynchronousGraphSpiNet : public torch::nn::Module
{
public:
    AsynchronousGraphSpiNet(
        double v_rest1,
        double v_r1,
        double v_rest2,
        double v_r2,
        int sample_num1,
        int sample_num2,
        double dropout_rate,
        double speed,
        double threshold_init1,
        double threshold_init2,
        double refractory_period_init,
        double decay_alpha,
        int batchSize,
        int time_step_init,
        double max_rate_init,
        double beta_init,
        int input_features,
        int transformed_features,
        int numClass,
        torch::Tensor edge_weights1,
        torch::Tensor edge_weights2)
        : conv1(std::make_shared<AsynchronousConvLayer>(v_rest1, v_r1, sample_num1, speed, threshold_init1,
                                                        refractory_period_init, decay_alpha, time_step_init,
                                                        max_rate_init, beta_init, edge_weights1)),
          conv2(std::make_shared<AsynchronousConvLayer>(v_rest2, v_r2, sample_num2, speed, threshold_init2,
                                                        refractory_period_init, decay_alpha, time_step_init,
                                                        max_rate_init, beta_init, edge_weights2)),
          dropout(torch::nn::Dropout(dropout_rate)),
          bn_conv1(torch::nn::BatchNorm1d(6)),
          bn_conv2(torch::nn::BatchNorm1d(6)),
          lin(torch::nn::Linear(input_features, 128)),
          lin2(torch::nn::Linear(6, 128)),
          fc(torch::nn::Linear(6, numClass))
    {

        this->v_rest1 = v_rest1;
        this->v_r1 = v_r1;
        this->v_rest2 = v_rest2;
        this->v_r2 = v_r2;
        this->sample_num1 = sample_num1;
        this->sample_num2 = sample_num2;
        this->dropout_rate = dropout_rate;
        this->speed = speed;
        this->threshold_init1 = threshold_init1;
        this->threshold_init2 = threshold_init2;
        this->refractory_period_init = refractory_period_init;
        this->decay_alpha = decay_alpha;
        this->batchSize = batchSize;
        this->time_step_init = time_step_init;
        this->max_rate_init = max_rate_init;
        this->beta_init = beta_init;
        this->input_features = input_features;
        this->transformed_features = transformed_features;
        this->numClass = numClass;
        this->edge_weights1 = edge_weights1;
        this->edge_weights2 = edge_weights2;

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("dropout", dropout);
        register_module("bn_conv1", bn_conv1);
        register_module("bn_conv2", bn_conv2);
        register_module("lin", lin);   // 注册线性层
        register_module("lin2", lin2); // 注册线性层
        register_module("fc", fc);     // 注册线性层

        // 遍历所有参数并设置 requires_grad 为 true
        for (auto &param : this->named_parameters())
        {
            param.value().set_requires_grad(true);
        }
    }

    std::unordered_map<std::string, torch::Tensor> get_state_dict();

    void load_state_dict(std::unordered_map<std::string, torch::Tensor> state_dict);

    void print_grad();

    const std::shared_ptr<AsynchronousConvLayer> &getConv1() const;

    const std::shared_ptr<AsynchronousConvLayer> &getConv2() const;
    // Get methods
    const torch::nn::Dropout &getDropout() const
    {
        return dropout;
    }

    const torch::nn::BatchNorm1d &getBN1() const
    {
        return bn_conv1;
    }

    const torch::nn::BatchNorm1d &getBN2() const
    {
        return bn_conv2;
    }
    torch::Tensor forward(const torch::Tensor &node_feature, const torch::Tensor &edge_index,
                          const torch::Tensor &pos,
                          const torch::Tensor &new_node_indices, const torch::Tensor &raw_features,
                          const torch::Tensor &node_indices)
    {
        const auto &new_nodes_gpu = new_node_indices;
        const auto &pos_gpu = pos;
        const auto &node_feature_gpu = node_feature;
        const auto &edge_index_gpu = edge_index;
        const auto &raw_features_gpu = raw_features;
        const auto &node_indices_gpu = node_indices;

        auto transformed_feature = lin->forward(node_feature); // 使用线性层变换特征
        auto stand_feature1 = standardize_features(transformed_feature);
        // 异步脉冲卷积层
        torch::Tensor conv1_output = conv1->propagation(new_nodes_gpu, pos_gpu, stand_feature1,
                                                        edge_index_gpu, raw_features_gpu, node_indices_gpu,
                                                        batchSize);
        torch::Tensor stand_feature2 = standardize_conv_features(conv1_output);
        torch::Tensor bn_conv1_output = bn_conv1->forward(stand_feature2).to(torch::kFloat);
        torch::Tensor relu_output1 = torch::elu(bn_conv1_output).to(torch::kFloat);
        if (conv1->getSpikeNodes().numel() == 0)
        {
            torch::Tensor dropout_output = dropout->forward(relu_output1).to(torch::kFloat);
            torch::Tensor fc_output = fc->forward(dropout_output);
            return fc_output;
        }
        else
        {
            auto sorted = torch::sort(conv1->getSpikeNodes());
            torch::Tensor sorted_spike_nodes = std::get<0>(sorted);
            torch::Tensor transformed_feature2 = lin2->forward(relu_output1);
            auto stand_feature3 = standardize_features(transformed_feature2);
            torch::Tensor conv2_output = conv2->propagation(sorted_spike_nodes, pos_gpu, stand_feature3,
                                                            edge_index_gpu, raw_features_gpu, node_indices_gpu,
                                                            batchSize);
            auto stand_feature4 = standardize_conv_features(conv2_output);
            torch::Tensor bn_conv2_output = bn_conv2->forward(stand_feature4).to(torch::kFloat);
            torch::Tensor relu_output3 = torch::elu(bn_conv2_output).to(torch::kFloat);
            torch::Tensor dropout_output = dropout->forward(relu_output3).to(torch::kFloat);
            torch::Tensor fc_output = fc->forward(dropout_output);
            return fc_output;
        }
    }

    std::vector<torch::Tensor> layer_parameters() const
    {
        std::vector<torch::Tensor> result;
        for (const auto &pair : named_parameters())
        {
            if (pair.key().find("conv1.") == 0 || pair.key().find("conv2.") == 0)
            {
                continue;
            }
            result.push_back(pair.value());
        }
        return result;
    }

    std::vector<torch::Tensor> conv_parameters() const
    {
        std::vector<torch::Tensor> result;
        for (const auto &pair : named_parameters())
        {
            if (pair.key().find("conv1.") == 0 || pair.key().find("conv2.") == 0)
            {
                result.push_back(pair.value());
            }
        }
        return result;
    }

    std::vector<torch::Tensor> parameters() const
    {
        std::vector<torch::Tensor> result;
        for (const auto &pair : named_parameters())
        {
            result.push_back(pair.value());
        }
        return result;
    }

private:
    std::shared_ptr<AsynchronousConvLayer> conv1;
    std::shared_ptr<AsynchronousConvLayer> conv2;
    torch::nn::BatchNorm1d bn_conv1;
    torch::nn::BatchNorm1d bn_conv2;
    torch::nn::Dropout dropout;
    torch::nn::Linear lin;
    torch::nn::Linear lin2;
    torch::nn::Linear fc;
    torch::Tensor edge_weights1;
    torch::Tensor edge_weights2;
    int batchSize;
    int time_step_init;
    double max_rate_init;
    double v_rest1;
    double v_r1;
    double v_rest2;
    double v_r2;
    int sample_num1;
    int sample_num2;
    double dropout_rate;
    double speed;
    double threshold_init1;
    double threshold_init2;
    double refractory_period_init;
    double decay_alpha;
    double beta_init;
    int input_features;
    int transformed_features;
    int numClass;

public:
    torch::Tensor x;
};

#endif // SPIKE_ASYNCHRONOUSGRAPHSPINET_H
