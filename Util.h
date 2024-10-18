//
// Created by fengfan on 23/03/2024.
//

#ifndef SPIKE_UTIL_H
#define SPIKE_UTIL_H

#include <unordered_map>
#include <torch/torch.h>
#include "AsynchronousConvLayer.h"

namespace std
{
    template <>
    struct hash<std::pair<int, int>>
    {
        size_t operator()(const std::pair<int, int> &p) const
            noexcept
        {
            auto hash1 = std::hash<int>{}(p.first);
            auto hash2 = std::hash<int>{}(p.second);
            return hash1 ^ (hash2 << 1);
        }
    };
}

struct EdgeData
{
    int src_index;
    int tgt_index;
    torch::Tensor distance;
    torch::Tensor spike;
};

torch::Tensor unique_tensor(const torch::Tensor &input);

torch::Tensor get_sampled_target_nodes(int node_index,
                                       const torch::Tensor &edge_index,
                                       int num_sample,
                                       const torch::Tensor &edge_weights);

torch::Tensor calculate_edge_length_torch(int src_index, int tgt_index,
                                          const torch::Tensor &node_positions);

std::pair<std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>>, std::vector<std::pair<double, int>>>
concatenate_spikes_by_distance(
    const std::unordered_map<std::pair<int, int>, torch::Tensor> &spi_dic,
    const torch::Tensor &node_positions, double transmission_speed);

torch::Tensor standardize_features(const torch::Tensor &input);

torch::Tensor standardize_conv_features(torch::Tensor input);
#endif // SPIKE_UTIL_H
