//
// Created by fengfan on 23/03/2024.
//

#include "Util.h"


torch::Tensor unique_tensor(const torch::Tensor &input) {
    std::unordered_set<int> seen;
    std::vector<int> unique;

    for (int i = 0; i < input.size(0); ++i) {
        auto element = input[i].item<int>();
        if (seen.find(element) == seen.end()) {
            seen.insert(element);
            unique.emplace_back(element);
        }
    }

    return torch::tensor(unique);
}

torch::Tensor get_sampled_target_nodes(int node_index,
                                       const torch::Tensor &edge_index,
                                       int num_sample,
                                       const torch::Tensor &edge_weights) {
    torch::Tensor mask = edge_index.select(0, 0) == node_index;
    torch::Tensor neighbor_nodes_unique = unique_tensor(torch::masked_select(edge_index.select(0, 1), mask));
    torch::Tensor neighbor_weights = torch::masked_select(edge_weights, mask);
    auto num_neighbors = neighbor_nodes_unique.numel();

    if (num_neighbors == 0) {
        return torch::tensor({});
    }

    if (num_neighbors <= num_sample) {
        return neighbor_nodes_unique;
    }

    torch::Tensor total_weight = torch::sum(neighbor_weights);
    if (total_weight.item<double>() == 0) {
        return torch::tensor({});
    }
    torch::Tensor probabilities = neighbor_weights / total_weight;
    if ((probabilities < 0).any().item<bool>() || torch::isnan(probabilities).any().item<bool>() ||
        torch::isinf(probabilities).any().item<bool>()) {
        return torch::tensor({});
    }
    if (std::abs(1.0 - probabilities.sum().item<double>()) > 1e-6) {
        throw std::runtime_error("Sum of probabilities is not approximately 1.");
    }
    torch::Tensor sampled_indices = torch::multinomial(probabilities, num_sample, false);
    torch::Tensor sampled_nodes = torch::index_select(neighbor_nodes_unique, 0, sampled_indices);
    return sampled_nodes;
}

torch::Tensor calculate_edge_length_torch(int src_index, int tgt_index,
                                          const torch::Tensor &node_positions) {
    const auto &start_positions = node_positions[src_index];
    const auto &end_positions = node_positions[tgt_index];
    torch::Tensor distance = torch::norm(start_positions - end_positions, 2);

    return distance;
}

std::pair<std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>>, std::vector<std::pair<double, int>>>
concatenate_spikes_by_distance(
        const std::unordered_map<std::pair<int, int>, torch::Tensor> &spi_dic,
        const torch::Tensor &node_positions, double transmission_speed) {

    std::unordered_map<int, std::vector<EdgeData>> edge_data_map;
    std::vector<std::pair<double, int>> node_with_time;

    for (const auto &item: spi_dic) {
        int src_index = item.first.first;
        int tgt_index = item.first.second;

        // // Debug: 打印grad_fn以确保梯度依赖没有丢失
        // if (item.second.grad_fn()) {
        //     std::cout << "Grad Function for item.second: " << item.second.grad_fn()->name() << std::endl;
        // } else {
        //     std::cerr << "item.second (input_spikes) has no gradient function!" << std::endl;
        // }

        torch::Tensor distance = calculate_edge_length_torch(src_index, tgt_index, node_positions);
        edge_data_map[tgt_index].emplace_back(EdgeData{src_index, tgt_index, distance, item.second});
    }

    std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>> result;

    for (auto &item: edge_data_map) {
        int tgt_index = item.first;
        auto &edges = item.second;

        std::vector<std::pair<double, torch::Tensor>> sorted_spikes;

        for (const auto &edge: edges) {
            double time = edge.distance.item<double>() / transmission_speed;
            sorted_spikes.emplace_back(time, edge.spike);
            node_with_time.emplace_back(time, edge.src_index);
        }

        std::sort(sorted_spikes.begin(), sorted_spikes.end(),
                  [](const std::pair<double, torch::Tensor> &a,
                     const std::pair<double, torch::Tensor> &b) {
                      return a.first < b.first;
                  });

        result[tgt_index] = sorted_spikes;
    }

    std::sort(node_with_time.begin(), node_with_time.end(),
              [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                  return a.first < b.first;
              });

    return std::make_pair(result, node_with_time);
}

torch::Tensor standardize_features(const torch::Tensor &input){
     // 计算每个特征的最大值和最小值
    auto min = input.min();
    auto max = input.max();
    // 最小-最大归一化
    torch::Tensor normalized = (input - min) / (max - min + 1e-6); // 防止除以零
    return normalized;
}

torch::Tensor standardize_conv_features(torch::Tensor input) {
    auto t = input.select(1, 0);
    auto position_x = input.select(1, 1);
    auto position_y = input.select(1, 2);
    auto polarity = input.select(1, 3);
    auto polarityy = input.select(1, 4);
    auto polarityx = input.select(1, 5);

    //对时间进行归一化
    torch::Tensor t_min = t.min();
    torch::Tensor t_max = t.max();
    if (t_min.item<double>() == t_max.item<double>()) {
        t = torch::zeros_like(t);
    } else {
        t = (t - t_min) / (t_max - t_min);
    }

    // 对横坐标进行归一化
    torch::Tensor x_min = position_x.min();
    torch::Tensor x_max = position_x.max();
    if (x_min.item<double>() == x_max.item<double>()) {
        position_x = torch::zeros_like(position_x);
    } else {
        position_x = (position_x - x_min) / (x_max - x_min);
    }

    //对纵坐标进行归一化
    torch::Tensor y_min = position_y.min();
    torch::Tensor y_max = position_y.max();
    if (y_min.item<double>() == y_max.item<double>()) {
        position_y = torch::zeros_like(position_y);
    } else {
        position_y = (position_y - y_min) / (y_max - y_min);
    }

        //对纵坐标进行归一化
    torch::Tensor p_min = polarity.min();
    torch::Tensor p_max = polarity.max();
    if (p_min.item<double>() == p_max.item<double>()) {
        polarity = torch::zeros_like(polarity);
    } else {
        polarity = (polarity- p_min) / (p_max - p_min);
    }

            //对纵坐标进行归一化
    torch::Tensor py_min = polarityy.min();
    torch::Tensor py_max = polarityy.max();
    if (py_min.item<double>() == py_max.item<double>()) {
        polarityy = torch::zeros_like(polarityy);
    } else {
        polarityy = (polarityy- py_min) / (py_max - py_min);
    }

    //对纵坐标进行归一化
    torch::Tensor px_min = polarityx.min();
    torch::Tensor px_max = polarityx.max();
    if (py_min.item<double>() == py_max.item<double>()) {
        polarityx = torch::zeros_like(polarityx);
    } else {
        polarityx = (polarityx- px_min) / (px_max - px_min);
    }

    //拼接特征
    torch::Tensor output = torch::stack(
            {t, position_x,
             position_y, polarity,polarityy,polarityx},
            1).to(torch::kFloat);

    return output;
}

