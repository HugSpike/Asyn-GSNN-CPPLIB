//
// Created by fengfan on 06/01/2024.
//

#include "AsynchronousConvLayer.h"

torch::Tensor
AsynchronousConvLayer::structured_node_features(const torch::Tensor &node_indices, const torch::Tensor &edge_index,
                                                const torch::Tensor &raw_features)
{
    int64_t num_nodes = raw_features.size(0);
    std::vector<std::vector<int>> adjacency_list(num_nodes); // 使用vector代替unordered_map

#pragma omp parallel for default(none) shared(edge_index, adjacency_list)
    for (int64_t i = 0; i < edge_index.size(1); ++i)
    {
        int src = edge_index[0][i].item<int>();
        int tgt = edge_index[1][i].item<int>();
#pragma omp critical // 保证线程安全
        {
            adjacency_list[src].push_back(tgt);
        }
    }
    std::vector<torch::Tensor> enhanced_features;
    enhanced_features.reserve(node_indices.size(0));
#pragma omp parallel default(none) shared(adjacency_list, enhanced_features, node_indices, raw_features, std::cout)
    {
        // 使用局部变量收集每个线程的结果
        std::vector<torch::Tensor> local_enhanced_features;
#pragma omp for nowait // 确保循环没有隐式的同步
        for (int64_t i = 0; i < node_indices.size(0); ++i)
        {
            int node_idx = node_indices[i].item<int>();
            const auto &neighbors = adjacency_list[node_idx];
            double total_spikes = this->spike_num[node_idx];
            auto node_potential = this->final_potential[node_idx];
            auto total_potential = torch::tensor({0});
            for (int neighbor : neighbors)
            {
                total_spikes = total_spikes + this->spike_num[neighbor];
                total_potential = total_potential + this->final_potential[neighbor];
            }

            int64_t num_neighbors = neighbors.size();
            torch::Tensor all_spikes_tensor = torch::tensor({total_spikes});
            torch::Tensor avg_potential = num_neighbors > 0 ? total_potential / num_neighbors : torch::tensor({0});
            torch::Tensor avg_spikes = num_neighbors > 0 ? all_spikes_tensor / num_neighbors : torch::tensor({0});

            torch::Tensor node_feature = torch::stack({avg_spikes.unsqueeze(0), torch::tensor({this->first_arrival_time[node_idx]}).unsqueeze(0),
                                                       avg_potential.unsqueeze(0), node_potential.unsqueeze(0),
                                                       torch::tensor({this->last_spike_time[node_idx]}).unsqueeze(0), this->membrane_potential[node_idx].unsqueeze(0)},
                                                      0);
            local_enhanced_features.push_back(node_feature.transpose(0, 1));
        }
// 合并局部结果到全局vector
#pragma omp critical
        enhanced_features.insert(enhanced_features.end(), local_enhanced_features.begin(),
                                 local_enhanced_features.end());
    }
    torch::Tensor final = torch::cat(enhanced_features, 0).to(torch::kFloat);
    return final.squeeze(2);
}

std::unordered_map<std::pair<int, int>, torch::Tensor> AsynchronousConvLayer::message(
    const torch::Tensor &node_feature,
    const torch::Tensor &new_node_indices,
    const torch::Tensor &edge_index,
    int num_sample)
{
    torch::Tensor real_spike = rate(node_feature, this->time_step.item<int>());
    std::unordered_map<std::pair<int, int>, torch::Tensor> input_spikes;
    std::unordered_map<std::pair<int, int>, int> edge_dict;

    auto edge_index_t = edge_index.t();
    for (int i = 0; i < edge_index_t.size(0); ++i)
    {
        auto edge = edge_index_t[i];
        edge_dict[{edge[0].item<int>(), edge[1].item<int>()}] = i;
    }
    for (int i = 0; i < new_node_indices.size(0); ++i)
    {
        auto node_index = new_node_indices[i].item<int>();
        auto sampled_targets = get_sampled_target_nodes(node_index, edge_index, num_sample, this->weights);

        if (sampled_targets.numel() == 0)
        {
            continue;
        }
        for (int j = 0; j < sampled_targets.size(0); ++j)
        {
            auto spike = real_spike.index({torch::indexing::Slice(), sampled_targets[j].item<int>(), torch::indexing::Slice()});
            std::pair<int, int> sampled_edge_index = {node_index, sampled_targets[j].item<int>()};
            auto input_signal = spike;
            input_spikes[sampled_edge_index] = SpikeFunction::apply(input_signal,
                                                                    node_feature[sampled_targets[j].item<int>()],
                                                                    this->beta);
        }
    }
    return input_spikes;
}

torch::Tensor AsynchronousConvLayer::aggregate(int node_index,
                                               const std::vector<std::pair<double, torch::Tensor>> &node_spike_with_time,
                                               double timestamp,
                                               const torch::Tensor &node_feature)
{
    auto device = node_feature.device();
    {
        std::lock_guard<std::mutex> lock2(this->lock_last_spike_time);
        std::lock_guard<std::mutex> lock3(this->lock_refractory_period);
        this->refractory_period_begin[node_index] = this->last_spike_time[node_index];
        this->refractory_period_end[node_index] = (this->last_spike_time[node_index] != 0) ? (this->last_spike_time[node_index] +
                                                                                              this->refractory_period.item<double>())
                                                                                           : 0;
    }
    auto sorted_spike_iter = node_spike_with_time.begin();
    if (sorted_spike_iter == node_spike_with_time.end())
    {
        std::cerr << "Error: Spike iterator is at the end." << std::endl;
        return torch::empty({0}, device);
    }

    double arrival_time;
    if (last_spike_time[node_index] == 0)
    {
        arrival_time = sorted_spike_iter->first + timestamp;
        {
            std::lock_guard<std::mutex> lock5(this->lock_first_arrival_time);
            this->first_arrival_time[node_index] = arrival_time;
        }
    }
    else
    {
        arrival_time = sorted_spike_iter->first + timestamp;
    }
    {
        std::lock_guard<std::mutex> lock_potential(this->potential_mutex);

        // 遍历 map 并打印每个元素
        this->membrane_potential[node_index] = this->membrane_potential[node_index].to(device);
        this->alpha = this->alpha.to(device);
        this->v_threshold = this->v_threshold.to(device);
        this->beta = this->beta.to(device);
        torch::Tensor node_index_feature = node_feature[node_index].to(device);
        // 分离 node_spike_with_time 中的时间戳和 spike 张量
        std::vector<double> spike_times_vector;
        std::vector<torch::Tensor> spikes_vector;

        for (const auto &entry : node_spike_with_time)
        {
            spike_times_vector.push_back(entry.first);
            spikes_vector.push_back(entry.second.to(device)); // 确保张量在正确的设备上
        }
        // 将时间戳转换为 Tensor
        auto spike_times_tensor = torch::tensor(spike_times_vector).to(device);

        // 将 spikes 向量堆叠为一个 Tensor
        auto stacked_spikes_tensor = torch::stack(spikes_vector);

        // 使用分离后的时间戳和 spike 调用 MembranePotentialFunction
        auto new_membrane_potential = MembranePotentialFunction::apply(
            this->membrane_potential[node_index],   // membrane_potential
            this->alpha,                            // alpha
            this->v_threshold,                      // v_threshold
            this->v_rest,                           // v_rest
            stacked_spikes_tensor,                  // spikes (vector of tensors)
            spike_times_tensor,                     // spike_times (vector of doubles)
            timestamp,                              // arrival_time
            this->refractory_period_end[node_index] // refractory_period_end
        );
        {
            std::lock_guard<std::mutex> lock_final(this->final_potential_mutex);
            this->final_potential[node_index] = torch::relu(new_membrane_potential - this->v_threshold);
        }
        if (this->membrane_potential[node_index].item<double>() >= this->v_threshold.item<double>())
        {
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            {
                std::lock_guard<std::mutex> lock2(lock_last_spike_time);
                this->last_spike_time[node_index] = static_cast<double>(now_time_t);
            }
            {
                std::lock_guard<std::mutex> lock_flag(this->flag_mutex);
                this->v_flag[node_index] += 1;
            }

            {
                std::lock_guard<std::mutex> lock_flag(this->flag_mutex);
                this->membrane_potential[node_index] = torch::tensor({this->v_r}).to(device);
            }
            {
                std::lock_guard<std::mutex> lock7(this->lock_refractory_period);
                this->refractory_period_begin[node_index] = arrival_time;
                this->refractory_period_end[node_index] = arrival_time + this->refractory_period.item<double>();
            }
            {
                std::lock_guard<std::mutex> lock8(this->lock_spike_num);
                std::lock_guard<std::mutex> lock_flag(this->flag_mutex);
                this->spike_num[node_index] += 1;
            }
            return torch::tensor({node_index}).to(device);
        }
        else
        {
            return torch::empty({0}, device);
        }
    }
}

torch::Tensor
AsynchronousConvLayer::propagation(const torch::Tensor &spike_nodes,
                                   const torch::Tensor &pos,
                                   const torch::Tensor &node_feature,
                                   const torch::Tensor &edge_index, const torch::Tensor &raw_features,
                                   const torch::Tensor &node_indices, int batchSize)
{
    torch::Tensor new_nodes_copy = spike_nodes.clone();
    std::vector<torch::Tensor> spiking_nodes_indices;
    auto spike_dic = message(node_feature, new_nodes_copy, edge_index, sample_num);
    auto pair = concatenate_spikes_by_distance(spike_dic, pos, v.item<double>());
    std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>> spike_with_time = pair.first;
    std::vector<std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>>> batches;
    auto batch_count = (spike_with_time.size() + batchSize - 1) / batchSize;
    auto it = spike_with_time.begin();
    for (int i = 0; i < batch_count; ++i)
    {
        std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>> batch;
        for (int j = 0; j < batchSize && it != spike_with_time.end(); ++j, ++it)
        {
            batch.insert(*it);
        }
        batches.emplace_back(batch);
    }
    for (auto &item : batches)
    {
        auto current_batch_result = process_node_batch(item, node_feature);
        spiking_nodes_indices.push_back(current_batch_result);
    }
    torch::Tensor all_spiking_nodes;
    if (!spiking_nodes_indices.empty())
    {
        all_spiking_nodes = torch::cat(spiking_nodes_indices);
        all_spiking_nodes = unique_tensor(all_spiking_nodes);
    }
    this->spike_nodes = all_spiking_nodes;
    torch::Tensor combined_tensor = structured_node_features(node_indices, edge_index, raw_features);
    return combined_tensor;
}

torch::Tensor AsynchronousConvLayer::process_node_batch(
    std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>> &spike_with_time,
    const torch::Tensor &node_feature)
{
    auto device = node_feature.device();
    std::size_t pool_size = spike_with_time.size();
    ThreadPool pool(pool_size);
    std::vector<std::future<torch::Tensor>> results;
    for (const auto &pair : spike_with_time)
    {
        int node_index = pair.first;
        auto node_spike_with_time = pair.second;

        results.emplace_back(
            pool.enqueue(
                [this, node_index, node_spike_with_time, node_feature, &device]()
                {
                    auto now = std::chrono::system_clock::now();
                    auto now_time_t = std::chrono::system_clock::to_time_t(now);
                    auto timestamp = static_cast<double>(now_time_t);
                    auto aggregated_result = this->aggregate(node_index, node_spike_with_time, timestamp,
                                                             node_feature)
                                                 .to(device);
                    if (!aggregated_result.defined() || aggregated_result.numel() == 0)
                    {
                        return torch::empty({0}, torch::device(device));
                    }
                    return aggregated_result;
                }));
    }
    std::vector<torch::Tensor> tensors;
    for (auto &result : results)
    {
        try
        {
            torch::Tensor tensor = result.get();
            if (!tensor.defined())
            {
                std::cerr << "Error: Received an undefined tensor from a future." << std::endl;
                continue;
            }
            tensors.push_back(tensor);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in processing results: " << e.what() << std::endl;
        }
    }

    if (tensors.empty())
    {
        std::cerr << "Warning: tensors vector is empty!" << std::endl;
        return torch::empty({0}, device);
    }
    return torch::cat(tensors).to(device);
}

void AsynchronousConvLayer::init_param(const torch::Tensor &node_indices)
{

    for (int index = 0; index < node_indices.size(0); ++index)
    {
        this->spike_num[index] = 0;
        this->v_flag[index] = 0;
        this->first_arrival_time[index] = 0.0;
        this->last_spike_time[index] = 0.0;
        this->refractory_period_begin[index] = 0.0;
        this->refractory_period_end[index] = 0.0;
        this->membrane_potential[index] = torch::tensor({v_r}, torch::requires_grad(true));
        this->final_potential[index] = torch::tensor({0.0}, torch::requires_grad(true));
    }
}

void AsynchronousConvLayer::reset_param(std::shared_ptr<AsynchronousConvLayer> source, const torch::Tensor &node_indices)
{
    this->spike_num = source->spike_num;
    this->v_flag = source->v_flag;
    for (int i = 0; i < node_indices.size(0); ++i)
    {
        this->first_arrival_time[i] = 0.0;
        this->last_spike_time[i] = 0.0;
        this->refractory_period_begin[i] = 0.0;
        this->refractory_period_end[i] = 0.0;
        this->membrane_potential[i] = torch::tensor({v_r}, torch::requires_grad(true));
        this->final_potential[i] = torch::tensor({0.0}, torch::requires_grad(true));
    }
}

const torch::Tensor &AsynchronousConvLayer::getSpikeNodes() const
{
    return spike_nodes;
}

const torch::Tensor &AsynchronousConvLayer::getTimeStep() const
{
    return time_step;
}

const torch::Tensor &AsynchronousConvLayer::getMaxRate() const
{
    return max_rate;
}

const torch::Tensor &AsynchronousConvLayer::getBeta() const
{
    return beta;
}

const std::unordered_map<int, int> &AsynchronousConvLayer::getSpikeNum() const
{
    return spike_num;
}

const torch::Tensor &AsynchronousConvLayer::getVThreshold() const
{
    return v_threshold;
}
