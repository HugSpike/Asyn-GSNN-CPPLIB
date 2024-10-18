//
// Created by fengfan on 09/01/2024.
//

#ifndef SPIKE_ASYNCHRONOUSCONVLAYER_H
#define SPIKE_ASYNCHRONOUSCONVLAYER_H

#include <torch/torch.h>
#include <mutex>
#include <unordered_map>
#include "Util.h"
#include "MembranePotential.h"
#include "ThreadPool.h"
#include "Spike.h"
#include "SpikeGen.h"



class AsynchronousConvLayer : public torch::nn::Module {
public:
    AsynchronousConvLayer(
            double v_rest,
            double v_r,
            int sample_num,
            double speed_init,
            double threshold_init,
            double refractory_period_init,
            double decay_alpha_init,
            int time_step_init,
            double max_rate_init,
            double beta_init,
            torch::Tensor edge_weights
    ) : v_rest(v_rest),
        v_r(v_r),
        sample_num(sample_num),
        speed_init(speed_init),
        threshold_init(threshold_init),
        refractory_period_init(refractory_period_init),
        decay_alpha_init(decay_alpha_init),
        time_step_init(time_step_init),
        max_rate_init(max_rate_init),
        beta_init(beta_init),
        edge_weights(edge_weights) {

    }

    std::unordered_map<std::pair<int, int>, torch::Tensor> message(
            const torch::Tensor &node_feature,
            const torch::Tensor &new_node_indices,
            const torch::Tensor &edge_index,
            int num_sample);



    torch::Tensor aggregate(int node_index,
                            const std::vector<std::pair<double, torch::Tensor>> &node_spike_with_time,
                            double timestamp,
                            const torch::Tensor &node_feature);

    torch::Tensor propagation(const torch::Tensor &spike_nodes,
                              const torch::Tensor &pos,
                              const torch::Tensor &node_feature,
                              const torch::Tensor &edge_index,
                              const torch::Tensor &raw_features,
                              const torch::Tensor &node_indices,
                              int batchSize);

    torch::Tensor process_node_batch(
            std::unordered_map<int, std::vector<std::pair<double, torch::Tensor>>> &spike_with_time,
            const torch::Tensor &node_feature);
    
    void init_param(const torch::Tensor &node_indices);

    void reset_param(std::shared_ptr<AsynchronousConvLayer> source,const torch::Tensor &node_indices);

    torch::Tensor structured_node_features(const torch::Tensor &node_indices, const torch::Tensor &edge_index,
                                           const torch::Tensor &raw_features);


private:
    double v_rest;
    double v_r;
    int sample_num;
    torch::Tensor edge_weights;
    double speed_init;
    double threshold_init;
    double refractory_period_init;
    double decay_alpha_init;
    int time_step_init;
    double max_rate_init;
    double beta_init;

    torch::Tensor v = torch::tensor({speed_init});
    torch::Tensor alpha = torch::tensor({decay_alpha_init});
    torch::Tensor refractory_period = torch::tensor({refractory_period_init});
    torch::Tensor time_step = torch::tensor({time_step_init});
public:
    const torch::Tensor &getTimeStep() const;

private:
    torch::Tensor max_rate = torch::tensor({max_rate_init});
public:
    const torch::Tensor &getMaxRate() const;

private:
    torch::Tensor beta = torch::tensor({beta_init});
public:
    const torch::Tensor &getBeta() const;

private:
    torch::Tensor v_threshold = register_parameter("v_threshold", torch::tensor({threshold_init}, torch::requires_grad(
            true)));
            
    torch::Tensor weights = edge_weights;
public:
    const torch::Tensor &getVThreshold() const;

public:
    std::unordered_map<int, double> last_spike_time;
    std::unordered_map<int, double> refractory_period_begin;
    std::unordered_map<int, double> refractory_period_end;
    std::unordered_map<int, int> spike_num;
public:
    const std::unordered_map<int, int> &getSpikeNum() const;

public:
    std::unordered_map<int, double> first_arrival_time;
    std::unordered_map<int, int> v_flag;
    std::unordered_map<int, torch::Tensor> membrane_potential;
    std::unordered_map<int, torch::Tensor> final_potential;
    torch::Tensor spike_nodes;
public:
    const torch::Tensor &getSpikeNodes() const;

private:
    std::mutex lock_last_spike_time;
    std::mutex lock_refractory_period;
    std::mutex lock_spike_num;
    std::mutex lock_first_arrival_time;
    std::mutex flag_mutex;
    std::mutex potential_mutex;
    std::mutex final_potential_mutex;
};

#endif //SPIKE_ASYNCHRONOUSCONVLAYER_H
