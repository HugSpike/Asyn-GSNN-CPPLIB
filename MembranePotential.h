//
// Created by fengfan on 23/03/2024.
//

#ifndef SPIKE_MEMBRANEPOTENTIAL_H
#define SPIKE_MEMBRANEPOTENTIAL_H

#include <torch/torch.h>

class MembranePotentialFunction : public torch::autograd::Function<MembranePotentialFunction>
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 const torch::Tensor &membrane_potential,
                                 const torch::Tensor &alpha,
                                 const torch::Tensor &v_threshold,
                                 double v_rest,
                                 const torch::Tensor &stacked_spikes, // 堆叠的 spikes
                                 const torch::Tensor &spike_times,    // 堆叠的 spike_times 也作为 Tensor
                                 double arrival_time,
                                 double refractory_period_end)
    {
        auto new_membrane_potential = membrane_potential.clone();
        std::vector<torch::Tensor> potentials;

        // 保存需要在 backward 中使用的变量
        ctx->save_for_backward({membrane_potential, alpha, v_threshold, stacked_spikes});
        // While 循环移至 forward 内部
        for (int64_t i = 0; i < stacked_spikes.size(0); i++)
        {
            auto spike_time = spike_times.select(0, i).squeeze(0).item<double>();
            if (new_membrane_potential.item<double>() >= v_threshold.item<double>() ||
                spike_time + arrival_time <= refractory_period_end)
            {
                break;
            }

            // 使用 select 获取 stacked_spikes[i]
            auto spike = stacked_spikes.select(0, i);
            new_membrane_potential =
                alpha * (new_membrane_potential - v_rest) + v_rest + spike.norm(2);

            potentials.push_back(new_membrane_potential);
        }

        // 保存所有的中间值用于 backward
        ctx->saved_data["potentials"] = potentials;

        return new_membrane_potential;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_output)
    {
        auto saved = ctx->get_saved_variables();
        auto membrane_potential = saved[0];
        auto alpha = saved[1];
        auto v_threshold = saved[2];
        auto stacked_spikes = saved[3];

        auto potentials = ctx->saved_data["potentials"].toTensorVector();
        auto grad_membrane_potential = torch::zeros_like(membrane_potential);
        auto grad_stacked_spikes = torch::zeros_like(stacked_spikes);

        // 遍历保存的 potentials 和 stacked_spikes，累积计算梯度
        for (int64_t i = 0; i < potentials.size(); i++)
        {
            auto spike = stacked_spikes.select(0, i);
            // 计算针对 spike 参数的梯度
            auto grad_spike = grad_output[0] * spike / spike.norm(2);
            grad_stacked_spikes = grad_stacked_spikes + grad_spike;
            grad_membrane_potential = grad_membrane_potential + grad_output[0] * alpha;
        }

        // 返回与 forward 函数中的参数顺序和数量相匹配的梯度
        return {
            grad_membrane_potential, // 对应 membrane_potential
            torch::Tensor(),         // 对应 alpha (不计算梯度)
            torch::Tensor(),         // 对应 v_threshold (不计算梯度)
            torch::Tensor(),         // 对应 v_rest (double 类型，不计算梯度)
            grad_stacked_spikes,     // 对应 stacked_spikes
            torch::Tensor(),         // 对应 spike_times (不计算梯度)
            torch::Tensor(),         // 对应 arrival_time (不计算梯度)
            torch::Tensor()          // 对应 refractory_period_end (不计算梯度)
        };
    }
};

#endif // SPIKE_MEMBRANEPOTENTIAL_H
