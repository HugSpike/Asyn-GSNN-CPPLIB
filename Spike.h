//
// Created by fengfan on 31/03/2024.
//

#ifndef SPIKE_SPIKE_H
#define SPIKE_SPIKE_H

#include "torch/torch.h"

class SpikeFunction : public torch::autograd::Function<SpikeFunction> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &input_signal,
            const torch::Tensor &node_index_feature,
            const torch::Tensor &beta) {
        ctx->save_for_backward({node_index_feature, beta});
        return input_signal;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_output) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto beta = saved[1];
        auto sigmoid_input = torch::sigmoid(beta * input);
        auto grad = beta * sigmoid_input * (1 - sigmoid_input);
        auto grad_input = grad * grad_output[0];
        // 返回 input_signal 的梯度为传入的 grad_output[0]
        auto grad_input_signal = grad_output[0];

        return {grad_input_signal, grad_input, torch::Tensor()};
    }
};

#endif //SPIKE_SPIKE_H
