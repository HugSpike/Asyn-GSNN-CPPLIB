//
// Created by fengfan on 04/04/2024.
//

#ifndef SPIKE_SPIKEGEN_H
#define SPIKE_SPIKEGEN_H

#include "torch/torch.h"

torch::Tensor rate_conv(torch::Tensor data);

torch::Tensor rate(
        torch::Tensor data,
        int num_steps = 0,
        float gain = 1.0,
        float offset = 0.0,
        int first_spike_time = 0,
        bool time_var_input = false
);

#endif //SPIKE_SPIKEGEN_H
