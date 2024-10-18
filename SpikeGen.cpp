#include <torch/torch.h>
#include "SpikeGen.h"

torch::Tensor rate_conv(torch::Tensor data) {
    torch::Tensor clipped_data = torch::clamp(data, 0, 1);

    torch::Tensor spike_data = torch::bernoulli(clipped_data);

    return spike_data;
}

torch::Tensor rate(
        torch::Tensor data,
        int num_steps,
        float gain,
        float offset,
        int first_spike_time,
        bool time_var_input
) {
    if (first_spike_time < 0 || num_steps < 0) {
        throw std::invalid_argument("first_spike_time and num_steps cannot be negative.");
    }

    if (!time_var_input && first_spike_time > num_steps - 1) {
        throw std::invalid_argument("first_spike_time must be less than num_steps.");
    }

    if (time_var_input && num_steps > 0) {
        throw std::invalid_argument("num_steps should not be specified if input is time-varying.");
    }

    torch::Tensor spike_data;

    if (time_var_input) {
        spike_data = rate_conv(data);
        if (first_spike_time > 0) {
            auto zeros = torch::zeros({first_spike_time, data.size(1), data.size(2)},
                                      data.options().dtype(torch::kFloat));
            spike_data = torch::cat({zeros, spike_data}, 0);
        }
    } else {
        if (num_steps > 0) {
            auto repeated_data = data.repeat({num_steps, 1, 1});
            repeated_data = repeated_data * gain + offset;
            spike_data = rate_conv(repeated_data);
            if (first_spike_time > 0) {
                spike_data.slice(0, 0, first_spike_time).fill_(0);
            }
        } else {
            throw std::invalid_argument("num_steps must be greater than 0 for time-static input data.");
        }
    }

    return spike_data;
}