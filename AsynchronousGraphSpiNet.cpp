//
// Created by fengfan on 09/01/2024.
//

#include "AsynchronousGraphSpiNet.h"

const std::shared_ptr<AsynchronousConvLayer> &AsynchronousGraphSpiNet::getConv1() const {
    return conv1;
}

const std::shared_ptr<AsynchronousConvLayer> &AsynchronousGraphSpiNet::getConv2() const {
    return conv2;
}


std::unordered_map<std::string, torch::Tensor> AsynchronousGraphSpiNet::get_state_dict() {
    std::unordered_map<std::string, torch::Tensor> stateDict;
    for (const auto &p: named_parameters()) {
        stateDict[p.key()] = p.value();
    }
    return stateDict;
}

void AsynchronousGraphSpiNet::load_state_dict(std::unordered_map<std::string, torch::Tensor> state_dict) {
    for (const auto &item: this->named_parameters()) {
        auto &name = item.key();
        auto &tensor = item.value();
        if (state_dict.find(name) != state_dict.end()) {
            tensor.copy_(state_dict.at(name));
        }
    }
}

void AsynchronousGraphSpiNet::print_grad() {
    for (auto &param: this->named_parameters()) {
        if (param.value().grad().defined()) {
            std::cout << "Param: " << param.key() << std::endl;
            std::cout << "Grad: " << param.value().grad() << std::endl;
        } else {
            std::cout << "Param: " << param.key() << " has no grad." << std::endl;
        }
    }
}

PYBIND11_MODULE(asynchronous_graph_spi_cpu, m)
{
    py::class_<AsynchronousGraphSpiNet, torch::nn::Module, std::shared_ptr<AsynchronousGraphSpiNet>>(m, "AsynchronousGraphSpiNet")
        .def(py::init<double, double, double, double, int, int,
                      double, double, double, double, double, double, int, int, double, double, int, int, int, torch::Tensor, torch::Tensor>())
        .def("to",
             static_cast<void (AsynchronousGraphSpiNet::*)(torch::Device, bool)>(&AsynchronousGraphSpiNet::to),
             py::arg("device"), py::arg("non_blocking") = false)
        .def("forward", &AsynchronousGraphSpiNet::forward)
        .def("layer_parameters", &AsynchronousGraphSpiNet::layer_parameters)
        .def("conv_parameters", &AsynchronousGraphSpiNet::conv_parameters)
        .def("parameters", &AsynchronousGraphSpiNet::parameters)
        .def("train", &AsynchronousGraphSpiNet::train)
        .def("eval", &AsynchronousGraphSpiNet::eval)
        .def("get_conv1", &AsynchronousGraphSpiNet::getConv1)
        .def("get_conv2", &AsynchronousGraphSpiNet::getConv2)
        .def("get_dropout", &AsynchronousGraphSpiNet::getDropout)
        .def("get_bn_conv1", &AsynchronousGraphSpiNet::getBN1)
        .def("get_bn_conv2", &AsynchronousGraphSpiNet::getBN2)
        .def("state_dict", &AsynchronousGraphSpiNet::get_state_dict)
        .def("load_state_dict", &AsynchronousGraphSpiNet::load_state_dict)
        .def("print_grad", &AsynchronousGraphSpiNet::print_grad);
    py::class_<AsynchronousConvLayer, std::shared_ptr<AsynchronousConvLayer>>(m, "AsynchronousConvLayer")
        .def(py::init<double, double, int, double, double, double, double, int, double, double, torch::Tensor>())
        .def("reset_param", &AsynchronousConvLayer::reset_param)
        .def("init_param", &AsynchronousConvLayer::init_param)
        .def("get_threshold", &AsynchronousConvLayer::getVThreshold);
}