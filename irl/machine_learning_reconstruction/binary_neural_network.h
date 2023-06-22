// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <torch/torch.h>

namespace IRL 
{
    struct binary_model : torch::nn::Module 
    {
        int size;
        binary_model(int s) 
        {
            size = s;
            l1 = register_module("l1", torch::nn::Linear(size, 100));
            l2 = register_module("l2", torch::nn::Linear(100, 100));
            l3 = register_module("l3", torch::nn::Linear(100, 100));
            l4 = register_module("l4", torch::nn::Linear(100, 100));
            l5 = register_module("l5", torch::nn::Linear(100, 100));
            l6 = register_module("l6", torch::nn::Linear(100, 100));
            l7 = register_module("l7", torch::nn::Linear(100, 100));
            l8 = register_module("l8", torch::nn::Linear(100, 100));
            l9 = register_module("l9", torch::nn::Linear(100, 100));
            l10 = register_module("l10", torch::nn::Linear(100, 1));
        }
        torch::Tensor forward(torch::Tensor x) 
        {
            x = torch::nn::functional::relu(l1(x));
            x = torch::nn::functional::relu(l2(x));
            x = torch::nn::functional::relu(l3(x));
            x = torch::nn::functional::relu(l4(x));
            x = torch::nn::functional::relu(l5(x));
            x = torch::nn::functional::relu(l6(x));
            x = torch::nn::functional::relu(l7(x));
            x = torch::nn::functional::relu(l8(x));
            x = torch::nn::functional::relu(l9(x));
            x = torch::sigmoid(l10(x));
            return x;
        }
        torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, l5{nullptr}, l6{nullptr}, l7{nullptr}, l8{nullptr}, l9{nullptr}, l10{nullptr};

        int getSize()
        {
            return size;
        }
    };
}