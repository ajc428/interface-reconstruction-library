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
    struct model : torch::nn::Module 
    {
        model() 
        {
            l1 = register_module("l1", torch::nn::Linear(108, 100));
            l2 = register_module("l2", torch::nn::Linear(100, 100));
            l3 = register_module("l3", torch::nn::Linear(100, 100));
            l4 = register_module("l4", torch::nn::Linear(100, 8));
        }
        torch::Tensor forward(torch::Tensor x) 
        {
            x = torch::nn::functional::relu(l1(x));
            x = torch::nn::functional::relu(l2(x));
            x = torch::nn::functional::relu(l3(x));
            x = l4(x);
            return x;
        }
        torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr};
    };
}