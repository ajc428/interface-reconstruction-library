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
        int size;
        int out;
        int depth;
        model(int s, int o, int d) 
        {
            size = s;
            out = o;
            depth = d;
            l1 = register_module("l1", torch::nn::Linear(size, 100));
            for (int i = 0; i < d; ++i)
            {
                layers.push_back(register_module("l" + std::to_string(i+2), torch::nn::Linear(100, 100)));
            }
            l4 = register_module("l" + std::to_string(d+2), torch::nn::Linear(100, out));
        }
        torch::Tensor forward(torch::Tensor x) 
        {
            x = torch::nn::functional::relu(l1(x));
            for (int i = 0; i < depth; ++i)
            {
                x = torch::nn::functional::relu(layers[i](x));
            }
            x = l4(x);
            return x;
        }
        std::vector<torch::nn::Linear> layers;
        torch::nn::Linear l1{nullptr}, l4{nullptr};

        int getSize()
        {
            return size;
        }

        int getOutput()
        {
            return out;
        }
    };
}